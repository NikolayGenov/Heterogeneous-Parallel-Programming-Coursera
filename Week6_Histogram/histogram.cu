// Histogram Equalization
#include <wb.h>

#define wbCheck(stmt)                                                      \
    do {                                                                   \
        cudaError_t err = stmt;                                            \
        if (err != cudaSuccess) {                                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                    \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
            return -1;                                                     \
        }                                                                  \
    } while (0)

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256

// Probability
__device__ float p(unsigned int x, int W, int H) {
    return x / (float)(W * H);
}

__device__ float correctFloat(unsigned char val, float* cdf, float* cdfmin) {
    return (cdf[val] - *cdfmin) / (1 - *cdfmin);
}

__global__ void floatTouChar(float* floatImage, unsigned char* ucharImage, int W, int H, int Channels) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < W * H * Channels) {
        ucharImage[index] = (unsigned char)(255 * floatImage[index]);
    }
}

__global__ void produceOutput(unsigned char* inputImage, float* outputImage, float* cdf, int W, int H, int Channels) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < W * H * Channels) {
        outputImage[index] = correctFloat(inputImage[index], cdf, &cdf[0]);
    }
}


// Convert the image from RGB to GrayScale
__global__ void rgbToGrayScale(unsigned char* ucharImage, unsigned char* grayImage, int W, int H, int Channels) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < W * H * Channels) {
        float r = ucharImage[index * Channels];
        float g = ucharImage[index * Channels + 1];
        float b = ucharImage[index * Channels + 2];
        grayImage[index] = (unsigned char)(0.21 * r + 0.71 * g + 0.07 * b);
    }
}


__global__ void histo_kernel(unsigned char* grayImage, long size, unsigned int* histo) {
    __shared__ unsigned int private_histo[HISTOGRAM_LENGTH];

    int tx = threadIdx.x;
    if (tx < HISTOGRAM_LENGTH)
        private_histo[tx] = 0;

    __syncthreads();

    int i = blockIdx.x * blockDim.x + tx;
    int stride = blockDim.x * gridDim.x;

    while (i < size) {
        atomicAdd(&(private_histo[grayImage[i]]), 1);
        i += stride;
    }

    __syncthreads();

    if (threadIdx.x < HISTOGRAM_LENGTH)
        atomicAdd(&(histo[tx]), private_histo[tx]);
}

__global__ void scan(unsigned int* input, float* output, int len, int W, int H) {
    __shared__ float XY[2 * BLOCK_SIZE];
    int tx = threadIdx.x;
    int bd = blockDim.x;
    int i = bd * blockIdx.x + tx;

    if (i < len)
        XY[tx] = p(input[i], W, H);
    else
        XY[tx] = 0;

    __syncthreads();

    for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
        int index = (tx + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE)
            XY[index] += XY[index - stride];

        __syncthreads();
    }

    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        __syncthreads();

        int index = (tx + 1) * stride * 2 - 1;
        if (index + stride < 2 * BLOCK_SIZE)
            XY[index + stride] += XY[index];
    }

    __syncthreads();

    if (i < len) {
        output[i] = XY[tx];
    }
}


int main(int argc, char** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float* hostInputImageData;
    float* hostOutputImageData;
    float* deviceInputImageData;
    float* deviceOutputImageData;
    float* cdf;
    unsigned int* histogram;
    unsigned char* deviceGrayScaleData;
    unsigned char* deviceUcharData;

    const char* inputImageFile;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbLog(TRACE,
          "The dimensions of image are ",
          imageWidth,
          " x ",
          imageHeight,
          " with ",
          imageChannels,
          " number of "
          "channels");

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void**)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void**)&deviceUcharData, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
    cudaMalloc((void**)&deviceGrayScaleData, imageWidth * imageHeight * sizeof(unsigned char));
    cudaMalloc((void**)&histogram, imageWidth * imageHeight * sizeof(unsigned int));
    cudaMalloc((void**)&cdf, HISTOGRAM_LENGTH * sizeof(float));
    cudaMalloc((void**)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbCheck(cudaMemset(deviceGrayScaleData, 0, imageWidth * imageHeight * sizeof(unsigned char)));
    wbCheck(cudaMemset(histogram, 0, imageWidth * imageHeight * sizeof(unsigned int)));

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Float to Char on the GPU");
    // Float to uChar
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid((imageWidth * imageHeight * imageChannels - 1) / BLOCK_SIZE + 1, 1, 1);

    floatTouChar<<<dimGrid, dimBlock>>> (deviceInputImageData, deviceUcharData, imageWidth, imageHeight, imageChannels);
    wbTime_stop(Compute, "Float to Char on the GPU");
    cudaDeviceSynchronize();

    // RGB to GrayScale
    wbTime_start(Compute, "Doing the Grayscale on the GPU");
    rgbToGrayScale<<<dimGrid, dimBlock>>> (deviceUcharData, deviceGrayScaleData, imageWidth, imageHeight, imageChannels);
    wbTime_stop(Compute, "Doing the Grayscale on the GPU");
    cudaDeviceSynchronize();

    // Histogram
    dim3 dimBlock1(HISTOGRAM_LENGTH, 1, 1);
    dim3 dimGrid1((imageWidth * imageHeight - 1) / BLOCK_SIZE + 1, 1, 1);

    wbTime_start(Compute, "Doing the Histogram on the GPU");
    histo_kernel<<<dimGrid1, dimBlock1>>> (deviceGrayScaleData, imageWidth * imageHeight, histogram);
    wbTime_stop(Compute, "Doing the Histogram on the GPU");
    cudaDeviceSynchronize();

    // CDF scan
    dim3 dimBlock2(HISTOGRAM_LENGTH, 1, 1);
    dim3 dimGrid2(1, 1, 1);

    wbTime_start(Compute, "Doing the CDF on the GPU");
    scan<<<dimGrid2, dimBlock2>>> (histogram, cdf, HISTOGRAM_LENGTH, imageWidth, imageHeight);
    wbTime_stop(Compute, "Doing the CDF on the GPU");
    cudaDeviceSynchronize();

    // Normalize the image with CDF
    wbTime_start(Compute, "Producing the result of the image on the GPU");
    produceOutput<<<dimGrid, dimBlock>>> (deviceUcharData, deviceOutputImageData, cdf, imageWidth, imageHeight, imageChannels);
    wbTime_stop(Compute, "Producing the result of the image on the GPU");
    cudaDeviceSynchronize();

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(histogram);
    cudaFree(cdf);
    cudaFree(deviceUcharData);
    cudaFree(deviceGrayScaleData);

    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

