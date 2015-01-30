#include    <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Channels 3
#define Mask_width 5
#define Mask_radius Mask_width / 2
#define O_TILE_WIDTH 12
#define BLOCK_WIDTH (O_TILE_WIDTH + 4)
#define TILE_WIDTH BLOCK_WIDTH

__device__ float zeroOneClamp(float s) {
    if (s < 0)
        return 0;
    if (s > 1)
        return 1;
    return s;
}

__global__ void imageConvolution(float* input, float* output, const float* __restrict__ M, int W, int H) {

    __shared__ float sm_color[Channels][TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int chan = blockIdx.z;

    int row_o = by * O_TILE_WIDTH + ty;
    int col_o = bx * O_TILE_WIDTH + tx;
    int index_o = (row_o * W + col_o) * Channels + chan;

    int row_i = row_o - Mask_radius;
    int col_i = col_o - Mask_radius;
    int index_i = (row_i * W + col_i) * Channels + chan;

    if ((row_i >= 0) && (row_i < H) && (col_i >= 0) && (col_i < W))
        sm_color[chan][ty][tx] = input[index_i];
    else
        sm_color[chan][ty][tx] = 0.0f;

    __syncthreads();

    float out = 0.0f;

    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH && row_o < H && col_o < W) {
        for (int i = 0; i < Mask_width; i++)
            for (int j = 0; j < Mask_width; j++)
                out += M[i * Mask_width + j] * sm_color[chan][i + ty][j + tx];

        __syncthreads();

        output[index_o] = zeroOneClamp(out);
    }
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char* inputImageFile;
    char* inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float* hostInputImageData;
    float* hostOutputImageData;
    float* hostMaskData;
    float* deviceInputImageData;
    float* deviceOutputImageData;
    float* deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float*)wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbLog(TRACE, "Dims: ", imageWidth, " x ", imageHeight, " with ", imageChannels, " number of " "channels");

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void**)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void**)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void**)&deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData, hostMaskData, maskRows * maskColumns * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid((imageWidth - 1) / O_TILE_WIDTH + 1, (imageHeight - 1) / O_TILE_WIDTH + 1, Channels);

    wbTime_start(Compute, "Doing the computation on the GPU");
    imageConvolution << <dimGrid, dimBlock>>> (deviceInputImageData, deviceOutputImageData, deviceMaskData, imageWidth, imageHeight);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
