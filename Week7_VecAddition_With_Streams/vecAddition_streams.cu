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


#define NUMBER_OF_STREAMS 2
#define SEG_SIZE 2048
#define BLOCK_SIZE 256


__global__ void vecAdd(float* in1, float* in2, float* out, int len) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len)
        out[i] = in1[i] + in2[i];
}

int main(int argc, char** argv) {
    wbArg_t args;
    int inputLength;
    float* h_A;
    float* h_B;
    float* h_Out;
    float* d_A[NUMBER_OF_STREAMS];
    float* d_B[NUMBER_OF_STREAMS];
    float* d_Out[NUMBER_OF_STREAMS];
    cudaStream_t streams[NUMBER_OF_STREAMS];

    for (int i = 0; i < NUMBER_OF_STREAMS; ++i)
        cudaStreamCreate(&streams[i]);

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    h_A = (float*)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    h_B = (float*)wbImport(wbArg_getInputFile(args, 1), &inputLength);
    h_Out = (float*)malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

    wbTime_start(GPU, "Allocating GPU memory.");
    int seg_size = SEG_SIZE * sizeof(float);
    for (int i = 0; i < NUMBER_OF_STREAMS; ++i) {
        wbCheck(cudaMalloc((void**)&d_A[i], seg_size));
        wbCheck(cudaMalloc((void**)&d_B[i], seg_size));
        wbCheck(cudaMalloc((void**)&d_Out[i], seg_size));
    }
    wbTime_stop(GPU, "Allocating GPU memory.");

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(SEG_SIZE / BLOCK_SIZE);

    wbTime_start(GPU, "Copy + Kernel + Copy on GPU.");

    for (int i = 0; i < inputLength; i += SEG_SIZE * NUMBER_OF_STREAMS) {

        wbTime_start(GPU, "Copy to device on GPU.");
        for (int k = 0; k < NUMBER_OF_STREAMS; ++k) {
            cudaMemcpyAsync(d_A[k], h_A + i + k * SEG_SIZE, seg_size, cudaMemcpyHostToDevice, streams[k]);
            cudaMemcpyAsync(d_B[k], h_B + i + k * SEG_SIZE, seg_size, cudaMemcpyHostToDevice, streams[k]);
        }
        wbTime_stop(GPU, "Copy to device on GPU.");

        wbTime_start(GPU, "Kernel call.");
        for (int k = 0; k < NUMBER_OF_STREAMS; ++k) {
            vecAdd << <dimGrid, dimBlock, 0, streams[k]>>> (d_A[k], d_B[k], d_Out[k], SEG_SIZE);
        }
        wbTime_stop(GPU, "Kernel call.");

        wbTime_start(GPU, "Copy to host.");
        for (int k = 0; k < NUMBER_OF_STREAMS; ++k) {
            cudaMemcpyAsync(h_Out + i + k * SEG_SIZE, d_Out[k], seg_size, cudaMemcpyDeviceToHost, streams[k]);
        }
        wbTime_stop(GPU, "Copy to host.");
    }

    wbTime_stop(GPU, "Copy + Kernel + Copy on GPU.");

    wbSolution(args, h_Out, inputLength);

    wbTime_start(Generic, "Clear memory and streams");
    for (int i = 0; i < NUMBER_OF_STREAMS; ++i) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_Out[i]);
    }

    for (int i = 0; i < NUMBER_OF_STREAMS; ++i)
        cudaStreamDestroy(streams[i]);

    free(h_A);
    free(h_B);
    free(h_Out);
    wbTime_stop(Generic, "Clear memory and streams");

    return 0;
}
