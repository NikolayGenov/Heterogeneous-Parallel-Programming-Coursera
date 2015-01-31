// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void addBlockSums(float* output, float* blockSums, int len) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < len && blockIdx.x > 0)
        for (int bl = 0; bl < blockIdx.x; ++bl)
            output[i] += blockSums[bl];
}

__global__ void scan(float* input, float* output, float* blockSums, int len) {
    //@@ Modify the body of this function to complete the functionality of
    __shared__ float XY[2 * BLOCK_SIZE];
    int tx = threadIdx.x;
    int bd = blockDim.x;
    int i = bd * blockIdx.x + tx;

    if (i < len)
        XY[tx] = input[i];
    else
        XY[tx] = 0;

    __syncthreads();

    //@@ the scan on the device
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
        if ((i + 1) % bd == 0)
            blockSums[blockIdx.x] = output[i];
    }
}

int main(int argc, char** argv) {
    wbArg_t args;
    float* hostInput;  // The input 1D list
    float* hostOutput; // The output list
    float* deviceInput;
    float* deviceOutput;
    float* blockSums;
    int numElements; // number of elements in the list
    int numBlocks;
    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float*)wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*)malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    numBlocks = (numElements - 1) / BLOCK_SIZE + 1;
    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbLog(TRACE, "The number of blocks is ", numBlocks);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements * sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements * sizeof(float)));
    wbCheck(cudaMalloc((void**)&blockSums, numBlocks * sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
    wbCheck(cudaMemset(blockSums, 0, numBlocks * sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce

    scan << <dimGrid, dimBlock>>> (deviceInput, deviceOutput, blockSums, numElements);
    cudaDeviceSynchronize();
    addBlockSums << <dimGrid, dimBlock>>> (deviceOutput, blockSums, numElements);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(blockSums);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
