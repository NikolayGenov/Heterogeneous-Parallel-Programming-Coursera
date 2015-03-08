#include <wb.h>

int main(int argc, char** argv) {
    wbArg_t args;
    int inputLength;
    float* h_A;
    float* h_B;
    float* h_O;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    h_A = (float*)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    h_B = (float*)wbImport(wbArg_getInputFile(args, 1), &inputLength);
    h_O = (float*)malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

    wbTime_start(GPU, "Computing data on the GPU");

    int workers = 256;

    #pragma acc parallel copyin(h_A[0 : inputLength], h_B[0 : inputLength]), copyout(h_O[0 : inputLength]), num_workers(workers)
    {
        #pragma acc loop worker
        int i;
        for (i = 0; i < inputLength; i++) {
            h_O[i] = h_A[i] + h_B[i];
        }
    }

    wbTime_stop(GPU, "Computing data on the GPU");
    wbSolution(args, h_O, inputLength);

    free(h_A);
    free(h_B);
    free(h_O);

    return 0;
}
