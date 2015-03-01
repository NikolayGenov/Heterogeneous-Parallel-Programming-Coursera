#include <wb.h> //@@ wb include opencl.h for you
#include <math.h>

//@@ OpenCL Kernel
const char* vaddsrc = "__kernel void vadd(__global const float *a,__global const float *b,__global float *result){ "
                      "int id = get_global_id(0);result[id] = a[id] + b[id];}";

int main(int argc, char** argv) {
    wbArg_t args;
    int inputLength;
    float* h_A;
    float* h_B;
    float* h_O;
    cl_mem d_A;
    cl_mem d_B;
    cl_mem d_O;

    args = wbArg_read(argc, argv);
    wbTime_start(Generic, "Importing data and creating memory on host");
    h_A = (float*)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    h_B = (float*)wbImport(wbArg_getInputFile(args, 1), &inputLength);
    h_O = (float*)malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
    wbLog(TRACE, "The input length is ", inputLength);
    int size = inputLength * sizeof(float);

    //Open CL contex setup
    size_t parmsz;
    cl_int clerr;
    cl_context clctx;
    cl_command_queue clcmdq;
    cl_program clpgm;
    cl_kernel clkern;

    cl_uint numPlatforms;
    clerr = clGetPlatformIDs(0, NULL, &numPlatforms);
    cl_platform_id platforms[numPlatforms];
    clerr = clGetPlatformIDs(numPlatforms, platforms, NULL);
   
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (unsigned long)platforms[0], 0};
    clctx = clCreateContextFromType(properties, CL_DEVICE_TYPE_ALL, NULL, NULL, &clerr);
   
    clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, 0, NULL, &parmsz);
    cl_device_id* cldevs = (cl_device_id*)malloc(parmsz);
    
    clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, parmsz, cldevs, NULL);
    clcmdq = clCreateCommandQueue(clctx, cldevs[0], 0, &clerr);
    
    //Complie and build program, and create kernel
    clpgm = clCreateProgramWithSource(clctx, 1, &vaddsrc, NULL, &clerr);
    char clcompileflags[4096];
    sprintf(clcompileflags, "-cl-mad-enable");
    clerr = clBuildProgram(clpgm, 0, NULL, clcompileflags, NULL, NULL);
    clkern = clCreateKernel(clpgm, "vadd", &clerr);

    wbTime_start(GPU, "Allocating + copy GPU memory.");
    //@@ Allocate GPU memory here Copy memory to the GPU here
    d_A = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_A, NULL);
    d_B = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_B, NULL);
    d_O = clCreateBuffer(clctx, CL_MEM_WRITE_ONLY, size, NULL, NULL);
    wbTime_stop(GPU, "Allocating + copy GPU memory.");

    size_t globalSize, localSize;
    localSize = 64;
    globalSize = ceil(inputLength / (float)localSize) * localSize;

    wbTime_start(Compute, "Performing CUDA computation");
    // Give params to kernel
    clerr = clSetKernelArg(clkern, 0, sizeof(cl_mem), (void*)&d_A);
    clerr = clSetKernelArg(clkern, 1, sizeof(cl_mem), (void*)&d_B);
    clerr = clSetKernelArg(clkern, 2, sizeof(cl_mem), (void*)&d_O);
    clerr = clSetKernelArg(clkern, 3, sizeof(int), &inputLength);

    //Run kernel and wait for it to finish
    cl_event event = NULL;
    clerr = clEnqueueNDRangeKernel(clcmdq, clkern, 1, NULL, &globalSize, &localSize, 0, NULL, &event);
    clerr = clWaitForEvents(1, &event);

    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    clerr = clEnqueueReadBuffer(clcmdq, d_O, CL_TRUE, 0, size, h_O, 0, NULL, NULL);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_O);
    wbTime_stop(GPU, "Freeing GPU Memory");
    wbSolution(args, h_O, inputLength);
    free(h_A);
    free(h_B);
    free(h_O);

    return 0;
}

