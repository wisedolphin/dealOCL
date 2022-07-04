#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h> 

#ifdef __APPLE__
#include <CL/cl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE (0x100000)

//gcc main.c -o vectorAddition -l OpenCL
 
 void error_proc(const char * msg, cl_int error_number)
{
    printf("Error! ");
    printf("Message: %s, ", msg);
    printf("Error number: %d, ", error_number);
	printf("\n");
}
//enum { MAX_SOURCE_SIZE = 1000 }; 
int main(void) {

    // Create the two input vectors
    int i;
    const int LIST_SIZE = 64*1024*1024;
    double*A = (double*)malloc(sizeof(double)*LIST_SIZE);
    double*B = (double*)malloc(sizeof(double)*LIST_SIZE);
    for(i = 0; i < LIST_SIZE; i++) {
        A[i] = (double)i;
        B[i] = (double)LIST_SIZE - (double)i;
    }
 
    // Load the kernel source code into the array of chars source_str
 //   FILE *fp;
 //   char *source_str;
 //   size_t source_size;
 //   size_t err_no = 0;
 //   err_no = fopen_s(&fp, "vector_add_kernel.cl", "r");
 //   if (err_no != 0) { printf("Something wrong with file %d\n", err_no); }
 //   if (!fp) {
 //       fprintf(stderr, "Failed to load kernel code.\n");
 //       exit(1);
 //   }
 //   source_str = (char*)malloc(MAX_SOURCE_SIZE);
 //   source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	//fclose( fp );
    //string with kernel source code
    #define STRINGIFY(...) #__VA_ARGS__
    const char* source_str = STRINGIFY(__kernel void vector_add_kernel(
        __global double* A, __global double* B, __global double* C) {
        int i = get_global_id(0);
        C[i] = A[i] + B[i];
    });
    size_t source_size = strlen(source_str);


    // Get platform and device information
    cl_platform_id platform_id = NULL;
	const char * platform_name = "NVIDIA CUDA"; //platform to choose 
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    // Geting number of platforms
    cl_int ret = clGetPlatformIDs(0, 0, &ret_num_platforms);
    if (ret != CL_SUCCESS){error_proc("Getting number of platforms", ret);}
	//mem allocation for platform list
    cl_platform_id *platformList = (cl_platform_id*)malloc(sizeof(cl_platform_id)*ret_num_platforms);
	//getting list of platforms
    ret = clGetPlatformIDs(ret_num_platforms, platformList, 0);
    if (ret != CL_SUCCESS){error_proc("Getting number of platforms", ret);}
    #define STR_SIZE 1024
    char nameBUF[STR_SIZE];
    //choosing platform with name platform_name
    for (cl_uint i = 0; i < ret_num_platforms; ++i)
    {
        ret = clGetPlatformInfo(platformList[i], CL_PLATFORM_NAME, STR_SIZE, nameBUF,0);
		if (!strcmp(platform_name, nameBUF)) {platform_id = platformList[i];}
        if (ret != CL_SUCCESS){error_proc("Getting platform names", ret); return(1);}
        printf("Platform: %d",i);
		printf(", Platform name: ");
		printf(nameBUF);
		printf("\n");
    }	
    //getiing device id for platform
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, 
            &device_id, &ret_num_devices);
	
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
	if (ret != CL_SUCCESS){error_proc("clCreateContext", ret); return(1);}
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
	if (ret != CL_SUCCESS){error_proc("clCreateCommandQueue", ret); return(1);}
    
    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            LIST_SIZE * sizeof(double), NULL, &ret);
	if (ret != CL_SUCCESS){error_proc("clCreateBuffer", ret); return(1);}
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            LIST_SIZE * sizeof(double), NULL, &ret);
	if (ret != CL_SUCCESS){error_proc("clCreateBuffer", ret); return(1);}
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
            LIST_SIZE * sizeof(double), NULL, &ret);
	if (ret != CL_SUCCESS){error_proc("clCreateBuffer", ret); return(1);}
	
    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
            LIST_SIZE * sizeof(double), A, 0, NULL, NULL);
	if (ret != CL_SUCCESS){error_proc("clEnqueueWriteBuffer", ret); return(1);}
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
            LIST_SIZE * sizeof(double), B, 0, NULL, NULL);
	if (ret != CL_SUCCESS){error_proc("clEnqueueWriteBuffer", ret); return(1);}
    
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t*)&source_size, &ret);
	if (ret != CL_SUCCESS){error_proc("clCreateProgramWithSource", ret); return(1);}
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS){error_proc("clBuildProgram", ret);}
    // if failed get building log
    if (ret != CL_SUCCESS)
    {
        printf("Program Build failed\n");
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* build_log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        build_log[log_size] = '\0';
        printf("--- Build log ---\n %s \n", build_log);
        exit(1);
    }
    
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add_kernel", &ret);
	if (ret != CL_SUCCESS){error_proc("clCreateKernel", ret); return(1);}
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    if (ret != CL_SUCCESS){error_proc("clSetKernelArg", ret); return(1);}
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    if (ret != CL_SUCCESS){error_proc("clSetKernelArg", ret); return(1);}
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
    if (ret != CL_SUCCESS){error_proc("clSetKernelArg", ret); return(1);}
    
    
    clock_t t;
    t = clock();
    // Execute the OpenCL kernel on the list
    size_t global_item_size = LIST_SIZE; // Process the entire lists
    size_t local_item_size = 64; // Divide work items into groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, NULL);
    if (ret != CL_SUCCESS){error_proc("clEnqueueNDRangeKernel", ret); return(1);}
    t = clock() - t;
    int time_taken = ((int)t) / CLOCKS_PER_SEC; // calculate the elapsed time
    printf("The kernel execution took %f seconds\n", time_taken);


    // Read the memory buffer C on the device to the local variable C
    double*C = (double*)malloc(sizeof(double)*LIST_SIZE);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
            LIST_SIZE * sizeof(double), C, 0, NULL, NULL);
    if (ret != CL_SUCCESS){error_proc("clEnqueueReadBuffer", ret); return(1);}
    // Display the result to the screen
    //for(i = 0; i < LIST_SIZE; i++)
        //printf("%lf + %lf = %lf \n", A[i], B[i], C[i]);
 
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    //Making the same calc on CPU
    t = clock();
    for (i = 0; i < LIST_SIZE; i++) {
        C[i] = A[i] + B[i];
    }
    t = clock() - t;
    time_taken = ((int)t) / CLOCKS_PER_SEC; // calculate the elapsed time
    printf("The CPU execution took %f seconds\n", time_taken);

    free(A);
    free(B);
    free(C);
    return 0;
}
