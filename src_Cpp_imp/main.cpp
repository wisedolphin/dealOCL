#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <iostream>
#include <CL/opencl.hpp>
#include <string>
/*
create tmp dir      mkdir tmp
go to tmp           cd tmp
launch              cmake ../ -G "CodeBlocks - Unix Makefiles"
buld                cmake --build .
*/
using namespace std;

int main()
{
    // serching for platforms and choose NVIDIA CUDA
    cl::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform plat;
    for (auto &p : platforms)
    {
        std::string platver = p.getInfo<CL_PLATFORM_VERSION>();
        std::string platname = p.getInfo<CL_PLATFORM_NAME>();
        std::cout << "Found platform: ";
        std::cout<<platname<<std::endl;
        if (platname.find("NVIDIA CUDA") != std::string::npos)
        {
            std::cout << "Found needed platform" << std::endl;
            std::cout << "Platform OCL version: " << platver << std::endl;
            plat = p;
        }
    }
    //choosing device
    cl::vector<cl::Device> devices;
    plat.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices.front();
    std::cout << "Found device on a platform: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    //creating context
    cl::Context context({device});
    // Kernel raw string
    std::string kernel{R"CLC(
    __kernel void vector_add_kernel(__global const double *A,
                                    __global const double *B,
                                    __global double *C)
    {
    // Get the index of the current element to be processed
    int i = get_global_id(0);
    // Do the operation
    C[i] = A[i] + B[i];
    C[i] = C[i] / A[i];
    C[i] = C[i] * B[i];
    }
    )CLC"};
    //Building programm
    cl::Program prog(context, kernel);
    std::cout << "Building programm ...\n";
    try{
        prog.build();
    }
    catch (...){
        cl_int build_err = CL_SUCCESS;
        auto build_info = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&build_err);
        for (auto &pair : build_info) std::cerr << pair.second << std::endl;
        return 1;
    }
    std::cout << "Building done\n";
    // creating command queue
    cl::CommandQueue queue(context, device);
    // creating arrays
    int N = 1024*1024*64;
    double* A = new double[N];
    double* B = new double[N];
    double* C = new double[N];
    for (int i = 0; i!=N; ++i)
    {
        A[i] = i;
        B[i] = N-i;
    }
    // creating buffer
    cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(double)*N);
    cl::Buffer bufferB(context, CL_MEM_READ_WRITE, sizeof(double)*N);
    cl::Buffer bufferC(context, CL_MEM_READ_ONLY, sizeof(double)*N);
    //enqueue buffers
    queue.enqueueWriteBuffer(bufferA,CL_TRUE,0,sizeof(double)*N,A);
    queue.enqueueWriteBuffer(bufferB,CL_TRUE,0,sizeof(double)*N,B);
    //kernel exec
    cl::Kernel add_double = cl::Kernel(prog, "vector_add_kernel");
    add_double.setArg(0,bufferA);
    add_double.setArg(1,bufferB);
    add_double.setArg(2,bufferC);
    clock_t t = clock();
    queue.enqueueNDRangeKernel(add_double,cl::NullRange,cl::NDRange(N),cl::NullRange);
    queue.finish();
    queue.enqueueReadBuffer(bufferC,CL_TRUE,0,sizeof(double)*N,C);
    t = clock() - t;
    std::cout << "GPU execution done\n";
    std::cout << "Time: " << ((double)t/CLOCKS_PER_SEC) << std::endl;
    std::cout << "Kernel execution done\n \n";
    //printing result
    /*
    for (int i = 0; i!=N; ++i)
    {
        std::cout << A[i] << " + ";
        std::cout << B[i] << " = ";
        std::cout << C[i] << std::endl;
    }
    */
    //CPU_version
    t = clock();
    for (int i = 0; i!=N; ++i)
    {
        C[i] = A[i] + B[i];
        C[i] = C[i] / A[i];
        C[i] = C[i] * B[i];
    }
    t = clock() - t;
    std::cout << "CPU execution done\n";
    std::cout << "Time: " << ((double)t/CLOCKS_PER_SEC) << std::endl;
    return 0;
}
