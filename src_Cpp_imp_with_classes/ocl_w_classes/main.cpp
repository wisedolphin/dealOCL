/*
Enspired by https://github.com/tilir
He has a great cource on youtube https://youtu.be/Ccclo1GCX0A
*/

#define CL_HPP_ENABLE_EXCEPTIONS

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 200
#endif // CL_HPP_TARGET_OPENCL_VERSION

#include <iostream>
#include <CL/opencl.hpp>
#include <string>
#include <numeric>
#include <chrono>

constexpr size_t ARR_SIZE = 64*1024*1024;
constexpr size_t LOCAL_SIZE = 64;

std::string vecaddkernel{R"CLC(
__kernel void vector_add(__global const int *A,
                                __global const int *B,
                                __global int *C)
{
// Get the index of the current element to be processed
int i = get_global_id(0);
// Do the operation
C[i] = A[i] + B[i];
}
)CLC"};


// Encapsulate platform, context and queue
class OclApp
{
    cl::Platform P_;
    cl::Context C_;
    cl::CommandQueue Q_;

    static cl::Platform select_platform();
    static cl::Context get_context(cl_platform_id);

    //special functor that get all params of kernel (A,B,C) +
    // + enqueue_args - args for queue
    using vadd_t = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>;

public:
    OclApp() : P_(select_platform()), C_(get_context(P_())), Q_(C_, CL_QUEUE_PROFILING_ENABLE)
    {
        cl::string name = P_.getInfo<CL_PLATFORM_NAME>();
        cl::string profile = P_.getInfo<CL_PLATFORM_PROFILE>();
        std::cout << "Selected: " << name << ": " << profile << std::endl;
    }

    cl::Event vadd(cl_int const *A, cl_int const *B, cl_int *C, size_t Sz);
};

cl::Platform OclApp::select_platform()
{
    cl::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (auto &p : platforms)
    {
        cl::vector<cl::Device> devices;
        p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.size() > 0)
        {
            return cl::Platform(p);
            //return p;
        }
    }
    throw std::runtime_error("Platforms with GPU not found");
}

cl::Context OclApp::get_context(cl_platform_id Pid)
{
    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM,
                                            reinterpret_cast<cl_context_properties>(Pid),
                                            0 // end of propeties list
                                         };
    return cl::Context(CL_DEVICE_TYPE_GPU, properties);
}

cl::Event OclApp::vadd(cl_int const * Aptr, cl_int const *Bptr, cl_int *CPtr, size_t Sz)
{
    // buffer size
    size_t BufSz = Sz * sizeof(cl_int);
    // allocate buffer on divice
    cl::Buffer A(C_, CL_MEM_READ_ONLY, BufSz);
    cl::Buffer B(C_, CL_MEM_READ_ONLY, BufSz);
    cl::Buffer C(C_, CL_MEM_READ_ONLY, BufSz);
    // copy arrays to allocated buffer
    cl::copy(Q_, Aptr, Aptr + Sz, A);
    cl::copy(Q_, Aptr, Aptr + Sz, B);
    // create programm with kernel string. "true" - build immediatly
    //cl::Program program(C_, vecaddkernel, true);
    //or crate programm and build after
    cl::Program program(C_, vecaddkernel);
    try{
        program.build();
    }
    catch(cl::BuildError &err)
    {
        if (err.err() == CL_BUILD_PROGRAM_FAILURE)
        {
            cl_int build_err = CL_SUCCESS;
            auto build_info = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&build_err);
            for (auto& pair : build_info)
            {
                std::cerr << pair.second << std::endl;
            }
        }
    }
    vadd_t add_vecs(program, "vector_add");
    cl::NDRange GlobalRange(Sz);
    cl::NDRange LocalRange(LOCAL_SIZE);
    // args for event
    cl::EnqueueArgs Args(Q_, GlobalRange, LocalRange);
    // enqueue event
    cl::Event evt = add_vecs(Args, A, B, C);
    // wait for result
    evt.wait();
    // copy result
    cl::copy(Q_, C, CPtr, CPtr + Sz);
    return evt;
}


int main()
try
{
    std::chrono::high_resolution_clock::time_point Tstart, Tend;
    cl_ulong GPUTstart, GPUTend;

    OclApp app;
    cl::vector<cl_int> source1(ARR_SIZE), source2(ARR_SIZE), dst(ARR_SIZE);
    //fills vector from it_beg to it_end from 0 to size-1
    std::iota(source1.begin(), source1.end(), 0);
    //fills vector from it_end to it_beg from 0 to size-1
    std::iota(source2.rbegin(), source2.rend(), 0);

    Tstart = std::chrono::high_resolution_clock::now();
    cl::Event evt = app.vadd(source1.data(), source2.data(), dst.data(), dst.size());
    Tend = std::chrono::high_resolution_clock::now();
    long Duration = std::chrono::duration_cast<std::chrono::milliseconds>(Tend - Tstart).count();
    std::cout << "GPU calc time with offload time: " << Duration << " ms\n";

    GPUTstart = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    GPUTend =   evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    long GPUDuration = (GPUTend - GPUTstart) / 1000000;
    std::cout << "GPU pure calc time: " << GPUDuration << " ms\n";

    //result check
    for(int i = 0; i!=ARR_SIZE; ++i)
    {
        auto kernel_res = dst.at(i);
        auto cpu_res = source1.at(i) + source2.at(i);
        if (kernel_res == cpu_res)
        {
            std::cerr << "Error at index" << i << ": " << kernel_res << " != " << cpu_res << std::endl;
            return -1;
        }
    }


    //CPU execution
    Tstart = std::chrono::high_resolution_clock::now();
    for(int i = 0; i!=ARR_SIZE; ++i)
    {
        dst[i] = source1[i] + source2[i];
    }
    Tend = std::chrono::high_resolution_clock::now();
    Duration = std::chrono::duration_cast<std::chrono::milliseconds>(Tend - Tstart).count();
    std::cout << "CPU calc time: " << Duration << " ms\n";


    std::cout << "All jobs done!";
}
catch (cl::Error &err)
{
    std::cerr<< "OCL Error" << err.err() << ":" << err.what() << std::endl;
    return -1;
}
catch (std::runtime_error &err)
{
    std::cerr<< "Runtime Error" << err.what() << std::endl;
    return -1;
}
catch(...)
{
    std::cerr<< "Unknown error \n";
    return -1;
}

