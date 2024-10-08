#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <ctime>

#include "common.hpp"
#include "cudnn.h"
#include "device_alternate.hpp"
#include "simple_log.hpp"

namespace ferrari
{

Caffe& Caffe::Get()
{
    static Caffe instance;
    return instance;
}

Caffe::Caffe() : cublas_handle_(NULL), cudnn_handle_(NULL), mode_(Caffe::GPU)
{
    // Try to create a cublas handler, and report an error if failed (but we will
    // keep the program running as one might just want to run CPU code).
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS)
    {
        LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
    }

    if (cudnnCreate(&cudnn_handle_) != CUDNN_STATUS_SUCCESS)
    {
        LOG(ERROR) << "Cannot create Cudnn handle. Cudnn won't be available.";
    }
    else
    {
        size_t cudnn_version = cudnnGetVersion();

        // Print the cuDNN version
        LOG(INFO) << "cuDNN Version: " << cudnn_version;
    }
}

Caffe::~Caffe()
{
    if (cublas_handle_)
        CUBLAS_CHECK(cublasDestroy(cublas_handle_));
    if (cudnn_handle_)
        CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
}

void Caffe::SetDevice(const int device_id)
{
    cudaFree(0);
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (current_device == device_id)
    {
        return;
    }
    // The call to cudaSetDevice must come before any calls to Get, which
    // may perform initialization using the GPU.
    CUDA_CHECK(cudaSetDevice(device_id));
    if (Get().cublas_handle_)
        CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
    CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_));

    if (Get().cudnn_handle_)
        CUDNN_CHECK(cudnnDestroy(Get().cudnn_handle_));
    CUDNN_CHECK(cudnnCreate(&Get().cudnn_handle_));
}

void Caffe::DeviceQuery()
{
    cudaDeviceProp prop;
    int            device;
    if (cudaSuccess != cudaGetDevice(&device))
    {
        printf("No cuda device present.\n");
        return;
    }
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    LOG(INFO) << "Device id:                     " << device;
    LOG(INFO) << "Major revision number:         " << prop.major;
    LOG(INFO) << "Minor revision number:         " << prop.minor;
    LOG(INFO) << "Name:                          " << prop.name;
    LOG(INFO) << "Total global memory:           " << prop.totalGlobalMem;
    LOG(INFO) << "Total shared memory per block: " << prop.sharedMemPerBlock;
    LOG(INFO) << "Total registers per block:     " << prop.regsPerBlock;
    LOG(INFO) << "Warp size:                     " << prop.warpSize;
    LOG(INFO) << "Maximum memory pitch:          " << prop.memPitch;
    LOG(INFO) << "Maximum threads per block:     " << prop.maxThreadsPerBlock;
    LOG(INFO) << "Maximum dimension of block:    " << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2];
    LOG(INFO) << "Maximum dimension of grid:     " << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2];
    LOG(INFO) << "Clock rate:                    " << prop.clockRate;
    LOG(INFO) << "Total constant memory:         " << prop.totalConstMem;
    LOG(INFO) << "Texture alignment:             " << prop.textureAlignment;
    LOG(INFO) << "Concurrent copy and execution: " << (prop.deviceOverlap ? "Yes" : "No");
    LOG(INFO) << "Number of multiprocessors:     " << prop.multiProcessorCount;
    LOG(INFO) << "Kernel execution timeout:      "
              << (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
    return;
}

bool Caffe::CheckDevice(const int device_id)
{
    // This function checks the availability of GPU #device_id.
    // It attempts to create a context on the device by calling cudaFree(0).
    // cudaSetDevice() alone is not sufficient to check the availability.
    // It lazily records device_id, however, does not initialize a
    // context. So it does not know if the host thread has the permission to use
    // the device or not.
    //
    // In a shared environment where the devices are set to EXCLUSIVE_PROCESS
    // or EXCLUSIVE_THREAD mode, cudaSetDevice() returns cudaSuccess
    // even if the device is exclusively occupied by another process or thread.
    // Cuda operations that initialize the context are needed to check
    // the permission. cudaFree(0) is one of those with no side effect,
    // except the context initialization.
    bool r = ((cudaSuccess == cudaSetDevice(device_id)) && (cudaSuccess == cudaFree(0)));
    // reset any error that may have occurred.
    cudaGetLastError();
    return r;
}

int Caffe::FindDevice(const int start_id)
{
    // This function finds the first available device by checking devices with
    // ordinal from start_id to the highest available value. In the
    // EXCLUSIVE_PROCESS or EXCLUSIVE_THREAD mode, if it succeeds, it also
    // claims the device due to the initialization of the context.
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    for (int i = start_id; i < count; i++)
    {
        if (CheckDevice(i))
            return i;
    }
    return -1;
}

const char* cublasGetErrorString(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    }
    return "Unknown cublas status";
}

}  // namespace ferrari