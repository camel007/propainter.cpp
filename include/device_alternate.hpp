#ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
#define CAFFE_UTIL_DEVICE_ALTERNATE_H_

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#ifdef USE_CUDNN           // cuDNN acceleration library.
#include "caffe/util/cudnn.hpp"
#endif

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition)                                             \
    /* Code block avoids redefinition of cudaError_t error */             \
    do                                                                    \
    {                                                                     \
        cudaError_t error = condition;                                    \
        CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
    } while (0)

#define CUBLAS_CHECK(condition)                                                                \
    do                                                                                         \
    {                                                                                          \
        cublasStatus_t status = condition;                                                     \
        CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " << caffe::cublasGetErrorString(status); \
    } while (0)

#define CURAND_CHECK(condition)                                                                \
    do                                                                                         \
    {                                                                                          \
        curandStatus_t status = condition;                                                     \
        CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " << caffe::curandGetErrorString(status); \
    } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace ferrari
{

// CUDA: library error reporting.
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N)
{
    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

}  // namespace ferrari

#endif  // CAFFE_UTIL_DEVICE_ALTERNATE_H_