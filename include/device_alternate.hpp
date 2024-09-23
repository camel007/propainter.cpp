#ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
#define CAFFE_UTIL_DEVICE_ALTERNATE_H_

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>  // cuda driver types

#include "cudnn.h"
#include "simple_log.hpp"

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition)                                              \
    /* Code block avoids redefinition of cudaError_t error */              \
    do                                                                     \
    {                                                                      \
        cudaError_t error = condition;                                     \
        FCHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
    } while (0)

#define CUBLAS_CHECK(condition)                                                                   \
    do                                                                                            \
    {                                                                                             \
        cublasStatus_t status = condition;                                                        \
        FCHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " << ferrari::cublasGetErrorString(status); \
    } while (0)

#define CUDNN_CHECK(condition)                                                         \
    do                                                                                 \
    {                                                                                  \
        cudnnStatus_t status = condition;                                              \
        FCHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " " << cudnnGetErrorString(status); \
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

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N)
{
    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

}  // namespace ferrari

#endif  // CAFFE_UTIL_DEVICE_ALTERNATE_H_