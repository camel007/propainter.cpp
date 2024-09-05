#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <stdint.h>

#include <cmath>  // for std::fabs and std::signbit

#include "common.hpp"
#include "device_alternate.hpp"
#include "glog/logging.h"

namespace ferrari
{
template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y);

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* X);

inline void caffe_memset(const size_t N, const int alpha, void* X)
{
    memset(X, alpha, N);  // NOLINT(caffe/alt_fn)
}

inline void caffe_gpu_memcpy(const size_t N, const void* X, void* Y)
{
    if (X != Y)
    {
        CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
    }
}
inline void caffe_gpu_memset(const size_t N, const int alpha, void* X)
{
#ifndef CPU_ONLY
    CUDA_CHECK(cudaMemset(X, alpha, N));  // NOLINT(caffe/alt_fn)
#else
    NO_GPU;
#endif
}

}  // namespace ferrari

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_