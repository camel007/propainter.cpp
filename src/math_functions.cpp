#include "math_functions.hpp"

#include <limits>
#include <cuda_runtime.h>
#include "common.hpp"
#include "device_alternate.hpp"

namespace ferrari
{

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y)
{
    if (X != Y)
    {
        if (Caffe::mode() == Caffe::GPU)
        {
#ifndef CPU_ONLY
            // NOLINT_NEXT_LINE(caffe/alt_fn)
            CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
            NO_GPU;
#endif
        }
        else
        {
            memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
        }
    }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X, unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

}  // namespace ferrari