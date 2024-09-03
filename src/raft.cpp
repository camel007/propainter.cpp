#include "raft.hpp"

#include "cuda_functional.hpp"

namespace raft
{
CorrBlock::CorrBlock(int batch, int dim, int ht, int wd, int num_levels, int radius)
    : batch_(batch), dim_(dim), ht_(ht), wd_(wd), num_levels_(num_levels), radius_(radius)
{
    cudnnCreate(&cudnn_handle_);
    cublasCreate(&cublas_handle_);

    corr_pyramid_.resize(num_levels_);
    int total_bytes = sizeof(float) * ht_ * wd_;
    for (int i = 0; i < num_levels; ++i)
    {
        cudaMalloc(&corr_pyramid_[i], total_bytes);
        total_bytes >= 2;
    }

    uint32_t delta_size = (2 * radius_ + 1) ^ 2;
    cudaMalloc(&delta_lvl_, sizeof(float) * delta_size);
}

CorrBlock::~CorrBlock()
{
    for (auto& d_mem : corr_pyramid_)
    {
        cudaFree(d_mem);
    }
    cudaFree(delta_lvl_);
    cublasDestroy(cublas_handle_);
    cudnnDestroy(cudnn_handle_);
}

int CorrBlock::computeCorr(const float* fmap1, const float* fmap2)
{
    // 设置cuBLAS参数
    const float alpha = 1.0f / sqrt(dim_);
    const float beta  = 0.0f;
    int         m     = ht_ * wd_;
    int         k     = dim_;
    int         n     = ht_ * wd_;

    for (int b = 0; b < batch_; ++b)
    {
        // 执行矩阵乘法 C = alpha * (A^T * B) + beta * C
        cublasSgemm(cublas_handle_,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    &alpha,
                    fmap1 + b * dim_ * m * k,
                    k,
                    fmap2 + b * dim_ * m * k,
                    k,
                    &beta,
                    corr_pyramid_[0] + b * m * n,
                    m);
    }

    buildCorrPyramid();

    return 0;
}

int CorrBlock::buildCorrPyramid()
{
    cudnnTensorDescriptor_t  in_desc, out_desc;
    cudnnPoolingDescriptor_t pool_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnCreatePoolingDescriptor(&pool_desc);

    cudnnSetPooling2dDescriptor(pool_desc,
                                CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                CUDNN_NOT_PROPAGATE_NAN,
                                2,
                                2,  // window height and width
                                0,
                                0,  // vertical and horizontal padding
                                2,
                                2);  // vertical and horizontal stride

    for (int n = 1; n < num_levels_; ++n)
    {
        int batch_size = batch_;
        int channels   = 1;
        int height     = wd_ / (2 ^ (n - 1));
        int width      = ht_ / (2 ^ (n - 1));

        cudnnSetTensor4dDescriptor(
            in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width);

        int pooled_height = height / 2;
        int pooled_width  = width / 2;

        cudnnSetTensor4dDescriptor(out_desc,
                                   CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,
                                   batch_size,
                                   channels,
                                   pooled_height,
                                   pooled_width);

        float* d_input  = corr_pyramid_[n - 1];
        float* d_output = corr_pyramid_[n];

        // Assuming d_input is already filled with data
        float alpha = 1.0f, beta = 0.0f;

        cudnnPoolingForward(
            cudnn_handle_, pool_desc, &alpha, in_desc, d_input, &beta, out_desc, d_output);
    }

    cudnnDestroyPoolingDescriptor(pool_desc);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);

    return cudaGetLastError();
}

int CorrBlock::generateDelta()
{
    create_delta(delta_lvl_, radius_);

    return 0;
}

int CorrBlock::call(const float* coords, float* output)
{
    int ret = 0;

    return ret;
}

}  // namespace raft