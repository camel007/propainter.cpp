#include "raft.hpp"

#include <memory>

#include "common.hpp"
#include "cuda_functional.hpp"
#include "device_alternate.hpp"
#include "glog/logging.h"

namespace ferrari
{
CorrBlock::CorrBlock(int batch, int dim, int ht, int wd, int num_levels, int radius)
    : batch_(batch), dim_(dim), ht_(ht), wd_(wd), num_levels_(num_levels), radius_(radius)
{
    for (int i = 0; i < num_levels_; ++i)
    {
        int                          n = batch * ht * wd;
        int                          c = 1;
        int                          h = (int)(ht / std::pow(2, i));
        int                          w = (int)(wd / std::pow(2, i));
        std::shared_ptr<Blob<float>> b =
            std::make_shared<Blob<float>>(std::vector<int>({n, c, h, w}));
        corr_pyramid_.push_back(b);
    }

    delta_lvl_ =
        std::make_shared<Blob<float>>(std::vector<int>({1, 2 * radius_ + 1, 2 * radius_ + 1, 2}));
}

CorrBlock::~CorrBlock() {}

#if 0
// 输入为的fmap1, fmap2均为 row-major, 形状分别为 k * m, k * n;
// 在 culbas中，形状分别为 m * k, n * k
// 执行 A * B^T = C, 在 cublas中， C的形状为 m * n，为 column-major
int CorrBlock::computeCorr(const std::shared_ptr<Blob<float>>& fmap1,
                           const std::shared_ptr<Blob<float>>& fmap2)
{
    const float* d_fmap1 = fmap1->gpu_data();                     // Pointer to fmap1 data on device
    const float* d_fmap2 = fmap2->gpu_data();                     // Pointer to fmap2 data on device
    float*       d_corr  = corr_pyramid_[0]->mutable_gpu_data();  // Output pointer

    int batch = fmap1->shape(0);  // Batch size
    int dim   = fmap1->shape(1);  // DIM
    int ht    = fmap1->shape(2);  // Height
    int wd    = fmap1->shape(3);  // Width

    // Scalars for cuBLAS
    float alpha = 1.0f / sqrtf(static_cast<float>(dim));
    float beta  = 0.0f;

    int hw = ht * wd;  // Number of pixels
    int m  = hw;       // Number of rows of op(A) and C
    int n  = hw;       // Number of columns of op(B) and C
    int k  = dim;      // Shared dimension

    // Adjust leading dimensions for row-major storage
    int lda = m;  // Leading dimension of A
    int ldb = n;  // Leading dimension of B
    int ldc = n;  // Leading dimension of C

    // Strides between batches (not needed for loop over cublasSgemm)
    long long int strideA = static_cast<long long>(m) * k;
    long long int strideB = static_cast<long long>(n) * k;
    long long int strideC = static_cast<long long>(m) * n;

    cublasOperation_t transa = CUBLAS_OP_N;  // No transpose on A
    cublasOperation_t transb = CUBLAS_OP_T;  // Transpose on B

    // Loop over batches and call cublasSgemm for each batch
    for (int b = 0; b < batch; ++b)
    {
        const float* batch_A = d_fmap1 + b * strideA;
        const float* batch_B = d_fmap2 + b * strideB;
        float*       batch_C = d_corr + b * strideC;

        cublasStatus_t status = cublasSgemm(Caffe::cublas_handle(),
                                            transa,  // Operation on A
                                            transb,  // Operation on B
                                            m,       // Number of rows of op(A) and C
                                            n,       // Number of columns of op(B) and C
                                            k,       // Number of columns of op(A) and rows of op(B)
                                            &alpha,  // Scalar alpha
                                            batch_A,  // Pointer to A
                                            lda,      // Leading dimension of A
                                            batch_B,  // Pointer to B
                                            ldb,      // Leading dimension of B
                                            &beta,    // Scalar beta
                                            batch_C,  // Pointer to C
                                            ldc       // Leading dimension of C
        );

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr << "cuBLAS Sgemm failed at batch " << b << std::endl;
            return -1;
        }
    }

    for(int i = 0; i < 32; ++i){
        LOG << corr_pyramid_[0]->data_at(0, 0, 0, i) << "\t";
    }
    LOG << std::endl;

    // buildCorrPyramid();
    corr_pyramid_[0]->SaveToNPY("corr0.npy");
    return 0;
}
#else
// 输入为的fmap1, fmap2均为 row-major, 形状分别为 k * m, k * n;
// 在 culbas中，形状分别为 m * k, n * k
// 执行 B * A^T = C^T, 在 cublas中， C^T的形状为 n * m，C^T为 row-major
int CorrBlock::computeCorr(const std::shared_ptr<Blob<float>>& fmap1,
                           const std::shared_ptr<Blob<float>>& fmap2)
{
    const float* d_fmap1 = fmap1->gpu_data();                     // Pointer to fmap1 data on
    const float* d_fmap2 = fmap2->gpu_data();                     // Pointer to fmap2 data on
    float*       d_corr  = corr_pyramid_[0]->mutable_gpu_data();  // Output pointer

    int batch = fmap1->shape(0);  // Batch size
    int dim   = fmap1->shape(1);  // DIM
    int ht    = fmap1->shape(2);  // Height
    int wd    = fmap1->shape(3);  // Width

    // Scalars for cuBLAS
    float alpha = 1.0f / sqrtf(static_cast<float>(dim));
    float beta  = 0.0f;

    int hw = ht * wd;  // Number of pixels
    int m  = hw;       // Number of rows of op(A) and C
    int n  = hw;       // Number of columns of op(B) and C
    int k  = dim;      // Shared dimension

    // Adjust leading dimensions for row-major storage
    int lda = n;  // Leading dimension of A
    int ldb = m;  // Leading dimension of B
    int ldc = m;  // Leading dimension of C

    // Strides between batches (not needed for loop over cublasSgemm)
    long long int strideA = static_cast<long long>(m) * k;
    long long int strideB = static_cast<long long>(n) * k;
    long long int strideC = static_cast<long long>(m) * n;

    cublasOperation_t transa = CUBLAS_OP_N;  // No transpose on A
    cublasOperation_t transb = CUBLAS_OP_T;  // Transpose on B

    // Loop over batches and call cublasSgemm for each batch
    for (int b = 0; b < batch; ++b)
    {
        const float* batch_A = d_fmap1 + b * strideA;
        const float* batch_B = d_fmap2 + b * strideB;
        float*       batch_C = d_corr + b * strideC;

        cublasStatus_t status = cublasSgemm(Caffe::cublas_handle(),
                                            transa,  // Operation on A
                                            transb,  // Operation on B
                                            m,       // Number of rows of op(A) and C
                                            n,       // Number of columns of op(B) and C
                                            k,       // Number of columns of op(A) and rows of
                                            &alpha,
                                            batch_B,  // Pointer to A
                                            lda,      // Leading dimension of A
                                            batch_A,  // Pointer to B
                                            ldb,      // Leading dimension of B
                                            &beta,    // Scalar beta
                                            batch_C,  // Pointer to C
                                            ldc       // Leading dimension of C
        );

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr << "cuBLAS Sgemm failed at batch " << b << std::endl;
            return -1;
        }
    }

    buildCorrPyramid();
    // corr_pyramid_[0]->SaveToNPY("corr0_row.npy");
    return 0;
}

#endif
/*
int CorrBlock::computeCorr(const std::shared_ptr<Blob<float>>& fmap1,
                           const std::shared_ptr<Blob<float>>& fmap2)
{
    const float* d_fmap1 = fmap1->gpu_data();                     // Pointer to fmap1 data on
device const float* d_fmap2 = fmap2->gpu_data();                     // Pointer to fmap2 data on
device float*       d_corr  = corr_pyramid_[0]->mutable_gpu_data();  // Output pointer

    int batch = fmap1->shape(0);  // Batch size
    int dim   = fmap1->shape(1);  // DIM
    int ht    = fmap1->shape(2);  // Height
    int wd    = fmap1->shape(3);  // Width

    // Scalars for cuBLAS
    float alpha = 1.0f / sqrtf(static_cast<float>(dim));
    float beta  = 0.0f;

    int hw = ht * wd;  // Number of pixels
    int m  = hw;       // Number of rows of op(A)
    int n  = hw;       // Number of columns of op(B)
    int k  = dim;      // Shared dimension

    // Adjust leading dimensions for row-major storage
    int lda = k;  // Leading dimension of A
    int ldb = k;  // Leading dimension of B
    int ldc = m;  // Leading dimension of C

    // Strides between batches
    long long int strideA = static_cast<long long>(m) * k;
    long long int strideB = static_cast<long long>(n) * k;
    long long int strideC = static_cast<long long>(m) * n;

    cublasOperation_t transa = CUBLAS_OP_N;  // No transpose on A
    cublasOperation_t transb = CUBLAS_OP_T;  // Transpose on B

    // Call cublasSgemmStridedBatched
    cublasStatus_t status =
        cublasSgemmStridedBatched(Caffe::cublas_handle(),
                                  transa,   // Operation on A
                                  transb,   // Operation on B
                                  m,        // Number of rows of op(A) and C
                                  n,        // Number of columns of op(B) and C
                                  k,        // Number of columns of op(A) and rows of op(B)
                                  &alpha,   // Scalar alpha
                                  d_fmap1,  // Pointer to A
                                  lda,      // Leading dimension of A
                                  strideA,  // Stride between matrices of A
                                  d_fmap2,  // Pointer to B
                                  ldb,      // Leading dimension of B
                                  strideB,  // Stride between matrices of B
                                  &beta,    // Scalar beta
                                  d_corr,   // Pointer to C
                                  ldc,      // Leading dimension of C
                                  strideC,  // Stride between matrices of C
                                  batch     // Batch count
        );

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS SgemmStridedBatched failed" << std::endl;
        return -1;
    }
    // buildCorrPyramid();
    corr_pyramid_[0]->SaveToNPY("corr0.npy");

    return 0;
}
*/

int CorrBlock::buildCorrPyramid()
{
    // 池化参数
    int windowHeight     = 2;
    int windowWidth      = 2;
    int verticalStride   = 2;
    int horizontalStride = 2;
    int padHeight        = 0;
    int padWidth         = 0;

    // 创建池化描述符
    cudnnPoolingDescriptor_t poolingDesc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc,
                                            CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                            CUDNN_NOT_PROPAGATE_NAN,
                                            windowHeight,
                                            windowWidth,
                                            padHeight,
                                            padWidth,
                                            verticalStride,
                                            horizontalStride));

    // 定义 alpha 和 beta
    float alpha = 1.0f;
    float beta  = 0.0f;

    // 创建张量描述符（在循环外创建以提高效率）
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));

    int N = corr_pyramid_[0]->shape(0);
    int C = corr_pyramid_[0]->shape(1);
    int H = corr_pyramid_[0]->shape(2);
    int W = corr_pyramid_[0]->shape(3);

    // 循环进行池化操作
    for (int i = 1; i < num_levels_; ++i)
    {
        // 设置输入张量描述符
        CUDNN_CHECK(
            cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

        // 计算输出尺寸
        int outN = N;
        int outC = C;
        int outH = (H + 2 * padHeight - windowHeight) / verticalStride + 1;
        int outW = (W + 2 * padWidth - windowWidth) / horizontalStride + 1;

        // 设置输出张量描述符
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outN, outC, outH, outW));

        const float* d_input  = corr_pyramid_[i - 1]->gpu_data();
        float*       d_output = corr_pyramid_[i]->mutable_gpu_data();

        // 执行池化操作
        CUDNN_CHECK(cudnnPoolingForward(Caffe::cudnn_handle(),
                                        poolingDesc,
                                        &alpha,
                                        inputDesc,
                                        d_input,
                                        &beta,
                                        outputDesc,
                                        d_output));

        H = outH;
        W = outW;
    }
    // 清理资源
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(inputDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(outputDesc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));

    return 0;
}

int CorrBlock::generateDelta()
{
    // create_delta(delta_lvl_, radius_);

    return 0;
}

int CorrBlock::call(const std::shared_ptr<Blob<float>>& coords,
                    std::shared_ptr<Blob<float>>&       output)
{
    int ret = 0;

    return ret;
}

}  // namespace ferrari