#pragma once

#include <cublas_v2.h>
#include <cudnn.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

namespace ferrari
{

class CorrBlock
{
public:
    CorrBlock(int batch, int dim, int ht, int wd, int num_levels = 4, int radius = 4);
    ~CorrBlock();

    int computeCorr(const float* fmap1, const float* fmap2);

    int call(const float* coords, float* output);

private:
    int buildCorrPyramid();
    int generateDelta();

private:
    int                 num_levels_;
    int                 radius_;
    int                 batch_, dim_, ht_, wd_;
    std::vector<float*> corr_pyramid_;
    cublasHandle_t      cublas_handle_;
    cudnnHandle_t       cudnn_handle_;
    float*              delta_lvl_;
};

}  // namespace ferrari