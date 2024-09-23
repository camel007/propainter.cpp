#pragma once

#include <cublas_v2.h>
#include <cudnn.h>

#include <vector>

#include "blob.hpp"

namespace ferrari
{

class CorrBlock
{
public:
    CorrBlock(int batch, int dim, int ht, int wd, int num_levels = 4, int radius = 4);
    ~CorrBlock();

    int computeCorr(const std::shared_ptr<Blob<float>>& fmap1,
                    const std::shared_ptr<Blob<float>>& fmap2);

    int call(const float* coords, float* output);
    int call(const std::shared_ptr<Blob<float>>& coords, std::shared_ptr<Blob<float>>& output);

private:
    int buildCorrPyramid();
    int generateDelta();

private:
    int                                       num_levels_;
    int                                       radius_;
    int                                       batch_, dim_, ht_, wd_;
    std::vector<std::shared_ptr<Blob<float>>> corr_pyramid_;
    std::shared_ptr<Blob<float>>              delta_lvl_;
};

}  // namespace ferrari