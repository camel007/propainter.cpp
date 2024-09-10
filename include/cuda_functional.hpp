#pragma once

#include <cuda_runtime.h>

#include "blob.hpp"

namespace ferrari
{

int grid_sample(std::shared_ptr<Blob<float>>& input,
                std::shared_ptr<Blob<float>>& grid,
                std::shared_ptr<Blob<float>>& output);

void create_delta(std::shared_ptr<Blob<float>>& delta, int r);

void create_coords_grid(std::shared_ptr<Blob<float>>& coords);

__global__ void compute_grid(const float* coords,
                             int          len,
                             const float* delta,
                             int          delta_len,
                             float        scale,
                             int          W,
                             int          H,
                             float*       output);

void broadcast_add(const std::shared_ptr<Blob<float>>& coords,
                   const std::shared_ptr<Blob<float>>& delta,
                   int                                 iter,
                   int                                 W,
                   int                                 H,
                   std::shared_ptr<Blob<float>>&       output);

}  // namespace ferrari