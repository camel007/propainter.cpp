#include <cuda_runtime.h>

__global__ void grid_sample_kernel(const float* __restrict__ input,
                                   const float* __restrict__ grid,
                                   float* __restrict__ output,
                                   int N,
                                   int C,
                                   int H_in,
                                   int W_in,
                                   int H_out,
                                   int W_out);
int             grid_sample(const float* input,
                            const float* grid,
                            float*       output,
                            int          N,
                            int          C,
                            int          H_in,
                            int          W_in,
                            int          H_out,
                            int          W_out);

__global__ void generate_delta(float* delta, int r);
void            create_delta(float* d_delta, int r, cudaStream_t stream = 0);

__global__ void generate_coords_grid(float* grid, int w, int h);

void create_coords_grid(float* grid, int w, int h);

__global__ void compute_grid(
    float* coords, int len, float* delta, int delta_len, float scale, int W, int H, float* output);

void broadcast_add(
    float* corrds, int len, float* delta, int delta_len, int iter, int W, int H, float* output);
