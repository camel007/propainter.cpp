#include <cmath>

#include "cuda_functional.hpp"

namespace ferrari
{
// CUDA 内核函数定义
__global__ void grid_sample_kernel(const float* __restrict__ input,
                                   const float* __restrict__ grid,
                                   float* __restrict__ output,
                                   int N,
                                   int C,
                                   int H_in,
                                   int W_in,
                                   int H_out,
                                   int W_out)
{
    // 计算全局线程索引
    int n     = blockIdx.x;                             // 批次索引
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;  // 输出高度索引
    int w_out = blockIdx.z * blockDim.z + threadIdx.x;  // 输出宽度索引

    // 检查输出坐标是否在有效范围内
    if (h_out >= H_out || w_out >= W_out)
        return;

    // 计算 grid 中的索引
    int grid_idx = n * H_out * W_out * 2 + h_out * W_out * 2 + w_out * 2;

    // 从 grid 中获取归一化坐标
    float x = grid[grid_idx];      // 范围在 [-1, 1]
    float y = grid[grid_idx + 1];  // 范围在 [-1, 1]

    // 将归一化坐标映射到输入图像坐标系
    float x_in = 0.5f * (x + 1.0f) * (W_in - 1);
    float y_in = 0.5f * (y + 1.0f) * (H_in - 1);

    // 计算邻近的四个像素坐标
    int x0 = floorf(x_in);
    int x1 = x0 + 1;
    int y0 = floorf(y_in);
    int y1 = y0 + 1;

    // 计算插值权重
    float wx = x_in - x0;
    float wy = y_in - y0;

    // 初始化输出值为零（针对超出边界的情况）
    for (int c = 0; c < C; ++c)
    {
        float value = 0.0f;

        // 检查四个邻近像素是否在输入范围内
        bool valid_x0 = (x0 >= 0 && x0 < W_in);
        bool valid_x1 = (x1 >= 0 && x1 < W_in);
        bool valid_y0 = (y0 >= 0 && y0 < H_in);
        bool valid_y1 = (y1 >= 0 && y1 < H_in);

        // 获取四个邻近像素的值，如果超出边界则视为零
        float v00 = (valid_x0 && valid_y0)
                        ? input[n * C * H_in * W_in + c * H_in * W_in + y0 * W_in + x0]
                        : 0.0f;
        float v01 = (valid_x0 && valid_y1)
                        ? input[n * C * H_in * W_in + c * H_in * W_in + y1 * W_in + x0]
                        : 0.0f;
        float v10 = (valid_x1 && valid_y0)
                        ? input[n * C * H_in * W_in + c * H_in * W_in + y0 * W_in + x1]
                        : 0.0f;
        float v11 = (valid_x1 && valid_y1)
                        ? input[n * C * H_in * W_in + c * H_in * W_in + y1 * W_in + x1]
                        : 0.0f;

        // 进行双线性插值计算
        float val_top    = v00 * (1 - wx) + v10 * wx;
        float val_bottom = v01 * (1 - wx) + v11 * wx;
        value            = val_top * (1 - wy) + val_bottom * wy;

        // 将结果写入输出张量
        output[n * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_out] = value;
    }
}

int grid_sample(std::shared_ptr<Blob<float>>& input,
                std::shared_ptr<Blob<float>>& grid,
                std::shared_ptr<Blob<float>>& output)
{
    int N    = input->shape(0);
    int C    = input->shape(1);
    int H_in = input->shape(2);
    int W_in = input->shape(3);

    int H_out = grid->shape(1);
    int W_out = grid->shape(2);

    // 定义 CUDA 网格和线程配置
    dim3 blockDim(16, 16);  // 每个线程块 16x16 个线程
    dim3 gridDim(N, (H_out + blockDim.y - 1) / blockDim.y, (W_out + blockDim.x - 1) / blockDim.x);

    // 调用 CUDA 内核
    grid_sample_kernel<<<gridDim, blockDim>>>(input->gpu_data(),
                                              grid->gpu_data(),
                                              output->mutable_gpu_data(),
                                              N,
                                              C,
                                              H_in,
                                              W_in,
                                              H_out,
                                              W_out);

    return 0;
}

__global__ void generate_delta(float* delta, int r)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int size = 2 * r + 1;
    if (x < size && y < size)
    {
        float dx = -r + x * (2.0f * r / (size - 1));
        float dy = -r + y * (2.0f * r / (size - 1));

        int idx            = y * size + x;
        delta[idx * 2]     = dx;  // stack along last dimension (x)
        delta[idx * 2 + 1] = dy;  // stack along last dimension (y)
    }
}

void create_delta(std::shared_ptr<Blob<float>>& delta, int r)
{
    int  size = 2 * r + 1;
    dim3 blockDim(16, 16);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x, (size + blockDim.y - 1) / blockDim.y);

    // Generate delta
    generate_delta<<<gridDim, blockDim>>>(delta->mutable_gpu_data(), r);
}

__global__ void generate_coords_grid(float* coords, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h)
    {
        int idx             = y * w + x;
        coords[idx * 2]     = x;  // stack along last dimension (x)
        coords[idx * 2 + 1] = y;  // stack along last dimension (y)
    }
}

void create_coords_grid(std::shared_ptr<Blob<float>>& coords)
{
    int w = coords->shape(0);
    int h = coords->shape(1);

    dim3 blockDim(16, 16);
    dim3 gridDim((w + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);

    // Generate delta
    generate_coords_grid<<<gridDim, blockDim>>>(coords->mutable_gpu_data(), w, h);
}

__global__ void compute_grid(const float* coords,
                             int          len,
                             const float* delta,
                             int          delta_len,
                             float        scale,
                             int          W,
                             int          H,
                             float*       output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < len)
    {
        float x = coords[2 * idx] * scale;
        float y = coords[2 * idx + 1] * scale;

        int   offset   = 2 * delta_len * idx;
        float x_factor = 2.0f / (W - 1);
        float y_factor = 2.0f / (H - 1);
        for (int i = 0; i < delta_len; ++i)
        {
            float ax                   = x_factor * (x + delta[2 * i]);
            float ay                   = y_factor * (y + delta[2 * i + 1]);
            output[offset + 2 * i]     = ax - 1.0f;
            output[offset + 2 * i + 1] = ay - 1.0f;
        }
    }
}

void broadcast_add(const std::shared_ptr<Blob<float>>& coords,
                   const std::shared_ptr<Blob<float>>& delta,
                   int                                 iter,
                   int                                 W,
                   int                                 H,
                   std::shared_ptr<Blob<float>>&       output)
{
    float  scale             = 1.0f / std::pow(2, (iter - 1));
    size_t block_per_threads = 16;

    int len       = coords->count() / 2;
    int delta_len = delta->count() / 2;

    compute_grid<<<(len + block_per_threads - 1) / block_per_threads, block_per_threads>>>(
        coords->gpu_data(),
        len,
        delta->gpu_data(),
        delta_len,
        scale,
        W,
        H,
        output->mutable_gpu_data());
}

__global__ void reshapeKernel(const float* input, float* output, int batch, int dim, int ht, int wd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dim * ht * wd;

    if (idx < total) {
        int b = idx / (dim * ht * wd);
        int d = (idx % (dim * ht * wd)) / (ht * wd);
        int h = (idx % (ht * wd)) / wd;
        int w = idx % wd;

        output[b * dim * ht * wd + d * ht * wd + h * wd + w] = input[idx];
    }
}

void convert_row2colomn_major(const std::shared_ptr<Blob<float>>& input,
                              std::shared_ptr<Blob<float>>&       output)
{
    int batch = input->shape(0);
    int dim   = input->shape(1);
    int ht    = input->shape(2);
    int wd    = input->shape(3);
    // Compute the total number of elements
    int totalElements = batch * dim * ht * wd;

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid   = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    output->Reshape(batch, dim, ht, wd);

    // Launch the kernel
    reshapeKernel<<<blocksPerGrid, threadsPerBlock>>>(
        input->gpu_data(), output->mutable_gpu_data(), batch, dim, ht, wd);

    // Synchronize to ensure the kernel has finished executing
    cudaDeviceSynchronize();
}

}  // namespace ferrari