#include "cuda_functional.hpp"

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

// 定义主机函数来调用 CUDA 内核
int grid_sample(const float* input,
                const float* grid,
                float*       output,
                int          N,
                int          C,
                int          H_in,
                int          W_in,
                int          H_out,
                int          W_out)
{
    // 定义 CUDA 网格和线程配置
    dim3 blockDim(16, 16);  // 每个线程块 16x16 个线程
    dim3 gridDim(N, (H_out + blockDim.y - 1) / blockDim.y, (W_out + blockDim.x - 1) / blockDim.x);

    // 调用 CUDA 内核
    grid_sample_kernel<<<gridDim, blockDim>>>(input, grid, output, N, C, H_in, W_in, H_out, W_out);

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();

    return err;
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

void create_delta(float* d_delta, int r, cudaStream_t stream)
{
    // Define grid and block dimensions
    int  size = 2 * r + 1;
    dim3 blockDim(16, 16);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x, (size + blockDim.y - 1) / blockDim.y);

    // Generate delta
    generate_delta<<<gridDim, blockDim, 0, stream>>>(d_delta, r);
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

void create_coords_grid(float* d_coords, int w, int h, cudaStream_t stream = 0)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((w + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);

    // Generate delta
    generate_coords_grid<<<gridDim, blockDim, 0, stream>>>(d_coords, w, h);
}
__global__ void compute_grid(
    float* coords, int len, float* delta, int delta_len, float scale, int W, int H, float* output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < len)
    {
        float x = coords[2 * idx] * scale;
        float y = coords[2 * idx + 1] * scale;

        int   offset   = 2 * delta_len * idx;
        float x_factor = 2.0f / (W - 1);
        float y_factor = 2.0f / (H - 1);
        for (size_t i = 0; i < delta_len; ++i)
        {
            float ax                   = x_factor * (x + delta[2 * i]);
            float ay                   = y_factor * (y + delta[2 * i + 1]);
            output[offset + 2 * i]     = ax;
            output[offset + 2 * i + 1] = ay;
        }
    }
}

void broadcast_add(
    float* coords, int len, float* delta, int delta_len, int iter, int W, int H, float* output)
{
    float  scale             = 1.0f / (2 << iter);
    size_t block_per_threads = 256;
    compute_grid<<<(len + block_per_threads - 1) / block_per_threads, block_per_threads>>>(
        coords, len, delta, delta_len, scale, W, H, output);
}