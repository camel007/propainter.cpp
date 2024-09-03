#define CATCH_CONFIG_MAIN
#include <cuda_runtime.h>

#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>

#include "cuda_functional.hpp"

TEST_CASE("grid_sample 基本功能测试", "[grid_sample]")
{
    // 设置输入参数
    const int N = 1, C = 1, H_in = 4, W_in = 4, H_out = 2, W_out = 2;
    const int input_size  = N * C * H_in * W_in;
    const int grid_size   = N * H_out * W_out * 2;
    const int output_size = N * C * H_out * W_out;

    // 分配主机内存
    float* h_input           = new float[input_size];
    float* h_grid            = new float[grid_size];
    float* h_output          = new float[output_size];
    float* h_expected_output = new float[output_size];

    // 初始化输入数据
    for (int i = 0; i < input_size; ++i)
    {
        h_input[i] = static_cast<float>(i);
    }

    // 设置网格数据（简单的恒等变换）
    h_grid[0] = -1.0f;
    h_grid[1] = -1.0f;  // 左上
    h_grid[2] = 1.0f;
    h_grid[3] = -1.0f;  // 右上
    h_grid[4] = -1.0f;
    h_grid[5] = 1.0f;  // 左下
    h_grid[6] = 1.0f;
    h_grid[7] = 1.0f;  // 右下

    // 设置预期输出
    h_expected_output[0] = 0.0f;   // 左上角
    h_expected_output[1] = 3.0f;   // 右上角
    h_expected_output[2] = 12.0f;  // 左下角
    h_expected_output[3] = 15.0f;  // 右下角

    // 分配设备内存
    float *d_input, *d_grid, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_grid, grid_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    // 将数据复制到设备
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid, h_grid, grid_size * sizeof(float), cudaMemcpyHostToDevice);

    // 调用grid_sample函数
    int error = grid_sample(d_input, d_grid, d_output, N, C, H_in, W_in, H_out, W_out);

    // 检查是否有CUDA错误
    REQUIRE(error == cudaSuccess);

    // 将结果复制回主机
    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    for (int i = 0; i < output_size; ++i)
    {
        REQUIRE(h_output[i] == Catch::Approx(h_expected_output[i]).epsilon(0.01f));
    }

    // 释放内存
    delete[] h_input;
    delete[] h_grid;
    delete[] h_output;
    delete[] h_expected_output;
    cudaFree(d_input);
    cudaFree(d_grid);
    cudaFree(d_output);
}