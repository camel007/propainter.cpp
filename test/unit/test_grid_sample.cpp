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

// ... 现有代码 ...

TEST_CASE("grid_sample resize 测试", "[grid_sample]")
{
    // 设置输入参数
    const int N = 1, C = 1, H_in = 4, W_in = 4, H_out = 2, W_out = 3;
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

    // 设置网格数据（均匀分布的采样点）
    for (int i = 0; i < H_out; ++i)
    {
        for (int j = 0; j < W_out; ++j)
        {
            h_grid[(i * W_out + j) * 2]     = -1.0f + 2.0f * j / (W_out - 1);  // x
            h_grid[(i * W_out + j) * 2 + 1] = -1.0f + 2.0f * i / (H_out - 1);  // y
        }
    }

    // 设置预期输出
    h_expected_output[0] = 0.0f;   // 左上
    h_expected_output[1] = 1.5f;   // 中上
    h_expected_output[2] = 3.0f;   // 右上
    h_expected_output[3] = 12.0f;  // 左下
    h_expected_output[4] = 13.5f;  // 中下
    h_expected_output[5] = 15.0f;  // 右下

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
TEST_CASE("grid_sample zoom 测试", "[grid_sample]")
{
    float h_input[] = {
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    float h_grid[]            = {-1.0,
                                 -1.0,
                                 -0.7142857313156128,
                                 -1.0,
                                 -0.4285714030265808,
                                 -1.0,
                                 -0.14285707473754883,
                                 -1.0,
                                 0.14285707473754883,
                                 -1.0,
                                 0.4285714030265808,
                                 -1.0,
                                 0.7142857313156128,
                                 -1.0,
                                 1.0,
                                 -1.0,
                                 -1.0,
                                 -0.7142857313156128,
                                 -0.7142857313156128,
                                 -0.7142857313156128,
                                 -0.4285714030265808,
                                 -0.7142857313156128,
                                 -0.14285707473754883,
                                 -0.7142857313156128,
                                 0.14285707473754883,
                                 -0.7142857313156128,
                                 0.4285714030265808,
                                 -0.7142857313156128,
                                 0.7142857313156128,
                                 -0.7142857313156128,
                                 1.0,
                                 -0.7142857313156128,
                                 -1.0,
                                 -0.4285714030265808,
                                 -0.7142857313156128,
                                 -0.4285714030265808,
                                 -0.4285714030265808,
                                 -0.4285714030265808,
                                 -0.14285707473754883,
                                 -0.4285714030265808,
                                 0.14285707473754883,
                                 -0.4285714030265808,
                                 0.4285714030265808,
                                 -0.4285714030265808,
                                 0.7142857313156128,
                                 -0.4285714030265808,
                                 1.0,
                                 -0.4285714030265808,
                                 -1.0,
                                 -0.14285707473754883,
                                 -0.7142857313156128,
                                 -0.14285707473754883,
                                 -0.4285714030265808,
                                 -0.14285707473754883,
                                 -0.14285707473754883,
                                 -0.14285707473754883,
                                 0.14285707473754883,
                                 -0.14285707473754883,
                                 0.4285714030265808,
                                 -0.14285707473754883,
                                 0.7142857313156128,
                                 -0.14285707473754883,
                                 1.0,
                                 -0.14285707473754883,
                                 -1.0,
                                 0.14285707473754883,
                                 -0.7142857313156128,
                                 0.14285707473754883,
                                 -0.4285714030265808,
                                 0.14285707473754883,
                                 -0.14285707473754883,
                                 0.14285707473754883,
                                 0.14285707473754883,
                                 0.14285707473754883,
                                 0.4285714030265808,
                                 0.14285707473754883,
                                 0.7142857313156128,
                                 0.14285707473754883,
                                 1.0,
                                 0.14285707473754883,
                                 -1.0,
                                 0.4285714030265808,
                                 -0.7142857313156128,
                                 0.4285714030265808,
                                 -0.4285714030265808,
                                 0.4285714030265808,
                                 -0.14285707473754883,
                                 0.4285714030265808,
                                 0.14285707473754883,
                                 0.4285714030265808,
                                 0.4285714030265808,
                                 0.4285714030265808,
                                 0.7142857313156128,
                                 0.4285714030265808,
                                 1.0,
                                 0.4285714030265808,
                                 -1.0,
                                 0.7142857313156128,
                                 -0.7142857313156128,
                                 0.7142857313156128,
                                 -0.4285714030265808,
                                 0.7142857313156128,
                                 -0.14285707473754883,
                                 0.7142857313156128,
                                 0.14285707473754883,
                                 0.7142857313156128,
                                 0.4285714030265808,
                                 0.7142857313156128,
                                 0.7142857313156128,
                                 0.7142857313156128,
                                 1.0,
                                 0.7142857313156128,
                                 -1.0,
                                 1.0,
                                 -0.7142857313156128,
                                 1.0,
                                 -0.4285714030265808,
                                 1.0,
                                 -0.14285707473754883,
                                 1.0,
                                 0.14285707473754883,
                                 1.0,
                                 0.4285714030265808,
                                 1.0,
                                 0.7142857313156128,
                                 1.0,
                                 1.0,
                                 1.0};
    float h_expected_output[] = {0.0,
                                 0.4285714030265808,
                                 0.8571429252624512,
                                 1.2857143878936768,
                                 1.7142856121063232,
                                 2.142857074737549,
                                 2.5714285373687744,
                                 3.0,
                                 1.7142856121063232,
                                 2.142857074737549,
                                 2.5714285373687744,
                                 3.0,
                                 3.4285712242126465,
                                 3.857142448425293,
                                 4.285714149475098,
                                 4.714285850524902,
                                 3.4285717010498047,
                                 3.8571431636810303,
                                 4.285714626312256,
                                 4.714285850524902,
                                 5.142857074737549,
                                 5.5714287757873535,
                                 6.0,
                                 6.428571701049805,
                                 5.142857551574707,
                                 5.571429252624512,
                                 6.000000476837158,
                                 6.428571701049805,
                                 6.857143402099609,
                                 7.285714149475098,
                                 7.7142863273620605,
                                 8.142857551574707,
                                 6.857142448425293,
                                 7.285714149475098,
                                 7.714284896850586,
                                 8.14285659790039,
                                 8.571428298950195,
                                 8.999999046325684,
                                 9.428571701049805,
                                 9.857142448425293,
                                 8.571428298950195,
                                 8.999999046325684,
                                 9.428571701049805,
                                 9.857141494750977,
                                 10.285713195800781,
                                 10.714284896850586,
                                 11.142857551574707,
                                 11.571428298950195,
                                 10.285714149475098,
                                 10.714285850524902,
                                 11.14285659790039,
                                 11.571428298950195,
                                 12.0,
                                 12.428571701049805,
                                 12.85714340209961,
                                 13.285714149475098,
                                 12.0,
                                 12.428571701049805,
                                 12.85714340209961,
                                 13.285715103149414,
                                 13.714284896850586,
                                 14.14285659790039,
                                 14.571428298950195,
                                 15.0};

    // 设置输入参数
    const int N = 1, C = 1, H_in = 4, W_in = 4, H_out = 8, W_out = 8;
    const int input_size  = N * C * H_in * W_in;
    const int grid_size   = N * H_out * W_out * 2;
    const int output_size = N * C * H_out * W_out;
    float*    h_output    = new float[output_size];

    REQUIRE(sizeof(h_input) == input_size * sizeof(float));

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
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_grid);
    cudaFree(d_output);
}

// ... 现有代码 ...