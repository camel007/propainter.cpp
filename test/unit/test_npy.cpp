#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "blob.hpp"
#include "npy.hpp"

const int Nx = 128;
const int Ny = 64;
const int Nz = 32;
TEST_CASE("blob test", "[test_blob]")
{
    float matrix[] = {0.77891346,
                      0.17474553,
                      0.74379612,
                      0.03985735,
                      0.70188098,
                      0.05263425,
                      0.44053011,
                      0.09026388,
                      0.29514856,
                      0.51547146,
                      0.82821231,
                      0.28403936,
                      0.55605025,
                      0.46127602,
                      0.24216507};

    ferrari::Blob<float> b;
    b.LoadFromNPY("test_data/3x5.npy");
    REQUIRE(sizeof(matrix) / sizeof(float) == b.count());
    const float* data = b.cpu_data();

    for (size_t i = 0; i < b.count(); ++i)
    {
        REQUIRE(data[i] == matrix[i]);
    }

    b.SaveToNPY("./temp.npy");

    cnpy::NpyArray array = cnpy::npy_load("./temp.npy");
    const float* d = array.data<float>();
    for (size_t i = 0; i < b.count(); ++i)
    {
        REQUIRE(d[i] == matrix[i]);
    }
}
