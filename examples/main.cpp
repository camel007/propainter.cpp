#include "npy.hpp"
#include "raft.hpp"

using namespace ferrari;
using namespace cnpy;

int main()
{
    CorrBlock cb(11, 256, 30, 54, 4, 4);

    std::vector<float> fmap1 = npy_load("../data/npy_files/fmap1.npy").as_vec<float>();
    std::vector<float> fmap2 = npy_load("../data/npy_files/fmap2.npy").as_vec<float>();

    // for (int i = 0; i < 11 * 256 * 30 * 54; ++i)
    // {
    //     std::cout << fmap1[i] << " ";
    // }f
    // std::cout << std::endl;
    // for (int i = 0; i < 11 * 256 * 30 * 54; ++i)
    // {
    //     std::cout << fmap2[i] << " ";
    // }
    // std::cout << std::endl;

    cb.computeCorr(fmap1.data(), fmap2.data());

    return 0;
}