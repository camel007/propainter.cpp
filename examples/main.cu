#include <iterator>
#include <memory>
#include <ostream>

#include "blob.hpp"
#include "cuda_functional.hpp"
#include "raft.hpp"

using namespace ferrari;

int main()
{
    Caffe::DeviceQuery();
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(0);

    std::shared_ptr<Blob<float>> fmap1_blob = std::make_shared<Blob<float>>();
    fmap1_blob->LoadFromNPY("../../../pt/npy_files/fmap1.npy");
    std::cout << fmap1_blob->shape_string() << std::endl;
    std::shared_ptr<Blob<float>> fmap2_blob = std::make_shared<Blob<float>>();
    fmap2_blob->LoadFromNPY("../../../pt/npy_files/fmap2.npy");
    std::cout << fmap2_blob->shape_string() << std::endl;

    // std::shared_ptr<Blob<float>> fmap1_t_blob = std::make_shared<Blob<float>>();
    // convert_row2colomn_major(fmap1_blob, fmap1_t_blob);

    // std::shared_ptr<Blob<float>> fmap2_t_blob = std::make_shared<Blob<float>>();
    // convert_row2colomn_major(fmap2_blob, fmap2_t_blob);

    CorrBlock corr(11, 256, 30, 54, 4, 4);
    corr.computeCorr(fmap1_blob, fmap2_blob);
    std::cout << __LINE__ << std::endl;
    return 0;
}
