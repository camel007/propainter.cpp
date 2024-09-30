#include <iterator>
#include <memory>
#include <ostream>

#include "blob.hpp"
#include "common.hpp"
#include "cuda_functional.hpp"
#include "raft.hpp"

using namespace ferrari;

int main()
{
    Caffe::DeviceQuery();
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(0);

    // std::shared_ptr<Blob<float>> fmap1_blob = std::make_shared<Blob<float>>();
    // fmap1_blob->LoadFromNPY("../../../pt/npy_files/fmap1.npy");
    // std::cout << fmap1_blob->shape_string() << std::endl;
    // std::shared_ptr<Blob<float>> fmap2_blob = std::make_shared<Blob<float>>();
    // fmap2_blob->LoadFromNPY("../../../pt/npy_files/fmap2.npy");
    // std::cout << fmap2_blob->shape_string() << std::endl;

    // CorrBlock corr(11, 256, 30, 54, 4, 4);
    // corr.computeCorr(fmap1_blob, fmap2_blob);

    TrtInfer infer;
    infer.loadEngine("../models/fnet");

    SharedBlob<float> image1_blob = std::make_shared<Blob<float>>();
    image1_blob->LoadFromNPY("../data/image1.npy");
    SharedBlob<float> fmap1_blob = std::make_shared<Blob<float>>();
    fmap1_blob->LoadFromNPY("../data/fmap1.npy");

    SharedBlob<float>              output_blob = std::make_shared<Blob<float>>(fmap1_blob->shape());
    std::vector<SharedBlob<float>> outputs     = {output_blob};

    infer.infer({image1_blob}, outputs);

    output_blob->SaveToNPY("output.npy");

    std::cout << "Success!" << std::endl;
    return 0;
}
