#include <iterator>
#include <ostream>

#include "blob.hpp"
#include "npy.hpp"
using namespace ferrari;

int main()
{
    Caffe::DeviceQuery();
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(0);

    // cnpy::NpyArray array = cnpy::npy_load("../../../pt/npy_files/fmap1.npy");

    // std::vector<size_t> shape = array.shape;
    // std::copy(shape.begin(), shape.end(), std::ostream_iterator<size_t>(std::cout, "\t"));
    // std::cout << std::endl;

    Blob<float> b;
    b.LoadFromNPY("../../../pt/npy_files/fmap1.npy");
    std::cout << b.shape_string() << std::endl;

    b.SaveToNPY("test.npy");

    return 0;
}
