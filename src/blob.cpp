#include "blob.hpp"

#include <climits>
#include <vector>

#include "common.hpp"
#include "math_functions.hpp"
#include "syncedmem.hpp"

namespace ferrari
{

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height, const int width)
{
    vector<int> shape(4);
    shape[0] = num;
    shape[1] = channels;
    shape[2] = height;
    shape[3] = width;
    Reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape)
{
    CHECK_LE(shape.size(), kMaxBlobAxes);
    count_ = 1;
    shape_.resize(shape.size());
    if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int))
    {
        shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
    }
    int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
    for (int i = 0; i < shape.size(); ++i)
    {
        CHECK_GE(shape[i], 0);
        if (count_ != 0)
        {
            CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
        }
        count_ *= shape[i];
        shape_[i]     = shape[i];
        shape_data[i] = shape[i];
    }
    if (count_ > capacity_)
    {
        capacity_ = count_;
        data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    }
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape)
{
    CHECK_LE(shape.dim_size(), kMaxBlobAxes);
    vector<int> shape_vec(shape.dim_size());
    for (int i = 0; i < shape.dim_size(); ++i)
    {
        shape_vec[i] = shape.dim(i);
    }
    Reshape(shape_vec);
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other)
{
    Reshape(other.shape());
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height, const int width)
    // capacity_ must be initialized before calling Reshape
    : capacity_(0)
{
    Reshape(num, channels, height, width);
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
    // capacity_ must be initialized before calling Reshape
    : capacity_(0)
{
    Reshape(shape);
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_shape() const
{
    CHECK(shape_data_);
    return (const int*)shape_data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const
{
    CHECK(data_);
    return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data)
{
    CHECK(data);
    // Make sure CPU and GPU sizes remain equal
    size_t size = count_ * sizeof(Dtype);
    if (data_->size() != size)
    {
        data_.reset(new SyncedMemory(size));
    }
    data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const
{
    CHECK(data_);
    return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_gpu_data(Dtype* data)
{
    CHECK(data);
    // Make sure CPU and GPU sizes remain equal
    size_t size = count_ * sizeof(Dtype);
    if (data_->size() != size)
    {
        data_.reset(new SyncedMemory(size));
    }
    data_->set_gpu_data(data);
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data()
{
    CHECK(data_);
    return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data()
{
    CHECK(data_);
    return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other)
{
    CHECK_EQ(count_, other.count());
    data_ = other.data();
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool reshape)
{
    if (source.count() != count_ || source.shape() != shape_)
    {
        if (reshape)
        {
            ReshapeLike(source);
        }
        else
        {
            LOG(FATAL) << "Trying to copy blobs of different sizes.";
        }
    }
    switch (Caffe::mode())
    {
        case Caffe::GPU:
            caffe_copy(count_, source.gpu_data(), static_cast<Dtype*>(data_->mutable_gpu_data()));
            break;
        case Caffe::CPU:
            caffe_copy(count_, source.cpu_data(), static_cast<Dtype*>(data_->mutable_cpu_data()));
            break;
        default:
            LOG(FATAL) << "Unknown caffe mode.";
    }
}

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace ferrari
