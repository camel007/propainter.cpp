#pragma once

#include <string>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "blob.hpp"
#include "common.hpp"

namespace ferrari
{

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            delete obj;
        }
    }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, InferDeleter>;

class TrtInfer
{
public:
    TrtInfer() : runtime_(nullptr), engine_(nullptr) {}
    ~TrtInfer()
    {
        engine_.reset();   // Release the engine first
        runtime_.reset();  // Then release the runtime
    }
    bool loadEngine(const std::string& model_file_json);

    bool infer(const std::vector<SharedBlob<float>>& inputs,
               std::vector<SharedBlob<float>>&       outputs);

private:
    std::map<std::string, std::vector<int>> input_blob_name_to_index_;
    std::map<std::string, std::vector<int>> output_blob_name_to_index_;

    UniquePtr<nvinfer1::ICudaEngine> engine_;
    UniquePtr<nvinfer1::IRuntime>    runtime_;

    DISABLE_COPY_AND_ASSIGN(TrtInfer);
};

}  // namespace ferrari