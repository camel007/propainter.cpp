#include "trt_infer.hpp"

#include <rapidjson/istreamwrapper.h>

#include <fstream>

#include "NvInferPlugin.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "simple_log.hpp"

namespace ferrari
{
using namespace rapidjson;

class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override
    {
        if (severity <= Severity::kINFO)
        {
            LOG(INFO) << msg;
        }
    }
} logger;

bool TrtInfer::loadEngine(const std::string& model_dir)
{
    if (!initLibNvInferPlugins(nullptr, ""))
    {
        LOG(ERROR) << "Load trt plugin failed" << std::endl;
        return false;
    }
    runtime_.reset(nvinfer1::createInferRuntime(logger));
    assert(runtime_ != nullptr);

    const std::string parameter_json = model_dir + "/parameter.json";

    std::ifstream  ifs(parameter_json);
    IStreamWrapper isw(ifs);
    Document       model_config;
    model_config.ParseStream(isw);

    const std::string engine_filename =
        model_dir + "/model_files/" + model_config["model_files"]["name"].GetString();
    LOG(INFO) << engine_filename << std::endl;

    std::ifstream file(engine_filename, std::ios::binary);
    assert(file.good());
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), size, nullptr));
    assert(engine_ != nullptr);

    for (auto& value : model_config["model_files"]["input"].GetArray())
    {
        auto dims = engine_->getTensorShape(value.GetString());
        input_blob_name_to_index_.emplace(
            std::string(value.GetString()),
            std::vector<int>({dims.d[0], dims.d[1], dims.d[2], dims.d[3]}));
    }
    for (auto& value : model_config["model_files"]["output"].GetArray())
    {
        auto dims = engine_->getTensorShape(value.GetString());
        output_blob_name_to_index_.emplace(
            std::string(value.GetString()),
            std::vector<int>({dims.d[0], dims.d[1], dims.d[2], dims.d[3]}));
    }

    return true;
}
bool TrtInfer::infer(const std::vector<SharedBlob<float>>& inputs,
                     std::vector<SharedBlob<float>>&       outputs)
{
    auto context = UniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context)
    {
        return false;
    }
    std::vector<void*> bindings;
    for (auto& input : inputs)
    {
        void* input_mem = (void*)input->mutable_gpu_data();
        bindings.push_back(input_mem);
    }
    for (auto& output : outputs)
    {
        void* output_mem = (void*)output->mutable_gpu_data();
        bindings.push_back(output_mem);
    }

    bool status = context->executeV2(bindings.data());
    if (!status)
    {
        LOG(ERROR) << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    }

    return true;
}

}  // namespace ferrari