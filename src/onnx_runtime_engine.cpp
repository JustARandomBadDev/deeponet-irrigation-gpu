#include "onnx_runtime_engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include <cuda_runtime_api.h>
#include <onnxruntime_session_options_config_keys.h>

namespace deeponet {
namespace {

constexpr std::size_t kBranchDimension = 723;
constexpr std::size_t kMaximumIoBindingBatch = 128;
constexpr std::array<const char*, 3> kInputNames = {
    "branch_input",
    "horizon",
    "current_moisture",
};
constexpr std::array<const char*, 1> kOutputNames = {"prediction"};

struct ExpectedTensor {
    std::string_view name;
    std::array<std::int64_t, 2> shape;
};

constexpr std::array<ExpectedTensor, 3> kExpectedInputs = {{
    {"branch_input", {-1, 723}},
    {"horizon", {-1, 1}},
    {"current_moisture", {-1, 1}},
}};
constexpr ExpectedTensor kExpectedOutput = {"prediction", {-1, 1}};

std::string format_shape(const std::vector<std::int64_t>& p_shape) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t i = 0; i < p_shape.size(); i++) {
        if (i != 0) {
            stream << ',';
        }
        stream << p_shape[i];
    }
    stream << ']';
    return stream.str();
}

void validate_tensor_info(
    const std::string& p_name,
    const Ort::TypeInfo& p_type_info,
    const ExpectedTensor& p_expected
) {
    const auto tensor_info = p_type_info.GetTensorTypeAndShapeInfo();
    if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("Tensor '" + p_name + "' is not float32");
    }

    const auto shape = tensor_info.GetShape();
    const std::vector<std::int64_t> expected_shape(
        p_expected.shape.begin(),
        p_expected.shape.end()
    );
    if (shape != expected_shape) {
        throw std::runtime_error(
            "Tensor '" + p_name + "' has shape " + format_shape(shape) +
            ", expected " + format_shape(expected_shape)
        );
    }
}

const ExpectedTensor& expected_input(const std::string& p_name) {
    const auto item = std::find_if(
        kExpectedInputs.begin(),
        kExpectedInputs.end(),
        [&p_name](const ExpectedTensor& p_tensor) {
            return p_tensor.name == p_name;
        }
    );
    if (item == kExpectedInputs.end()) {
        throw std::runtime_error("Unexpected model input tensor: " + p_name);
    }
    return *item;
}

bool contains_provider(
    const std::vector<std::string>& p_providers,
    std::string_view p_name
) {
    return std::find(p_providers.begin(), p_providers.end(), p_name) !=
           p_providers.end();
}

void check_cuda(cudaError_t p_status, const char* p_operation) {
    if (p_status != cudaSuccess) {
        throw std::runtime_error(
            std::string(p_operation) + " failed: " + cudaGetErrorString(p_status)
        );
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t p_float_capacity)
        : float_capacity_(p_float_capacity) {
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&data_),
                float_capacity_ * sizeof(float)
            ),
            "cudaMalloc"
        );
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            const cudaError_t status = cudaFree(data_);
            if (status != cudaSuccess) {
                std::cerr << "cudaFree failed during cleanup: "
                          << cudaGetErrorString(status) << '\n';
            }
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] float* data() const {
        return data_;
    }

private:
    float* data_ = nullptr;
    std::size_t float_capacity_;
};

}  // namespace

struct OnnxRuntimeEngine::CudaIoResources {
    CudaIoResources()
        : branch(kMaximumIoBindingBatch * kBranchDimension),
          horizon(kMaximumIoBindingBatch),
          current_moisture(kMaximumIoBindingBatch),
          prediction(kMaximumIoBindingBatch) {
        tensors.reserve(4);
    }

    DeviceBuffer branch;
    DeviceBuffer horizon;
    DeviceBuffer current_moisture;
    DeviceBuffer prediction;
    std::unique_ptr<Ort::IoBinding> binding;
    std::vector<Ort::Value> tensors;
    std::size_t bound_batch = 0;
};

const char* provider_name(ExecutionProvider p_provider) {
    switch (p_provider) {
        case ExecutionProvider::Cpu:
            return "CPUExecutionProvider";
        case ExecutionProvider::Cuda:
            return "CUDAExecutionProvider";
    }
    throw std::logic_error("Unknown execution provider");
}

OnnxRuntimeEngine::OnnxRuntimeEngine(
    const std::filesystem::path& p_model_path,
    ExecutionProvider p_provider,
    int p_intra_op_threads
)
    : provider_(p_provider),
      environment_(ORT_LOGGING_LEVEL_WARNING, "deeponet_ort") {
    if (!std::filesystem::is_regular_file(p_model_path)) {
        throw std::runtime_error("ONNX model does not exist: " + p_model_path.string());
    }

    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (p_intra_op_threads < 0) {
        throw std::invalid_argument("intra-op thread count cannot be negative");
    }
    if (p_intra_op_threads > 0) {
        session_options_.SetIntraOpNumThreads(p_intra_op_threads);
        session_options_.SetInterOpNumThreads(1);
    }

    if (provider_ == ExecutionProvider::Cuda) {
        const auto providers = available_providers();
        if (!contains_provider(providers, "CUDAExecutionProvider")) {
            throw std::runtime_error(
                "CUDAExecutionProvider was requested but is not available in this "
                "ONNX Runtime build"
            );
        }

        Ort::CUDAProviderOptions cuda_options;
        cuda_options.Update({{"device_id", "0"}});
        session_options_.AppendExecutionProvider_CUDA_V2(*cuda_options);
        session_options_.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
    }

    session_ = Ort::Session(environment_, p_model_path.c_str(), session_options_);
    validate_model_contract();
}

OnnxRuntimeEngine::~OnnxRuntimeEngine() = default;

std::vector<float> OnnxRuntimeEngine::infer(
    const std::vector<float>& p_branch_input,
    const std::vector<float>& p_horizon,
    const std::vector<float>& p_current_moisture,
    std::size_t p_batch_size
) {
    validate_inputs(
        p_branch_input,
        p_horizon,
        p_current_moisture,
        p_batch_size
    );

    const auto batch = static_cast<std::int64_t>(p_batch_size);
    const std::array<std::int64_t, 2> branch_shape = {batch, 723};
    const std::array<std::int64_t, 2> scalar_shape = {batch, 1};
    const auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    );

    std::vector<Ort::Value> inputs;
    inputs.reserve(kInputNames.size());
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(p_branch_input.data()),
        p_branch_input.size(),
        branch_shape.data(),
        branch_shape.size()
    ));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(p_horizon.data()),
        p_horizon.size(),
        scalar_shape.data(),
        scalar_shape.size()
    ));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(p_current_moisture.data()),
        p_current_moisture.size(),
        scalar_shape.data(),
        scalar_shape.size()
    ));

    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        kInputNames.data(),
        inputs.data(),
        inputs.size(),
        kOutputNames.data(),
        kOutputNames.size()
    );
    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
        throw std::runtime_error("ONNX Runtime returned an invalid prediction output");
    }

    const auto output_info = outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<std::int64_t> expected_shape = {batch, 1};
    const auto output_shape = output_info.GetShape();
    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_shape != expected_shape ||
        output_info.GetElementCount() != p_batch_size) {
        throw std::runtime_error(
            "prediction has an unexpected runtime shape or element type"
        );
    }

    const float* output_data = outputs[0].GetTensorData<float>();
    std::vector<float> predictions(output_data, output_data + p_batch_size);
    if (!std::all_of(predictions.begin(), predictions.end(), [](float p_value) {
            return std::isfinite(p_value);
        })) {
        throw std::runtime_error("prediction contains a non-finite value");
    }
    return predictions;
}

void OnnxRuntimeEngine::infer_iobinding(
    const std::vector<float>& p_branch_input,
    const std::vector<float>& p_horizon,
    const std::vector<float>& p_current_moisture,
    std::span<float> p_predictions,
    std::size_t p_batch_size
) {
    if (provider_ != ExecutionProvider::Cuda) {
        throw std::logic_error("I/O Binding requires CUDAExecutionProvider");
    }
    validate_inputs(
        p_branch_input,
        p_horizon,
        p_current_moisture,
        p_batch_size
    );
    if (p_batch_size > kMaximumIoBindingBatch) {
        throw std::invalid_argument("I/O Binding supports a maximum batch of 128");
    }
    if (p_predictions.size() != p_batch_size) {
        throw std::invalid_argument("prediction span must contain batch_size floats");
    }

    prepare_iobinding(p_batch_size);
    const std::size_t branch_bytes = p_branch_input.size() * sizeof(float);
    const std::size_t scalar_bytes = p_batch_size * sizeof(float);
    check_cuda(
        cudaMemcpy(
            cuda_io_->branch.data(),
            p_branch_input.data(),
            branch_bytes,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy branch_input host-to-device"
    );
    check_cuda(
        cudaMemcpy(
            cuda_io_->horizon.data(),
            p_horizon.data(),
            scalar_bytes,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy horizon host-to-device"
    );
    check_cuda(
        cudaMemcpy(
            cuda_io_->current_moisture.data(),
            p_current_moisture.data(),
            scalar_bytes,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy current_moisture host-to-device"
    );

    session_.Run(Ort::RunOptions{nullptr}, *cuda_io_->binding);
    cuda_io_->binding->SynchronizeOutputs();
    check_cuda(
        cudaMemcpy(
            p_predictions.data(),
            cuda_io_->prediction.data(),
            scalar_bytes,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy prediction device-to-host"
    );
    if (!std::all_of(p_predictions.begin(), p_predictions.end(), [](float p_value) {
            return std::isfinite(p_value);
        })) {
        throw std::runtime_error("prediction contains a non-finite value");
    }
}

void OnnxRuntimeEngine::validate_inputs(
    const std::vector<float>& p_branch_input,
    const std::vector<float>& p_horizon,
    const std::vector<float>& p_current_moisture,
    std::size_t p_batch_size
) const {
    if (p_batch_size == 0) {
        throw std::invalid_argument("Batch size must be greater than zero");
    }
    if (p_batch_size > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        p_batch_size > std::numeric_limits<std::size_t>::max() / kBranchDimension) {
        throw std::invalid_argument("Batch size exceeds ONNX Runtime shape limits");
    }
    if (p_branch_input.size() != p_batch_size * kBranchDimension) {
        throw std::invalid_argument(
            "branch_input must contain batch_size * 723 float32 values"
        );
    }
    if (p_horizon.size() != p_batch_size) {
        throw std::invalid_argument("horizon must contain batch_size float32 values");
    }
    if (p_current_moisture.size() != p_batch_size) {
        throw std::invalid_argument(
            "current_moisture must contain batch_size float32 values"
        );
    }
}

void OnnxRuntimeEngine::prepare_iobinding(std::size_t p_batch_size) {
    if (!cuda_io_) {
        cuda_io_ = std::make_unique<CudaIoResources>();
    }
    if (cuda_io_->binding && cuda_io_->bound_batch == p_batch_size) {
        return;
    }

    const auto batch = static_cast<std::int64_t>(p_batch_size);
    const std::array<std::int64_t, 2> branch_shape = {batch, 723};
    const std::array<std::int64_t, 2> scalar_shape = {batch, 1};
    const Ort::MemoryInfo cuda_memory(
        "Cuda",
        OrtDeviceAllocator,
        0,
        OrtMemTypeDefault
    );

    cuda_io_->binding = std::make_unique<Ort::IoBinding>(session_);
    cuda_io_->tensors.clear();
    cuda_io_->tensors.push_back(Ort::Value::CreateTensor<float>(
        cuda_memory,
        cuda_io_->branch.data(),
        p_batch_size * kBranchDimension,
        branch_shape.data(),
        branch_shape.size()
    ));
    cuda_io_->tensors.push_back(Ort::Value::CreateTensor<float>(
        cuda_memory,
        cuda_io_->horizon.data(),
        p_batch_size,
        scalar_shape.data(),
        scalar_shape.size()
    ));
    cuda_io_->tensors.push_back(Ort::Value::CreateTensor<float>(
        cuda_memory,
        cuda_io_->current_moisture.data(),
        p_batch_size,
        scalar_shape.data(),
        scalar_shape.size()
    ));
    cuda_io_->tensors.push_back(Ort::Value::CreateTensor<float>(
        cuda_memory,
        cuda_io_->prediction.data(),
        p_batch_size,
        scalar_shape.data(),
        scalar_shape.size()
    ));

    cuda_io_->binding->BindInput(kInputNames[0], cuda_io_->tensors[0]);
    cuda_io_->binding->BindInput(kInputNames[1], cuda_io_->tensors[1]);
    cuda_io_->binding->BindInput(kInputNames[2], cuda_io_->tensors[2]);
    cuda_io_->binding->BindOutput(kOutputNames[0], cuda_io_->tensors[3]);
    cuda_io_->bound_batch = p_batch_size;
}

std::vector<std::string> OnnxRuntimeEngine::available_providers() {
    return Ort::GetAvailableProviders();
}

void OnnxRuntimeEngine::validate_model_contract() const {
    if (session_.GetInputCount() != kExpectedInputs.size()) {
        throw std::runtime_error("Model must expose exactly three input tensors");
    }
    if (session_.GetOutputCount() != 1) {
        throw std::runtime_error("Model must expose exactly one output tensor");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    for (std::size_t i = 0; i < session_.GetInputCount(); i++) {
        const auto name = session_.GetInputNameAllocated(i, allocator);
        const std::string input_name(name.get());
        const auto& expected = expected_input(input_name);
        validate_tensor_info(input_name, session_.GetInputTypeInfo(i), expected);
    }

    const auto output_name = session_.GetOutputNameAllocated(0, allocator);
    const std::string output(output_name.get());
    if (output != kExpectedOutput.name) {
        throw std::runtime_error("Unexpected model output tensor: " + output);
    }
    validate_tensor_info(output, session_.GetOutputTypeInfo(0), kExpectedOutput);
}

}  // namespace deeponet

