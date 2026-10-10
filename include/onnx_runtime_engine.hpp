#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace deeponet {

enum class ExecutionProvider {
    Cpu,
    Cuda,
};

[[nodiscard]] const char* provider_name(ExecutionProvider p_provider);

class OnnxRuntimeEngine {
public:
    OnnxRuntimeEngine(
        const std::filesystem::path& p_model_path,
        ExecutionProvider p_provider,
        int p_intra_op_threads = 0
    );
    ~OnnxRuntimeEngine();

    OnnxRuntimeEngine(const OnnxRuntimeEngine&) = delete;
    OnnxRuntimeEngine& operator=(const OnnxRuntimeEngine&) = delete;

    std::vector<float> infer(
        const std::vector<float>& p_branch_input,
        const std::vector<float>& p_horizon,
        const std::vector<float>& p_current_moisture,
        std::size_t p_batch_size
    );

    void infer_iobinding(
        const std::vector<float>& p_branch_input,
        const std::vector<float>& p_horizon,
        const std::vector<float>& p_current_moisture,
        std::span<float> p_predictions,
        std::size_t p_batch_size
    );

    [[nodiscard]] static std::vector<std::string> available_providers();

private:
    struct CudaIoResources;

    void validate_inputs(
        const std::vector<float>& p_branch_input,
        const std::vector<float>& p_horizon,
        const std::vector<float>& p_current_moisture,
        std::size_t p_batch_size
    ) const;
    void prepare_iobinding(std::size_t p_batch_size);
    void validate_model_contract() const;

    ExecutionProvider provider_;
    Ort::Env environment_;
    Ort::SessionOptions session_options_;
    Ort::Session session_{nullptr};
    std::unique_ptr<CudaIoResources> cuda_io_;
};

}  // namespace deeponet

