#include "golden_fixture.hpp"
#include "onnx_runtime_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr float kTolerance = 1.0e-4F;

enum class GoldenMode {
    Cpu,
    Cuda,
    CudaIoBinding,
};

struct ErrorSummary {
    float maximum = 0.0F;
    double total = 0.0;
    std::size_t count = 0;

    void add(float p_error) {
        maximum = std::max(maximum, p_error);
        total += p_error;
        count++;
    }

    [[nodiscard]] double mean() const {
        return count == 0 ? 0.0 : total / static_cast<double>(count);
    }
};

GoldenMode parse_mode(int p_argc, char** p_argv) {
    if (p_argc != 2) {
        throw std::invalid_argument(
            "Usage: ./build/deeponet_golden_test "
            "--cpu|--cuda|--cuda-iobinding"
        );
    }
    const std::string_view argument(p_argv[1]);
    if (argument == "--cpu") {
        return GoldenMode::Cpu;
    }
    if (argument == "--cuda") {
        return GoldenMode::Cuda;
    }
    if (argument == "--cuda-iobinding") {
        return GoldenMode::CudaIoBinding;
    }
    throw std::invalid_argument(
        "Provider must be --cpu, --cuda, or --cuda-iobinding"
    );
}

deeponet::ExecutionProvider execution_provider(GoldenMode p_mode) {
    return p_mode == GoldenMode::Cpu
               ? deeponet::ExecutionProvider::Cpu
               : deeponet::ExecutionProvider::Cuda;
}

const char* mode_name(GoldenMode p_mode) {
    switch (p_mode) {
        case GoldenMode::Cpu:
            return "cpu";
        case GoldenMode::Cuda:
            return "cuda";
        case GoldenMode::CudaIoBinding:
            return "cuda_iobinding";
    }
    throw std::logic_error("Unknown golden-test mode");
}

std::vector<float> run_batch(
    deeponet::OnnxRuntimeEngine& p_engine,
    const deeponet::GoldenFixture& p_fixture,
    std::size_t p_batch_size,
    bool p_use_iobinding
) {
    std::vector<float> branch;
    std::vector<float> horizon;
    std::vector<float> current;
    branch.reserve(p_batch_size * p_fixture.branch_dimension);
    horizon.reserve(p_batch_size);
    current.reserve(p_batch_size);

    for (std::size_t i = 0; i < p_batch_size; i++) {
        const auto& sample = p_fixture.cases[i];
        branch.insert(branch.end(), sample.branch_input.begin(), sample.branch_input.end());
        horizon.push_back(sample.horizon);
        current.push_back(sample.current_moisture);
    }
    if (!p_use_iobinding) {
        return p_engine.infer(branch, horizon, current, p_batch_size);
    }
    std::vector<float> predictions(p_batch_size);
    p_engine.infer_iobinding(
        branch,
        horizon,
        current,
        predictions,
        p_batch_size
    );
    return predictions;
}

ErrorSummary validate_batch(
    deeponet::OnnxRuntimeEngine& p_engine,
    const deeponet::GoldenFixture& p_fixture,
    std::size_t p_batch_size,
    bool p_use_iobinding
) {
    const auto predictions = run_batch(
        p_engine,
        p_fixture,
        p_batch_size,
        p_use_iobinding
    );
    ErrorSummary summary;
    std::cout << "Batch " << p_batch_size << ":\n";
    for (std::size_t i = 0; i < p_batch_size; i++) {
        const auto& sample = p_fixture.cases[i];
        const float error = std::abs(predictions[i] - sample.expected_prediction);
        summary.add(error);
        std::cout << "  case=" << i
                  << " horizon_hours=" << sample.horizon * 24.0F
                  << " expected=" << sample.expected_prediction
                  << " actual=" << predictions[i]
                  << " absolute_error=" << error << '\n';
    }
    std::cout << "  max_absolute_error=" << summary.maximum
              << " mean_absolute_error=" << summary.mean() << '\n';
    return summary;
}

ErrorSummary compare_batches(
    deeponet::OnnxRuntimeEngine& p_left,
    deeponet::OnnxRuntimeEngine& p_right,
    const deeponet::GoldenFixture& p_fixture,
    const std::vector<std::size_t>& p_batch_sizes,
    bool p_left_iobinding,
    bool p_right_iobinding
) {
    ErrorSummary summary;
    for (const std::size_t batch_size : p_batch_sizes) {
        const auto left = run_batch(
            p_left,
            p_fixture,
            batch_size,
            p_left_iobinding
        );
        const auto right = run_batch(
            p_right,
            p_fixture,
            batch_size,
            p_right_iobinding
        );
        for (std::size_t i = 0; i < batch_size; i++) {
            summary.add(std::abs(left[i] - right[i]));
        }
    }
    return summary;
}

void print_available_providers() {
    std::cout << "ONNX Runtime version: " << Ort::GetVersionString() << '\n';
    std::cout << "Available ONNX Runtime providers:";
    for (const auto& provider : deeponet::OnnxRuntimeEngine::available_providers()) {
        std::cout << ' ' << provider;
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto mode = parse_mode(argc, argv);
        const auto provider = execution_provider(mode);
        const bool use_iobinding = mode == GoldenMode::CudaIoBinding;
        const std::filesystem::path model = "models/deeponet_reference.onnx";
        const auto fixture = deeponet::load_golden_fixture(
            "models/deeponet_reference_golden.bin"
        );
        if (fixture.cases.size() < 8) {
            throw std::runtime_error("Golden fixture needs at least eight cases");
        }

        print_available_providers();
        std::cout << "Requested mode: " << mode_name(mode) << '\n'
                  << "Requested provider: " << deeponet::provider_name(provider) << '\n'
                  << "Golden cases: " << fixture.cases.size() << '\n'
                  << std::fixed << std::setprecision(8);

        deeponet::OnnxRuntimeEngine engine(model, provider);
        const std::vector<std::size_t> batch_sizes = {1, 8, fixture.cases.size()};
        ErrorSummary overall;
        for (const std::size_t batch_size : batch_sizes) {
            const auto result = validate_batch(
                engine,
                fixture,
                batch_size,
                use_iobinding
            );
            overall.maximum = std::max(overall.maximum, result.maximum);
            overall.total += result.total;
            overall.count += result.count;
        }

        std::cout << "Overall max_absolute_error=" << overall.maximum << '\n'
                  << "Overall mean_absolute_error=" << overall.mean() << '\n'
                  << "Tolerance=" << kTolerance << '\n';
        if (overall.maximum > kTolerance) {
            std::cerr << "FAIL: golden-vector tolerance exceeded\n";
            return 1;
        }

        if (provider == deeponet::ExecutionProvider::Cuda) {
            deeponet::OnnxRuntimeEngine cpu_engine(
                model,
                deeponet::ExecutionProvider::Cpu
            );
            const auto cross_provider = compare_batches(
                cpu_engine,
                engine,
                fixture,
                batch_sizes,
                false,
                use_iobinding
            );
            std::cout << "CPU_vs_CUDA max_absolute_difference="
                      << cross_provider.maximum << '\n'
                      << "CPU_vs_CUDA mean_absolute_difference="
                      << cross_provider.mean() << '\n';
            if (cross_provider.maximum > kTolerance) {
                std::cerr << "FAIL: CPU/CUDA cross-provider tolerance exceeded\n";
                return 1;
            }
        }
        if (mode == GoldenMode::CudaIoBinding) {
            const auto naive_vs_iobinding = compare_batches(
                engine,
                engine,
                fixture,
                batch_sizes,
                false,
                true
            );
            std::cout << "CUDA_naive_vs_iobinding max_absolute_difference="
                      << naive_vs_iobinding.maximum << '\n'
                      << "CUDA_naive_vs_iobinding mean_absolute_difference="
                      << naive_vs_iobinding.mean() << '\n';
            if (naive_vs_iobinding.maximum > kTolerance) {
                std::cerr << "FAIL: naive/I/O Binding tolerance exceeded\n";
                return 1;
            }
        }
        std::cout << "PASS: all golden-vector checks passed\n";
        return 0;
    } catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}

