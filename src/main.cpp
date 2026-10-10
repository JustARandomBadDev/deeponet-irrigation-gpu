#include "golden_fixture.hpp"
#include "onnx_runtime_engine.hpp"

#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

deeponet::ExecutionProvider parse_provider(int p_argc, char** p_argv) {
    if (p_argc != 2) {
        throw std::invalid_argument("Usage: ./build/deeponet_ort --cpu|--cuda");
    }
    const std::string_view argument(p_argv[1]);
    if (argument == "--cpu") {
        return deeponet::ExecutionProvider::Cpu;
    }
    if (argument == "--cuda") {
        return deeponet::ExecutionProvider::Cuda;
    }
    throw std::invalid_argument("Provider must be --cpu or --cuda");
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
        const auto provider = parse_provider(argc, argv);
        const std::filesystem::path model = "models/deeponet_reference.onnx";
        const std::filesystem::path fixture_path =
            "models/deeponet_reference_golden.bin";
        const auto fixture = deeponet::load_golden_fixture(fixture_path);
        const auto& sample = fixture.cases.front();

        print_available_providers();
        std::cout << "Requested provider: " << deeponet::provider_name(provider) << '\n';
        deeponet::OnnxRuntimeEngine engine(model, provider);
        const auto prediction = engine.infer(
            sample.branch_input,
            {sample.horizon},
            {sample.current_moisture},
            1
        );
        const float absolute_error = std::abs(prediction.front() - sample.expected_prediction);

        std::cout << std::fixed << std::setprecision(8)
                  << "Horizon (hours): " << sample.horizon * 24.0F << '\n'
                  << "Current moisture: " << sample.current_moisture << '\n'
                  << "Expected prediction: " << sample.expected_prediction << '\n'
                  << "ONNX Runtime prediction: " << prediction.front() << '\n'
                  << "Absolute error: " << absolute_error << '\n';
        return absolute_error <= 1.0e-4F ? 0 : 1;
    } catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}

