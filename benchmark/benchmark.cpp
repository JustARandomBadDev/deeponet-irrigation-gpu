#include "golden_fixture.hpp"
#include "onnx_runtime_engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>

#ifndef DEEPONET_BUILD_TYPE
#define DEEPONET_BUILD_TYPE "Unknown"
#endif

namespace {

constexpr float kCorrectnessTolerance = 1.0e-4F;
constexpr std::array<std::size_t, 4> kBatchSizes = {1, 8, 32, 128};

enum class BenchmarkMode {
    Cpu,
    Cuda,
    CudaIoBinding,
};

struct Options {
    std::string provider = "all";
    std::size_t warmup_iterations = 100;
    std::size_t measured_iterations = 1000;
    std::size_t repetitions = 3;
};

struct BenchmarkInput {
    std::vector<float> branch;
    std::vector<float> horizon;
    std::vector<float> current_moisture;
    std::vector<float> expected;
};

struct Statistics {
    double mean_ms;
    double median_ms;
    double p95_ms;
    double minimum_ms;
    double maximum_ms;
    double standard_deviation_ms;
    double throughput_samples_per_second;
};

struct BenchmarkResult {
    BenchmarkMode mode;
    std::size_t batch_size;
    std::size_t repetition;
    bool aggregate;
    std::size_t warmup_iterations;
    std::size_t measured_iterations;
    Statistics statistics;
};

const char* mode_identifier(BenchmarkMode p_mode) {
    switch (p_mode) {
        case BenchmarkMode::Cpu:
            return "cpu";
        case BenchmarkMode::Cuda:
            return "cuda";
        case BenchmarkMode::CudaIoBinding:
            return "cuda_iobinding";
    }
    throw std::logic_error("Unknown benchmark mode");
}

const char* mode_label(BenchmarkMode p_mode) {
    switch (p_mode) {
        case BenchmarkMode::Cpu:
            return "ORT CPU";
        case BenchmarkMode::Cuda:
            return "ORT CUDA naive";
        case BenchmarkMode::CudaIoBinding:
            return "ORT CUDA I/O Binding";
    }
    throw std::logic_error("Unknown benchmark mode");
}

deeponet::ExecutionProvider execution_provider(BenchmarkMode p_mode) {
    return p_mode == BenchmarkMode::Cpu
               ? deeponet::ExecutionProvider::Cpu
               : deeponet::ExecutionProvider::Cuda;
}

std::size_t parse_count(const char* p_text, const char* p_option) {
    const std::string text(p_text);
    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    if (consumed != text.size() || value == 0) {
        throw std::invalid_argument(std::string(p_option) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

Options parse_options(int p_argc, char** p_argv) {
    Options options;
    for (int i = 1; i < p_argc; i++) {
        const std::string_view argument(p_argv[i]);
        if (argument == "--provider" && i + 1 < p_argc) {
            options.provider = p_argv[i + 1];
            i++;
        } else if (argument == "--warmup" && i + 1 < p_argc) {
            options.warmup_iterations = parse_count(p_argv[i + 1], "--warmup");
            i++;
        } else if (argument == "--iterations" && i + 1 < p_argc) {
            options.measured_iterations = parse_count(
                p_argv[i + 1],
                "--iterations"
            );
            i++;
        } else if (argument == "--repetitions" && i + 1 < p_argc) {
            options.repetitions = parse_count(p_argv[i + 1], "--repetitions");
            i++;
        } else {
            throw std::invalid_argument(
                "Usage: ./build/deeponet_benchmark --provider "
                "cpu|cuda|cuda-iobinding|all "
                "[--warmup N] [--iterations N] [--repetitions N]"
            );
        }
    }

    const std::array<std::string_view, 4> valid = {
        "cpu",
        "cuda",
        "cuda-iobinding",
        "all",
    };
    if (std::find(valid.begin(), valid.end(), options.provider) == valid.end()) {
        throw std::invalid_argument("Unknown benchmark provider: " + options.provider);
    }
    return options;
}

std::vector<BenchmarkMode> selected_modes(const Options& p_options) {
    if (p_options.provider == "cpu") {
        return {BenchmarkMode::Cpu};
    }
    if (p_options.provider == "cuda") {
        return {BenchmarkMode::Cuda};
    }
    if (p_options.provider == "cuda-iobinding") {
        return {BenchmarkMode::CudaIoBinding};
    }
    return {
        BenchmarkMode::Cpu,
        BenchmarkMode::Cuda,
        BenchmarkMode::CudaIoBinding,
    };
}

BenchmarkInput prepare_input(
    const deeponet::GoldenFixture& p_fixture,
    std::size_t p_batch_size
) {
    BenchmarkInput input;
    input.branch.reserve(p_batch_size * p_fixture.branch_dimension);
    input.horizon.reserve(p_batch_size);
    input.current_moisture.reserve(p_batch_size);
    input.expected.reserve(p_batch_size);
    for (std::size_t i = 0; i < p_batch_size; i++) {
        const auto& sample = p_fixture.cases[i % p_fixture.cases.size()];
        input.branch.insert(
            input.branch.end(),
            sample.branch_input.begin(),
            sample.branch_input.end()
        );
        input.horizon.push_back(sample.horizon);
        input.current_moisture.push_back(sample.current_moisture);
        input.expected.push_back(sample.expected_prediction);
    }
    return input;
}

void run_inference(
    deeponet::OnnxRuntimeEngine& p_engine,
    BenchmarkMode p_mode,
    const BenchmarkInput& p_input,
    std::vector<float>& p_predictions,
    std::size_t p_batch_size
) {
    if (p_mode == BenchmarkMode::CudaIoBinding) {
        p_engine.infer_iobinding(
            p_input.branch,
            p_input.horizon,
            p_input.current_moisture,
            p_predictions,
            p_batch_size
        );
    } else {
        p_predictions = p_engine.infer(
            p_input.branch,
            p_input.horizon,
            p_input.current_moisture,
            p_batch_size
        );
    }
}

void verify_predictions(
    const std::vector<float>& p_predictions,
    const std::vector<float>& p_expected,
    BenchmarkMode p_mode,
    std::size_t p_batch_size
) {
    if (p_predictions.size() != p_expected.size()) {
        throw std::runtime_error("Benchmark correctness output size mismatch");
    }
    float maximum_error = 0.0F;
    for (std::size_t i = 0; i < p_predictions.size(); i++) {
        maximum_error = std::max(
            maximum_error,
            std::abs(p_predictions[i] - p_expected[i])
        );
    }
    if (maximum_error > kCorrectnessTolerance) {
        throw std::runtime_error(
            std::string(mode_identifier(p_mode)) + " batch " +
            std::to_string(p_batch_size) +
            " failed correctness before timing; max error=" +
            std::to_string(maximum_error)
        );
    }
}

double percentile(const std::vector<double>& p_sorted, double p_fraction) {
    const double position = p_fraction * static_cast<double>(p_sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return p_sorted[lower] * (1.0 - weight) + p_sorted[upper] * weight;
}

Statistics calculate_statistics(
    const std::vector<double>& p_latencies_ms,
    std::size_t p_batch_size
) {
    if (p_latencies_ms.empty()) {
        throw std::invalid_argument("Cannot summarize empty latency samples");
    }
    std::vector<double> sorted = p_latencies_ms;
    std::sort(sorted.begin(), sorted.end());
    const double mean = std::accumulate(
        sorted.begin(),
        sorted.end(),
        0.0
    ) / static_cast<double>(sorted.size());
    double squared_sum = 0.0;
    for (const double value : sorted) {
        const double difference = value - mean;
        squared_sum += difference * difference;
    }
    const double standard_deviation = std::sqrt(
        squared_sum / static_cast<double>(sorted.size())
    );
    return {
        .mean_ms = mean,
        .median_ms = percentile(sorted, 0.50),
        .p95_ms = percentile(sorted, 0.95),
        .minimum_ms = sorted.front(),
        .maximum_ms = sorted.back(),
        .standard_deviation_ms = standard_deviation,
        .throughput_samples_per_second =
            static_cast<double>(p_batch_size) / (mean / 1000.0),
    };
}

std::vector<BenchmarkResult> run_mode(
    BenchmarkMode p_mode,
    const deeponet::GoldenFixture& p_fixture,
    const Options& p_options,
    int p_cpu_threads
) {
    const int intra_op_threads = p_mode == BenchmarkMode::Cpu ? p_cpu_threads : 0;
    deeponet::OnnxRuntimeEngine engine(
        "models/deeponet_reference.onnx",
        execution_provider(p_mode),
        intra_op_threads
    );
    std::vector<BenchmarkResult> results;

    for (const std::size_t batch_size : kBatchSizes) {
        const auto input = prepare_input(p_fixture, batch_size);
        std::vector<float> predictions(batch_size);
        run_inference(engine, p_mode, input, predictions, batch_size);
        verify_predictions(predictions, input.expected, p_mode, batch_size);

        std::vector<double> aggregate_latencies;
        aggregate_latencies.reserve(
            p_options.repetitions * p_options.measured_iterations
        );
        for (std::size_t repetition = 1;
             repetition <= p_options.repetitions;
             repetition++) {
            for (std::size_t i = 0; i < p_options.warmup_iterations; i++) {
                run_inference(engine, p_mode, input, predictions, batch_size);
            }

            std::vector<double> latencies;
            latencies.reserve(p_options.measured_iterations);
            for (std::size_t i = 0; i < p_options.measured_iterations; i++) {
                const auto start = std::chrono::steady_clock::now();
                run_inference(engine, p_mode, input, predictions, batch_size);
                const auto stop = std::chrono::steady_clock::now();
                const double latency_ms =
                    std::chrono::duration<double, std::milli>(stop - start).count();
                latencies.push_back(latency_ms);
            }
            verify_predictions(predictions, input.expected, p_mode, batch_size);
            aggregate_latencies.insert(
                aggregate_latencies.end(),
                latencies.begin(),
                latencies.end()
            );
            results.push_back({
                .mode = p_mode,
                .batch_size = batch_size,
                .repetition = repetition,
                .aggregate = false,
                .warmup_iterations = p_options.warmup_iterations,
                .measured_iterations = p_options.measured_iterations,
                .statistics = calculate_statistics(latencies, batch_size),
            });
        }
        results.push_back({
            .mode = p_mode,
            .batch_size = batch_size,
            .repetition = 0,
            .aggregate = true,
            .warmup_iterations = p_options.warmup_iterations,
            .measured_iterations = aggregate_latencies.size(),
            .statistics = calculate_statistics(aggregate_latencies, batch_size),
        });
    }
    return results;
}

const BenchmarkResult& aggregate_result(
    const std::vector<BenchmarkResult>& p_results,
    BenchmarkMode p_mode,
    std::size_t p_batch_size
) {
    const auto result = std::find_if(
        p_results.begin(),
        p_results.end(),
        [p_mode, p_batch_size](const BenchmarkResult& p_item) {
            return p_item.aggregate && p_item.mode == p_mode &&
                   p_item.batch_size == p_batch_size;
        }
    );
    if (result == p_results.end()) {
        throw std::runtime_error("Missing aggregate benchmark result");
    }
    return *result;
}

void print_results(const std::vector<BenchmarkResult>& p_results) {
    std::cout << "\nProvider                 Batch    Mean ms     P50 ms     P95 ms"
                 "      Samples/s\n"
              << "---------------------------------------------------------------"
                 "-------------\n";
    for (const auto& result : p_results) {
        if (!result.aggregate) {
            continue;
        }
        const auto& stats = result.statistics;
        std::cout << std::left << std::setw(24) << mode_label(result.mode)
                  << std::right << std::setw(6) << result.batch_size
                  << std::setw(12) << std::fixed << std::setprecision(4)
                  << stats.mean_ms
                  << std::setw(12) << stats.median_ms
                  << std::setw(12) << stats.p95_ms
                  << std::setw(15) << std::setprecision(1)
                  << stats.throughput_samples_per_second << '\n';
    }
}

void print_speedups(const std::vector<BenchmarkResult>& p_results) {
    static constexpr std::array<BenchmarkMode, 3> modes = {
        BenchmarkMode::Cpu,
        BenchmarkMode::Cuda,
        BenchmarkMode::CudaIoBinding,
    };
    const bool has_all_modes = std::all_of(
        kBatchSizes.begin(),
        kBatchSizes.end(),
        [&p_results](std::size_t p_batch_size) {
            return std::all_of(
                modes.begin(),
                modes.end(),
                [&p_results, p_batch_size](BenchmarkMode p_mode) {
                    return std::any_of(
                        p_results.begin(),
                        p_results.end(),
                        [p_mode, p_batch_size](const BenchmarkResult& p_item) {
                            return p_item.aggregate && p_item.mode == p_mode &&
                                   p_item.batch_size == p_batch_size;
                        }
                    );
                }
            );
        }
    );
    if (!has_all_modes) {
        return;
    }

    std::cout << "\nBatch  CUDA/CPU speedup  I/O Binding/CPU speedup"
                 "  I/O Binding improvement\n"
              << "---------------------------------------------------------------"
                 "----------\n";
    for (const std::size_t batch_size : kBatchSizes) {
        const double cpu = aggregate_result(
            p_results,
            BenchmarkMode::Cpu,
            batch_size
        ).statistics.mean_ms;
        const double cuda = aggregate_result(
            p_results,
            BenchmarkMode::Cuda,
            batch_size
        ).statistics.mean_ms;
        const double iobinding = aggregate_result(
            p_results,
            BenchmarkMode::CudaIoBinding,
            batch_size
        ).statistics.mean_ms;
        const double improvement = (cuda - iobinding) / cuda * 100.0;
        std::cout << std::setw(5) << batch_size
                  << std::setw(19) << std::fixed << std::setprecision(3)
                  << cpu / cuda << 'x'
                  << std::setw(26) << cpu / iobinding << 'x'
                  << std::setw(24) << std::setprecision(2)
                  << improvement << "%\n";
    }
}

std::string json_escape(std::string_view p_value) {
    std::string escaped;
    for (const char character : p_value) {
        switch (character) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            default:
                escaped += character;
        }
    }
    return escaped;
}

std::string cpu_model() {
    std::ifstream cpu_info("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpu_info, line)) {
        if (line.rfind("model name", 0) == 0) {
            const auto separator = line.find(':');
            if (separator != std::string::npos) {
                return line.substr(separator + 2);
            }
        }
    }
    return "unknown";
}

std::string cuda_version_string(int p_version) {
    return std::to_string(p_version / 1000) + "." +
           std::to_string((p_version % 1000) / 10);
}

std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    std::ostringstream stream;
    stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

void write_environment(
    const std::filesystem::path& p_path,
    const Options& p_options,
    int p_cpu_threads
) {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
        throw std::runtime_error("Could not query CUDA device 0");
    }
    int runtime_version = 0;
    int driver_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) != cudaSuccess ||
        cudaDriverGetVersion(&driver_version) != cudaSuccess) {
        throw std::runtime_error("Could not query CUDA runtime/driver versions");
    }

    std::ofstream output(p_path);
    if (!output) {
        throw std::runtime_error("Could not write environment metadata");
    }
    output << "{\n"
           << "  \"timestamp_utc\": \"" << utc_timestamp() << "\",\n"
           << "  \"gpu_model\": \"" << json_escape(properties.name) << "\",\n"
           << "  \"cpu_model\": \"" << json_escape(cpu_model()) << "\",\n"
           << "  \"cuda_runtime_version\": \""
           << cuda_version_string(runtime_version) << "\",\n"
           << "  \"cuda_driver_api_version\": \""
           << cuda_version_string(driver_version) << "\",\n"
           << "  \"onnxruntime_version\": \"" << Ort::GetVersionString()
           << "\",\n"
           << "  \"build_type\": \"" << DEEPONET_BUILD_TYPE << "\",\n"
           << "  \"cpu_intra_op_threads\": " << p_cpu_threads << ",\n"
           << "  \"cpu_inter_op_threads\": 1,\n"
           << "  \"warmup_iterations_per_repetition\": "
           << p_options.warmup_iterations << ",\n"
           << "  \"measured_iterations_per_repetition\": "
           << p_options.measured_iterations << ",\n"
           << "  \"repetitions\": " << p_options.repetitions << ",\n"
           << "  \"benchmark_modes\": [\"cpu\", \"cuda\", "
              "\"cuda_iobinding\"],\n"
           << "  \"batch_sizes\": [1, 8, 32, 128],\n"
           << "  \"input_source\": "
              "\"deterministic repetition of existing golden vectors\",\n"
           << "  \"available_execution_providers\": [";
    const auto providers = deeponet::OnnxRuntimeEngine::available_providers();
    for (std::size_t i = 0; i < providers.size(); i++) {
        if (i != 0) {
            output << ", ";
        }
        output << '"' << json_escape(providers[i]) << '"';
    }
    output << "]\n}\n";
}

void write_csv(
    const std::filesystem::path& p_path,
    const std::vector<BenchmarkResult>& p_results
) {
    std::ofstream output(p_path);
    if (!output) {
        throw std::runtime_error("Could not write benchmark CSV");
    }
    output << "scope,repetition,provider,batch_size,warmup_iterations,"
              "measured_iterations,mean_latency_ms,median_latency_ms,"
              "p95_latency_ms,min_latency_ms,max_latency_ms,"
              "stddev_latency_ms,throughput_samples_per_second,"
              "microseconds_per_sample\n";
    output << std::setprecision(10);
    for (const auto& result : p_results) {
        const auto& stats = result.statistics;
        output << (result.aggregate ? "aggregate" : "repetition") << ','
               << result.repetition << ','
               << mode_identifier(result.mode) << ','
               << result.batch_size << ','
               << result.warmup_iterations << ','
               << result.measured_iterations << ','
               << stats.mean_ms << ','
               << stats.median_ms << ','
               << stats.p95_ms << ','
               << stats.minimum_ms << ','
               << stats.maximum_ms << ','
               << stats.standard_deviation_ms << ','
               << stats.throughput_samples_per_second << ','
               << stats.mean_ms * 1000.0 / static_cast<double>(result.batch_size)
               << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (std::string_view(DEEPONET_BUILD_TYPE) != "Release") {
            throw std::runtime_error(
                "Benchmarks require -DCMAKE_BUILD_TYPE=Release; current build is "
                DEEPONET_BUILD_TYPE
            );
        }
        const Options options = parse_options(argc, argv);
        const auto fixture = deeponet::load_golden_fixture(
            "models/deeponet_reference_golden.bin"
        );
        const unsigned int hardware_threads = std::thread::hardware_concurrency();
        const int cpu_threads = static_cast<int>(
            hardware_threads == 0 ? 1 : hardware_threads
        );

        std::cout << "Build type: " << DEEPONET_BUILD_TYPE << '\n'
                  << "ONNX Runtime: " << Ort::GetVersionString() << '\n'
                  << "CPU intra-op threads: " << cpu_threads << '\n'
                  << "Warmup/iterations/repetitions: "
                  << options.warmup_iterations << '/'
                  << options.measured_iterations << '/'
                  << options.repetitions << '\n';

        std::vector<BenchmarkResult> results;
        for (const BenchmarkMode mode : selected_modes(options)) {
            std::cout << "Running " << mode_identifier(mode) << "...\n";
            auto mode_results = run_mode(
                mode,
                fixture,
                options,
                cpu_threads
            );
            results.insert(
                results.end(),
                mode_results.begin(),
                mode_results.end()
            );
        }

        print_results(results);
        print_speedups(results);
        const std::filesystem::path output_directory = "results/benchmark";
        std::filesystem::create_directories(output_directory);
        write_csv(output_directory / "onnx_runtime_benchmark.csv", results);
        write_environment(
            output_directory / "environment.json",
            options,
            cpu_threads
        );
        std::cout << "\nCSV: results/benchmark/onnx_runtime_benchmark.csv\n"
                  << "Environment: results/benchmark/environment.json\n";
        return 0;
    } catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}

