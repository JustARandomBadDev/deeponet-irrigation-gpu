cmake_minimum_required(VERSION 3.24)

project(deeponet_irrigation_gpu VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
find_package(CUDAToolkit REQUIRED)

set(ONNXRUNTIME_VERSION "1.30.0")
set(ONNXRUNTIME_ARCHIVE
    "onnxruntime-linux-x64-gpu_cuda13-${ONNXRUNTIME_VERSION}.tgz")
set(ONNXRUNTIME_URL
    "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_ARCHIVE}")

FetchContent_Declare(
    onnxruntime
    URL "${ONNXRUNTIME_URL}"
    URL_HASH SHA256=382d79133112388cf94ce5855789b7c9bef12bef76a08b6b277e5a317213adcd
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# The official archive is a binary distribution, not a CMake subproject.
# MakeAvailable only downloads/extracts it; the imported target is defined here.
FetchContent_MakeAvailable(onnxruntime)

set(ONNXRUNTIME_INCLUDE_DIR "${onnxruntime_SOURCE_DIR}/include")
set(ONNXRUNTIME_LIBRARY_DIR "${onnxruntime_SOURCE_DIR}/lib")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_LIBRARY_DIR}/libonnxruntime.so")
set(ONNXRUNTIME_CUDA_PROVIDER
    "${ONNXRUNTIME_LIBRARY_DIR}/libonnxruntime_providers_cuda.so")

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime C++ headers were not found in ${ONNXRUNTIME_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
    message(FATAL_ERROR "ONNX Runtime shared library was not found in ${ONNXRUNTIME_LIBRARY_DIR}")
endif()
if(NOT EXISTS "${ONNXRUNTIME_CUDA_PROVIDER}")
    message(FATAL_ERROR "The downloaded ONNX Runtime archive has no CUDA provider")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(
    onnxruntime::onnxruntime
    PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

add_library(
    deeponet_ort_runtime
    STATIC
    src/onnx_runtime_engine.cpp
    src/golden_fixture.cpp
)
target_include_directories(deeponet_ort_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(
    deeponet_ort_runtime
    PUBLIC onnxruntime::onnxruntime
    PRIVATE CUDA::cudart
)
target_compile_options(deeponet_ort_runtime PRIVATE -Wall -Wextra -Wpedantic)

add_executable(deeponet_ort src/main.cpp)
target_link_libraries(deeponet_ort PRIVATE deeponet_ort_runtime)
target_compile_options(deeponet_ort PRIVATE -Wall -Wextra -Wpedantic)

add_executable(deeponet_golden_test src/golden_test.cpp)
target_link_libraries(deeponet_golden_test PRIVATE deeponet_ort_runtime)
target_compile_options(deeponet_golden_test PRIVATE -Wall -Wextra -Wpedantic)

add_executable(deeponet_benchmark benchmark/benchmark.cpp)
target_link_libraries(deeponet_benchmark PRIVATE deeponet_ort_runtime CUDA::cudart)
target_compile_options(deeponet_benchmark PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(
    deeponet_benchmark
    PRIVATE DEEPONET_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
)

# The provider libraries live beside libonnxruntime.so in the fetched archive.
# Embedding that directory makes root-launched build executables work without
# requiring a manually configured LD_LIBRARY_PATH.
set_target_properties(
    deeponet_ort
    deeponet_golden_test
    deeponet_benchmark
    PROPERTIES BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}"
)

