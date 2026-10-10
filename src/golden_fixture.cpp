#include "golden_fixture.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace deeponet {
namespace {

constexpr std::array<char, 8> kMagic = {'D', 'O', 'N', 'G', 'O', 'L', 'D', '1'};

template <typename T>
void read_exact(std::ifstream& p_stream, T* p_destination, std::size_t p_count) {
    const auto bytes = static_cast<std::streamsize>(sizeof(T) * p_count);
    p_stream.read(reinterpret_cast<char*>(p_destination), bytes);
    if (!p_stream) {
        throw std::runtime_error("Golden fixture ended unexpectedly");
    }
}

}  // namespace

GoldenFixture load_golden_fixture(const std::filesystem::path& p_fixture_path) {
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("Golden fixture requires a little-endian host");
    }

    std::ifstream stream(p_fixture_path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(
            "Could not open golden fixture: " + p_fixture_path.string()
        );
    }

    std::array<char, 8> magic{};
    std::uint32_t case_count = 0;
    std::uint32_t branch_dimension = 0;
    read_exact(stream, magic.data(), magic.size());
    read_exact(stream, &case_count, 1);
    read_exact(stream, &branch_dimension, 1);

    if (magic != kMagic) {
        throw std::runtime_error("Golden fixture has an invalid magic header");
    }
    if (case_count == 0) {
        throw std::runtime_error("Golden fixture contains no cases");
    }
    if (branch_dimension != kBranchDimension) {
        throw std::runtime_error("Golden fixture branch dimension must be 723");
    }

    GoldenFixture fixture{
        .branch_dimension = branch_dimension,
        .cases = {},
    };
    fixture.cases.reserve(case_count);
    for (std::uint32_t i = 0; i < case_count; i++) {
        GoldenCase item{
            .branch_input = std::vector<float>(branch_dimension),
            .horizon = 0.0F,
            .current_moisture = 0.0F,
            .expected_prediction = 0.0F,
        };
        read_exact(stream, item.branch_input.data(), item.branch_input.size());
        read_exact(stream, &item.horizon, 1);
        read_exact(stream, &item.current_moisture, 1);
        read_exact(stream, &item.expected_prediction, 1);
        fixture.cases.push_back(std::move(item));
    }

    char trailing_byte = 0;
    if (stream.read(&trailing_byte, 1)) {
        throw std::runtime_error("Golden fixture contains trailing data");
    }
    if (!stream.eof()) {
        throw std::runtime_error("Could not finish reading golden fixture");
    }
    return fixture;
}

}  // namespace deeponet

