#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace deeponet {

inline constexpr std::size_t kBranchDimension = 723;

struct GoldenCase {
    std::vector<float> branch_input;
    float horizon;
    float current_moisture;
    float expected_prediction;
};

struct GoldenFixture {
    std::size_t branch_dimension;
    std::vector<GoldenCase> cases;
};

[[nodiscard]] GoldenFixture load_golden_fixture(
    const std::filesystem::path& p_fixture_path
);

}  // namespace deeponet

