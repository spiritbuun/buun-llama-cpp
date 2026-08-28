// Minimal, dependency-free subset of vLLM's scalar type descriptor used by
// the Apache-2.0 Marlin CUDA templates vendored in this directory.
#pragma once

#include <cstdint>

namespace vllm {

class ScalarType {
public:
    using Id = int64_t;

    constexpr ScalarType(Id value = 0, int bits = 0) : value_(value), bits_(bits) {}

    constexpr Id id() const { return value_; }
    constexpr int64_t size_bits() const { return bits_; }
    constexpr bool operator==(const ScalarType & other) const { return value_ == other.value_; }
    constexpr bool operator!=(const ScalarType & other) const { return !(*this == other); }

    static constexpr ScalarType from_id(Id id) {
        switch (id) {
            case 1: return {1, 4};
            case 2: return {2, 4};
            case 3: return {3, 4};
            case 4: return {4, 8};
            case 5: return {5, 8};
            case 6: return {6, 8};
            case 7: return {7, 4};
            case 8: return {8, 8};
            case 9: return {9, 8};
            case 10: return {10, 16};
            case 11: return {11, 16};
            default: return {};
        }
    }

private:
    Id value_;
    int bits_;
};

using ScalarTypeId = ScalarType::Id;

inline constexpr ScalarType kS4        {1, 4};
inline constexpr ScalarType kU4        {2, 4};
inline constexpr ScalarType kU4B8      {3, 4};
inline constexpr ScalarType kS8        {4, 8};
inline constexpr ScalarType kU8        {5, 8};
inline constexpr ScalarType kU8B128    {6, 8};
inline constexpr ScalarType kFE2M1f    {7, 4};
inline constexpr ScalarType kFE4M3fn   {8, 8};
inline constexpr ScalarType kFE8M0fnu  {9, 8};
inline constexpr ScalarType kFloat16   {10, 16};
inline constexpr ScalarType kBFloat16  {11, 16};

} // namespace vllm
