#pragma once

#include <humming/utils/base.cuh>

template <uint32_t lut>
CUDA_INLINE uint32_t lop3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
};

CUDA_INLINE uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
  return lop3<(0xF0 & 0xCC) | 0xAA>(a, b, c);
};

CUDA_INLINE uint32_t lop3_and_xor(uint32_t a, uint32_t b, uint32_t c) {
  return lop3<(0xF0 & 0xCC) ^ 0xAA>(a, b, c);
};
