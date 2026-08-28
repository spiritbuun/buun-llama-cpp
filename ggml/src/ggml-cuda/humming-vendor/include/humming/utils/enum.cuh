#pragma once


enum class WeightScaleType : uint32_t {
  GROUP,
  BLOCK,
  CHANNEL,
  TENSOR,
};


enum class WeightScale2Type : uint32_t {
  NONE,
  CHANNEL,
  TENSOR,
};


enum class MmaType : uint32_t {
  MMA,
  WGMMA,
  MXMMA
};


enum class GemmType : uint32_t {
  DENSE,
  INDEXED,
  GROUPED_CONTIGUOUS,
  GROUPED_MASKED,
};
