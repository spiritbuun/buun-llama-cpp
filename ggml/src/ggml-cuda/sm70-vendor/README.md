# Vendored third-party headers — SM70 D256 attention plugin

## cute/ + cutlass/ (152 headers)
- Source: https://github.com/NVIDIA/cutlass
- Commit: 62750a2b75c802660e4894434dc55e839f322277
- License: Apache-2.0 (see LICENSE-cute-cutlass)
- Why: the 1Cat SM70 D256 Split-D flash-attention kernel is written in CuTe
  (MMA_Atom<SM70_8x8x4> etc). llama.cpp does not vendor cute; this plugin
  needs the transitive include closure (152 headers, header-only).

## flash/ (8 headers)
- Source: https://github.com/zhinianqin/flash-attention-v100
- Commit: c2eda5e6115b98c3ba4bfd181570668742eece22
- License: BSD-3-Clause (FlashAttention lineage; see LICENSE-flash-attention)
- Why: the kernel uses the FA2 base layer for online softmax
  (sm70_reduce_max / sm70_row_slot / sm70_row_allreduce_8 /
  sm70_scale_apply_exp2 in softmax.h), masking (mask.h), and kernel
  traits (kernel_traits.h). These were part of the base fork the 1Cat
  D256 patch builds against (see 1CatAI/1Cat-vLLM vllm_flash_attn.cmake).

## The kernel itself
- fattn-sm70-d256.cu is adapted from the 1CatAI D256 Split-D kernel
  (1Cat-vLLM v1.3.0, cmake/patches/sm70_flash_attn_d256_pipeline.patch),
  (c) 1CatAI, BSD-3-Clause compatible; see the header of that file.

Nothing here is compiled beyond what fattn-sm70-d256.cu includes; all are
header-only.
