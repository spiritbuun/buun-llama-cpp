# Embedded SM120 Gated Delta Net kernels

These six SM120 cubins implement the exact-shape prefill route used by
`gated_delta_net_fla_ptx.cu`: D=128, H=48, HK=16, BT=64.
The host dispatcher owns supported token counts, state modes and layout gates.
Other shapes use the ordinary backend.

The source is vLLM 0.28.0's vendored flash-linear-attention implementation,
compiled with Triton 3.7.1. The source files carry Apache-2.0 notices and the
original MIT notice for Songlin Yang and Yu Zhang. Both license texts are
included here. `manifest.json` records the AOT metadata, signatures and constants
from the compiled artifacts.

On supported ELF x86-64 builds CMake embeds these objects in ggml-cuda.
Normal execution needs no external kernel directory.
`GGML_CUDA_GDN_FLA_PTX_DIR` remains a developer override, not an enable switch.

SHA-256 digests:

```text
114b291fa96a8b4a13a2ba2b3ecb2306e51716a4d3c4d7b3c29afd7243d0f4ae  sm120_chunk_fwd_kernel_o.cubin
9984eea2ce05603613841c505c44a640dedde866e974b7717bbe0d278171d341  sm120_chunk_gated_delta_rule_fwd_kernel_h_blockdim64.cubin
2899d7efa96d268733053585dbe17065275bd51a342cc656d3853d01a3ed2481  sm120_chunk_local_cumsum_scalar_kernel.cubin
000ac6e8a96de4ed0065589c25667677f59e8bbbccb15bcf5c310ae0ff02aa0f  sm120_chunk_scaled_dot_kkt_fwd_kernel.cubin
aabbca571a3457c69756f6ce17d4f88310d5671e8d1f6d373f5865c3ddf9bc58  sm120_merge_16x16_to_64x64_inverse_kernel.cubin
f471f9642f8e9342cc1ad81f5091a05e72213de6f705ecf1dcd43fa985cc18ff  sm120_recompute_w_u_fwd_kernel.cubin
```
