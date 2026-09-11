# Embedded SM80 Gated Delta Net kernels

These cubins are the six exact-shape Triton kernels selected by vLLM 0.28.0
on an A100 for the Qwen3.6/3.8-27B prefill Gated Delta Net shape (`T=512`,
`H=48`, `HK=16`, `D=128`, `BT=64`). They are loaded only on an SM80 device;
all other devices and shapes use their matching specialization or the portable
GGML kernels.

The source kernels are vLLM's vendored copy of flash-linear-attention. vLLM's
files carry `SPDX-License-Identifier: Apache-2.0`, retain the original MIT
copyright notice for Songlin Yang and Yu Zhang, and identify the copied
flash-linear-attention source. Both license texts are included here.

The binaries were produced by vLLM 0.28.0's Triton AOT cache after autotuning
the exact model shape on an A100-SXM4-80GB. Their SHA-256 digests are:

```text
e61c32fa907368784df7f8b45543c19e3caaaaa1bf0c5476f0bc6848d7b3aa10  sm80_chunk_local_cumsum_scalar_kernel.cubin
039ab63ac0da4d72b349ffe85d02644d32ec1542a28fa24ed54a76f72a096563  sm80_chunk_scaled_dot_kkt_fwd_kernel.cubin
3066814a99115a72fc29a404f7bd749a3b30e97fb6124b281ce5073d1e68f8cf  sm80_merge_16x16_to_64x64_inverse_kernel.cubin
bad96ec12439c0267787b293b3114b6b1ce73b6bd9af8013ea7ebc658bd7e923  sm80_recompute_w_u_fwd_kernel.cubin
d7def6e6bc23d5b67f3b09358baa682074b34b2513c92db4646c564bd84a61b5  sm80_chunk_gated_delta_rule_fwd_kernel_h_blockdim64.cubin
ef69b15f55b5880ac35adaf932cf2d92bc32425a8ce28e1e5863c49094ca265c  sm80_chunk_fwd_kernel_o.cubin
```

On ELF x86-64 builds CMake wraps these files as read-only objects and links
them into `ggml-cuda`. The optional `GGML_CUDA_GDN_FLA_PTX_DIR` environment
variable remains available for developer A/B tests and overrides the embedded
modules when set.
