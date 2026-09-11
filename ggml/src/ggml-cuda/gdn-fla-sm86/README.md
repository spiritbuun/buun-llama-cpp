# Embedded SM86 Gated Delta Net kernels

These cubins are the six exact-shape Triton kernels used by vLLM 0.28.0 for
the Qwen3.6-27B prefill Gated Delta Net shape (`T=512`, `H=48`, `HK=16`,
`D=128`, `BT=64`). They were generated for CUDA SM86 and are loaded only on
an SM86 device; all other devices and shapes use the portable GGML kernels.

The source kernels are vLLM's vendored copy of flash-linear-attention. vLLM's
files carry `SPDX-License-Identifier: Apache-2.0`, retain the original MIT
copyright notice for Songlin Yang and Yu Zhang, and identify the copied
flash-linear-attention source. Both license texts are included in this
directory.

The binaries were produced by vLLM 0.28.0's Triton AOT cache. Their SHA-256
digests are:

```text
925808052216f67c8dde8050a516b9a9c0ca8fda2c880fb610d6f735c9896831  chunk_local_cumsum_scalar_kernel.cubin
b8b271e8d67abfd19494bcdcd694a8696638f65c097f455e485986100dd8674e  chunk_scaled_dot_kkt_fwd_kernel.cubin
8423cfb63c213f9c3694a32f34658b8040f9d7c957d0a82b5e3e0d11c8a1092a  merge_16x16_to_64x64_inverse_kernel.cubin
58c3c18729e037e090223de08e4efd250fe1dc9d2c2b49ab84d38aca4c12c61d  recompute_w_u_fwd_kernel.cubin
f9972516780c54f3e0f62f6270f5a35f59c5b2dc6bb89f8f4394f07222ef5127  chunk_gated_delta_rule_fwd_kernel_h_blockdim64.cubin
8cd2af5f085ecb92f680c93ea65e6e621855a2b59b56a8d020c67fd8a8df81dc  chunk_fwd_kernel_o.cubin
```

On ELF x86-64 builds CMake wraps these files as read-only objects and links
them into `ggml-cuda`. The optional `GGML_CUDA_GDN_FLA_PTX_DIR` environment
variable remains available for developer A/B tests and overrides the embedded
modules when set.
