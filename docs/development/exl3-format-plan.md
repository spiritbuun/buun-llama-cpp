# EXL3 weights in the native safetensors path — design

Source of truth: exllamav3 (MIT, turboderp) `exllamav3_ext/quant/{codebook,exl3_dq,exl3_gemv_kernel,
hadamard_inner,reconstruct,pack}.cu*` and `modules/quant/exl3_lib/quantize.py`; QTIP (arXiv 2406.11235).

## Placement

CUDA supports one GPU, multi-GPU `--split-mode layer`, and `--split-mode tensor`.
Tensor splitting preserves whole 128-value Hadamard blocks and the corresponding
input/output sign-scale vectors. Packed column transfers select complete 16x16
tiles; expert parallelism keeps each expert's weights and vectors together and
skips routed rows assigned to another GPU. No additional EXL3 enable flag is needed.

Different GPU layouts can change floating-point reduction order and, consequently,
quantized activations and MoE routing. Cross-layout logits are not promised to be
identical. The CUDA tests check packed shard bytes, numerical operator agreement,
and repeatability within each layout. Model qualification includes dense EXL3 and
Flash-Next MoE, including multi-slot MTP serving.

CPU execution of EXL3 weights is also unsupported; do not assume that another
format's MoE CPU-cache/offload results establish EXL3 support.

## Format (as stored, e.g. turboderp/Qwen3.8-27B-exl3 @ 4.00bpw)
Per linear module `M` (in features `k`, out features `n`, both multiples of 128):
- `M.trellis` int16 `[k/16, n/16, 16*K]` — one 16x16 weight tile per `[kt][nt]`, `256*K` bits.
  The tile's 256 weights are ordered in tensor-core lane order (`tensor_core_perm`): for lane `t`
  (0..31) and slot `j` (0..7): rows `r = (t%4)*2 + {0,1}` and `+8`, cols `c = t/4` and `+8`.
  Bits are packed MSB-first in 16-bit words (`pack_trellis_kernel`, then a 16-bit word swap per
  uint32). Weight `t` of the stream is the 16-bit window ending at bit `(t+1)*K` of the circular
  stream (tail-biting): `w_t = bits[(t*K + K - 16) mod 256K, +16)` (`exl3_dq.cuh: dq`).
- codebook: 16-bit window `x` -> value. cb2 "mul1" (this checkpoint, marker `M.mul1` int32 scalar):
  `x *= 0x83DCD12D; s = dp4a(x, 0x01010101, 0x6400); v = fp16(s) * fp16(0x1eee) + fp16(0xc931)`
  (all fp16 ops). cb1 "mcg": `x *= 0xCBAC1FED; lop3(x, 0x8fff8fff, 0x3b603b60, 0x6a); v = lo+hi halves`.
  cb0 "3inst": `x = x*89226354 + 64248484; lop3 ...; v = lo+hi`.
- `M.suh` fp16 `[k]`, `M.svh` fp16 `[n]`: sign/scale vectors ("out_scales: always" folds a per-column
  scale into svh). Weight in the original basis: `W = H_k^T diag(suh) T diag(svh) H_n`-style; the
  inference contract is exactly exllamav3's:
  `xh = had128(x * suh)` (128-blocks along k, unit scale) ; `y_inner = xh @ T` ; `y = had128(y_inner) * svh`.
  `had128` = warp-level Sylvester-order Hadamard (`had_hf_r_128_inner`, r_scale = 1.0).
- Quant config: `quantization_config.json` {quant_method: exl3, bits, head_bits, codebook, out_scales}.
  K per tensor = `trellis.shape[2] / 16` (layers K=4, lm_head K=6 here; SC variants mix K).

## ggml representation
- Types `GGML_TYPE_EXL3_1 .. EXL3_8` (K = bits): logical `blck_size 16`, `type_size 2*K` bytes. Tensor
  `[ne0 = k, ne1 = n]`; `row_size = k*K/8`. Data layout is the tile stream in **n-tile-major** order
  `[n/16][k/16][32*K bytes]` so a 16-row group is contiguous. Rows are not independently decodable
  (each weight's window overlaps its neighbours across the tile), so the CPU backend does not support
  the type (same as exllamav3: GPU only); `to_float` aborts with a clear message.
- Side tensors reuse the fork's scale plumbing: `M.svh -> <target>.scale` (F16 `[n]`, MUL_MAT src[2]),
  `M.suh -> <target>.input_scale` (F16 `[k]`, MUL_MAT src[3]; the `{1}` scalar assumption is relaxed
  for EXL3 weights). Row transforms (QKV_ROWS/V_ROWS: 128-row head blocks) permute n-tiles + svh;
  V_COLUMNS (128-col blocks) permutes k-tiles + suh — both are 128-aligned, and block-Hadamard commutes
  with whole-block permutation.

## CUDA executors
- Decode (m <= 8): port of `exl3_gemv_kernel` (warps split k, no block barriers, register prefetch
  ring, one m16n8k16 MMA pair per tile, fp16 accumulate folded to fp32) with the input Hadamard done
  per warp in the prologue (it is warp-local: 128 elements x suh) and the cross-block k reduction into
  an F32 `y_inner`; a second small kernel applies `had128 * svh`. No cooperative launch (CUDA graphs).
- Prefill: `reconstruct` tiles to BF16/F16 rows (`W[n][k]`) in chunks + cuBLAS, Hadamards as kernels
  on x and y (like the Marlin GEMM route).
- Fidelity reference: numpy decode of the codebook/trellis (fp16-exact) vs our dequant kernel, then
  KLD against the fork's exact BF16 anchor; cross-check with the repo's `kld_table.json`.
