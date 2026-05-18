// Instantiate BSA hdim=128 bf16 non-causal forward kernel.
// Slow compile (cutlass templates) — separate TU for incremental builds.

// Disable -Werror for this file (third-party cutlass/BSA code)
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wmissing-declarations"

#include "flash_fwd_block_hdim128_bf16_sm80.cu"
