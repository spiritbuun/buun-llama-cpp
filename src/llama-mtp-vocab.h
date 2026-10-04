#pragma once

#include <cstdint>
#include <vector>

struct llama_model;
struct llama_vocab;

// Built-in MTP draft-vocabulary shortlist (the llamAmpere "qwen3.8-27b-65536"
// map; the same bitmap common/mtp-vocab-trim embeds for sidecar derivative
// GGUFs, sha256-verified against the llamAmpere source data). Lets built-in-head
// models (nextn block inside the main GGUF, no sidecar d2t) draft against a
// 65K-row gather of the LM head instead of the full vocabulary head.
constexpr int64_t LLAMA_MTP_VOCAB_BUILTIN_N_SEL   = 65536;
constexpr int64_t LLAMA_MTP_VOCAB_BUILTIN_N_VOCAB = 248320;

// True when the loaded vocabulary is byte-identical (content digest) to the
// tokenizer the built-in map was derived from, and has the expected size.
bool llama_mtp_vocab_builtin_matches(const llama_vocab & vocab);

// Expand the built-in bitmap into ascending token ids (n_sel entries).
std::vector<int32_t> llama_mtp_vocab_builtin_ids();

// Build the model's compact MTP draft head once: ids (I32 [n_sel]) plus a
// row-gathered copy of model.output in the head's own quant type
// ([n_embd, n_sel]). Idempotent; safe to call on every draft-context creation.
// Returns false (no side effects, one-time log) when the model is ineligible:
// sidecar d2t present, no nextn layers, arch not wired, tokenizer mismatch,
// LoRA adapters active, speculative-scale head tensors, split/host head buffer,
// LLAMA_MTP_VOCAB_MAP=0, or allocation failure. Consumers: arch graph builds
// under cparams.embeddings_nextn_masked (draft decode only).
bool llama_model_init_mtp_draft_vocab(const struct llama_model * model);
