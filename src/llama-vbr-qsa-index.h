#pragma once

#include "llama-vbr-artifact-adopt.h"
#include "llama-vbr-explicit-capture.h"

#include <array>
#include <memory>

class artifact_segment_chain;
class llama_memory_hybrid_idx;

uint32_t vbr_qsa_index_companion_format_version() noexcept;

std::array<uint8_t, 32>
vbr_qsa_index_companion_build_identity() noexcept;

bool vbr_qsa_index_companion_terminal(
    const void * data, size_t size, llama_pos & output) noexcept;

vbr_explicit_companion_provider vbr_qsa_index_capture_provider(
    llama_memory_hybrid_idx & owner) noexcept;

bool vbr_parse_qsa_index_companion(
    const void * context,
    const vbr_artifact_companion_payload & descriptor,
    const artifact_segment_chain & source,
    const vbr_target_companion_snapshot & target,
    std::unique_ptr<vbr_parsed_companion_image> & output) noexcept;

vbr_companion_adoption_provider vbr_qsa_index_adoption_provider(
    llama_memory_hybrid_idx & owner,
    uint32_t attention_child_id) noexcept;
