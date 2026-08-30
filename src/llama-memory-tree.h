#pragma once

#include "llama-vbr-generation-types.h"

#include <cstdint>
#include <vector>

class llama_kv_cache;
class llama_memory_i;
class llama_memory_hybrid_idx;
class llama_memory_recurrent;

// Canonical pre-order view of the memory tree. Both checkpoint generation
// capture and explicit artifact capture consume this one walk.
struct llama_memory_tree_child {
    uint32_t child_id = 0;
    llama_kv_cache * attention = nullptr;
    llama_memory_recurrent * recurrent = nullptr;
    checkpoint_child_dependency_mode dependency_mode =
        checkpoint_child_dependency_mode::absent;
    // Optional fixed-precision index state that must travel atomically with
    // this attention child. It is a companion, never another VBR controller.
    llama_memory_hybrid_idx * qsa_index_owner = nullptr;
};

bool llama_memory_tree_collect(
    llama_memory_i * memory,
    std::vector<llama_memory_tree_child> & output) noexcept;
