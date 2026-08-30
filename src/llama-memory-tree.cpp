#include "llama-memory-tree.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

namespace {

bool collect_impl(
        llama_memory_i * memory,
        std::vector<llama_memory_tree_child> & output) {
    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(memory)) {
        output.push_back({ uint32_t(output.size()), iswa->get_base(), nullptr,
            checkpoint_child_dependency_mode::live_guarded, nullptr });
        output.push_back({ uint32_t(output.size()), iswa->get_swa(), nullptr,
            checkpoint_child_dependency_mode::payload_complete, nullptr });
        return true;
    }
    if (auto * hybrid_iswa =
            dynamic_cast<llama_memory_hybrid_iswa *>(memory)) {
        if (!collect_impl(hybrid_iswa->get_mem_attn(), output)) {
            return false;
        }
        output.push_back({ uint32_t(output.size()), nullptr,
            hybrid_iswa->get_mem_recr(),
            checkpoint_child_dependency_mode::absent, nullptr });
        return true;
    }
    if (auto * indexed = dynamic_cast<llama_memory_hybrid_idx *>(memory)) {
        output.push_back({ uint32_t(output.size()), indexed->get_mem_attn(), nullptr,
            checkpoint_child_dependency_mode::live_guarded,
            indexed->get_mem_idx() ? indexed : nullptr });
        output.push_back({ uint32_t(output.size()), nullptr,
            indexed->get_mem_recr(),
            checkpoint_child_dependency_mode::absent, nullptr });
        return true;
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
        output.push_back({ uint32_t(output.size()), hybrid->get_mem_attn(), nullptr,
            checkpoint_child_dependency_mode::live_guarded, nullptr });
        output.push_back({ uint32_t(output.size()), nullptr,
            hybrid->get_mem_recr(),
            checkpoint_child_dependency_mode::absent, nullptr });
        return true;
    }
    if (auto * kv = dynamic_cast<llama_kv_cache *>(memory)) {
        // A prompt artifact must bind the selected sequence's logical rows to
        // their physical cells.  Even though a plain KV cache has no sibling
        // state dependency, treating it as payload-complete discards that
        // placement and leaves the resulting artifact unusable for restore.
        output.push_back({ uint32_t(output.size()), kv, nullptr,
            checkpoint_child_dependency_mode::live_guarded, nullptr });
        return true;
    }
    if (auto * recurrent = dynamic_cast<llama_memory_recurrent *>(memory)) {
        output.push_back({ uint32_t(output.size()), nullptr, recurrent,
            checkpoint_child_dependency_mode::absent, nullptr });
        return true;
    }
    return false;
}

} // namespace

bool llama_memory_tree_collect(
        llama_memory_i * memory,
        std::vector<llama_memory_tree_child> & output) noexcept {
    output.clear();
    if (memory == nullptr) {
        return false;
    }
    try {
        return collect_impl(memory, output) && !output.empty();
    } catch (...) {
        output.clear();
        return false;
    }
}
