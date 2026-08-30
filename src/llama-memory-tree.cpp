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
            checkpoint_child_dependency_mode::live_guarded });
        output.push_back({ uint32_t(output.size()), iswa->get_swa(), nullptr,
            checkpoint_child_dependency_mode::payload_complete });
        return true;
    }
    if (auto * hybrid_iswa =
            dynamic_cast<llama_memory_hybrid_iswa *>(memory)) {
        if (!collect_impl(hybrid_iswa->get_mem_attn(), output)) {
            return false;
        }
        output.push_back({ uint32_t(output.size()), nullptr,
            hybrid_iswa->get_mem_recr(),
            checkpoint_child_dependency_mode::absent });
        return true;
    }
    // Indexed hybrid memory has a third, indexed attention component whose
    // artifact semantics are not represented by the two-child hybrid shape.
    // Refuse it explicitly until the indexed topology has its own collector;
    // falling through to llama_memory_hybrid would silently omit that state.
    if (dynamic_cast<llama_memory_hybrid_idx *>(memory)) {
        return false;
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory)) {
        output.push_back({ uint32_t(output.size()), hybrid->get_mem_attn(), nullptr,
            checkpoint_child_dependency_mode::live_guarded });
        output.push_back({ uint32_t(output.size()), nullptr,
            hybrid->get_mem_recr(),
            checkpoint_child_dependency_mode::absent });
        return true;
    }
    if (auto * kv = dynamic_cast<llama_kv_cache *>(memory)) {
        // A prompt artifact must bind the selected sequence's logical rows to
        // their physical cells.  Even though a plain KV cache has no sibling
        // state dependency, treating it as payload-complete discards that
        // placement and leaves the resulting artifact unusable for restore.
        output.push_back({ uint32_t(output.size()), kv, nullptr,
            checkpoint_child_dependency_mode::live_guarded });
        return true;
    }
    if (auto * recurrent = dynamic_cast<llama_memory_recurrent *>(memory)) {
        output.push_back({ uint32_t(output.size()), nullptr, recurrent,
            checkpoint_child_dependency_mode::absent });
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
