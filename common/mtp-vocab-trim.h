#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class common_mtp_vocab_trim_status {
    not_applicable,
    cached,
    created,
    failed,
};

struct common_mtp_vocab_trim_result {
    std::string                  path;
    common_mtp_vocab_trim_status status = common_mtp_vocab_trim_status::not_applicable;
    std::string                  detail;
    bool                         native_head = false;
    // Additional model-buffer residency for native/shared-weight MTP. External
    // derivatives are measured as child models and report zero here.
    size_t                       resident_bytes = 0;
};

// Prepare a cached FR-Spec-style artifact from either a supported standalone
// Qwen-27B MTP GGUF or a combined target+MTP GGUF. A standalone source becomes
// a compact child; a combined source emits only the compact native head/map.
common_mtp_vocab_trim_result common_mtp_vocab_trim_prepare(const std::string & source_path,
                                                           const std::string & vocab_pack);

// Narrow model-free seam used by the GGUF codec test. Production callers must
// use common_mtp_vocab_trim_prepare(), which owns model admission and the map.
bool common_mtp_vocab_trim_repack_for_test(const std::string &          source_path,
                                           const std::string &          destination_path,
                                           const std::vector<int64_t> & draft_to_target,
                                           std::string &                error);

bool common_mtp_vocab_trim_head_for_test(const std::string &          source_path,
                                         const std::string &          destination_path,
                                         const std::vector<int64_t> & draft_to_target,
                                         std::string &                error);

// Canonical tokenizer-token digest seam. Production admission reads the same
// serialization directly from GGUF metadata without materializing this vector.
std::string common_mtp_vocab_trim_tokenizer_digest_for_test(const std::vector<std::string> & tokens);

// Production-pipeline seam for a compact synthetic fixture. This uses the same
// admission, identity, cache, digest, publication, and validation implementation
// as common_mtp_vocab_trim_prepare(), with only the locked model dimensions/map
// and cache root supplied by the test.
common_mtp_vocab_trim_result common_mtp_vocab_trim_prepare_for_test(
        const std::string &          source_path,
        const std::string &          cache_dir,
        const std::vector<int64_t> & draft_to_target,
        size_t                       tokenizer_size,
        int64_t                      embedding_length,
        const std::string &          tokenizer_token_digest);
