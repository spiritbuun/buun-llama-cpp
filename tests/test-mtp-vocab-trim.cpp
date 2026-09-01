#include "ggml.h"
#include "gguf.h"
#include "mtp-vocab-trim.h"
#include "speculative.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

void test_mtp_carry_lifecycle() {
    common_speculative_mtp_carry_lifecycle carry;
    float                                  pending_h = 12345.0f;

    assert(!carry.draft_ready());
    assert(carry.draft_carry(&pending_h) == nullptr);
    assert(carry.target_process_mode(0) ==
           common_speculative_mtp_carry_lifecycle::process_mode::cold_zero);
    assert(carry.target_process_mode(1) ==
           common_speculative_mtp_carry_lifecycle::process_mode::target_only);

    carry.target_process_refreshed();
    assert(carry.draft_ready());
    assert(carry.draft_carry(&pending_h) == &pending_h);
    assert(carry.target_process_mode(17) ==
           common_speculative_mtp_carry_lifecycle::process_mode::retained_carry);

    const common_speculative_sequence_event transitions[] = {
        common_speculative_sequence_event::prompt_rewind,
        common_speculative_sequence_event::target_restored_without_draft,
        common_speculative_sequence_event::draft_image_restored,
        common_speculative_sequence_event::composite_image_restored,
        common_speculative_sequence_event::live_range_shift,
        common_speculative_sequence_event::target_replaced,
        common_speculative_sequence_event::full_clear,
    };

    for (size_t i = 0; i < sizeof(transitions) / sizeof(transitions[0]); ++i) {
        // Poison the old branch's carry. Every external lifecycle transition,
        // including the target+draft image used by child clone, must suppress
        // its consumption until a target process publishes a fresh row.
        pending_h = 20000.0f + (float) i;
        carry.sequence_transition(transitions[i]);
        assert(!carry.draft_ready());
        assert(carry.draft_carry(&pending_h) == nullptr);
        assert(carry.target_process_mode(37) ==
               common_speculative_mtp_carry_lifecycle::process_mode::target_only);

        carry.target_process_skipped();
        assert(!carry.draft_ready());
        assert(carry.draft_carry(&pending_h) == nullptr);

        pending_h = 100.0f + (float) i;
        carry.target_process_refreshed();
        assert(carry.draft_ready());
        assert(carry.draft_carry(&pending_h) == &pending_h);
    }

    std::vector<common_speculative_mtp_carry_lifecycle> batch_carry(2);
    const std::vector<int32_t> active = { 0, 1 };
    llama_pos positions[] = { 0, 0 };
    assert(common_speculative_mtp_process_preflight_resolve(
               batch_carry, active, positions) ==
           common_speculative_mtp_process_preflight::cold_or_retained);

    positions[1] = 37;
    assert(common_speculative_mtp_process_preflight_resolve(
               batch_carry, active, positions) ==
           common_speculative_mtp_process_preflight::target_only);

    batch_carry[1].target_process_refreshed();
    assert(common_speculative_mtp_process_preflight_resolve(
               batch_carry, active, positions) ==
           common_speculative_mtp_process_preflight::cold_or_retained);

    batch_carry[0].target_process_refreshed();
    positions[0] = 91;
    assert(common_speculative_mtp_process_preflight_resolve(
               batch_carry, active, positions) ==
           common_speculative_mtp_process_preflight::cold_or_retained);

    const std::vector<int32_t> only_first = { 0, -1 };
    batch_carry[0].sequence_transition(
        common_speculative_sequence_event::prompt_rewind);
    assert(common_speculative_mtp_process_preflight_resolve(
               batch_carry, only_first, positions) ==
           common_speculative_mtp_process_preflight::target_only);
}

struct files_cleanup {
    std::vector<std::filesystem::path> paths;

    ~files_cleanup() {
        std::error_code ec;
        for (const auto & path : paths) {
            std::filesystem::remove(path, ec);
        }
    }
};

}  // namespace

int main() {
    test_mtp_carry_lifecycle();

    std::vector<std::string> digest_tokens = { "!", "hello", "▁world" };
    assert(common_mtp_vocab_trim_tokenizer_digest_for_test(digest_tokens) ==
           "7954b97c711bbcb1c5197e525208e499600bf8e405c2724d8afa8e29e626f119");
    digest_tokens[1] = "Hello";
    assert(common_mtp_vocab_trim_tokenizer_digest_for_test(digest_tokens) !=
           "7954b97c711bbcb1c5197e525208e499600bf8e405c2724d8afa8e29e626f119");

    const auto disabled = common_mtp_vocab_trim_prepare("missing.gguf", 0);
    assert(disabled.status == common_mtp_vocab_trim_status::not_applicable);
    assert(disabled.path == "missing.gguf");
    const auto unsupported_size = common_mtp_vocab_trim_prepare("missing.gguf", 16384);
    assert(unsupported_size.status == common_mtp_vocab_trim_status::failed);
    assert(unsupported_size.detail == "draft vocabulary size must be 0 (disabled) or 32768");

    const auto                  nonce   = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::filesystem::path source  = "test-mtp-vocab-trim-source-" + std::to_string(nonce) + ".gguf";
    const std::filesystem::path output  = "test-mtp-vocab-trim-output-" + std::to_string(nonce) + ".gguf";
    const std::filesystem::path refused = "test-mtp-vocab-trim-refused-" + std::to_string(nonce) + ".gguf";
    files_cleanup               cleanup{
                      { source, output, refused }
    };

    ggml_init_params tensor_params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * tensor_ctx = ggml_init(tensor_params);
    assert(tensor_ctx != nullptr);
    gguf_context * source_gguf = gguf_init_empty();
    assert(source_gguf != nullptr);

    gguf_set_val_str(source_gguf, "general.architecture", "test");
    gguf_set_val_str(source_gguf, "test.marker", "preserved");
    // The derivative writer always normalizes its declaration and layout to
    // GGUF_DEFAULT_ALIGNMENT, even when the source declares it explicitly.
    gguf_set_val_u32(source_gguf, GGUF_KEY_GENERAL_ALIGNMENT, GGUF_DEFAULT_ALIGNMENT);

    constexpr int64_t n_embd     = 32;
    constexpr int64_t n_vocab    = 64;
    ggml_tensor *     token_embd = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_F32, n_embd, n_vocab);
    ggml_set_name(token_embd, "token_embd.weight");
    std::memset(token_embd->data, 0x5a, ggml_nbytes(token_embd));
    gguf_add_tensor(source_gguf, token_embd);

    ggml_tensor * output_weight = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_Q8_0, n_embd, n_vocab);
    ggml_set_name(output_weight, "output.weight");
    const size_t row_size = ggml_row_size(output_weight->type, n_embd);
    for (int64_t token = 0; token < n_vocab; ++token) {
        std::memset(static_cast<uint8_t *>(output_weight->data) + token * row_size, static_cast<int>(token), row_size);
    }
    gguf_add_tensor(source_gguf, output_weight);

    ggml_tensor * mtp_marker = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, 8);
    ggml_set_name(mtp_marker, "blk.64.nextn_eh_proj.weight");
    std::memset(mtp_marker->data, 0xa5, ggml_nbytes(mtp_marker));
    gguf_add_tensor(source_gguf, mtp_marker);

    assert(gguf_write_to_file(source_gguf, source.string().c_str(), false));
    gguf_free(source_gguf);
    ggml_free(tensor_ctx);

    const std::vector<int64_t> map = { 0, 2, 3, 31, 50, 63 };
    std::string                error;
    assert(common_mtp_vocab_trim_repack_for_test(source.string(), output.string(), map, error));
    assert(error.empty());

    ggml_context *   output_meta_raw = nullptr;
    gguf_init_params read_params     = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &output_meta_raw,
    };
    gguf_context * output_gguf = gguf_init_from_file(output.string().c_str(), read_params);
    assert(output_gguf != nullptr);
    assert(output_meta_raw != nullptr);
    assert(gguf_get_n_tensors(output_gguf) == 4);
    const int64_t marker_id = gguf_find_key(output_gguf, "test.marker");
    assert(marker_id >= 0);
    assert(std::string(gguf_get_val_str(output_gguf, marker_id)) == "preserved");
    assert(gguf_find_key(output_gguf, GGUF_KEY_GENERAL_ALIGNMENT) < 0);

    ggml_tensor * trimmed = ggml_get_tensor(output_meta_raw, "output.weight");
    assert(trimmed != nullptr);
    assert(trimmed->type == GGML_TYPE_Q8_0);
    assert(trimmed->ne[0] == n_embd);
    assert(trimmed->ne[1] == static_cast<int64_t>(map.size()));
    for (size_t row = 0; row < map.size(); ++row) {
        const uint8_t   expected = static_cast<uint8_t>(map[row]);
        const uint8_t * data     = static_cast<const uint8_t *>(trimmed->data) + row * row_size;
        for (size_t i = 0; i < row_size; ++i) {
            assert(data[i] == expected);
        }
    }

    ggml_tensor * d2t = ggml_get_tensor(output_meta_raw, "d2t");
    assert(d2t != nullptr);
    assert(d2t->type == GGML_TYPE_I32);
    assert(d2t->ne[0] == static_cast<int64_t>(map.size()));
    const std::vector<int32_t> expected_d2t(map.begin(), map.end());
    assert(std::memcmp(d2t->data, expected_d2t.data(), expected_d2t.size() * sizeof(expected_d2t[0])) == 0);

    ggml_tensor * copied_marker = ggml_get_tensor(output_meta_raw, "blk.64.nextn_eh_proj.weight");
    assert(copied_marker != nullptr);
    for (size_t i = 0; i < ggml_nbytes(copied_marker); ++i) {
        assert(static_cast<const uint8_t *>(copied_marker->data)[i] == 0xa5);
    }

    gguf_free(output_gguf);
    ggml_free(output_meta_raw);

    const std::vector<int64_t> duplicate_map = { 0, 2, 2, 3 };
    assert(!common_mtp_vocab_trim_repack_for_test(source.string(), refused.string(), duplicate_map, error));
    assert(!error.empty());
    assert(!std::filesystem::exists(refused));

    std::cout << "test-mtp-vocab-trim: PASS\n";
    return 0;
}
