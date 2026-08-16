#include "ggml.h"
#include "gguf.h"
#include "mtp-vocab-trim.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct files_cleanup {
    std::vector<std::filesystem::path> paths;

    ~files_cleanup() {
        std::error_code ec;
        for (const auto & path : paths) {
            std::filesystem::remove_all(path, ec);
        }
    }
};

std::vector<std::string> fixture_tokens(size_t n_vocab) {
    std::vector<std::string> tokens;
    tokens.reserve(n_vocab);
    for (size_t i = 0; i < n_vocab; ++i) {
        tokens.push_back("fixture-token-" + std::to_string(i));
    }
    return tokens;
}

void write_supported_fixture(
        const std::filesystem::path & path,
        uint8_t payload_seed,
        bool scaled = false,
        bool native = true) {
    constexpr int64_t n_embd = 32;
    constexpr int64_t n_vocab = 64;
    ggml_init_params tensor_params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * tensor_ctx = ggml_init(tensor_params);
    assert(tensor_ctx != nullptr);
    gguf_context * gguf = gguf_init_empty();
    assert(gguf != nullptr);

    gguf_set_val_str(gguf, "general.architecture", "qwen35");
    gguf_set_val_u32(gguf, "qwen35.block_count", 65);
    gguf_set_val_u32(gguf, "qwen35.embedding_length", n_embd);
    gguf_set_val_u32(gguf, "qwen35.nextn_predict_layers", 1);
    gguf_set_val_str(gguf, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(gguf, "tokenizer.ggml.pre", "qwen35");
    const std::vector<std::string> tokens = fixture_tokens(n_vocab);
    std::vector<const char *> token_ptrs;
    token_ptrs.reserve(tokens.size());
    for (const std::string & token : tokens) {
        token_ptrs.push_back(token.c_str());
    }
    gguf_set_arr_str(gguf, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());

    auto add_matrix = [&](const char * name, uint8_t fill) {
        ggml_tensor * tensor = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_F32, n_embd, n_vocab);
        ggml_set_name(tensor, name);
        std::memset(tensor->data, fill, ggml_nbytes(tensor));
        gguf_add_tensor(gguf, tensor);
    };
    add_matrix("token_embd.weight", 0x31);
    add_matrix("output.weight", payload_seed);

    const std::vector<const char *> block_tensors = native
            ? std::vector<const char *> { "blk.0.attn_norm.weight", "blk.64.attn_norm.weight" }
            : std::vector<const char *> { "blk.64.attn_norm.weight" };
    for (const char * name : block_tensors) {
        ggml_tensor * tensor = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, n_embd);
        ggml_set_name(tensor, name);
        std::memset(tensor->data, 0x42, ggml_nbytes(tensor));
        gguf_add_tensor(gguf, tensor);
    }
    if (scaled) {
        ggml_tensor * scale = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, 1);
        ggml_set_name(scale, "output.scale");
        gguf_add_tensor(gguf, scale);
    }

    assert(gguf_write_to_file(gguf, path.string().c_str(), false));
    gguf_free(gguf);
    ggml_free(tensor_ctx);
}

void corrupt_output_payload(const std::filesystem::path & artifact) {
    gguf_init_params params = { /* .no_alloc = */ true, /* .ctx = */ nullptr };
    gguf_context * gguf = gguf_init_from_file(artifact.string().c_str(), params);
    assert(gguf != nullptr);
    const int64_t output_id = gguf_find_tensor(gguf, "output.weight");
    assert(output_id >= 0);
    const uint64_t offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, output_id);
    gguf_free(gguf);

    std::fstream file(artifact, std::ios::binary | std::ios::in | std::ios::out);
    assert(file);
    file.seekg(static_cast<std::streamoff>(offset));
    char byte = 0;
    file.read(&byte, 1);
    assert(file);
    byte ^= 0x5a;
    file.seekp(static_cast<std::streamoff>(offset));
    file.write(&byte, 1);
    file.flush();
    assert(file);
}

}  // namespace

int main() {
    std::vector<std::string> digest_tokens = { "!", "hello", "▁world" };
    assert(common_mtp_vocab_trim_tokenizer_digest_for_test(digest_tokens) ==
           "7954b97c711bbcb1c5197e525208e499600bf8e405c2724d8afa8e29e626f119");
    digest_tokens[1] = "Hello";
    assert(common_mtp_vocab_trim_tokenizer_digest_for_test(digest_tokens) !=
           "7954b97c711bbcb1c5197e525208e499600bf8e405c2724d8afa8e29e626f119");

    const auto disabled = common_mtp_vocab_trim_prepare("missing.gguf", "");
    assert(disabled.status == common_mtp_vocab_trim_status::not_applicable);
    assert(disabled.path == "missing.gguf");
    const auto unsupported_pack = common_mtp_vocab_trim_prepare("missing.gguf", "balanced");
    assert(unsupported_pack.status == common_mtp_vocab_trim_status::failed);
    assert(unsupported_pack.detail == "unknown Qwen-27B MTP vocabulary pack");
    for (const char * pack : { "mix", "eng", "code", "cn", "jp", "kr" }) {
        const auto recognized_pack = common_mtp_vocab_trim_prepare("missing.gguf", pack);
        assert(recognized_pack.status == common_mtp_vocab_trim_status::failed);
        assert(recognized_pack.detail == "could not open or identify the MTP source");
    }

    const auto                  nonce   = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::filesystem::path source  = "test-mtp-vocab-trim-source-" + std::to_string(nonce) + ".gguf";
    const std::filesystem::path output  = "test-mtp-vocab-trim-output-" + std::to_string(nonce) + ".gguf";
    const std::filesystem::path extract = "test-mtp-vocab-trim-extract-" + std::to_string(nonce) + ".gguf";
    const std::filesystem::path refused = "test-mtp-vocab-trim-refused-" + std::to_string(nonce) + ".gguf";
    const std::filesystem::path production = "test-mtp-vocab-trim-production-" + std::to_string(nonce) + ".gguf";
    const std::filesystem::path replacement = "test-mtp-vocab-trim-replacement-" + std::to_string(nonce) + ".gguf";
    const std::filesystem::path scaled = "test-mtp-vocab-trim-scaled-" + std::to_string(nonce) + ".gguf";
    const std::filesystem::path external = "test-mtp-vocab-trim-external-" + std::to_string(nonce) + ".gguf";
    const std::filesystem::path cache = "test-mtp-vocab-trim-cache-" + std::to_string(nonce);
    files_cleanup               cleanup{
                      { source, output, extract, refused, production, replacement, scaled, external, cache }
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
    ggml_set_name(mtp_marker, "blk.64.nextn.eh_proj.weight");
    std::memset(mtp_marker->data, 0xa5, ggml_nbytes(mtp_marker));
    gguf_add_tensor(source_gguf, mtp_marker);

    ggml_tensor * trunk_marker = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, 8);
    ggml_set_name(trunk_marker, "blk.0.attn_norm.weight");
    gguf_add_tensor(source_gguf, trunk_marker);

    ggml_tensor * private_embd = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_F32, n_embd, n_vocab);
    ggml_set_name(private_embd, "blk.64.nextn.embed_tokens.weight");
    gguf_add_tensor(source_gguf, private_embd);

    ggml_tensor * private_head = ggml_new_tensor_2d(tensor_ctx, GGML_TYPE_F32, n_embd, n_vocab);
    ggml_set_name(private_head, "blk.64.nextn.shared_head_head.weight");
    gguf_add_tensor(source_gguf, private_head);

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
    assert(gguf_get_n_tensors(output_gguf) == 7);
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

    ggml_tensor * copied_marker = ggml_get_tensor(output_meta_raw, "blk.64.nextn.eh_proj.weight");
    assert(copied_marker != nullptr);
    for (size_t i = 0; i < ggml_nbytes(copied_marker); ++i) {
        assert(static_cast<const uint8_t *>(copied_marker->data)[i] == 0xa5);
    }

    gguf_free(output_gguf);
    ggml_free(output_meta_raw);

    // A combined target+MTP GGUF produces a compact-head artifact only. Native MTP
    // continues sharing the target embedding and NextN layer in memory.
    assert(common_mtp_vocab_trim_head_for_test(source.string(), extract.string(), map, error));
    ggml_context * extract_meta_raw = nullptr;
    read_params.ctx = &extract_meta_raw;
    gguf_context * extract_gguf = gguf_init_from_file(extract.string().c_str(), read_params);
    assert(extract_gguf != nullptr);
    assert(extract_meta_raw != nullptr);
    assert(gguf_get_n_tensors(extract_gguf) == 2);
    assert(ggml_get_tensor(extract_meta_raw, "output.weight") != nullptr);
    assert(ggml_get_tensor(extract_meta_raw, "d2t") != nullptr);
    assert(ggml_get_tensor(extract_meta_raw, "token_embd.weight") == nullptr);
    assert(ggml_get_tensor(extract_meta_raw, "blk.64.nextn.eh_proj.weight") == nullptr);
    assert(ggml_get_tensor(extract_meta_raw, "blk.0.attn_norm.weight") == nullptr);
    assert(ggml_get_tensor(extract_meta_raw, "blk.64.nextn.embed_tokens.weight") == nullptr);
    assert(ggml_get_tensor(extract_meta_raw, "blk.64.nextn.shared_head_head.weight") == nullptr);
    gguf_free(extract_gguf);
    ggml_free(extract_meta_raw);

    const std::vector<int64_t> duplicate_map = { 0, 2, 2, 3 };
    assert(!common_mtp_vocab_trim_repack_for_test(source.string(), refused.string(), duplicate_map, error));
    assert(!error.empty());
    assert(!std::filesystem::exists(refused));

    // Exercise the real admission/cache/publication pipeline with a compact
    // Qwen-shaped fixture. Only the locked dimensions/token digest/map and cache
    // root are overridden; converter control flow is production-identical.
    write_supported_fixture(production, 0x11);
    const std::vector<std::string> prod_tokens = fixture_tokens(64);
    const std::string prod_digest = common_mtp_vocab_trim_tokenizer_digest_for_test(prod_tokens);
    const auto created = common_mtp_vocab_trim_prepare_for_test(
            production.string(), cache.string(), map, 64, 32, prod_digest);
    assert(created.status == common_mtp_vocab_trim_status::created);
    assert(created.native_head);
    assert(created.resident_bytes == map.size() * (32 * sizeof(float) + sizeof(int32_t)));
    assert(std::filesystem::is_regular_file(created.path));

    const auto cached = common_mtp_vocab_trim_prepare_for_test(
            production.string(), cache.string(), map, 64, 32, prod_digest);
    assert(cached.status == common_mtp_vocab_trim_status::cached);
    assert(cached.path == created.path);

    // A schema-preserving payload mutation must invalidate and recreate the cache.
    corrupt_output_payload(created.path);
    const auto repaired = common_mtp_vocab_trim_prepare_for_test(
            production.string(), cache.string(), map, 64, 32, prod_digest);
    assert(repaired.status == common_mtp_vocab_trim_status::created);
    assert(repaired.path == created.path);

    // Atomic same-size/same-mtime source replacement must get a different cache
    // identity even though the old size+mtime key would have collided.
    const auto original_mtime = std::filesystem::last_write_time(production);
    const auto original_size = std::filesystem::file_size(production);
    write_supported_fixture(replacement, 0x22);
    assert(std::filesystem::file_size(replacement) == original_size);
    std::filesystem::last_write_time(replacement, original_mtime);
    std::error_code replace_ec;
    std::filesystem::rename(replacement, production, replace_ec);
#if defined(_WIN32)
    if (replace_ec) {
        // Windows does not permit rename-over-existing on every filesystem. A
        // remove+rename still exercises stable file identity independently of
        // the deliberately preserved size and mtime.
        std::filesystem::remove(production, replace_ec);
        assert(!replace_ec);
        std::filesystem::rename(replacement, production, replace_ec);
    }
#endif
    assert(!replace_ec);
    const auto replaced = common_mtp_vocab_trim_prepare_for_test(
            production.string(), cache.string(), map, 64, 32, prod_digest);
    assert(replaced.status == common_mtp_vocab_trim_status::created);
    assert(replaced.path != created.path);

    // Native scaled heads are rejected before resident-byte accounting/attach.
    write_supported_fixture(scaled, 0x33, true);
    const auto scaled_result = common_mtp_vocab_trim_prepare_for_test(
            scaled.string(), cache.string(), map, 64, 32, prod_digest);
    assert(scaled_result.status == common_mtp_vocab_trim_status::not_applicable);
    assert(scaled_result.resident_bytes == 0);

    // Exercise the other production admission branch: a standalone MTP child
    // has no block-0 trunk tensor, so its derivative must retain the child model
    // and compact only output.weight. It is measured as its own model and must
    // not reserve a second output attachment beside the target.
    write_supported_fixture(external, 0x44, false, false);
    const auto external_created = common_mtp_vocab_trim_prepare_for_test(
            external.string(), cache.string(), map, 64, 32, prod_digest);
    assert(external_created.status == common_mtp_vocab_trim_status::created);
    assert(!external_created.native_head);
    assert(external_created.resident_bytes == 0);
    assert(std::filesystem::is_regular_file(external_created.path));

    ggml_context * external_meta_raw = nullptr;
    read_params.ctx = &external_meta_raw;
    gguf_context * external_gguf = gguf_init_from_file(external_created.path.c_str(), read_params);
    assert(external_gguf != nullptr);
    assert(external_meta_raw != nullptr);
    ggml_tensor * external_output = ggml_get_tensor(external_meta_raw, "output.weight");
    assert(external_output != nullptr && external_output->ne[1] == static_cast<int64_t>(map.size()));
    assert(ggml_get_tensor(external_meta_raw, "d2t") != nullptr);
    assert(ggml_get_tensor(external_meta_raw, "token_embd.weight") != nullptr);
    assert(ggml_get_tensor(external_meta_raw, "blk.64.attn_norm.weight") != nullptr);
    assert(ggml_get_tensor(external_meta_raw, "blk.0.attn_norm.weight") == nullptr);
    gguf_free(external_gguf);
    ggml_free(external_meta_raw);

    const auto external_cached = common_mtp_vocab_trim_prepare_for_test(
            external.string(), cache.string(), map, 64, 32, prod_digest);
    assert(external_cached.status == common_mtp_vocab_trim_status::cached);
    assert(external_cached.path == external_created.path);
    assert(!external_cached.native_head);
    assert(external_cached.resident_bytes == 0);

    std::cout << "test-mtp-vocab-trim: PASS\n";
    return 0;
}
