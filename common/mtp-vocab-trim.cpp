#include "mtp-vocab-trim.h"

#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-sha256.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <system_error>
#include <vector>

namespace {

constexpr size_t       QWEN_DRAFT_VOCAB_SIZE   = 32768;
constexpr size_t       QWEN65K_DRAFT_VOCAB_SIZE = 65536;
constexpr size_t       QWEN_TOKENIZER_SIZE     = 248320;
constexpr int64_t      QWEN_EMBEDDING_LENGTH = 5120;
constexpr const char * MAP_VERSION           = "qwen27b-public-balanced-v2-i32-tokenizer-fbbabd";
constexpr const char * TOKENIZER_DOMAIN      = "buun.qwen27b-tokenizer-tokens/v1";
constexpr const char * TOKENIZER_DIGEST      = "fbbabd2048cbbddc2db0bd24f8812fa65215649d3d4b54223021ca47ae6be487";
constexpr const char * META_VERSION          = "buun.mtp_vocab_trim.version";
constexpr const char * META_MAP_SIZE         = "buun.mtp_vocab_trim.draft_vocab_size";
constexpr const char * META_SOURCE_SIZE      = "buun.mtp_vocab_trim.source_size";
constexpr const char * META_SOURCE_MTIME     = "buun.mtp_vocab_trim.source_mtime";

#include "mtp-vocab-qwen27b-balanced.inc"
#include "mtp-vocab-qwen27b-65k.inc"

constexpr size_t bit_count(uint64_t value) {
    size_t count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
}

constexpr size_t qwen27b_map_size() {
    size_t count = 0;
    for (uint64_t word : QWEN27B_BALANCED_VOCAB) {
        count += bit_count(word);
    }
    return count;
}

static_assert(qwen27b_map_size() == QWEN_DRAFT_VOCAB_SIZE,
              "the embedded Qwen-27B public balanced vocabulary must contain exactly 32768 tokens");

constexpr size_t qwen27b_65k_map_size() {
    size_t count = 0;
    for (uint64_t word : QWEN27B_65K_VOCAB) {
        count += bit_count(word);
    }
    return count;
}

static_assert(qwen27b_65k_map_size() == QWEN65K_DRAFT_VOCAB_SIZE,
              "the embedded Qwen-27B 65K vocabulary must contain exactly 65536 tokens");

// Both maps cover the same 248320-token Qwen tokenizer, so a single array
// type covers both sizes.
const std::array<uint64_t, 3880> * qwen27b_vocab_map(size_t draft_vocab_size) {
    if (draft_vocab_size == QWEN_DRAFT_VOCAB_SIZE) {
        return &QWEN27B_BALANCED_VOCAB;
    }
    if (draft_vocab_size == QWEN65K_DRAFT_VOCAB_SIZE) {
        return &QWEN27B_65K_VOCAB;
    }
    return nullptr;
}

using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using ggml_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

struct source_identity {
    uint64_t size  = 0;
    int64_t  mtime = 0;
};

struct tensor_schema {
    std::string            name;
    ggml_type              type = GGML_TYPE_COUNT;
    std::array<int64_t, 4> ne   = {};
    size_t                 size = 0;
};

struct qwen27b_admission {
    std::vector<int64_t>       map;
    std::vector<tensor_schema> tensors;
    ggml_type                  output_type = GGML_TYPE_COUNT;
};

bool same_identity(const source_identity & lhs, const source_identity & rhs) {
    return lhs.size == rhs.size && lhs.mtime == rhs.mtime;
}

bool read_source_identity(const std::string & path, source_identity & out) {
    std::error_code ec;
    const auto      size = std::filesystem::file_size(path, ec);
    if (ec) {
        return false;
    }
    const auto mtime = std::filesystem::last_write_time(path, ec);
    if (ec) {
        return false;
    }
    out.size  = size;
    out.mtime = mtime.time_since_epoch().count();
    return true;
}

std::string hex_digest(const std::array<uint8_t, 32> & digest) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string           out;
    out.reserve(digest.size() * 2);
    for (uint8_t byte : digest) {
        out.push_back(hex[byte >> 4]);
        out.push_back(hex[byte & 0x0f]);
    }
    return out;
}

std::array<uint8_t, 32> tokenizer_digest(const std::vector<std::string> & tokens) {
    llama_sha256_writer writer;
    writer.string(TOKENIZER_DOMAIN, std::strlen(TOKENIZER_DOMAIN));
    writer.u64(tokens.size());
    for (const std::string & token : tokens) {
        writer.string(token.data(), token.size());
    }
    return writer.finish();
}

bool tokenizer_matches(const gguf_context * source, std::string & reason) {
    const int64_t model_id  = gguf_find_key(source, "tokenizer.ggml.model");
    const int64_t pre_id    = gguf_find_key(source, "tokenizer.ggml.pre");
    const int64_t tokens_id = gguf_find_key(source, "tokenizer.ggml.tokens");
    if (model_id < 0 || pre_id < 0 || tokens_id < 0 ||
        gguf_get_kv_type(source, model_id) != GGUF_TYPE_STRING ||
        gguf_get_kv_type(source, pre_id) != GGUF_TYPE_STRING ||
        gguf_get_kv_type(source, tokens_id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(source, tokens_id) != GGUF_TYPE_STRING ||
        gguf_get_arr_n(source, tokens_id) != QWEN_TOKENIZER_SIZE ||
        std::strcmp(gguf_get_val_str(source, model_id), "gpt2") != 0 ||
        std::strcmp(gguf_get_val_str(source, pre_id), "qwen35") != 0) {
        reason = "not the supported Qwen-27B tokenizer";
        return false;
    }

    llama_sha256_writer writer;
    writer.string(TOKENIZER_DOMAIN, std::strlen(TOKENIZER_DOMAIN));
    writer.u64(QWEN_TOKENIZER_SIZE);
    for (size_t i = 0; i < QWEN_TOKENIZER_SIZE; ++i) {
        const char * token = gguf_get_arr_str(source, tokens_id, i);
        writer.string(token, std::strlen(token));
    }
    if (hex_digest(writer.finish()) != TOKENIZER_DIGEST) {
        reason = "Qwen-27B tokenizer token identity does not match the balanced vocabulary map";
        return false;
    }
    return true;
}

std::string cache_key(const std::string & source_path, const source_identity & identity, size_t draft_vocab_size) {
    std::error_code ec;
    std::string     canonical = std::filesystem::weakly_canonical(source_path, ec).string();
    if (ec) {
        canonical = source_path;
    }
    llama_sha256 hash;
    hash.update(MAP_VERSION, std::strlen(MAP_VERSION));
    hash.update(canonical.data(), canonical.size());
    hash.update(&identity.size, sizeof(identity.size));
    hash.update(&identity.mtime, sizeof(identity.mtime));
    hash.update(&draft_vocab_size, sizeof(draft_vocab_size));
    return hex_digest(hash.finish()).substr(0, 32);
}

bool copy_bytes(std::ifstream & input, std::ofstream & output, uint64_t offset, uint64_t size, std::string & error) {
    std::array<char, 1024 * 1024> buffer;
    input.clear();
    input.seekg(static_cast<std::streamoff>(offset));
    if (!input) {
        error = "failed to seek source GGUF";
        return false;
    }
    while (size > 0) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(size, buffer.size()));
        input.read(buffer.data(), static_cast<std::streamsize>(chunk));
        if (input.gcount() != static_cast<std::streamsize>(chunk)) {
            error = "short read from source GGUF";
            return false;
        }
        output.write(buffer.data(), static_cast<std::streamsize>(chunk));
        if (!output) {
            error = "failed to write derived GGUF";
            return false;
        }
        size -= chunk;
    }
    return true;
}

size_t read_gguf_at(void * userdata, void * output, uint64_t offset, size_t size) {
    auto & input = *static_cast<std::ifstream *>(userdata);
    input.clear();
    input.seekg(static_cast<std::streamoff>(offset));
    if (!input) {
        return 0;
    }
    input.read(static_cast<char *>(output), static_cast<std::streamsize>(size));
    return static_cast<size_t>(input.gcount());
}

bool write_padding(std::ofstream & output, size_t size, std::string & error) {
    static constexpr std::array<char, GGUF_DEFAULT_ALIGNMENT> zeros = {};
    if (size > zeros.size()) {
        error = "invalid GGUF alignment padding";
        return false;
    }
    output.write(zeros.data(), static_cast<std::streamsize>(size));
    if (!output) {
        error = "failed to write GGUF alignment padding";
        return false;
    }
    return true;
}

bool validate_map(const std::vector<int64_t> & map, int64_t n_vocab, std::string & error) {
    if (map.empty() || static_cast<int64_t>(map.size()) >= n_vocab) {
        error = "draft vocabulary must be non-empty and smaller than the target vocabulary";
        return false;
    }
    int64_t previous = -1;
    for (int64_t token : map) {
        if (token < 0 || token >= n_vocab || token > std::numeric_limits<int32_t>::max()) {
            error = "draft vocabulary contains an out-of-range token id";
            return false;
        }
        if (token <= previous) {
            error = "draft vocabulary token ids must be unique and sorted";
            return false;
        }
        previous = token;
    }
    return true;
}

bool repack(const std::string &          source_path,
            const std::string &          destination_path,
            const std::vector<int64_t> & map,
            const source_identity *      identity,
            std::string &                error) {
    error.clear();

    source_identity observed_identity;
    if (!read_source_identity(source_path, observed_identity)) {
        error = "failed to stat source GGUF";
        return false;
    }
    if (identity && !same_identity(*identity, observed_identity)) {
        error = "source GGUF changed before repacking";
        return false;
    }
    std::ifstream input(source_path, std::ios::binary);
    if (!input) {
        error = "failed to open source GGUF";
        return false;
    }

    ggml_context *   raw_meta = nullptr;
    gguf_init_params params   = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &raw_meta,
    };
    gguf_ptr source(gguf_init_from_callback(
                        read_gguf_at, &input, 1024 * 1024, observed_identity.size, params),
                    gguf_free);
    ggml_ptr source_meta(raw_meta, ggml_free);
    if (!source || !source_meta) {
        error = "failed to read source GGUF metadata";
        return false;
    }

    const int64_t output_id = gguf_find_tensor(source.get(), "output.weight");
    if (output_id < 0 || gguf_find_tensor(source.get(), "d2t") >= 0) {
        error = output_id < 0 ? "source has no output.weight" : "source already has d2t";
        return false;
    }
    ggml_tensor * output_source = ggml_get_tensor(source_meta.get(), "output.weight");
    if (!output_source || ggml_n_dims(output_source) != 2) {
        error = "output.weight is not a matrix";
        return false;
    }
    const int64_t n_embd  = output_source->ne[0];
    const int64_t n_vocab = output_source->ne[1];
    if (!validate_map(map, n_vocab, error)) {
        return false;
    }
    if (n_embd <= 0 || n_embd % ggml_blck_size(output_source->type) != 0) {
        error = "output.weight row is incompatible with its GGML type";
        return false;
    }
    const size_t row_size = ggml_row_size(output_source->type, n_embd);
    if (row_size == 0 || static_cast<uint64_t>(n_vocab) > std::numeric_limits<size_t>::max() / row_size ||
        ggml_nbytes(output_source) != row_size * static_cast<size_t>(n_vocab)) {
        error = "output.weight rows are not independently copyable";
        return false;
    }

    const size_t     tensor_count = static_cast<size_t>(gguf_get_n_tensors(source.get())) + 1;
    const size_t     meta_size    = ggml_tensor_overhead() * (tensor_count + 8) + 4096;
    ggml_init_params meta_params  = {
        /* .mem_size   = */ meta_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_ptr replacement_meta(ggml_init(meta_params), ggml_free);
    if (!replacement_meta) {
        error = "failed to allocate derived GGUF tensor metadata";
        return false;
    }

    gguf_ptr destination(gguf_init_empty(), gguf_free);
    if (!destination) {
        error = "failed to allocate derived GGUF metadata";
        return false;
    }
    gguf_set_kv(destination.get(), source.get());
    // gguf_init_empty() lays out tensors at the default alignment. Copying a
    // non-default general.alignment KV does not change that internal layout, so
    // omit the declaration and make the derivative truthfully default-aligned.
    gguf_remove_key(destination.get(), GGUF_KEY_GENERAL_ALIGNMENT);
    gguf_set_val_str(destination.get(), META_VERSION, MAP_VERSION);
    gguf_set_val_u32(destination.get(), META_MAP_SIZE, map.size());
    if (identity) {
        gguf_set_val_u64(destination.get(), META_SOURCE_SIZE, identity->size);
        gguf_set_val_i64(destination.get(), META_SOURCE_MTIME, identity->mtime);
    }

    for (int64_t i = 0; i < gguf_get_n_tensors(source.get()); ++i) {
        const char * name = gguf_get_tensor_name(source.get(), i);
        if (i == output_id) {
            ggml_tensor * output = ggml_new_tensor_2d(replacement_meta.get(), output_source->type, n_embd, map.size());
            ggml_set_name(output, name);
            gguf_add_tensor(destination.get(), output);
        } else {
            gguf_add_tensor(destination.get(), ggml_get_tensor(source_meta.get(), name));
        }
    }
    ggml_tensor * d2t = ggml_new_tensor_1d(replacement_meta.get(), GGML_TYPE_I32, map.size());
    ggml_set_name(d2t, "d2t");
    gguf_add_tensor(destination.get(), d2t);

    std::vector<int32_t> d2t_map;
    d2t_map.reserve(map.size());
    for (int64_t token : map) {
        d2t_map.push_back(static_cast<int32_t>(token));
    }

    std::ofstream output(destination_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "failed to open destination GGUF";
        return false;
    }

    std::vector<uint8_t> metadata(gguf_get_meta_size(destination.get()));
    gguf_get_meta_data(destination.get(), metadata.data());
    output.write(reinterpret_cast<const char *>(metadata.data()), static_cast<std::streamsize>(metadata.size()));
    if (!output) {
        error = "failed to write derived GGUF metadata";
        return false;
    }

    const uint64_t source_data = gguf_get_data_offset(source.get());
    for (int64_t i = 0; i < gguf_get_n_tensors(destination.get()); ++i) {
        const char * name          = gguf_get_tensor_name(destination.get(), i);
        size_t       bytes_written = 0;
        if (std::strcmp(name, "d2t") == 0) {
            output.write(reinterpret_cast<const char *>(d2t_map.data()),
                         static_cast<std::streamsize>(d2t_map.size() * sizeof(d2t_map[0])));
            if (!output) {
                error = "failed to write d2t tensor";
                return false;
            }
            bytes_written = d2t_map.size() * sizeof(d2t_map[0]);
        } else if (std::strcmp(name, "output.weight") == 0) {
            size_t begin = 0;
            while (begin < map.size()) {
                size_t end = begin + 1;
                while (end < map.size() && map[end] == map[end - 1] + 1) {
                    ++end;
                }
                const uint64_t source_offset = source_data + gguf_get_tensor_offset(source.get(), output_id) +
                                               static_cast<uint64_t>(map[begin]) * row_size;
                const uint64_t run_size = static_cast<uint64_t>(end - begin) * row_size;
                if (!copy_bytes(input, output, source_offset, run_size, error)) {
                    return false;
                }
                bytes_written += run_size;
                begin = end;
            }
        } else {
            const int64_t source_id = gguf_find_tensor(source.get(), name);
            if (source_id < 0) {
                error = "derived GGUF references an unknown source tensor";
                return false;
            }
            bytes_written = gguf_get_tensor_size(source.get(), source_id);
            if (!copy_bytes(input, output, source_data + gguf_get_tensor_offset(source.get(), source_id), bytes_written,
                            error)) {
                return false;
            }
        }
        const size_t padding = GGML_PAD(bytes_written, GGUF_DEFAULT_ALIGNMENT) - bytes_written;
        if (!write_padding(output, padding, error)) {
            return false;
        }
    }
    output.flush();
    if (!output) {
        error = "failed to finalize derived GGUF";
        return false;
    }
    return true;
}

bool get_u32(const gguf_context * ctx, const char * key, uint32_t & value) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_UINT32) {
        return false;
    }
    value = gguf_get_val_u32(ctx, id);
    return true;
}

bool qwen27b_map(const std::string & source_path,
                 size_t              draft_vocab_size,
                 qwen27b_admission & admission,
                 std::string &       reason) {
    ggml_context *   raw_meta = nullptr;
    gguf_init_params params   = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &raw_meta,
    };
    gguf_ptr source(gguf_init_from_file(source_path.c_str(), params), gguf_free);
    ggml_ptr source_meta(raw_meta, ggml_free);
    if (!source || !source_meta) {
        reason = "unreadable GGUF metadata";
        return false;
    }
    const int64_t split_count_id = gguf_find_key(source.get(), "split.count");
    if (split_count_id >= 0) {
        reason = "split MTP sidecars are not supported by the derivative cache";
        return false;
    }
    const int64_t arch_id = gguf_find_key(source.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(source.get(), arch_id) != GGUF_TYPE_STRING ||
        std::strcmp(gguf_get_val_str(source.get(), arch_id), "qwen35") != 0) {
        reason = "not qwen35";
        return false;
    }
    uint32_t block_count      = 0;
    uint32_t embedding_length = 0;
    uint32_t nextn_layers     = 0;
    if (!get_u32(source.get(), "qwen35.block_count", block_count) || block_count != 65 ||
        !get_u32(source.get(), "qwen35.embedding_length", embedding_length) || embedding_length != QWEN_EMBEDDING_LENGTH ||
        !get_u32(source.get(), "qwen35.nextn_predict_layers", nextn_layers) || nextn_layers != 1) {
        reason = "not a supported Qwen-27B MTP shape";
        return false;
    }
    if (gguf_find_tensor(source.get(), "blk.0.attn_norm.weight") >= 0 || gguf_find_tensor(source.get(), "d2t") >= 0) {
        reason = gguf_find_tensor(source.get(), "d2t") >= 0 ? "already trimmed" : "not an MTP-only sidecar";
        return false;
    }
    if (gguf_find_tensor(source.get(), "blk.64.nextn.shared_head_head.weight") >= 0) {
        reason = "MTP sidecar has a separate shared LM head";
        return false;
    }
    const int64_t output_id     = gguf_find_tensor(source.get(), "output.weight");
    const int64_t token_embd_id = gguf_find_tensor(source.get(), "token_embd.weight");
    if (output_id < 0 || token_embd_id < 0) {
        reason = "MTP sidecar needs independent output and token embedding tensors";
        return false;
    }
    const int64_t * output_ne = gguf_get_tensor_ne(source.get(), output_id);
    const int64_t * token_ne  = gguf_get_tensor_ne(source.get(), token_embd_id);
    if (output_ne[0] != QWEN_EMBEDDING_LENGTH || token_ne[0] != QWEN_EMBEDDING_LENGTH ||
        output_ne[1] != QWEN_TOKENIZER_SIZE || token_ne[1] != QWEN_TOKENIZER_SIZE ||
        output_ne[2] != 1 || output_ne[3] != 1 || token_ne[2] != 1 || token_ne[3] != 1) {
        reason = "unexpected Qwen-27B vocabulary shape";
        return false;
    }
    const int64_t n_vocab = output_ne[1];
    const ggml_type output_type = gguf_get_tensor_type(source.get(), output_id);
    if (QWEN_EMBEDDING_LENGTH % ggml_blck_size(output_type) != 0) {
        reason = "Qwen-27B output head is incompatible with its GGML type";
        return false;
    }

    if (!tokenizer_matches(source.get(), reason)) {
        return false;
    }

    const std::array<uint64_t, 3880> * vocab_map = qwen27b_vocab_map(draft_vocab_size);
    if (vocab_map == nullptr) {
        reason = "the Qwen-27B vocabulary maps have exactly 32768 and 65536 entries";
        return false;
    }
    admission.map.clear();
    admission.map.reserve(draft_vocab_size);
    for (int64_t token = 0; token < n_vocab; ++token) {
        const size_t word = static_cast<size_t>(token) / 64;
        const size_t bit  = static_cast<size_t>(token) % 64;
        if (word < vocab_map->size() && ((*vocab_map)[word] & (UINT64_C(1) << bit)) != 0) {
            admission.map.push_back(token);
        }
    }
    admission.output_type = output_type;
    admission.tensors.clear();
    admission.tensors.reserve(static_cast<size_t>(gguf_get_n_tensors(source.get())));
    for (int64_t i = 0; i < gguf_get_n_tensors(source.get()); ++i) {
        tensor_schema schema;
        schema.name = gguf_get_tensor_name(source.get(), i);
        schema.type = gguf_get_tensor_type(source.get(), i);
        std::copy_n(gguf_get_tensor_ne(source.get(), i), schema.ne.size(), schema.ne.begin());
        schema.size = gguf_get_tensor_size(source.get(), i);
        admission.tensors.push_back(std::move(schema));
    }
    return admission.map.size() == draft_vocab_size;
}

bool cached_file_valid(const std::string & path, const source_identity & identity, size_t draft_vocab_size,
                       const qwen27b_admission & admission) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        return false;
    }
    gguf_init_params params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };
    gguf_ptr cached(gguf_init_from_file(path.c_str(), params), gguf_free);
    if (!cached) {
        return false;
    }
    const int64_t output_id = gguf_find_tensor(cached.get(), "output.weight");
    const int64_t d2t_id    = gguf_find_tensor(cached.get(), "d2t");
    if (output_id < 0 || d2t_id < 0) {
        return false;
    }
    const int64_t * output_ne = gguf_get_tensor_ne(cached.get(), output_id);
    const int64_t * d2t_ne    = gguf_get_tensor_ne(cached.get(), d2t_id);
    const size_t expected_output_size = ggml_row_size(admission.output_type, QWEN_EMBEDDING_LENGTH) * draft_vocab_size;
    if (gguf_get_n_tensors(cached.get()) != static_cast<int64_t>(admission.tensors.size() + 1) ||
        gguf_get_tensor_type(cached.get(), output_id) != admission.output_type ||
        gguf_get_tensor_type(cached.get(), d2t_id) != GGML_TYPE_I32 ||
        output_ne[0] != QWEN_EMBEDDING_LENGTH || output_ne[1] != static_cast<int64_t>(draft_vocab_size) ||
        output_ne[2] != 1 || output_ne[3] != 1 ||
        gguf_get_tensor_size(cached.get(), output_id) != expected_output_size ||
        d2t_ne[0] != static_cast<int64_t>(draft_vocab_size) || d2t_ne[1] != 1 || d2t_ne[2] != 1 || d2t_ne[3] != 1 ||
        gguf_get_tensor_size(cached.get(), d2t_id) != draft_vocab_size * sizeof(int32_t)) {
        return false;
    }
    for (const tensor_schema & expected : admission.tensors) {
        if (expected.name == "output.weight") {
            continue;
        }
        const int64_t id = gguf_find_tensor(cached.get(), expected.name.c_str());
        if (id < 0 || gguf_get_tensor_type(cached.get(), id) != expected.type ||
            gguf_get_tensor_size(cached.get(), id) != expected.size ||
            !std::equal(expected.ne.begin(), expected.ne.end(), gguf_get_tensor_ne(cached.get(), id))) {
            return false;
        }
    }
    std::string tokenizer_reason;
    if (!tokenizer_matches(cached.get(), tokenizer_reason)) {
        return false;
    }
    const int64_t version_id  = gguf_find_key(cached.get(), META_VERSION);
    const int64_t map_size_id = gguf_find_key(cached.get(), META_MAP_SIZE);
    const int64_t size_id     = gguf_find_key(cached.get(), META_SOURCE_SIZE);
    const int64_t mtime_id    = gguf_find_key(cached.get(), META_SOURCE_MTIME);
    if (version_id < 0 || map_size_id < 0 || size_id < 0 || mtime_id < 0 ||
        gguf_get_kv_type(cached.get(), version_id) != GGUF_TYPE_STRING ||
        gguf_get_kv_type(cached.get(), map_size_id) != GGUF_TYPE_UINT32 ||
        gguf_get_kv_type(cached.get(), size_id) != GGUF_TYPE_UINT64 ||
        gguf_get_kv_type(cached.get(), mtime_id) != GGUF_TYPE_INT64) {
        return false;
    }

    uint64_t required_size = gguf_get_data_offset(cached.get());
    for (int64_t i = 0; i < gguf_get_n_tensors(cached.get()); ++i) {
        required_size = std::max<uint64_t>(required_size, gguf_get_data_offset(cached.get()) +
                                                              gguf_get_tensor_offset(cached.get(), i) +
                                                              gguf_get_tensor_size(cached.get(), i));
    }
    const uint64_t actual_size = std::filesystem::file_size(path, ec);
    if (ec || actual_size < required_size ||
        std::strcmp(gguf_get_val_str(cached.get(), version_id), MAP_VERSION) != 0 ||
        gguf_get_val_u32(cached.get(), map_size_id) != draft_vocab_size ||
        gguf_get_val_u64(cached.get(), size_id) != identity.size ||
        gguf_get_val_i64(cached.get(), mtime_id) != identity.mtime) {
        return false;
    }

    // d2t drives scatter row indices in the model graph. Validate the small map
    // itself, rather than trusting only its shape and cache metadata.
    std::ifstream input(path, std::ios::binary);
    input.seekg(
        static_cast<std::streamoff>(gguf_get_data_offset(cached.get()) + gguf_get_tensor_offset(cached.get(), d2t_id)));
    int64_t previous = -1;
    const std::array<uint64_t, 3880> * vocab_map = qwen27b_vocab_map(draft_vocab_size);
    for (size_t i = 0; i < draft_vocab_size; ++i) {
        int32_t token = -1;
        input.read(reinterpret_cast<char *>(&token), sizeof(token));
        const size_t word = token >= 0 ? static_cast<size_t>(token) / 64 : vocab_map->size();
        const size_t bit  = token >= 0 ? static_cast<size_t>(token) % 64 : 0;
        if (!input || token <= previous || word >= vocab_map->size() ||
            ((*vocab_map)[word] & (UINT64_C(1) << bit)) == 0) {
            return false;
        }
        previous = token;
    }
    return true;
}

}  // namespace

bool common_mtp_vocab_trim_repack_for_test(const std::string &          source_path,
                                           const std::string &          destination_path,
                                           const std::vector<int64_t> & draft_to_target,
                                           std::string &                error) {
    return repack(source_path, destination_path, draft_to_target, nullptr, error);
}

std::string common_mtp_vocab_trim_tokenizer_digest_for_test(const std::vector<std::string> & tokens) {
    return hex_digest(tokenizer_digest(tokens));
}

common_mtp_vocab_trim_result common_mtp_vocab_trim_prepare(const std::string & source_path,
                                                           uint32_t            requested_draft_vocab_size) {
    common_mtp_vocab_trim_result result;
    result.path = source_path;
    try {
        if (requested_draft_vocab_size == 0) {
            result.detail = "disabled by --spec-mtp-vocab-size 0";
            return result;
        }

        const size_t draft_vocab_size = requested_draft_vocab_size;
        if (draft_vocab_size != QWEN_DRAFT_VOCAB_SIZE && draft_vocab_size != QWEN65K_DRAFT_VOCAB_SIZE) {
            result.status = common_mtp_vocab_trim_status::failed;
            result.detail = "draft vocabulary size must be 0 (disabled), 32768, or 65536";
            return result;
        }

        source_identity identity;
        if (!read_source_identity(source_path, identity)) {
            result.status = common_mtp_vocab_trim_status::failed;
            result.detail = "could not stat the MTP sidecar";
            return result;
        }

        qwen27b_admission admission;
        if (!qwen27b_map(source_path, draft_vocab_size, admission, result.detail)) {
            return result;
        }
        source_identity identity_after_admission;
        if (!read_source_identity(source_path, identity_after_admission) ||
            !same_identity(identity, identity_after_admission)) {
            result.status = common_mtp_vocab_trim_status::failed;
            result.detail = "MTP sidecar changed while checking vocabulary-trim compatibility";
            return result;
        }

        const std::filesystem::path cache_dir =
            std::filesystem::path(fs_get_cache_directory()) / "mtp-vocab-trim-v2";
        std::error_code ec;
        std::filesystem::create_directories(cache_dir, ec);
        if (ec || !std::filesystem::is_directory(cache_dir, ec)) {
            result.status = common_mtp_vocab_trim_status::failed;
            result.detail = "could not create the MTP derivative cache directory";
            return result;
        }

        const std::filesystem::path destination =
            cache_dir / ("qwen27b-mtp-v" + std::to_string(draft_vocab_size) + "-" +
                         cache_key(source_path, identity, draft_vocab_size) + ".gguf");
        if (cached_file_valid(destination.string(), identity, draft_vocab_size, admission)) {
            result.path   = destination.string();
            result.status = common_mtp_vocab_trim_status::cached;
            result.detail = std::string(MAP_VERSION) + "/" + std::to_string(draft_vocab_size);
            return result;
        }

        const auto                  nonce     = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const std::filesystem::path temporary = destination.string() + ".tmp." + std::to_string(nonce);
        std::filesystem::remove(temporary, ec);
        std::string error;
        if (!repack(source_path, temporary.string(), admission.map, &identity, error)) {
            std::filesystem::remove(temporary, ec);
            result.status = common_mtp_vocab_trim_status::failed;
            result.detail = std::move(error);
            return result;
        }

        source_identity identity_after;
        if (!read_source_identity(source_path, identity_after) || !same_identity(identity, identity_after)) {
            std::filesystem::remove(temporary, ec);
            result.status = common_mtp_vocab_trim_status::failed;
            result.detail = "MTP sidecar changed while creating its vocabulary derivative";
            return result;
        }

        std::filesystem::rename(temporary, destination, ec);
        if (ec) {
            if (cached_file_valid(destination.string(), identity, draft_vocab_size, admission)) {
                std::filesystem::remove(temporary, ec);
            } else {
                std::filesystem::remove(destination, ec);
                ec.clear();
                std::filesystem::rename(temporary, destination, ec);
            }
        }
        if (ec || !cached_file_valid(destination.string(), identity, draft_vocab_size, admission)) {
            std::filesystem::remove(temporary, ec);
            result.status = common_mtp_vocab_trim_status::failed;
            result.detail = "could not publish or validate the derived MTP GGUF";
            return result;
        }

        result.path   = destination.string();
        result.status = common_mtp_vocab_trim_status::created;
        result.detail = std::string(MAP_VERSION) + "/" + std::to_string(draft_vocab_size);
        return result;
    } catch (const std::exception & error) {
        result.status = common_mtp_vocab_trim_status::failed;
        result.detail = error.what();
        return result;
    } catch (...) {
        result.status = common_mtp_vocab_trim_status::failed;
        result.detail = "unexpected error while preparing the MTP vocabulary derivative";
        return result;
    }
}
