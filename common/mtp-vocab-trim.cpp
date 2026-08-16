#include "mtp-vocab-trim.h"

#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-sha256.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <io.h>
#else
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace {

constexpr size_t       QWEN_DRAFT_VOCAB_SIZE = 32768;
constexpr size_t       QWEN_TOKENIZER_SIZE   = 248320;
constexpr int64_t      QWEN_EMBEDDING_LENGTH = 5120;
constexpr const char * MAP_VERSION_BASE      = "qwen27b-frequency-v5-native-head-i32-tokenizer-fbbabd-payload-sha256";
constexpr const char * TOKENIZER_DOMAIN      = "buun.qwen27b-tokenizer-tokens/v1";
constexpr const char * TOKENIZER_DIGEST      = "fbbabd2048cbbddc2db0bd24f8812fa65215649d3d4b54223021ca47ae6be487";
constexpr const char * PAYLOAD_DOMAIN        = "buun.mtp_vocab_trim.semantic_payload/v1";
constexpr const char * META_VERSION          = "buun.mtp_vocab_trim.version";
constexpr const char * META_MAP_SIZE         = "buun.mtp_vocab_trim.draft_vocab_size";
constexpr const char * META_SOURCE_SIZE      = "buun.mtp_vocab_trim.source_size";
constexpr const char * META_SOURCE_MTIME     = "buun.mtp_vocab_trim.source_mtime";
constexpr const char * META_SOURCE_MTIME_NS  = "buun.mtp_vocab_trim.source_mtime_ns";
constexpr const char * META_SOURCE_CHANGE    = "buun.mtp_vocab_trim.source_change";
constexpr const char * META_SOURCE_CHANGE_NS = "buun.mtp_vocab_trim.source_change_ns";
constexpr const char * META_SOURCE_ID_0      = "buun.mtp_vocab_trim.source_id_0";
constexpr const char * META_SOURCE_ID_1      = "buun.mtp_vocab_trim.source_id_1";
constexpr const char * META_SOURCE_ID_2      = "buun.mtp_vocab_trim.source_id_2";
constexpr const char * META_PAYLOAD_SHA256   = "buun.mtp_vocab_trim.payload_sha256";

#include "mtp-vocab-qwen27b-chinese.inc"
#include "mtp-vocab-qwen27b-code.inc"
#include "mtp-vocab-qwen27b-english.inc"
#include "mtp-vocab-qwen27b-japanese.inc"
#include "mtp-vocab-qwen27b-korean.inc"

constexpr size_t bit_count(uint64_t value) {
    size_t count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
}

template<size_t N>
constexpr size_t qwen27b_map_size(const std::array<uint64_t, N> & vocab) {
    size_t count = 0;
    for (uint64_t word : vocab) {
        count += bit_count(word);
    }
    return count;
}

static_assert(qwen27b_map_size(QWEN27B_ENGLISH_VOCAB)  == QWEN_DRAFT_VOCAB_SIZE);
static_assert(qwen27b_map_size(QWEN27B_CODE_VOCAB)     == QWEN_DRAFT_VOCAB_SIZE);
static_assert(qwen27b_map_size(QWEN27B_CHINESE_VOCAB)  == QWEN_DRAFT_VOCAB_SIZE);
static_assert(qwen27b_map_size(QWEN27B_JAPANESE_VOCAB) == QWEN_DRAFT_VOCAB_SIZE);
static_assert(qwen27b_map_size(QWEN27B_KOREAN_VOCAB)   == QWEN_DRAFT_VOCAB_SIZE);

using qwen27b_vocab = std::array<uint64_t, QWEN_TOKENIZER_SIZE / 64>;

const qwen27b_vocab * qwen27b_vocab_for_pack(const std::string & pack) {
    if (pack == "eng")      return &QWEN27B_ENGLISH_VOCAB;
    if (pack == "code")     return &QWEN27B_CODE_VOCAB;
    if (pack == "cn")       return &QWEN27B_CHINESE_VOCAB;
    if (pack == "jp")       return &QWEN27B_JAPANESE_VOCAB;
    if (pack == "kr")       return &QWEN27B_KOREAN_VOCAB;
    return nullptr;
}

using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using ggml_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

struct source_identity {
    uint64_t                size       = 0;
    int64_t                 mtime      = 0;
    int64_t                 mtime_ns   = 0;
    int64_t                 change     = 0;
    int64_t                 change_ns  = 0;
    std::array<uint64_t, 3> stable_id  = {};
};

struct trim_policy {
    size_t               draft_vocab_size = 0;
    size_t               tokenizer_size = 0;
    int64_t              embedding_length = 0;
    std::vector<int64_t> map;
    std::string          tokenizer_digest;
    std::string          version;
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
    bool                       native_head = false;
    std::string                version;
    int64_t                    embedding_length = 0;
};

bool qwen27b_native_head_tensor(const char * name) {
    return std::strcmp(name, "output.weight") == 0;
}

bool same_identity(const source_identity & lhs, const source_identity & rhs) {
    return lhs.size == rhs.size && lhs.mtime == rhs.mtime && lhs.mtime_ns == rhs.mtime_ns &&
           lhs.change == rhs.change && lhs.change_ns == rhs.change_ns && lhs.stable_id == rhs.stable_id;
}

struct file_closer {
    void operator()(FILE * file) const {
        if (file != nullptr) {
            std::fclose(file);
        }
    }
};

using file_ptr = std::unique_ptr<FILE, file_closer>;

bool file_identity(FILE * file, source_identity & out) {
#if defined(_WIN32)
    const intptr_t os_handle = _get_osfhandle(_fileno(file));
    if (os_handle == -1) {
        return false;
    }
    const HANDLE handle = reinterpret_cast<HANDLE>(os_handle);
    FILE_STANDARD_INFO standard = {};
    FILE_BASIC_INFO basic = {};
    FILE_ID_INFO id = {};
    if (!GetFileInformationByHandleEx(handle, FileStandardInfo, &standard, sizeof(standard)) ||
        !GetFileInformationByHandleEx(handle, FileBasicInfo, &basic, sizeof(basic)) ||
        !GetFileInformationByHandleEx(handle, FileIdInfo, &id, sizeof(id)) ||
        standard.EndOfFile.QuadPart < 0) {
        return false;
    }
    out.size      = static_cast<uint64_t>(standard.EndOfFile.QuadPart);
    out.mtime     = basic.LastWriteTime.QuadPart;
    out.change    = basic.ChangeTime.QuadPart;
    out.stable_id[0] = id.VolumeSerialNumber;
    std::memcpy(&out.stable_id[1], id.FileId.Identifier, sizeof(uint64_t));
    std::memcpy(&out.stable_id[2], id.FileId.Identifier + sizeof(uint64_t), sizeof(uint64_t));
#else
    struct stat st = {};
    if (fstat(fileno(file), &st) != 0 || st.st_size < 0) {
        return false;
    }
    out.size         = static_cast<uint64_t>(st.st_size);
    out.stable_id[0] = static_cast<uint64_t>(st.st_dev);
    out.stable_id[1] = static_cast<uint64_t>(st.st_ino);
#    if defined(__APPLE__)
    out.mtime     = st.st_mtimespec.tv_sec;
    out.mtime_ns  = st.st_mtimespec.tv_nsec;
    out.change    = st.st_ctimespec.tv_sec;
    out.change_ns = st.st_ctimespec.tv_nsec;
#    else
    out.mtime     = st.st_mtim.tv_sec;
    out.mtime_ns  = st.st_mtim.tv_nsec;
    out.change    = st.st_ctim.tv_sec;
    out.change_ns = st.st_ctim.tv_nsec;
#    endif
#endif
    return true;
}

file_ptr open_file(const std::string & path) {
    return file_ptr(ggml_fopen(path.c_str(), "rb"));
}

bool read_source_identity(const std::string & path, source_identity & out) {
    file_ptr file = open_file(path);
    return file && file_identity(file.get(), out);
}

bool seek_file(FILE * file, uint64_t offset) {
#if defined(_WIN32)
    return offset <= static_cast<uint64_t>(INT64_MAX) && _fseeki64(file, static_cast<int64_t>(offset), SEEK_SET) == 0;
#else
    return offset <= static_cast<uint64_t>(std::numeric_limits<off_t>::max()) &&
           fseeko(file, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
}

bool read_exact(FILE * file, void * data, size_t size) {
    return size == 0 || std::fread(data, 1, size, file) == size;
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

bool tokenizer_matches(const gguf_context * source, const trim_policy & policy, std::string & reason) {
    const int64_t model_id  = gguf_find_key(source, "tokenizer.ggml.model");
    const int64_t pre_id    = gguf_find_key(source, "tokenizer.ggml.pre");
    const int64_t tokens_id = gguf_find_key(source, "tokenizer.ggml.tokens");
    if (model_id < 0 || pre_id < 0 || tokens_id < 0 ||
        gguf_get_kv_type(source, model_id) != GGUF_TYPE_STRING ||
        gguf_get_kv_type(source, pre_id) != GGUF_TYPE_STRING ||
        gguf_get_kv_type(source, tokens_id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(source, tokens_id) != GGUF_TYPE_STRING ||
        gguf_get_arr_n(source, tokens_id) != policy.tokenizer_size ||
        std::strcmp(gguf_get_val_str(source, model_id), "gpt2") != 0 ||
        std::strcmp(gguf_get_val_str(source, pre_id), "qwen35") != 0) {
        reason = "not the supported Qwen-27B tokenizer";
        return false;
    }

    llama_sha256_writer writer;
    writer.string(TOKENIZER_DOMAIN, std::strlen(TOKENIZER_DOMAIN));
    writer.u64(policy.tokenizer_size);
    for (size_t i = 0; i < policy.tokenizer_size; ++i) {
        const char * token = gguf_get_arr_str(source, tokens_id, i);
        writer.string(token, std::strlen(token));
    }
    if (hex_digest(writer.finish()) != policy.tokenizer_digest) {
        reason = "Qwen-27B tokenizer token identity does not match the supported vocabulary packs";
        return false;
    }
    return true;
}

std::string cache_key(const std::string & source_path, const source_identity & identity,
                      const std::string & version) {
    std::error_code ec;
    std::string     canonical = std::filesystem::weakly_canonical(source_path, ec).string();
    if (ec) {
        canonical = source_path;
    }
    llama_sha256_writer identity_writer;
    identity_writer.string(version.data(), version.size());
    identity_writer.string(canonical.data(), canonical.size());
    identity_writer.u64(identity.size);
    identity_writer.u64(static_cast<uint64_t>(identity.mtime));
    identity_writer.u64(static_cast<uint64_t>(identity.mtime_ns));
    identity_writer.u64(static_cast<uint64_t>(identity.change));
    identity_writer.u64(static_cast<uint64_t>(identity.change_ns));
    for (uint64_t value : identity.stable_id) {
        identity_writer.u64(value);
    }
    return hex_digest(identity_writer.finish()).substr(0, 32);
}

bool copy_bytes(FILE * input, std::ofstream & output, uint64_t offset, uint64_t size, std::string & error) {
    std::array<char, 1024 * 1024> buffer;
    if (!seek_file(input, offset)) {
        error = "failed to seek source GGUF";
        return false;
    }
    while (size > 0) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(size, buffer.size()));
        if (!read_exact(input, buffer.data(), chunk)) {
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
    FILE * input = static_cast<FILE *>(userdata);
    if (!seek_file(input, offset)) {
        return 0;
    }
    return std::fread(output, 1, size, input);
}

void digest_tensor_header(llama_sha256_writer & writer, const gguf_context * ctx, int64_t tensor_id) {
    const char * name = gguf_get_tensor_name(ctx, tensor_id);
    writer.string(name, std::strlen(name));
    writer.u32(static_cast<uint32_t>(gguf_get_tensor_type(ctx, tensor_id)));
    const int64_t * ne = gguf_get_tensor_ne(ctx, tensor_id);
    for (size_t dim = 0; dim < 4; ++dim) {
        writer.u64(static_cast<uint64_t>(ne[dim]));
    }
    writer.u64(gguf_get_tensor_size(ctx, tensor_id));
}

bool digest_file_bytes(FILE * input, uint64_t offset, uint64_t size,
                       llama_sha256_writer & writer, std::string & error) {
    std::array<uint8_t, 1024 * 1024> buffer;
    if (!seek_file(input, offset)) {
        error = "failed to seek tensor payload while hashing";
        return false;
    }
    while (size > 0) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(size, buffer.size()));
        if (!read_exact(input, buffer.data(), chunk)) {
            error = "short tensor payload while hashing";
            return false;
        }
        writer.bytes(buffer.data(), chunk);
        size -= chunk;
    }
    return true;
}

bool artifact_payload_digest(FILE * input, const gguf_context * artifact,
                             std::string & digest, std::string & error) {
    llama_sha256_writer writer;
    writer.string(PAYLOAD_DOMAIN, std::strlen(PAYLOAD_DOMAIN));
    const int64_t n_tensors = gguf_get_n_tensors(artifact);
    writer.u64(static_cast<uint64_t>(n_tensors));
    const uint64_t data_offset = gguf_get_data_offset(artifact);
    for (int64_t i = 0; i < n_tensors; ++i) {
        digest_tensor_header(writer, artifact, i);
        if (!digest_file_bytes(input, data_offset + gguf_get_tensor_offset(artifact, i),
                               gguf_get_tensor_size(artifact, i), writer, error)) {
            return false;
        }
    }
    digest = hex_digest(writer.finish());
    return true;
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

bool repack(FILE *                       input,
            const source_identity &      observed_identity,
            const std::string &          destination_path,
            const std::vector<int64_t> & map,
            const source_identity *      identity,
            bool                         native_head,
            const std::string &          version,
            std::string &                error) {
    error.clear();

    if (identity && !same_identity(*identity, observed_identity)) {
        error = "source GGUF changed before repacking";
        return false;
    }

    ggml_context *   raw_meta = nullptr;
    gguf_init_params params   = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &raw_meta,
    };
    gguf_ptr source(gguf_init_from_callback(
                        read_gguf_at, input, 1024 * 1024, observed_identity.size, params),
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

    size_t tensor_count = 1; // d2t
    for (int64_t i = 0; i < gguf_get_n_tensors(source.get()); ++i) {
        if (!native_head || qwen27b_native_head_tensor(gguf_get_tensor_name(source.get(), i))) {
            ++tensor_count;
        }
    }
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
    gguf_set_val_str(destination.get(), META_VERSION, version.c_str());
    gguf_set_val_u32(destination.get(), META_MAP_SIZE, map.size());
    if (identity) {
        gguf_set_val_u64(destination.get(), META_SOURCE_SIZE, identity->size);
        gguf_set_val_i64(destination.get(), META_SOURCE_MTIME, identity->mtime);
        gguf_set_val_i64(destination.get(), META_SOURCE_MTIME_NS, identity->mtime_ns);
        gguf_set_val_i64(destination.get(), META_SOURCE_CHANGE, identity->change);
        gguf_set_val_i64(destination.get(), META_SOURCE_CHANGE_NS, identity->change_ns);
        gguf_set_val_u64(destination.get(), META_SOURCE_ID_0, identity->stable_id[0]);
        gguf_set_val_u64(destination.get(), META_SOURCE_ID_1, identity->stable_id[1]);
        gguf_set_val_u64(destination.get(), META_SOURCE_ID_2, identity->stable_id[2]);
    }

    for (int64_t i = 0; i < gguf_get_n_tensors(source.get()); ++i) {
        const char * name = gguf_get_tensor_name(source.get(), i);
        if (native_head && !qwen27b_native_head_tensor(name)) {
            continue;
        }
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

    // The digest covers the semantic tensor schema and payload, excluding GGUF
    // padding/metadata. Compute it before serializing metadata so the golden value
    // is embedded in the artifact itself.
    llama_sha256_writer payload_writer;
    payload_writer.string(PAYLOAD_DOMAIN, std::strlen(PAYLOAD_DOMAIN));
    payload_writer.u64(static_cast<uint64_t>(gguf_get_n_tensors(destination.get())));
    const uint64_t source_data = gguf_get_data_offset(source.get());
    for (int64_t i = 0; i < gguf_get_n_tensors(destination.get()); ++i) {
        const char * name = gguf_get_tensor_name(destination.get(), i);
        digest_tensor_header(payload_writer, destination.get(), i);
        if (std::strcmp(name, "d2t") == 0) {
            payload_writer.bytes(d2t_map.data(), d2t_map.size() * sizeof(d2t_map[0]));
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
                if (!digest_file_bytes(input, source_offset, run_size, payload_writer, error)) {
                    return false;
                }
                begin = end;
            }
        } else {
            const int64_t source_id = gguf_find_tensor(source.get(), name);
            if (source_id < 0 ||
                !digest_file_bytes(input, source_data + gguf_get_tensor_offset(source.get(), source_id),
                                   gguf_get_tensor_size(source.get(), source_id), payload_writer, error)) {
                if (error.empty()) {
                    error = "derived GGUF references an unknown source tensor";
                }
                return false;
            }
        }
    }
    const std::string payload_digest = hex_digest(payload_writer.finish());
    gguf_set_val_str(destination.get(), META_PAYLOAD_SHA256, payload_digest.c_str());

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

bool qwen27b_map(FILE *               input,
                 const source_identity & identity,
                 const trim_policy &  policy,
                 qwen27b_admission & admission,
                 std::string &       reason) {
    ggml_context *   raw_meta = nullptr;
    gguf_init_params params   = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &raw_meta,
    };
    gguf_ptr source(gguf_init_from_callback(read_gguf_at, input, 1024 * 1024, identity.size, params), gguf_free);
    ggml_ptr source_meta(raw_meta, ggml_free);
    if (!source || !source_meta) {
        reason = "unreadable GGUF metadata";
        return false;
    }
    const int64_t split_count_id = gguf_find_key(source.get(), "split.count");
    if (split_count_id >= 0) {
        reason = "split MTP sources are not supported by the artifact cache";
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
        !get_u32(source.get(), "qwen35.embedding_length", embedding_length) ||
            embedding_length != static_cast<uint32_t>(policy.embedding_length) ||
        !get_u32(source.get(), "qwen35.nextn_predict_layers", nextn_layers) || nextn_layers != 1) {
        reason = "not a supported Qwen-27B MTP shape";
        return false;
    }
    const bool full_model = gguf_find_tensor(source.get(), "blk.0.attn_norm.weight") >= 0;
    if (gguf_find_tensor(source.get(), "d2t") >= 0) {
        reason = "already trimmed";
        return false;
    }
    if (gguf_find_tensor(source.get(), "blk.64.nextn.shared_head_head.weight") >= 0) {
        reason = "MTP source has a separate shared LM head";
        return false;
    }
    if (full_model && gguf_find_tensor(source.get(), "output.scale") >= 0) {
        reason = "native MTP source has a scaled output tensor";
        return false;
    }
    if (gguf_find_tensor(source.get(), "blk.64.attn_norm.weight") < 0) {
        reason = "source has no Qwen-27B MTP layer";
        return false;
    }
    const int64_t output_id     = gguf_find_tensor(source.get(), "output.weight");
    const int64_t token_embd_id = gguf_find_tensor(source.get(), "token_embd.weight");
    if (output_id < 0 || token_embd_id < 0) {
        reason = "MTP source needs independent output and token embedding tensors";
        return false;
    }
    const int64_t * output_ne = gguf_get_tensor_ne(source.get(), output_id);
    const int64_t * token_ne  = gguf_get_tensor_ne(source.get(), token_embd_id);
    if (output_ne[0] != policy.embedding_length || token_ne[0] != policy.embedding_length ||
        output_ne[1] != static_cast<int64_t>(policy.tokenizer_size) ||
        token_ne[1] != static_cast<int64_t>(policy.tokenizer_size) ||
        output_ne[2] != 1 || output_ne[3] != 1 || token_ne[2] != 1 || token_ne[3] != 1) {
        reason = "unexpected Qwen-27B vocabulary shape";
        return false;
    }
    const int64_t n_vocab = output_ne[1];
    const ggml_type output_type = gguf_get_tensor_type(source.get(), output_id);
    if (policy.embedding_length % ggml_blck_size(output_type) != 0) {
        reason = "Qwen-27B output head is incompatible with its GGML type";
        return false;
    }

    if (!tokenizer_matches(source.get(), policy, reason)) {
        return false;
    }

    std::string map_error;
    if (!validate_map(policy.map, n_vocab, map_error) || policy.map.size() != policy.draft_vocab_size) {
        reason = map_error.empty() ? "invalid locked Qwen-27B vocabulary pack" : map_error;
        return false;
    }
    admission.map = policy.map;
    admission.output_type = output_type;
    admission.native_head = full_model;
    admission.version = policy.version;
    admission.embedding_length = policy.embedding_length;
    admission.tensors.clear();
    admission.tensors.reserve(full_model ? 1 : static_cast<size_t>(gguf_get_n_tensors(source.get())));
    for (int64_t i = 0; i < gguf_get_n_tensors(source.get()); ++i) {
        const char * name = gguf_get_tensor_name(source.get(), i);
        if (full_model && !qwen27b_native_head_tensor(name)) {
            continue;
        }
        tensor_schema schema;
        schema.name = name;
        schema.type = gguf_get_tensor_type(source.get(), i);
        std::copy_n(gguf_get_tensor_ne(source.get(), i), schema.ne.size(), schema.ne.begin());
        schema.size = gguf_get_tensor_size(source.get(), i);
        admission.tensors.push_back(std::move(schema));
    }
    return true;
}

bool cached_file_valid(const std::string & path, const source_identity & identity,
                       const trim_policy & policy, const qwen27b_admission & admission) {
    file_ptr input = open_file(path);
    source_identity artifact_identity;
    if (!input || !file_identity(input.get(), artifact_identity)) {
        return false;
    }
    gguf_init_params params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };
    gguf_ptr cached(gguf_init_from_callback(
            read_gguf_at, input.get(), 1024 * 1024, artifact_identity.size, params), gguf_free);
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
    const size_t draft_vocab_size = policy.draft_vocab_size;
    const size_t expected_output_size =
            ggml_row_size(admission.output_type, admission.embedding_length) * draft_vocab_size;
    if (gguf_get_n_tensors(cached.get()) != static_cast<int64_t>(admission.tensors.size() + 1) ||
        gguf_get_tensor_type(cached.get(), output_id) != admission.output_type ||
        gguf_get_tensor_type(cached.get(), d2t_id) != GGML_TYPE_I32 ||
        output_ne[0] != admission.embedding_length || output_ne[1] != static_cast<int64_t>(draft_vocab_size) ||
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
    if (!tokenizer_matches(cached.get(), policy, tokenizer_reason)) {
        return false;
    }
    const int64_t version_id  = gguf_find_key(cached.get(), META_VERSION);
    const int64_t map_size_id = gguf_find_key(cached.get(), META_MAP_SIZE);
    const int64_t size_id     = gguf_find_key(cached.get(), META_SOURCE_SIZE);
    const int64_t mtime_id    = gguf_find_key(cached.get(), META_SOURCE_MTIME);
    const int64_t mtime_ns_id = gguf_find_key(cached.get(), META_SOURCE_MTIME_NS);
    const int64_t change_id   = gguf_find_key(cached.get(), META_SOURCE_CHANGE);
    const int64_t change_ns_id = gguf_find_key(cached.get(), META_SOURCE_CHANGE_NS);
    const int64_t source_id_0 = gguf_find_key(cached.get(), META_SOURCE_ID_0);
    const int64_t source_id_1 = gguf_find_key(cached.get(), META_SOURCE_ID_1);
    const int64_t source_id_2 = gguf_find_key(cached.get(), META_SOURCE_ID_2);
    const int64_t payload_id  = gguf_find_key(cached.get(), META_PAYLOAD_SHA256);
    if (version_id < 0 || map_size_id < 0 || size_id < 0 || mtime_id < 0 || mtime_ns_id < 0 ||
        change_id < 0 || change_ns_id < 0 || source_id_0 < 0 || source_id_1 < 0 || source_id_2 < 0 ||
        payload_id < 0 ||
        gguf_get_kv_type(cached.get(), version_id) != GGUF_TYPE_STRING ||
        gguf_get_kv_type(cached.get(), map_size_id) != GGUF_TYPE_UINT32 ||
        gguf_get_kv_type(cached.get(), size_id) != GGUF_TYPE_UINT64 ||
        gguf_get_kv_type(cached.get(), mtime_id) != GGUF_TYPE_INT64 ||
        gguf_get_kv_type(cached.get(), mtime_ns_id) != GGUF_TYPE_INT64 ||
        gguf_get_kv_type(cached.get(), change_id) != GGUF_TYPE_INT64 ||
        gguf_get_kv_type(cached.get(), change_ns_id) != GGUF_TYPE_INT64 ||
        gguf_get_kv_type(cached.get(), source_id_0) != GGUF_TYPE_UINT64 ||
        gguf_get_kv_type(cached.get(), source_id_1) != GGUF_TYPE_UINT64 ||
        gguf_get_kv_type(cached.get(), source_id_2) != GGUF_TYPE_UINT64 ||
        gguf_get_kv_type(cached.get(), payload_id) != GGUF_TYPE_STRING) {
        return false;
    }

    uint64_t required_size = gguf_get_data_offset(cached.get());
    for (int64_t i = 0; i < gguf_get_n_tensors(cached.get()); ++i) {
        required_size = std::max<uint64_t>(required_size, gguf_get_data_offset(cached.get()) +
                                                              gguf_get_tensor_offset(cached.get(), i) +
                                                              gguf_get_tensor_size(cached.get(), i));
    }
    if (artifact_identity.size < required_size ||
        std::strcmp(gguf_get_val_str(cached.get(), version_id), admission.version.c_str()) != 0 ||
        gguf_get_val_u32(cached.get(), map_size_id) != draft_vocab_size ||
        gguf_get_val_u64(cached.get(), size_id) != identity.size ||
        gguf_get_val_i64(cached.get(), mtime_id) != identity.mtime ||
        gguf_get_val_i64(cached.get(), mtime_ns_id) != identity.mtime_ns ||
        gguf_get_val_i64(cached.get(), change_id) != identity.change ||
        gguf_get_val_i64(cached.get(), change_ns_id) != identity.change_ns ||
        gguf_get_val_u64(cached.get(), source_id_0) != identity.stable_id[0] ||
        gguf_get_val_u64(cached.get(), source_id_1) != identity.stable_id[1] ||
        gguf_get_val_u64(cached.get(), source_id_2) != identity.stable_id[2]) {
        return false;
    }

    // d2t drives scatter row indices in the model graph. Validate the small map
    // itself, rather than trusting only its shape and cache metadata.
    if (!seek_file(input.get(), gguf_get_data_offset(cached.get()) + gguf_get_tensor_offset(cached.get(), d2t_id))) {
        return false;
    }
    int64_t previous = -1;
    for (size_t i = 0; i < draft_vocab_size; ++i) {
        int32_t token = -1;
        if (!read_exact(input.get(), &token, sizeof(token)) || token <= previous ||
            token != admission.map[i]) {
            return false;
        }
        previous = token;
    }

    std::string actual_digest;
    std::string digest_error;
    if (!artifact_payload_digest(input.get(), cached.get(), actual_digest, digest_error) ||
        actual_digest != gguf_get_val_str(cached.get(), payload_id)) {
        return false;
    }
    return true;
}

trim_policy production_policy(const std::string & vocab_pack) {
    trim_policy policy;
    const qwen27b_vocab * vocab = qwen27b_vocab_for_pack(vocab_pack);
    if (!vocab) {
        return policy;
    }
    policy.draft_vocab_size = QWEN_DRAFT_VOCAB_SIZE;
    policy.tokenizer_size = QWEN_TOKENIZER_SIZE;
    policy.embedding_length = QWEN_EMBEDDING_LENGTH;
    policy.tokenizer_digest = TOKENIZER_DIGEST;
    policy.version = std::string(MAP_VERSION_BASE) + "-" + vocab_pack;
    policy.map.reserve(QWEN_DRAFT_VOCAB_SIZE);
    for (size_t token = 0; token < QWEN_TOKENIZER_SIZE; ++token) {
        if (((*vocab)[token / 64] & (UINT64_C(1) << (token % 64))) != 0) {
            policy.map.push_back(static_cast<int64_t>(token));
        }
    }
    return policy;
}

size_t native_resident_bytes(const trim_policy & policy, const qwen27b_admission & admission) {
    if (!admission.native_head) {
        return 0;
    }
    const size_t row_size = ggml_row_size(admission.output_type, admission.embedding_length);
    if (policy.draft_vocab_size == 0 || policy.draft_vocab_size > SIZE_MAX / sizeof(int32_t)) {
        return 0;
    }
    const size_t map_bytes = policy.draft_vocab_size * sizeof(int32_t);
    if (row_size > (SIZE_MAX - map_bytes) / policy.draft_vocab_size) {
        return 0;
    }
    return row_size * policy.draft_vocab_size + map_bytes;
}

common_mtp_vocab_trim_result prepare_with_policy(
        const std::string & source_path,
        const std::filesystem::path & cache_dir,
        const std::string & cache_label,
        const trim_policy & policy) {
    common_mtp_vocab_trim_result result;
    result.path = source_path;

    file_ptr source = open_file(source_path);
    source_identity identity;
    if (!source || !file_identity(source.get(), identity)) {
        result.status = common_mtp_vocab_trim_status::failed;
        result.detail = "could not open or identify the MTP source";
        return result;
    }

    qwen27b_admission admission;
    if (!qwen27b_map(source.get(), identity, policy, admission, result.detail)) {
        return result;
    }
    source_identity identity_after_admission;
    source_identity path_identity_after_admission;
    if (!file_identity(source.get(), identity_after_admission) ||
        !read_source_identity(source_path, path_identity_after_admission) ||
        !same_identity(identity, identity_after_admission) ||
        !same_identity(identity, path_identity_after_admission)) {
        result.status = common_mtp_vocab_trim_status::failed;
        result.detail = "MTP source changed while checking vocabulary-trim compatibility";
        return result;
    }

    std::error_code ec;
    std::filesystem::create_directories(cache_dir, ec);
    if (ec || !std::filesystem::is_directory(cache_dir, ec)) {
        result.status = common_mtp_vocab_trim_status::failed;
        result.detail = "could not create the MTP artifact cache directory";
        return result;
    }

    const std::filesystem::path destination =
        cache_dir / (cache_label + "-v" + std::to_string(policy.draft_vocab_size) + "-" +
                     cache_key(source_path, identity, admission.version) + ".gguf");
    if (cached_file_valid(destination.string(), identity, policy, admission)) {
        source_identity path_identity;
        if (!read_source_identity(source_path, path_identity) || !same_identity(identity, path_identity)) {
            result.status = common_mtp_vocab_trim_status::failed;
            result.detail = "MTP source changed while reusing its vocabulary artifact";
            return result;
        }
        result.path = destination.string();
        result.status = common_mtp_vocab_trim_status::cached;
        result.detail = admission.version + "/" + std::to_string(policy.draft_vocab_size);
        result.native_head = admission.native_head;
        result.resident_bytes = native_resident_bytes(policy, admission);
        return result;
    }

    const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
#if defined(_WIN32)
    const uint64_t process_id = GetCurrentProcessId();
#else
    const uint64_t process_id = static_cast<uint64_t>(getpid());
#endif
    const std::filesystem::path temporary =
            destination.string() + ".tmp." + std::to_string(process_id) + "." + std::to_string(nonce);
    std::filesystem::remove(temporary, ec);
    std::string error;
    if (!repack(source.get(), identity, temporary.string(), admission.map, &identity,
                admission.native_head, admission.version, error)) {
        std::filesystem::remove(temporary, ec);
        result.status = common_mtp_vocab_trim_status::failed;
        result.detail = std::move(error);
        return result;
    }

    source_identity identity_after;
    source_identity path_identity_after;
    if (!file_identity(source.get(), identity_after) ||
        !read_source_identity(source_path, path_identity_after) ||
        !same_identity(identity, identity_after) || !same_identity(identity, path_identity_after)) {
        std::filesystem::remove(temporary, ec);
        result.status = common_mtp_vocab_trim_status::failed;
        result.detail = "MTP source changed while creating its vocabulary artifact";
        return result;
    }

    std::filesystem::rename(temporary, destination, ec);
    if (ec) {
        if (cached_file_valid(destination.string(), identity, policy, admission)) {
            std::filesystem::remove(temporary, ec);
        } else {
            std::filesystem::remove(destination, ec);
            ec.clear();
            std::filesystem::rename(temporary, destination, ec);
        }
    }
    if (ec || !cached_file_valid(destination.string(), identity, policy, admission)) {
        std::filesystem::remove(temporary, ec);
        result.status = common_mtp_vocab_trim_status::failed;
        result.detail = "could not publish or validate the derived MTP GGUF";
        return result;
    }

    result.path = destination.string();
    result.status = common_mtp_vocab_trim_status::created;
    result.detail = admission.version + "/" + std::to_string(policy.draft_vocab_size);
    result.native_head = admission.native_head;
    result.resident_bytes = native_resident_bytes(policy, admission);
    return result;
}

}  // namespace

bool common_mtp_vocab_trim_repack_for_test(const std::string &          source_path,
                                           const std::string &          destination_path,
                                           const std::vector<int64_t> & draft_to_target,
                                           std::string &                error) {
    file_ptr source = open_file(source_path);
    source_identity identity;
    if (!source || !file_identity(source.get(), identity)) {
        error = "failed to open source GGUF";
        return false;
    }
    return repack(source.get(), identity, destination_path, draft_to_target, nullptr, false,
                  std::string(MAP_VERSION_BASE) + "-test", error);
}

bool common_mtp_vocab_trim_head_for_test(const std::string &          source_path,
                                         const std::string &          destination_path,
                                         const std::vector<int64_t> & draft_to_target,
                                         std::string &                error) {
    file_ptr source = open_file(source_path);
    source_identity identity;
    if (!source || !file_identity(source.get(), identity)) {
        error = "failed to open source GGUF";
        return false;
    }
    return repack(source.get(), identity, destination_path, draft_to_target, nullptr, true,
                  std::string(MAP_VERSION_BASE) + "-test", error);
}

std::string common_mtp_vocab_trim_tokenizer_digest_for_test(const std::vector<std::string> & tokens) {
    return hex_digest(tokenizer_digest(tokens));
}

common_mtp_vocab_trim_result common_mtp_vocab_trim_prepare(const std::string & source_path,
                                                           const std::string & vocab_pack) {
    common_mtp_vocab_trim_result result;
    result.path = source_path;
    try {
        if (vocab_pack.empty()) {
            result.detail = "disabled because no MTP vocabulary pack was selected";
            return result;
        }

        trim_policy policy = production_policy(vocab_pack);
        if (policy.map.empty()) {
            result.status = common_mtp_vocab_trim_status::failed;
            result.detail = "unknown Qwen-27B MTP vocabulary pack";
            return result;
        }
        const std::filesystem::path cache_dir =
            std::filesystem::path(fs_get_cache_directory()) / "mtp-vocab-trim-v5";
        return prepare_with_policy(source_path, cache_dir, "qwen27b-mtp-" + vocab_pack, policy);
    } catch (const std::exception & error) {
        result.status = common_mtp_vocab_trim_status::failed;
        result.detail = error.what();
        return result;
    } catch (...) {
        result.status = common_mtp_vocab_trim_status::failed;
        result.detail = "unexpected error while preparing the MTP vocabulary artifact";
        return result;
    }
}

common_mtp_vocab_trim_result common_mtp_vocab_trim_prepare_for_test(
        const std::string &          source_path,
        const std::string &          cache_dir,
        const std::vector<int64_t> & draft_to_target,
        size_t                       tokenizer_size,
        int64_t                      embedding_length,
        const std::string &          tokenizer_token_digest) {
    trim_policy policy;
    policy.draft_vocab_size = draft_to_target.size();
    policy.tokenizer_size = tokenizer_size;
    policy.embedding_length = embedding_length;
    policy.map = draft_to_target;
    policy.tokenizer_digest = tokenizer_token_digest;
    policy.version = std::string(MAP_VERSION_BASE) + "-production-fixture";
    return prepare_with_policy(source_path, cache_dir, "qwen27b-mtp-fixture", policy);
}
