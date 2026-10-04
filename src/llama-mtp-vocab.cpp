#include "llama-mtp-vocab.h"

#include "llama-arch.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-sha256.h"
#include "llama-vocab.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

// Same identity constants as common/mtp-vocab-trim.cpp: the map is valid only
// for the exact Qwen-27B tokenizer content it was derived from.
constexpr const char * TOKENIZER_DOMAIN = "buun.qwen27b-tokenizer-tokens/v1";
constexpr const char * TOKENIZER_DIGEST = "fbbabd2048cbbddc2db0bd24f8812fa65215649d3d4b54223021ca47ae6be487";

// defines QWEN27B_65K_VOCAB: std::array<uint64_t, 3880> selection bitmap over
// the 248320-token vocabulary (bit i of word j = token j*64 + i selected)
#include "mtp-vocab-qwen27b-65k.inc"

constexpr size_t bit_count(uint64_t value) {
    size_t count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
}

constexpr size_t builtin_map_size() {
    size_t count = 0;
    for (uint64_t word : QWEN27B_65K_VOCAB) {
        count += bit_count(word);
    }
    return count;
}

static_assert(builtin_map_size() == (size_t) LLAMA_MTP_VOCAB_BUILTIN_N_SEL,
              "the built-in 65K MTP draft vocabulary must contain exactly 65536 tokens");
static_assert(QWEN27B_65K_VOCAB.size() * 64 >= (size_t) LLAMA_MTP_VOCAB_BUILTIN_N_VOCAB,
              "the built-in 65K MTP draft vocabulary bitmap must cover the full tokenizer");

std::string hex_digest(const std::array<uint8_t, 32> & digest) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(digest.size() * 2);
    for (uint8_t byte : digest) {
        out.push_back(hex[byte >> 4]);
        out.push_back(hex[byte & 0x0f]);
    }
    return out;
}

} // namespace

bool llama_mtp_vocab_builtin_matches(const llama_vocab & vocab) {
    if ((int64_t) vocab.n_tokens() != LLAMA_MTP_VOCAB_BUILTIN_N_VOCAB) {
        return false;
    }
    llama_sha256_writer writer;
    writer.string(TOKENIZER_DOMAIN, std::strlen(TOKENIZER_DOMAIN));
    writer.u64(vocab.n_tokens());
    for (llama_token id = 0; id < (llama_token) vocab.n_tokens(); ++id) {
        const std::string & text = vocab.get_token_data(id).text;
        writer.string(text.data(), text.size());
    }
    return hex_digest(writer.finish()) == TOKENIZER_DIGEST;
}

std::vector<int32_t> llama_mtp_vocab_builtin_ids() {
    std::vector<int32_t> ids;
    ids.reserve((size_t) LLAMA_MTP_VOCAB_BUILTIN_N_SEL);
    for (size_t w = 0; w < QWEN27B_65K_VOCAB.size(); ++w) {
        for (int bit = 0; bit < 64; ++bit) {
            if (QWEN27B_65K_VOCAB[w] & (uint64_t(1) << bit)) {
                const int64_t id = (int64_t) w * 64 + bit;
                if (id >= LLAMA_MTP_VOCAB_BUILTIN_N_VOCAB) {
                    continue; // padding bits past the vocabulary
                }
                ids.push_back((int32_t) id);
            }
        }
    }
    GGML_ASSERT((int64_t) ids.size() == LLAMA_MTP_VOCAB_BUILTIN_N_SEL);
    return ids; // ascending by construction
}

bool llama_model_init_mtp_draft_vocab(const llama_model * model_c) {
    if (model_c == nullptr) {
        return false;
    }
    // lazy-build of a derived weight cache: logically const, mutates only the
    // model's mtp_draft_vocab member (guarded by `attempted`, built at most once)
    llama_model * model = const_cast<llama_model *>(model_c);

    auto & st = model->mtp_draft_vocab;
    if (st.compact != nullptr) {
        return true; // already built
    }
    if (st.attempted) {
        return false; // previous attempt decided against it (logged then)
    }
    st.attempted = true;

    const char * env = std::getenv("LLAMA_MTP_VOCAB_MAP");
    if (env != nullptr && env[0] == '0') {
        LLAMA_LOG_INFO("%s: disabled via LLAMA_MTP_VOCAB_MAP=0; drafts score the full head\n", __func__);
        return false;
    }
    if (model->d2t != nullptr) {
        return false; // sidecar derivative GGUF owns the draft-vocab trim
    }
    if (model->hparams.n_layer_nextn <= 0) {
        return false; // no built-in MTP head
    }
    if (model->arch != LLM_ARCH_QWEN35) {
        return false; // only the qwen35 build consumes mtp_draft_vocab today
    }
    if (!model->loras.empty()) {
        LLAMA_LOG_INFO("%s: LoRA adapters active; the compact draft head would bypass them, keeping the full head\n", __func__);
        return false;
    }

    ggml_tensor * head = model->output;
    if (head == nullptr || head->buffer == nullptr) {
        return false;
    }
    if (model->output_s != nullptr || model->output_in_s != nullptr) {
        LLAMA_LOG_INFO("%s: speculative-scale head tensors present; keeping the full head\n", __func__);
        return false;
    }
    if (!ggml_is_contiguous(head)) {
        LLAMA_LOG_INFO("%s: head %s is not contiguous; keeping the full head\n", __func__, ggml_get_name(head));
        return false;
    }
    if (head->ne[1] <= LLAMA_MTP_VOCAB_BUILTIN_N_SEL) {
        return false; // nothing to trim
    }
    const ggml_type_traits * ttraits = ggml_get_type_traits(head->type);
    if (ttraits == nullptr || ttraits->to_float == nullptr) {
        LLAMA_LOG_INFO("%s: head type %s has no row decoder; keeping the full head\n", __func__, ggml_type_name(head->type));
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(head->buffer);
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr || ggml_backend_dev_buffer_type(dev) != buft) {
        // a split (multi-device row) or host buffer cannot hold the compact head as one tensor
        LLAMA_LOG_INFO("%s: head %s is in a split or host buffer (%s); keeping the full head\n",
                __func__, ggml_get_name(head), ggml_backend_buft_name(buft));
        return false;
    }
    if (!llama_mtp_vocab_builtin_matches(model->vocab)) {
        LLAMA_LOG_INFO("%s: tokenizer does not match the built-in 65K draft vocabulary; keeping the full head\n", __func__);
        return false;
    }

    const int64_t n_sel = LLAMA_MTP_VOCAB_BUILTIN_N_SEL;
    const std::vector<int32_t> ids = llama_mtp_vocab_builtin_ids();

    const int64_t t0 = ggml_time_us();

    ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
    st.ctx.reset(ggml_init(ip));
    if (st.ctx == nullptr) {
        return false;
    }
    st.ids = ggml_new_tensor_1d(st.ctx.get(), GGML_TYPE_I32, n_sel);
    ggml_set_name(st.ids, "mtp_draft_vocab_ids");
    st.compact = ggml_new_tensor_2d(st.ctx.get(), head->type, head->ne[0], n_sel);
    ggml_set_name(st.compact, "mtp_draft_vocab_compact");

    st.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(st.ctx.get(), buft));
    if (st.buf == nullptr) {
        LLAMA_LOG_WARN("%s: failed to allocate the compact draft head (%.1f MiB); keeping the full head\n",
                __func__, (double) (ggml_row_size(head->type, head->ne[0]) * n_sel) / (1024.0*1024.0));
        st.ids     = nullptr;
        st.compact = nullptr;
        st.ctx.reset();
        return false;
    }
    ggml_backend_buffer_set_usage(st.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_backend_tensor_set(st.ids, ids.data(), 0, ids.size() * sizeof(int32_t));

    // row-gather: every ggml quant layout packs whole rows contiguously
    // (nb[1] == row_size), so selected rows copy verbatim and keep the head's
    // exact quant encoding - no dequant/requant round-trip, bit-identical rows.
    const size_t row_bytes = ggml_row_size(head->type, head->ne[0]);
    std::vector<uint8_t> head_bytes(ggml_nbytes(head));
    ggml_backend_tensor_get(head, head_bytes.data(), 0, head_bytes.size());
    std::vector<uint8_t> compact_bytes(row_bytes * (size_t) n_sel);
    for (int64_t i = 0; i < n_sel; ++i) {
        const size_t src = (size_t) ids[(size_t) i] * row_bytes;
        GGML_ASSERT(src + row_bytes <= head_bytes.size());
        std::memcpy(compact_bytes.data() + (size_t) i * row_bytes, head_bytes.data() + src, row_bytes);
    }
    ggml_backend_tensor_set(st.compact, compact_bytes.data(), 0, compact_bytes.size());

    LLAMA_LOG_INFO("%s: built compact MTP draft head: %lld/%lld rows, %s, %.1f MiB (%.0f ms)\n",
            __func__, (long long) n_sel, (long long) head->ne[1], ggml_type_name(head->type),
            (double) compact_bytes.size() / (1024.0*1024.0), (double) (ggml_time_us() - t0) / 1000.0);
    return true;
}
