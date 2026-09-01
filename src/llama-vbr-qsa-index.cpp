#include "llama-vbr-qsa-index.h"

#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-sha256.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace {

constexpr uint32_t QSA_INDEX_STATE_MAGIC = 0x49534151; // "QSAI"
constexpr uint32_t QSA_INDEX_STATE_VERSION = 1;
constexpr char QSA_INDEX_CODEC_DOMAIN[] = "buun.vbr.capture/qsa-index-codec/v1";

class counting_writer final : public llama_io_write_i {
public:
    void write(const void *, size_t size) override { add(size); }
    void write_tensor(ggml_tensor *, size_t, size_t size) override { add(size); }
    size_t n_bytes() override { return bytes_; }
private:
    void add(size_t size) {
        if (size > std::numeric_limits<size_t>::max() - bytes_) {
            throw std::bad_alloc();
        }
        bytes_ += size;
    }
    size_t bytes_ = 0;
};

class vector_writer final : public llama_io_write_i {
public:
    explicit vector_writer(std::vector<uint8_t> & output) : output_(output) {
        output_.clear();
    }
    void write(const void * source, size_t size) override {
        const auto * bytes = static_cast<const uint8_t *>(source);
        output_.insert(output_.end(), bytes, bytes+size);
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const size_t start = output_.size();
        output_.resize(start+size);
        ggml_backend_tensor_get(tensor, output_.data()+start, offset, size);
    }
    size_t n_bytes() override { return output_.size(); }
private:
    std::vector<uint8_t> & output_;
};

class chain_reader final : public llama_io_read_i {
public:
    chain_reader(const artifact_segment_chain & source, uint64_t offset) :
        source_(source), offset_(offset) {}

    void read(void * destination, size_t size) override {
        if (!source_.read(offset_, static_cast<uint8_t *>(destination), size)) {
            throw std::runtime_error("QSA index companion short read");
        }
        offset_ += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        constexpr size_t CHUNK = size_t(1) << 20;
        if (!tensor) {
            throw std::runtime_error("QSA index companion null tensor");
        }
        const size_t scratch_size = std::min(CHUNK, size);
        if (scratch_.size() < scratch_size) {
            scratch_.resize(scratch_size);
        }
        size_t consumed = 0;
        while (consumed < size) {
            const size_t take = std::min(scratch_.size(), size-consumed);
            if (!source_.read(offset_, scratch_.data(), take)) {
                throw std::runtime_error("QSA index companion tensor short read");
            }
            ggml_backend_tensor_set(
                tensor, scratch_.data(), offset+consumed, take);
            offset_ += take;
            consumed += take;
        }
    }

    size_t n_bytes() override { return size_t(offset_); }

private:
    const artifact_segment_chain & source_;
    uint64_t offset_ = 0;
    std::vector<uint8_t> scratch_;
};

class cursor {
public:
    explicit cursor(const artifact_segment_chain & source) : source_(source) {}
    template <typename T> bool scalar(T & value) noexcept {
        if (!source_.read(offset_, reinterpret_cast<uint8_t *>(&value), sizeof(T))) {
            return false;
        }
        offset_ += sizeof(T);
        return true;
    }
    uint64_t offset() const noexcept { return offset_; }
private:
    const artifact_segment_chain & source_;
    uint64_t offset_ = 0;
};

struct qsa_cell_key {
    llama_pos pos = -1;
    int32_t x = 0;
    int32_t y = 0;
    llama_token token = LLAMA_TOKEN_NULL;
};

bool operator<(const qsa_cell_key & lhs, const qsa_cell_key & rhs) noexcept {
    return std::tie(lhs.pos, lhs.x, lhs.y, lhs.token) <
           std::tie(rhs.pos, rhs.x, rhs.y, rhs.token);
}

struct qsa_cell_location {
    llama_pos pos = -1;
    int32_t x = 0;
    int32_t y = 0;
};

bool operator<(const qsa_cell_location & lhs,
               const qsa_cell_location & rhs) noexcept {
    return std::tie(lhs.pos, lhs.x, lhs.y) <
           std::tie(rhs.pos, rhs.x, rhs.y);
}

bool qsa_target_terminal(
        const llama_memory_hybrid_idx & target,
        llama_seq_id sequence,
        llama_pos & terminal) noexcept {
    terminal = -1;
    const auto * cache = target.get_mem_idx();
    if (!cache || sequence < 0) {
        return false;
    }
    const llama_pos first = cache->seq_pos_min(sequence);
    const llama_pos last = cache->seq_pos_max(sequence);
    if (first < 0 || last < first) {
        return false;
    }
    terminal = last;
    return true;
}

bool qsa_target_write(
        const llama_memory_hybrid_idx & target,
        llama_io_write_i & output,
        llama_seq_id sequence) {
    llama_pos terminal = -1;
    const auto * cache = target.get_mem_idx();
    if (!cache || !qsa_target_terminal(target, sequence, terminal)) {
        return false;
    }
    const uint32_t cache_size = cache->get_size();
    const uint32_t streams = cache->get_n_stream();
    const ggml_type type_k = cache->type_k();
    const ggml_type type_v = cache->type_v();
    const auto layers = cache->get_layer_ids();
    if (cache_size == 0 || streams == 0 || layers.empty() ||
        layers.size() > UINT32_MAX) {
        return false;
    }
    output.write(&QSA_INDEX_STATE_MAGIC, sizeof(QSA_INDEX_STATE_MAGIC));
    output.write(&QSA_INDEX_STATE_VERSION, sizeof(QSA_INDEX_STATE_VERSION));
    output.write(&cache_size, sizeof(cache_size));
    output.write(&streams, sizeof(streams));
    output.write(&type_k, sizeof(type_k));
    output.write(&type_v, sizeof(type_v));
    const uint32_t layer_count = uint32_t(layers.size());
    output.write(&layer_count, sizeof(layer_count));
    for (const uint32_t layer : layers) {
        const auto * storage = cache->get_k_storage(int32_t(layer));
        if (!storage || storage->ne[0] <= 0) {
            return false;
        }
        const uint64_t width = uint64_t(storage->ne[0]);
        output.write(&layer, sizeof(layer));
        output.write(&width, sizeof(width));
    }
    output.write(&terminal, sizeof(terminal));
    const uint32_t source_stream = cache->get_stream_for_seq(sequence);
    output.write(&source_stream, sizeof(source_stream));
    const auto & cells = cache->get_cells(sequence);
    uint32_t cell_count = 0;
    for (uint32_t cell = 0; cell < cells.size(); ++cell) {
        cell_count += cells.seq_has(cell, sequence);
    }
    if (cell_count == 0) {
        return false;
    }
    output.write(&cell_count, sizeof(cell_count));
    for (uint32_t cell = 0; cell < cells.size(); ++cell) {
        if (!cells.seq_has(cell, sequence)) {
            continue;
        }
        const llama_pos position = cells.pos_get(cell);
        const auto ext = cells.ext_get(cell);
        output.write(&position, sizeof(position));
        output.write(&ext, sizeof(ext));
    }
    cache->state_write(output, sequence, LLAMA_STATE_SEQ_FLAGS_NONE);
    llama_pos terminal_after = -1;
    return qsa_target_terminal(target, sequence, terminal_after) &&
        terminal_after == terminal;
}

class qsa_parsed_image final : public vbr_parsed_companion_image {
public:
    vbr_artifact_companion_kind kind() const noexcept override {
        return vbr_artifact_companion_kind::qsa_index;
    }
    uint32_t format_version() const noexcept override {
        return QSA_INDEX_STATE_VERSION;
    }

    llama_memory_hybrid_idx * target = nullptr;
    const artifact_segment_chain * source = nullptr;
    uint64_t native_offset = 0;
    uint32_t source_stream = 0;
    llama_pos terminal = -1;
    std::vector<qsa_cell_key> cells;
};

class qsa_prepared_image final : public vbr_prepared_companion_image {
public:
    llama_memory_hybrid_idx * target = nullptr;
    llama_seq_id destination = -1;
    llama_pos terminal = -1;
    vbr_companion_attention_layout layout;
    std::unique_ptr<qsa_parsed_image> recovery;
    bool destination_was_empty = true;

    static bool install(
            llama_memory_hybrid_idx & target,
            const qsa_parsed_image & parsed,
            llama_seq_id destination,
            const vbr_companion_attention_layout & layout) noexcept {
        try {
            auto * cache = target.get_mem_idx();
            if (!cache || parsed.target != &target || !parsed.source ||
                destination < 0 || parsed.cells.empty() ||
                parsed.source_stream >= cache->get_n_stream()) {
                return false;
            }
            using lookup_value = std::pair<uint32_t, uint32_t>;
            std::map<qsa_cell_location, lookup_value> target_cells;
            for (const auto & cell : layout.cells) {
                // Attention VBR intentionally does not carry the source token
                // in its cell metadata.  QSA does, and restores it from the
                // authenticated native payload below.  Physical relocation is
                // therefore keyed by the shared positional coordinates only.
                qsa_cell_location key {
                    cell.logical_position, cell.ext_x, cell.ext_y,
                };
                if (cell.stream >= cache->get_n_stream() ||
                    cell.physical_cell >= cache->get_size() ||
                    !target_cells.emplace(key,
                        lookup_value { cell.stream, cell.physical_cell }).second) {
                    return false;
                }
            }
            llama_kv_cache::slot_info_vec_t sinfos(cache->get_n_stream());
            uint32_t target_stream = UINT32_MAX;
            for (const auto & key : parsed.cells) {
                const auto found = target_cells.find({ key.pos, key.x, key.y });
                if (found == target_cells.end()) {
                    return false;
                }
                if (target_stream == UINT32_MAX) {
                    target_stream = found->second.first;
                } else if (target_stream != found->second.first) {
                    return false;
                }
                auto & sinfo = sinfos[parsed.source_stream];
                if (sinfo.empty()) {
                    sinfo.s0 = parsed.source_stream;
                    sinfo.s1 = parsed.source_stream;
                    sinfo.resize(1);
                    sinfo.strm[0] = parsed.source_stream;
                }
                sinfo.idxs[0].push_back(found->second.second);
            }
            if (target_stream != parsed.source_stream) {
                return false;
            }
            // state_read_sinfo rewrites the stream id to the destination
            // cache's stream while preserving the supplied physical indices.
            chain_reader reader(*parsed.source, parsed.native_offset);
            cache->state_read_sinfo(
                reader, destination, LLAMA_STATE_SEQ_FLAGS_NONE,
                nullptr, &sinfos);
            llama_pos terminal = -1;
            const bool terminal_ok =
                qsa_target_terminal(target, destination, terminal);
            const bool bytes_ok = uint64_t(reader.n_bytes()) == parsed.source->size();
            if (!terminal_ok || terminal != parsed.terminal || !bytes_ok) {
                return false;
            }
            return true;
        } catch (...) {
            if (auto * cache = target.get_mem_idx()) {
                cache->seq_rm(destination, -1, -1);
            }
            return false;
        }
    }

    static bool prepare_with_layout(
            const void * context,
            std::unique_ptr<vbr_parsed_companion_image> parsed_base,
            llama_seq_id destination,
            const vbr_companion_attention_layout & layout,
            std::unique_ptr<vbr_prepared_companion_image> & output) noexcept {
        output.reset();
        auto * target = static_cast<llama_memory_hybrid_idx *>(
            const_cast<void *>(context));
        auto * parsed = dynamic_cast<qsa_parsed_image *>(parsed_base.get());
        if (!target || !parsed || parsed->target != target ||
            !target->get_mem_idx() || !target->get_mem_idx()->state_empty()) {
            return false;
        }
        auto image = std::make_unique<qsa_prepared_image>();
        image->target = target;
        image->destination = destination;
        image->terminal = parsed->terminal;
        image->layout = layout;
        output = std::move(image);
        return install(*target, *parsed, destination, layout);
    }

    static bool prepare_replacement_with_layout(
            const void * context,
            std::unique_ptr<vbr_parsed_companion_image> incoming_base,
            std::unique_ptr<vbr_parsed_companion_image> recovery_base,
            llama_seq_id destination,
            const vbr_companion_attention_layout & layout,
            std::unique_ptr<vbr_prepared_companion_image> & output) noexcept {
        output.reset();
        auto * target = static_cast<llama_memory_hybrid_idx *>(
            const_cast<void *>(context));
        auto * incoming = dynamic_cast<qsa_parsed_image *>(incoming_base.get());
        auto * recovery = dynamic_cast<qsa_parsed_image *>(recovery_base.get());
        if (!target || !incoming || !recovery ||
            incoming->target != target || recovery->target != target) {
            return false;
        }
        auto image = std::make_unique<qsa_prepared_image>();
        image->target = target;
        image->destination = destination;
        image->terminal = incoming->terminal;
        image->layout = layout;
        image->destination_was_empty = false;
        image->recovery.reset(static_cast<qsa_parsed_image *>(
            recovery_base.release()));
        output = std::move(image);
        return install(*target, *incoming, destination, layout);
    }

    static bool target_empty(const void * context) noexcept {
        const auto * target = static_cast<const llama_memory_hybrid_idx *>(context);
        return target && target->get_mem_idx() &&
            target->get_mem_idx()->state_empty();
    }

    static bool recheck(
            const void * context,
            const vbr_prepared_companion_image & base) noexcept {
        const auto & image = static_cast<const qsa_prepared_image &>(base);
        if (!context || image.target != context || image.destination < 0) {
            return false;
        }
        llama_pos terminal = -1;
        auto * cache = image.target->get_mem_idx();
        if (!cache || !qsa_target_terminal(*image.target,
                image.destination, terminal) || terminal != image.terminal) {
            return false;
        }
        const auto & cells = cache->get_cells(image.destination);
        size_t observed = 0;
        for (const auto & expected : image.layout.cells) {
            if (expected.physical_cell >= cells.size() ||
                expected.stream >= cache->get_n_stream() ||
                !cells.seq_has(expected.physical_cell, image.destination) ||
                cells.pos_get(expected.physical_cell) !=
                    expected.logical_position) {
                return false;
            }
            const auto ext = cells.ext_get(expected.physical_cell);
            if (ext.x != expected.ext_x || ext.y != expected.ext_y) {
                return false;
            }
            ++observed;
        }
        size_t occupied = 0;
        for (uint32_t physical = 0; physical < cells.size(); ++physical) {
            occupied += cells.seq_has(physical, image.destination);
        }
        return observed != 0 && observed == occupied;
    }

    static void publish(
            const void *, vbr_prepared_companion_image & base) noexcept {
        auto & image = static_cast<qsa_prepared_image &>(base);
        GGML_ASSERT(image.target != nullptr);
        image.target = nullptr;
    }

    static bool rollback(
            const void *, vbr_prepared_companion_image & base) noexcept {
        auto & image = static_cast<qsa_prepared_image &>(base);
        if (!image.target || !image.target->get_mem_idx()) {
            return false;
        }
        image.target->get_mem_idx()->seq_rm(image.destination, -1, -1);
        if (image.destination_was_empty) {
            return true;
        }
        // Occupied replacement uses the incumbent attention layout, which is
        // not published/mutated during preparation. Reconstruct it directly.
        vbr_companion_attention_layout layout;
        const auto & cells = image.target->get_mem_attn()->get_cells(
            image.destination);
        const uint32_t stream = image.target->get_mem_attn()->get_stream_for_seq(
            image.destination);
        for (uint32_t physical = 0; physical < cells.size(); ++physical) {
            if (!cells.seq_has(physical, image.destination)) {
                continue;
            }
            const auto ext = cells.ext_get(physical);
            layout.cells.push_back({ stream, physical, cells.pos_get(physical),
                                     ext.x, ext.y, ext.tok });
        }
        return image.recovery && install(
            *image.target, *image.recovery, image.destination, layout);
    }
};

} // namespace

uint32_t vbr_qsa_index_companion_format_version() noexcept {
    return QSA_INDEX_STATE_VERSION;
}

std::array<uint8_t, 32>
vbr_qsa_index_companion_build_identity() noexcept {
    static const std::array<uint8_t, 32> identity = [] {
        llama_sha256_writer writer;
        writer.string(
            QSA_INDEX_CODEC_DOMAIN, std::strlen(QSA_INDEX_CODEC_DOMAIN));
        writer.u32(QSA_INDEX_STATE_VERSION);
        return writer.finish();
    }();
    return identity;
}

bool vbr_qsa_index_companion_terminal(
        const void * data, size_t size, llama_pos & output) noexcept {
    output = -1;
    if (!data) {
        return false;
    }
    const auto * bytes = static_cast<const uint8_t *>(data);
    size_t offset = 0;
    auto read = [&](auto & value) {
        if (offset > size || sizeof(value) > size-offset) {
            return false;
        }
        std::memcpy(&value, bytes+offset, sizeof(value));
        offset += sizeof(value);
        return true;
    };
    uint32_t magic = 0, version = 0, cache_size = 0, streams = 0;
    ggml_type type_k = GGML_TYPE_COUNT, type_v = GGML_TYPE_COUNT;
    uint32_t layers = 0;
    if (!read(magic) || !read(version) || !read(cache_size) ||
        !read(streams) || !read(type_k) || !read(type_v) || !read(layers) ||
        magic != QSA_INDEX_STATE_MAGIC || version != QSA_INDEX_STATE_VERSION ||
        cache_size == 0 || streams == 0 || layers == 0) {
        return false;
    }
    for (uint32_t i = 0; i < layers; ++i) {
        uint32_t layer = 0;
        uint64_t width = 0;
        if (!read(layer) || !read(width) || width == 0) {
            return false;
        }
    }
    return read(output) && output >= 0;
}

vbr_explicit_companion_provider vbr_qsa_index_capture_provider(
        llama_memory_hybrid_idx & owner) noexcept {
    vbr_explicit_companion_provider provider;
    provider.kind = vbr_artifact_companion_kind::qsa_index;
    provider.format_version = QSA_INDEX_STATE_VERSION;
    provider.build_identity_digest =
        vbr_qsa_index_companion_build_identity();
    provider.domain = {
        llama_cache_acct_residency::pageable_host,
        llama_cache_acct_domain_kind::not_applicable,
        UINT32_MAX, UINT16_MAX,
    };
    provider.required = true;
    provider.context = &owner;
    provider.size = [](const void * context, llama_seq_id sequence,
                       uint64_t & output) noexcept {
        output = 0;
        try {
            auto * target = static_cast<const llama_memory_hybrid_idx *>(context);
            counting_writer writer;
            if (!target || !qsa_target_write(*target, writer, sequence)) {
                return false;
            }
            output = writer.n_bytes();
            return output != 0;
        } catch (...) {
            return false;
        }
    };
    provider.capture_stream = [](const void * context, llama_seq_id sequence,
                                 llama_io_write_i & output) {
        const auto * target = static_cast<const llama_memory_hybrid_idx *>(context);
        return target && qsa_target_write(*target, output, sequence);
    };
    provider.capture = [](const void * context, llama_seq_id sequence,
                          std::vector<uint8_t> & output) noexcept {
        try {
            const auto * target = static_cast<const llama_memory_hybrid_idx *>(context);
            vector_writer writer(output);
            if (!target || !qsa_target_write(*target, writer, sequence)) {
                output.clear();
                return false;
            }
            return !output.empty();
        } catch (...) {
            output.clear();
            return false;
        }
    };
    provider.terminal_position = [](const void * context,
                                    llama_seq_id sequence,
                                    llama_pos & output) noexcept {
        const auto * target = static_cast<const llama_memory_hybrid_idx *>(context);
        return target && qsa_target_terminal(*target, sequence, output);
    };
    return provider;
}

bool vbr_parse_qsa_index_companion(
        const void *,
        const vbr_artifact_companion_payload & descriptor,
        const artifact_segment_chain & source,
        const vbr_target_companion_snapshot & target_snapshot,
        std::unique_ptr<vbr_parsed_companion_image> & output) noexcept {
    output.reset();
    try {
        auto * target = static_cast<llama_memory_hybrid_idx *>(
            const_cast<void *>(target_snapshot.target_cookie));
        auto * cache = target ? target->get_mem_idx() : nullptr;
        if (!target_snapshot.available || !target || !cache ||
            descriptor.kind != vbr_artifact_companion_kind::qsa_index ||
            descriptor.format_version != QSA_INDEX_STATE_VERSION ||
            descriptor.payload_bytes != source.size()) {
            return false;
        }
        cursor reader(source);
        uint32_t magic = 0, version = 0, cache_size = 0, streams = 0;
        ggml_type type_k = GGML_TYPE_COUNT, type_v = GGML_TYPE_COUNT;
        uint32_t layer_count = 0;
        if (!reader.scalar(magic) || !reader.scalar(version) ||
            !reader.scalar(cache_size) || !reader.scalar(streams) ||
            !reader.scalar(type_k) || !reader.scalar(type_v) ||
            !reader.scalar(layer_count) || magic != QSA_INDEX_STATE_MAGIC ||
            version != QSA_INDEX_STATE_VERSION ||
            cache_size != cache->get_size() ||
            streams != cache->get_n_stream() || type_k != cache->type_k() ||
            type_v != cache->type_v()) {
            return false;
        }
        const auto layers = cache->get_layer_ids();
        if (layer_count == 0 || layer_count != layers.size()) {
            return false;
        }
        for (uint32_t i = 0; i < layer_count; ++i) {
            uint32_t layer = 0;
            uint64_t width = 0;
            const auto * storage = cache->get_k_storage(int32_t(layers[i]));
            if (!reader.scalar(layer) || !reader.scalar(width) || !storage ||
                layer != layers[i] || width != uint64_t(storage->ne[0])) {
                return false;
            }
        }
        auto parsed = std::make_unique<qsa_parsed_image>();
        uint32_t cell_count = 0;
        if (!reader.scalar(parsed->terminal) ||
            !reader.scalar(parsed->source_stream) ||
            !reader.scalar(cell_count) || parsed->terminal < 0 ||
            parsed->source_stream >= streams || cell_count == 0 ||
            cell_count > cache_size) {
            return false;
        }
        parsed->cells.reserve(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            qsa_cell_key key;
            llama_kv_cell_ext ext;
            if (!reader.scalar(key.pos) || !reader.scalar(ext) || key.pos < 0) {
                return false;
            }
            key.x = ext.x;
            key.y = ext.y;
            key.token = ext.tok;
            parsed->cells.push_back(key);
        }
        std::vector<qsa_cell_location> unique;
        unique.reserve(parsed->cells.size());
        for (const auto & cell : parsed->cells) {
            unique.push_back({ cell.pos, cell.x, cell.y });
        }
        std::sort(unique.begin(), unique.end());
        if (std::adjacent_find(unique.begin(), unique.end(),
                [](const auto & a, const auto & b) {
                    return !(a < b) && !(b < a);
                }) != unique.end()) {
            return false;
        }
        parsed->target = target;
        parsed->source = &source;
        parsed->native_offset = reader.offset();
        output = std::move(parsed);
        return true;
    } catch (...) {
        output.reset();
        return false;
    }
}

vbr_companion_adoption_provider vbr_qsa_index_adoption_provider(
        llama_memory_hybrid_idx & owner,
        uint32_t attention_child_id) noexcept {
    vbr_companion_adoption_provider provider;
    provider.kind = vbr_artifact_companion_kind::qsa_index;
    provider.target_cookie = &owner;
    provider.context = &owner;
    provider.prepare_with_layout =
        qsa_prepared_image::prepare_with_layout;
    provider.prepare_replacement_with_layout =
        qsa_prepared_image::prepare_replacement_with_layout;
    provider.attention_child_id = attention_child_id;
    provider.target_empty = qsa_prepared_image::target_empty;
    provider.recheck = qsa_prepared_image::recheck;
    provider.publish_swap = qsa_prepared_image::publish;
    provider.rollback = qsa_prepared_image::rollback;
    return provider;
}
