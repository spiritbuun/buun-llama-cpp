#include "llama-memory-recurrent.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-batch.h"
#include "llama-model.h"
#include "llama-vram-demand.h"
#include "llama-vbr-artifact-adopt.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>

namespace {
struct ggml_backend_buft_comparator {
    bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
        return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
    }
};

bool recurrent_resize_test_fail_before_publish() {
    // TEST-ONLY, undocumented fault seam. Keep it local to this translation unit: the
    // recurrent cache has no public/internal test-control API, and the exact value is
    // checked only after a complete off-side resize has been staged.
    const char * value = std::getenv("LLAMA_RECURRENT_RESIZE_TEST_FAIL");
    return value != nullptr && std::strcmp(value, "before_publish") == 0;
}
} // namespace

namespace {

struct recurrent_tensor_row {
    bool present = false;
    ggml_type type = GGML_TYPE_COUNT;
    uint64_t row_bytes = 0;
    std::vector<uint8_t> bytes;
};

class recurrent_cursor {
  public:
    explicit recurrent_cursor(const artifact_segment_chain & source)
        : source_(source), size_(source.size()) {}

    template <typename T>
    bool scalar(T & value) noexcept {
        if (offset_ > size_ || sizeof(T) > size_ - offset_ ||
            !source_.read(offset_, reinterpret_cast<uint8_t *>(&value),
                          sizeof(T))) {
            return false;
        }
        offset_ += sizeof(T);
        return true;
    }

    bool block(std::vector<uint8_t> & value, size_t size) {
        if (offset_ > size_ || size > size_ - offset_) {
            return false;
        }
        value.resize(size);
        if (!source_.read(offset_, value.data(), size)) {
            value.clear();
            return false;
        }
        offset_ += size;
        return true;
    }

    bool finished() const noexcept { return offset_ == size_; }

  private:
    const artifact_segment_chain & source_;
    uint64_t size_ = 0;
    uint64_t offset_ = 0;
};

class vbr_recurrent_parsed_image final :
        public vbr_parsed_companion_image {
  public:
    vbr_artifact_companion_kind kind() const noexcept override {
        return vbr_artifact_companion_kind::recurrent;
    }
    uint32_t format_version() const noexcept override { return 1; }

    llama_memory_recurrent * target = nullptr;
    llama_pos position = -1;
    std::vector<recurrent_tensor_row> r;
    std::vector<recurrent_tensor_row> p;
    std::vector<recurrent_tensor_row> s;
};

} // namespace

class vbr_recurrent_prepared_image final :
        public vbr_prepared_companion_image {
  public:
    llama_memory_recurrent * target = nullptr;
    llama_seq_id destination = -1;
    std::vector<llama_memory_recurrent::mem_cell> cells;
    uint32_t head = 0;
    uint32_t used = 0;
    int32_t rs_z = -1;
    std::vector<uint32_t> rs_idx;
    std::vector<uint32_t> rollback_valid_depth;
    std::unique_ptr<vbr_recurrent_parsed_image> recovery;
    bool destination_was_empty = false;
    size_t replacement_physical = 0;
    llama_pos replacement_position = -1;
    uint64_t replacement_binding_epoch = 0;

    static bool parsed_compatible(
            const llama_memory_recurrent & target,
            const vbr_recurrent_parsed_image & parsed) noexcept {
        if (parsed.target != &target || parsed.position < 0 ||
            parsed.r.size() != target.r_l.size() ||
            parsed.p.size() != target.p_l.size() ||
            parsed.s.size() != target.s_l.size()) {
            return false;
        }
        const auto rows_compatible = [](
                const std::vector<ggml_tensor *> & tensors,
                const std::vector<recurrent_tensor_row> & rows) {
            for (size_t i = 0; i < tensors.size(); ++i) {
                const auto * tensor = tensors[i];
                const auto & row = rows[i];
                if ((tensor != nullptr) != row.present) {
                    return false;
                }
                if (tensor && (row.type != tensor->type ||
                    row.row_bytes != uint64_t(ggml_row_size(
                        tensor->type, tensor->ne[0])) ||
                    row.row_bytes == 0 ||
                    row.bytes.size() != row.row_bytes)) {
                    return false;
                }
            }
            return true;
        };
        return rows_compatible(target.r_l, parsed.r) &&
               rows_compatible(target.p_l, parsed.p) &&
               rows_compatible(target.s_l, parsed.s);
    }

    static bool live_location(
            const llama_memory_recurrent & target,
            llama_seq_id destination, size_t & physical,
            uint32_t & row) noexcept {
        if (destination < 0 || uint32_t(destination) >= target.n_seq_max ||
            target.used != 1) {
            return false;
        }
        physical = target.cells.size();
        for (size_t i = 0; i < target.cells.size(); ++i) {
            if (!target.cells[i].has_seq_id(destination)) {
                continue;
            }
            if (physical != target.cells.size()) {
                return false;
            }
            physical = i;
        }
        if (physical == target.cells.size()) {
            return false;
        }
        uint32_t rollback = 0;
        if (target.n_rs_seq != 0) {
            if (size_t(destination) >= target.rs_idx.size()) {
                return false;
            }
            rollback = target.rs_idx[size_t(destination)];
        }
        const int32_t source = target.cells[physical].src >= 0
            ? target.cells[physical].src : int32_t(physical);
        if (source < 0 || uint64_t(rollback)*target.size +
                uint32_t(source) > UINT32_MAX) {
            return false;
        }
        row = rollback*target.size + uint32_t(source);
        return true;
    }

    static bool write_rows(
            llama_memory_recurrent & target, uint32_t row,
            const vbr_recurrent_parsed_image & parsed) noexcept {
        try {
            if (!parsed_compatible(target, parsed)) {
                return false;
            }
            const auto write = [row](
                    const std::vector<ggml_tensor *> & tensors,
                    const std::vector<recurrent_tensor_row> & rows) {
                for (size_t i = 0; i < tensors.size(); ++i) {
                    if (!tensors[i]) {
                        continue;
                    }
                    if (uint64_t(row) >= uint64_t(tensors[i]->ne[1])) {
                        return false;
                    }
                    ggml_backend_tensor_set(
                        tensors[i], rows[i].bytes.data(),
                        size_t(row)*size_t(rows[i].row_bytes),
                        rows[i].bytes.size());
                }
                return true;
            };
            return write(target.r_l, parsed.r) &&
                   write(target.p_l, parsed.p) &&
                   write(target.s_l, parsed.s);
        } catch (...) {
            return false;
        }
    }

    static bool prepare_empty(
            const void * context,
            std::unique_ptr<vbr_parsed_companion_image> parsed_base,
            llama_seq_id destination,
            std::unique_ptr<vbr_prepared_companion_image> & output) noexcept {
        try {
            auto * target = static_cast<llama_memory_recurrent *>(
                const_cast<void *>(context));
            auto * parsed = dynamic_cast<vbr_recurrent_parsed_image *>(
                parsed_base.get());
            if (!target || !parsed || parsed->target != target ||
                destination < 0 ||
                uint32_t(destination) >= target->n_seq_max ||
                (target->used != 0 ||
                  std::any_of(target->cells.begin(), target->cells.end(),
                      [](const llama_memory_recurrent::mem_cell & cell) {
                          return !cell.is_empty();
                      })) ||
                parsed->r.size() != target->r_l.size() ||
                parsed->p.size() != target->p_l.size() ||
                parsed->s.size() != target->s_l.size()) {
                return false;
            }

            auto image = std::make_unique<vbr_recurrent_prepared_image>();
            image->target = target;
            image->destination = destination;
            image->cells.resize(target->size);
            image->head = 0;
            image->used = 1;
            image->rs_idx.assign(target->rs_idx.size(), 0);
            image->rollback_valid_depth.assign(
                target->rollback_valid_depth.size(), 0);
            auto & cell = image->cells[0];
            cell.pos = parsed->position;
            cell.src = 0;
            cell.seq_id.insert(destination);
            image->cells[size_t(destination)].tail = 0;

            if (!parsed_compatible(*target, *parsed)) {
                return false;
            }
            // An empty controller has no live row: its backing bytes are not
            // part of the logical state and cannot be observed until metadata
            // publication. Every successful import or first decode overwrites
            // the complete active R/S row. Therefore a failed construction-
            // empty import needs no D2H rollback journal; leaving the target
            // logically empty is the exact rollback state.
            image->destination_was_empty = true;
            image->replacement_physical = 0;
            image->replacement_position = parsed->position;
            image->replacement_binding_epoch = target->tensor_binding_epoch_;
            output = std::move(image);
            return write_rows(*target, 0, *parsed);
        } catch (...) {
            output.reset();
            return false;
        }
    }

    static bool prepare(
            const void * context,
            std::unique_ptr<vbr_parsed_companion_image> parsed,
            llama_seq_id destination,
            std::unique_ptr<vbr_prepared_companion_image> & output) noexcept {
        return prepare_empty(
            context, std::move(parsed), destination, output);
    }

    static bool live_matches(
            const llama_memory_recurrent & target,
            llama_seq_id destination,
            const vbr_recurrent_parsed_image & expected) noexcept {
        try {
            if (destination < 0 ||
                uint32_t(destination) >= target.n_seq_max ||
                expected.target != &target || target.used != 1 ||
                expected.r.size() != target.r_l.size() ||
                expected.p.size() != target.p_l.size() ||
                expected.s.size() != target.s_l.size()) {
                return false;
            }
            size_t physical = 0;
            uint32_t row = 0;
            if (!live_location(target, destination, physical, row) ||
                target.cells[physical].pos != expected.position) {
                return false;
            }
            size_t max_row_bytes = 0;
            for (const auto & value : expected.r) {
                max_row_bytes = std::max(
                    max_row_bytes, size_t(value.row_bytes));
            }
            for (const auto & value : expected.p) {
                max_row_bytes = std::max(
                    max_row_bytes, size_t(value.row_bytes));
            }
            for (const auto & value : expected.s) {
                max_row_bytes = std::max(
                    max_row_bytes, size_t(value.row_bytes));
            }
            if (max_row_bytes == 0) {
                return false;
            }
            std::vector<uint8_t> live(std::min<size_t>(
                size_t(1) << 20, max_row_bytes));
            const auto rows_match = [row, &live](
                    const std::vector<ggml_tensor *> & tensors,
                    const std::vector<recurrent_tensor_row> & rows) {
                for (size_t i = 0; i < tensors.size(); ++i) {
                    const auto * tensor = tensors[i];
                    const auto & expected_row = rows[i];
                    if ((tensor != nullptr) != expected_row.present) {
                        return false;
                    }
                    if (!tensor) {
                        continue;
                    }
                    const uint64_t row_bytes = ggml_row_size(
                        tensor->type, tensor->ne[0]);
                    if (expected_row.type != tensor->type ||
                        expected_row.row_bytes != row_bytes ||
                        expected_row.bytes.size() != row_bytes ||
                        uint64_t(row) >= uint64_t(tensor->ne[1])) {
                        return false;
                    }
                    for (size_t offset = 0; offset < row_bytes;) {
                        const size_t take = std::min(
                            live.size(), size_t(row_bytes)-offset);
                        ggml_backend_tensor_get(
                            tensor, live.data(),
                            size_t(row)*size_t(row_bytes)+offset, take);
                        if (std::memcmp(
                                live.data(), expected_row.bytes.data()+offset,
                                take) != 0) {
                            return false;
                        }
                        offset += take;
                    }
                }
                return true;
            };
            return rows_match(target.r_l, expected.r) &&
                   rows_match(target.p_l, expected.p) &&
                   rows_match(target.s_l, expected.s);
        } catch (...) {
            return false;
        }
    }

    static bool prepare_replacement(
            const void * context,
            std::unique_ptr<vbr_parsed_companion_image> incoming,
            std::unique_ptr<vbr_parsed_companion_image> recovery_base,
            llama_seq_id destination,
            std::unique_ptr<vbr_prepared_companion_image> & output) noexcept {
        output.reset();
        auto * target = static_cast<llama_memory_recurrent *>(
            const_cast<void *>(context));
        auto * parsed = dynamic_cast<vbr_recurrent_parsed_image *>(
            incoming.get());
        auto * recovery = dynamic_cast<vbr_recurrent_parsed_image *>(
            recovery_base.get());
        size_t physical = 0;
        uint32_t row = 0;
        if (!target || !parsed || !recovery ||
            !parsed_compatible(*target, *parsed) ||
            !live_matches(*target, destination, *recovery) ||
            !live_location(*target, destination, physical, row) ||
            row != physical) {
            return false;
        }
        // Occupied import already owns the controller operation exclusively.
        // Authenticate the incumbent row once, then overwrite that same row
        // and retain its parsed bytes for rollback.  Allocating a second
        // full-size recurrent tensor set here would create an unquoted device
        // peak even though the artifact contains only one active row.
        auto image = std::make_unique<vbr_recurrent_prepared_image>();
        image->target = target;
        image->destination = destination;
        image->destination_was_empty = false;
        image->replacement_physical = physical;
        image->replacement_position = parsed->position;
        image->replacement_binding_epoch = target->tensor_binding_epoch_;
        image->recovery.reset(static_cast<vbr_recurrent_parsed_image *>(
            recovery_base.release()));
        output = std::move(image);
        return write_rows(*target, uint32_t(physical), *parsed);
    }

    static void publish(
            const void * context,
            vbr_prepared_companion_image & base) noexcept {
        // BEGIN VBR_IMPORT_RECURRENT_METADATA_SWAP
        auto * target = static_cast<llama_memory_recurrent *>(
            const_cast<void *>(context));
        auto & image = static_cast<vbr_recurrent_prepared_image &>(base);
        GGML_ASSERT(target && image.target == target && image.destination >= 0);
        if (image.destination_was_empty) {
            target->cells.swap(image.cells);
            std::swap(target->head, image.head);
            std::swap(target->used, image.used);
            std::swap(target->rs_z, image.rs_z);
            target->rs_idx.swap(image.rs_idx);
            target->rollback_valid_depth.swap(
                image.rollback_valid_depth);
        } else {
            GGML_ASSERT(image.replacement_physical < target->cells.size());
            auto & cell = target->cells[image.replacement_physical];
            cell.pos = image.replacement_position;
            cell.src = int32_t(image.replacement_physical);
            cell.src0 = -1;
            target->reset_rollback_state(image.destination);
        }
        image.target = nullptr;
        // END VBR_IMPORT_RECURRENT_METADATA_SWAP
    }

    static bool recheck(
            const void * context,
            const vbr_prepared_companion_image & base) noexcept {
        const auto * image =
            static_cast<const vbr_recurrent_prepared_image *>(&base);
        if (!image || image->target != context) {
            return false;
        }
        const auto * target = static_cast<const llama_memory_recurrent *>(context);
        if (image->destination_was_empty) {
            return target->tensor_binding_epoch_ ==
                    image->replacement_binding_epoch &&
                target_empty(context);
        }
        size_t physical = 0;
        uint32_t row = 0;
        // The controller operation excludes decode writers until the no-fail
        // publish. Recheck structural/binding currency here; rescanning the
        // full row would duplicate the authenticated D2H comparison performed
        // immediately before the controlled write.
        return image->recovery &&
            target->tensor_binding_epoch_ ==
                image->replacement_binding_epoch &&
            live_location(*target, image->destination, physical, row) &&
            physical == image->replacement_physical && row == physical &&
            target->cells[physical].pos == image->recovery->position;
    }

    static bool rollback(
            const void *, vbr_prepared_companion_image & base) noexcept {
        auto & image = static_cast<vbr_recurrent_prepared_image &>(base);
        if (!image.target) {
            return false;
        }
        if (image.destination_was_empty) {
            return true;
        }
        return image.recovery &&
            image.replacement_physical <= UINT32_MAX &&
            write_rows(*image.target, uint32_t(image.replacement_physical),
                       *image.recovery);
    }

    static bool target_empty(const void * context) noexcept {
        const auto * target = static_cast<const llama_memory_recurrent *>(context);
        return target != nullptr && target->used == 0 &&
               std::all_of(target->cells.begin(), target->cells.end(),
                   [](const llama_memory_recurrent::mem_cell & cell) {
                       return cell.is_empty();
                   });
    }
};

bool vbr_parse_recurrent_companion(
        const void *,
        const vbr_artifact_companion_payload & descriptor,
        const artifact_segment_chain & source,
        const vbr_target_companion_snapshot & target_snapshot,
        std::unique_ptr<vbr_parsed_companion_image> & output) noexcept {
    try {
        output.reset();
        auto * target = static_cast<llama_memory_recurrent *>(
            const_cast<void *>(target_snapshot.target_cookie));
        if (!target || !target_snapshot.available ||
            descriptor.kind != vbr_artifact_companion_kind::recurrent ||
            descriptor.format_version != 1 ||
            descriptor.payload_bytes != source.size() ||
            source.size() > std::numeric_limits<size_t>::max()) {
            return false;
        }
        recurrent_cursor cursor(source);
        static constexpr uint32_t SEQUENCE_STATE_MAGIC = 0xaf143cd8;
        uint32_t magic = 0;
        llama_seq_id source_sequence = -1;
        uint32_t cell_count = 0;
        llama_pos position = -1;
        uint32_t n_seq_id = 0;
        uint32_t s_trans = 0;
        uint32_t n_layer = 0;
        if (!cursor.scalar(magic) || magic != SEQUENCE_STATE_MAGIC ||
            !cursor.scalar(source_sequence) || source_sequence < 0 ||
            !cursor.scalar(cell_count) || cell_count != 1 ||
            !cursor.scalar(position) || position < 0 ||
            !cursor.scalar(n_seq_id) || n_seq_id != 0 ||
            !cursor.scalar(s_trans) || s_trans != 0 ||
            !cursor.scalar(n_layer) || n_layer != target->r_l.size()) {
            return false;
        }
        auto parsed = std::make_unique<vbr_recurrent_parsed_image>();
        parsed->target = target;
        parsed->position = position;
        parsed->r.resize(n_layer);
        parsed->p.resize(n_layer);
        parsed->s.resize(n_layer);
        const auto read_typed_row = [&](ggml_tensor * tensor,
                                        recurrent_tensor_row & row) {
            int32_t type = -1;
            uint64_t row_bytes = 0;
            if (!cursor.scalar(type) || type != int32_t(tensor->type) ||
                !cursor.scalar(row_bytes) ||
                row_bytes != uint64_t(ggml_row_size(
                    tensor->type, tensor->ne[0])) ||
                row_bytes > std::numeric_limits<size_t>::max() ||
                !cursor.block(row.bytes, size_t(row_bytes))) {
                return false;
            }
            row.present = true;
            row.type = tensor->type;
            row.row_bytes = row_bytes;
            return true;
        };
        // PLE layers serialize one additional convolution-history row
        // immediately after their R row.  Unlike R/S, its type is implicit in
        // the model tensor; the native sequence-state format stores only the
        // row byte count before the payload.
        for (size_t i = 0; i < target->r_l.size(); ++i) {
            auto * recurrent = target->r_l[i];
            if (!recurrent) {
                continue;
            }
            if (!read_typed_row(recurrent, parsed->r[i])) {
                return false;
            }
            auto * tensor = target->p_l[i];
            if (!tensor) {
                continue;
            }
            uint64_t row_bytes = 0;
            auto & row = parsed->p[i];
            if (!cursor.scalar(row_bytes) ||
                row_bytes != uint64_t(ggml_row_size(
                    tensor->type, tensor->ne[0])) ||
                row_bytes > std::numeric_limits<size_t>::max() ||
                !cursor.block(row.bytes, size_t(row_bytes))) {
                return false;
            }
            row.present = true;
            row.type = tensor->type;
            row.row_bytes = row_bytes;
        }
        for (size_t i = 0; i < target->s_l.size(); ++i) {
            if (target->s_l[i] &&
                !read_typed_row(target->s_l[i], parsed->s[i])) {
                return false;
            }
        }
        if (!cursor.finished()) {
            return false;
        }
        output = std::move(parsed);
        return true;
    } catch (...) {
        output.reset();
        return false;
    }
}

vbr_companion_adoption_provider vbr_recurrent_companion_adoption_provider(
        llama_memory_recurrent & target) noexcept {
    vbr_companion_adoption_provider provider;
    provider.kind = vbr_artifact_companion_kind::recurrent;
    provider.target_cookie = &target;
    provider.context = &target;
    provider.prepare = &vbr_recurrent_prepared_image::prepare;
    provider.prepare_replacement =
        &vbr_recurrent_prepared_image::prepare_replacement;
    provider.target_empty = &vbr_recurrent_prepared_image::target_empty;
    provider.recheck = &vbr_recurrent_prepared_image::recheck;
    provider.publish_swap = &vbr_recurrent_prepared_image::publish;
    provider.rollback = &vbr_recurrent_prepared_image::rollback;
    return provider;
}

//
// llama_memory_recurrent
//

llama_memory_recurrent::llama_memory_recurrent(
        const llama_model & model,
                ggml_type   type_r,
                ggml_type   type_s,
                     bool   offload,
                 uint32_t   mem_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
    const layer_filter_cb & filter) : hparams(model.hparams), n_seq_max(n_seq_max) {
    const int32_t n_layer = hparams.n_layer();

    head = 0;
    size = mem_size;
    used = 0;

    this->n_rs_seq = n_rs_seq;
    rs_idx.assign(n_seq_max, 0);
    rollback_valid_depth.assign(n_seq_max, 0);

    cells.clear();
    cells.resize(mem_size);

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                // r and s per layer, plus the separate PLE conv row where the model has one
                /*.mem_size   =*/ size_t((hparams.ple_conv_state() > 0 ? 3u : 2u)*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    r_l.resize(n_layer);
    s_l.resize(n_layer);
    p_l.resize(n_layer);

    for (int i = 0; i < n_layer; i++) {
        if (filter && !filter(i)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, i);
            continue;
        }

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(i);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s, layer %3d: dev = %s\n", __func__, i, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for rs cache");
        }

        const uint32_t n_rows = mem_size * (1 + n_rs_seq);
        ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
        ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_rows);
        ggml_format_name(r, "cache_r_l%d", i);
        ggml_format_name(s, "cache_s_l%d", i);
        r_l[i] = r;
        s_l[i] = s;

        // the PLE history needs its own row: Meta must mirror it while the delta-net conv state next door stays split
        if (hparams.ple_conv_state() > 0 && hparams.is_ple(i)) {
            ggml_tensor * p = ggml_new_tensor_2d(ctx, type_r, hparams.ple_conv_state(), n_rows);
            ggml_format_name(p, "cache_ple_r_l%d", i);
            p_l[i] = p;
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = nullptr;
        if (hparams.no_alloc) {
            // memory-fit probe: report sizes without touching device memory (mirrors the KV
            // cache). Probes that really allocated here depressed the measured free VRAM by the
            // full RS size while the projection ALSO counted it — the fit double-billed RS and
            // shrank contexts for phantom deficits (np=4 27B: 619 "missing" MiB, ~449 phantom).
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // the scheduler must not try to allocate the cache tensors
            }
        } else {
            // co-tenancy hold-aware alloc: hybrid models' recurrent state is often the
            // second-largest context alloc — same patience as the attention KV twin
            buf = llama_vram_hold_alloc_ctx_tensors(ctx.get(), buft);
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for rs cache");
        }
        if (!hparams.no_alloc) {
            ggml_backend_buffer_clear(buf, 0);
        }
        LLAMA_LOG_INFO("%s: %10s RS buffer size = %8.2f MiB%s\n", __func__, ggml_backend_buffer_name(buf),
                (hparams.no_alloc ? ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft)
                                  : ggml_backend_buffer_get_size(buf)) / 1024.0 / 1024.0,
                hparams.no_alloc ? " (projected)" : "");

        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_r = size_r_bytes();
        const size_t memory_size_s = size_s_bytes();
        const size_t memory_size_p = size_p_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs %2u rs_seq), R (%s): %7.2f MiB, S (%s): %7.2f MiB, P (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_r + memory_size_s + memory_size_p) / (1024.0f * 1024.0f), mem_size, n_layer, n_seq_max, n_rs_seq,
                ggml_type_name(type_r), (float)memory_size_r / (1024.0f * 1024.0f),
                ggml_type_name(type_s), (float)memory_size_s / (1024.0f * 1024.0f),
                ggml_type_name(type_r), (float)memory_size_p / (1024.0f * 1024.0f));
    }
}

void llama_memory_recurrent::clear(bool data) {
    for (int32_t i = 0; i < (int32_t) size; ++i) {
        cells[i].pos = -1;
        cells[i].seq_id.clear();
        cells[i].src = -1;
        cells[i].tail = -1;
    }

    head = 0;
    used = 0;

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }

    std::fill(rs_idx.begin(), rs_idx.end(), 0);
    std::fill(rollback_valid_depth.begin(), rollback_valid_depth.end(), 0);
}

bool llama_memory_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0 && (uint32_t) seq_id >= this->n_seq_max) {
        LLAMA_LOG_ERROR("%s: invalid seq_id (%d) - larger than n_seq_max (%d)\n", __func__, seq_id, this->n_seq_max);
        return false;
    }

    const bool rm_all = p0 == 0 && p1 == std::numeric_limits<llama_pos>::max();
    if (rm_all) {
        if (seq_id >= 0) {
            reset_rollback_state(seq_id);
        } else {
            std::fill(rs_idx.begin(), rs_idx.end(), 0);
            std::fill(rollback_valid_depth.begin(), rollback_valid_depth.end(), 0);
        }
    }

    // models like Mamba or RWKV can't have a state partially erased at the end
    // of the sequence because their state isn't preserved for previous tokens
    if (seq_id >= (int64_t) size) {
        // could be fatal
        return false;
    }
    if (0 <= seq_id) {
        int32_t & tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];

            // partial rollback via per-token snapshot index (bounded by n_rs_seq)
            if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
                const llama_pos rollback = cell.pos - (p0 - 1);
                GGML_ASSERT((size_t) seq_id < rollback_valid_depth.size());
                GGML_ASSERT(rollback_valid_depth[seq_id] <= n_rs_seq);
                const bool pending = rs_idx[seq_id] != 0;
                if (rollback >= 1 && (pending || rollback > (llama_pos) rollback_valid_depth[seq_id])) {
                    return false;
                }
                if (rollback >= 1) {
                    GGML_ASSERT(rollback <= (llama_pos) n_rs_seq);
                    set_rs_idx(seq_id, (uint32_t) rollback);
                    cell.pos = p0 - 1;
                }
            }
            // invalidate tails which will be cleared
            if (p0 <= cell.pos && cell.pos < p1) {
                tail_id = -1;
            }
        }
    } else {
        // seq_id is negative, then the range should include everything or nothing
        if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
            return false;
        }
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].pos >= p0 && cells[i].pos < p1) {
            if (seq_id < 0) {
                cells[i].seq_id.clear();
            } else if (cells[i].has_seq_id(seq_id)) {
                cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cells[i].pos >= 0) {
                    used--;
                }
                cells[i].pos = -1;
                cells[i].src = -1;
                if (new_head == size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }

    return true;
}

void llama_memory_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    (void) try_seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

bool llama_memory_recurrent::try_seq_cp(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos    p0,
        llama_pos    p1) {
    if (seq_id_src == seq_id_dst) {
        return true;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if ((uint32_t) seq_id_dst >= size || (uint32_t) seq_id_src >= size) {
        LLAMA_LOG_ERROR("%s: invalid sequence ids for copy: src = %d, dst = %d, size = %u\n",
                __func__, seq_id_src, seq_id_dst, size);
        return false;
    }

    auto & tail_src_meta = cells[seq_id_src];
    auto & tail_dst_meta = cells[seq_id_dst];

    uint32_t next_empty_cell = size;
    if (tail_src_meta.tail >= 0) {
        // Recurrent states must not share a mutable cell. Reserve the replacement before
        // clearing the current destination so exhaustion leaves that destination untouched.
        for (uint32_t i = head; i < head + size; ++i) {
            const uint32_t idx = i % size;
            if (cells[idx].is_empty()) {
                next_empty_cell = idx;
                break;
            }
        }

        if (next_empty_cell == size) {
            LLAMA_LOG_ERROR("%s: failed to find available cell for copy\n", __func__);
            return false;
        }
    }

    if (tail_dst_meta.tail >= 0) {
        // Reservation succeeded (or the source is empty), so replacing the destination is safe.
        const bool removed = seq_rm(seq_id_dst, -1, -1);
        GGML_ASSERT(removed);
        GGML_UNUSED(removed);
    }

    if (tail_src_meta.tail < 0) {
        return true;
    }

    const int32_t src_cell_id = tail_src_meta.tail;
    auto & cell_src = cells[src_cell_id];
    auto & empty_cell = cells[next_empty_cell];

    // Copy tensors data
    copy_cell(src_cell_id, next_empty_cell);

    empty_cell.pos = cell_src.pos;
    empty_cell.src = next_empty_cell; // results in a copy in the graph if needed
    empty_cell.seq_id.insert(seq_id_dst);
    tail_dst_meta.tail = next_empty_cell;
    used += 1;

    // copy_cell() installs only the active row; rollback planes are not migrated.
    reset_rollback_state(seq_id_dst);
    GGML_ASSERT(rollback_valid_depth[seq_id_dst] == 0);

    return true;
}

void llama_memory_recurrent::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = size;

    for (llama_seq_id s = 0; (size_t) s < rollback_valid_depth.size(); ++s) {
        if (s != seq_id) {
            reset_rollback_state(s);
        }
    }

    for (uint32_t i = 0; i < size; ++i) {
        if ((llama_seq_id) i != seq_id) {
            cells[i].tail = -1;
        }

        if (!cells[i].has_seq_id(seq_id)) {
            if (cells[i].pos >= 0) {
                used--;
            }

            cells[i].pos = -1;
            cells[i].src = -1;
            cells[i].seq_id.clear();

            if (new_head == size){
                new_head = i;
            }
        } else {
            cells[i].seq_id.clear();
            cells[i].seq_id.insert(seq_id);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }
}

void llama_memory_recurrent::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (shift == 0) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be shifted
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos += shift;
            }
        }
    }
}

void llama_memory_recurrent::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be changed
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos /= d;
            }
        }
    }
}

llama_pos llama_memory_recurrent::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos result = std::numeric_limits<llama_pos>::max();

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::min(result, cells[i].pos);
        }
    }

    if (result == std::numeric_limits<llama_pos>::max()) {
        result = -1;
    }

    return result;
}

llama_pos llama_memory_recurrent::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos result = -1;

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);
        }
    }

    return result;
}

void llama_memory_recurrent::set_rs_idx(llama_seq_id seq_id, uint32_t idx) {
    if (seq_id < 0) {
        std::fill(rs_idx.begin(), rs_idx.end(), 0);
        return;
    }

    assert(n_seq_max == rs_idx.size());

    GGML_ASSERT((uint32_t) seq_id < n_seq_max);
    GGML_ASSERT(idx <= n_rs_seq);

    rs_idx[seq_id] = idx;
}

void llama_memory_recurrent::reset_rollback_state(llama_seq_id seq_id) {
    if (seq_id < 0 || (size_t) seq_id >= rollback_valid_depth.size()) {
        return;
    }

    set_rs_idx(seq_id, 0);
    rollback_valid_depth[seq_id] = 0;
}

void llama_memory_recurrent::bump_tensor_binding_epoch() noexcept {
    copy_graph_cache_.clear();
    tensor_binding_epoch_ = tensor_binding_epoch_ == UINT64_MAX
        ? 1
        : tensor_binding_epoch_ + 1;
}

void llama_memory_recurrent::invalidate_rollback(const llama_ubatch & ubatch) {
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(n_seq_tokens > 0);

    for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
        const uint32_t i = s * n_seq_tokens;
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];
            GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < rollback_valid_depth.size());

            // The graph about to run will overwrite the snapshot planes, so their old
            // validity ends here. Do not clear rs_idx: a preceding seq_rm() may have
            // selected one of those planes as this graph's input. s_copy() consumes and
            // clears that pending selector while constructing the graph.
            rollback_valid_depth[seq_id] = 0;
        }
    }
}

void llama_memory_recurrent::commit_rollback(const llama_ubatch & ubatch) {
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(n_seq_tokens > 0);

    const uint32_t valid_depth = std::min(n_rs_seq, n_seq_tokens - 1);
    GGML_ASSERT(valid_depth <= n_rs_seq);

    for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
        const uint32_t i = s * n_seq_tokens;
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];
            GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < rollback_valid_depth.size());
            rollback_valid_depth[seq_id] = valid_depth;
        }
    }
}

void llama_memory_recurrent::copy_cell(int32_t i_src, int32_t i_dst) {
    if (i_src == i_dst || i_src < 0 || i_dst < 0) {
        return;
    }

    const uint32_t n_recur = hparams.n_layer();

    ggml_backend_dev_t copy_dev = nullptr;
    bool compatible = true;
    size_t n_tensors = 0;
    auto inspect = [&](ggml_tensor * t) {
        if (!t || !compatible) {
            return;
        }
        if (!t->buffer || ggml_backend_buffer_is_host(t->buffer) ||
            ggml_backend_buffer_is_meta(t->buffer)) {
            compatible = false;
            return;
        }
        auto * buft = ggml_backend_buffer_get_type(t->buffer);
        auto * dev = buft ? ggml_backend_buft_get_device(buft) : nullptr;
        if (!dev || (copy_dev && copy_dev != dev)) {
            compatible = false;
            return;
        }
        copy_dev = dev;
        ++n_tensors;
    };
    for (uint32_t il = 0; il < n_recur; ++il) {
        inspect(r_l[il]);
        inspect(s_l[il]);
    }

    if (compatible && copy_dev && n_tensors > 0) {
        auto backend_it = copy_backends_.find(copy_dev);
        if (backend_it == copy_backends_.end()) {
            backend_it = copy_backends_.emplace(
                copy_dev, ggml_backend_ptr(ggml_backend_dev_init(copy_dev, nullptr))).first;
        }
        ggml_backend_t backend = backend_it->second.get();
        if (backend) {
            const auto key = std::make_tuple(copy_dev, i_src, i_dst);
            auto cached = copy_graph_cache_.find(key);
            if (cached != copy_graph_cache_.end()) {
                const ggml_status status = ggml_backend_graph_compute_async(
                    cached->second.backend, cached->second.graph);
                ggml_backend_synchronize(cached->second.backend);
                if (status == GGML_STATUS_SUCCESS) {
                    return;
                }
                copy_graph_cache_.erase(cached);
            }

            const size_t graph_size = n_tensors * 5 + 8;
            const size_t ctx_mem = 3 * n_tensors * ggml_tensor_overhead() +
                ggml_graph_overhead_custom(graph_size, false);
            ggml_init_params batch_params = {ctx_mem, nullptr, true};
            ggml_context * ctx = ggml_init(batch_params);
            ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);

            auto add_copy = [&](ggml_tensor * t) {
                if (!t) {
                    return;
                }
                ggml_tensor * src_v = ggml_view_1d(
                    ctx, t, t->ne[0], (size_t) i_src * t->nb[1]);
                ggml_tensor * dst_v = ggml_view_1d(
                    ctx, t, t->ne[0], (size_t) i_dst * t->nb[1]);
                src_v->buffer = t->buffer;
                dst_v->buffer = t->buffer;
                ggml_build_forward_expand(graph, ggml_cpy(ctx, src_v, dst_v));
            };
            for (uint32_t il = 0; il < n_recur; ++il) {
                add_copy(r_l[il]);
                add_copy(s_l[il]);
            }

            const ggml_status status =
                ggml_backend_graph_compute_async(backend, graph);
            if (status == GGML_STATUS_SUCCESS) {
                ggml_backend_synchronize(backend);
                copy_graph_cache_.emplace(key, copy_graph_entry {
                    ggml_context_ptr(ctx), graph, backend,
                });
                return;
            }
            ggml_backend_synchronize(backend);
            ggml_free(ctx);
        }
    }

    // create one shared ggml context for all fallback view pairs (meta caches
    // need two views per shard)
    ggml_init_params params = {
        /*.mem_size   =*/ size_t(64 * n_recur * ggml_tensor_overhead()),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    // a cell copy stays within one tensor: on a plain buffer it is a single row copy. On a
    // meta (tensor-split) buffer the cell row is sharded across devices, each shard with its
    // own row stride — decompose into device-local row copies on the shard tensors (the meta
    // buffer cannot serve ad-hoc views through get/set_tensor)
    auto copy_row = [&](ggml_tensor * t) {
        if (t->buffer != nullptr && ggml_backend_buffer_is_meta(t->buffer)) {
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(t->buffer);
            for (size_t j = 0; j < n_bufs; ++j) {
                ggml_tensor * st = ggml_backend_meta_buffer_simple_tensor(t, j);
                if (st == nullptr || st->ne[0] == 0) {
                    continue; // zero-width shard (rotation gave this device no slice)
                }
                ggml_tensor * src_v = ggml_view_1d(ctx, st, st->ne[0], i_src * st->nb[1]);
                ggml_tensor * dst_v = ggml_view_1d(ctx, st, st->ne[0], i_dst * st->nb[1]);
                src_v->buffer = st->buffer;
                dst_v->buffer = st->buffer;
                ggml_backend_tensor_copy(src_v, dst_v);
            }
            return;
        }
        ggml_tensor * src_v = ggml_view_1d(ctx, t, t->ne[0], i_src * t->nb[1]);
        ggml_tensor * dst_v = ggml_view_1d(ctx, t, t->ne[0], i_dst * t->nb[1]);
        src_v->buffer = t->buffer;
        dst_v->buffer = t->buffer;
        ggml_backend_tensor_copy(src_v, dst_v);
    };

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (r_l[il]) {
            copy_row(r_l[il]);
        }
        if (s_l[il]) {
            copy_row(s_l[il]);
        }
    }

    ggml_free(ctx);
}

int llama_memory_recurrent::get_cell_count(llama_seq_id seq_id) const {
    int count = 0;
    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            count++;
        }
    }
    return count;
}

bool llama_memory_recurrent::vbr_capture_readiness_cells(
        uint64_t,
        uint64_t & committed,
        uint64_t & projected,
        uint64_t & capacity) const {
    committed = used;
    projected = used;
    capacity = size;
    return capacity != 0;
}

bool llama_memory_recurrent::expand(uint32_t new_mem_size) {
    return new_mem_size <= size || resize(new_mem_size);
}

bool llama_memory_recurrent::shrink(uint32_t new_mem_size) {
    return new_mem_size >= size || resize(new_mem_size);
}

bool llama_memory_recurrent::resize(uint32_t new_mem_size) {
    if (new_mem_size == size) {
        return true;
    }
    if (new_mem_size == 0) {
        LLAMA_LOG_ERROR("%s: recurrent memory requires at least one resident cell\n", __func__);
        return false;
    }

    try {
        const int32_t n_layer = hparams.n_layer();
        const uint32_t old_size = size;
        const uint32_t n_copy = std::min(old_size, new_mem_size);
        const size_t tensor_overhead = ggml_tensor_overhead();

        if (new_mem_size < old_size) {
            // A logical sequence can live in any physical cell. Truncating a high
            // occupied cell (or a retained cell that still references one) would
            // silently discard recurrent state while the attention half survives.
            // Refuse atomically; callers may retry after clearing/compacting it.
            for (uint32_t i = 0; i < old_size; ++i) {
                const auto & cell = cells[i];
                if (!cell.seq_id.empty()) {
                    if (i >= new_mem_size || cell.src >= (int32_t) new_mem_size ||
                        cell.src0 >= (int32_t) new_mem_size) {
                        LLAMA_LOG_WARN("%s: cannot shrink recurrent memory while cell %u references truncated storage\n",
                                __func__, i);
                        return false;
                    }
                    for (llama_seq_id seq_id : cell.seq_id) {
                        if (seq_id < 0 || (uint32_t) seq_id >= new_mem_size) {
                            LLAMA_LOG_WARN("%s: cannot shrink recurrent memory while sequence %d remains live\n",
                                    __func__, seq_id);
                            return false;
                        }
                    }
                }
            }
            for (uint32_t seq_id = 0; seq_id < new_mem_size; ++seq_id) {
                if (cells[seq_id].tail >= (int32_t) new_mem_size) {
                    LLAMA_LOG_WARN("%s: cannot shrink recurrent memory while sequence %u has a high physical tail\n",
                            __func__, seq_id);
                    return false;
                }
            }
        }

        if (n_layer < 0 ||
            (tensor_overhead != 0 && size_t(n_layer) > SIZE_MAX / tensor_overhead / 2)) {
            LLAMA_LOG_ERROR("%s: resized rs context metadata size overflows\n", __func__);
            return false;
        }
        if (new_mem_size > n_seq_max ||
            new_mem_size > cells.max_size() ||
            new_mem_size > rs_idx.max_size() ||
            new_mem_size > rollback_valid_depth.max_size()) {
            LLAMA_LOG_ERROR("%s: requested rs cache size %u exceeds its capacity %u\n",
                    __func__, new_mem_size, n_seq_max);
            return false;
        }

        const uint64_t row_groups = uint64_t(n_rs_seq) + 1;
        if (new_mem_size != 0 && row_groups > UINT64_MAX / new_mem_size) {
            LLAMA_LOG_ERROR("%s: resized rs row count overflows\n", __func__);
            return false;
        }
        const uint64_t new_rows = uint64_t(new_mem_size) * row_groups;
        if (new_rows > INT64_MAX || new_rows > SIZE_MAX) {
            LLAMA_LOG_ERROR("%s: resized rs row count exceeds ggml dimensions\n", __func__);
            return false;
        }
        const size_t context_bytes = 2 * size_t(n_layer) * tensor_overhead;

        auto tensor_shape_fits = [&](ggml_type type, uint32_t n_embd) {
            if (n_embd != 0 && new_rows > uint64_t(INT64_MAX) / n_embd) {
                return false;
            }
            const size_t row_bytes = ggml_row_size(type, n_embd);
            return row_bytes == 0 || new_rows <= SIZE_MAX / row_bytes;
        };

        std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

        auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
            auto it = ctx_map.find(buft);
            if (it == ctx_map.end()) {
                ggml_init_params params = {
                    /*.mem_size   =*/ context_bytes,
                    /*.mem_buffer =*/ NULL,
                    /*.no_alloc   =*/ true,
                };
                ggml_context_ptr ctx(ggml_init(params));
                if (!ctx) {
                    return nullptr;
                }
                it = ctx_map.emplace(buft, std::move(ctx)).first;
            }
            return it->second.get();
        };

        const std::vector<ggml_tensor *> old_r_l = r_l;
        const std::vector<ggml_tensor *> old_s_l = s_l;
        std::vector<ggml_tensor *> new_r_l(r_l.size(), nullptr);
        std::vector<ggml_tensor *> new_s_l(s_l.size(), nullptr);

        for (int i = 0; i < n_layer; i++) {
            if (!old_r_l[i] && !old_s_l[i]) {
                continue;
            }

            if ((old_r_l[i] && !tensor_shape_fits(old_r_l[i]->type, hparams.n_embd_r())) ||
                (old_s_l[i] && !tensor_shape_fits(old_s_l[i]->type, hparams.n_embd_s()))) {
                LLAMA_LOG_ERROR("%s: resized rs tensor shape overflows\n", __func__);
                return false;
            }

            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(old_r_l[i] ? old_r_l[i]->buffer : old_s_l[i]->buffer);
            ggml_context * ctx = ctx_for_buft(buft);
            if (!ctx) {
                LLAMA_LOG_ERROR("%s: failed to create ggml context for resized rs cache\n", __func__);
                return false;
            }

            if (old_r_l[i]) {
                ggml_tensor * r = ggml_new_tensor_2d(ctx, old_r_l[i]->type, hparams.n_embd_r(), (int64_t) new_rows);
                if (!r) {
                    LLAMA_LOG_ERROR("%s: failed to create resized R tensor\n", __func__);
                    return false;
                }
                ggml_format_name(r, "cache_r_l%d", i);
                new_r_l[i] = r;
            }
            if (old_s_l[i]) {
                ggml_tensor * s = ggml_new_tensor_2d(ctx, old_s_l[i]->type, hparams.n_embd_s(), (int64_t) new_rows);
                if (!s) {
                    LLAMA_LOG_ERROR("%s: failed to create resized S tensor\n", __func__);
                    return false;
                }
                ggml_format_name(s, "cache_s_l%d", i);
                new_s_l[i] = s;
            }
        }

        std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> new_ctxs_bufs;
        new_ctxs_bufs.reserve(ctx_map.size());
        for (auto & [buft, ctx] : ctx_map) {
            ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
            if (!buf) {
                LLAMA_LOG_ERROR("%s: failed to allocate resized rs buffer\n", __func__);
                return false;
            }
            ggml_backend_buffer_clear(buf.get(), 0);
            new_ctxs_bufs.emplace_back(std::move(ctx), std::move(buf));
        }

        if (n_copy > 0) {
            const size_t n_copy_rows = size_t(uint64_t(n_copy) * row_groups);
            std::vector<uint8_t> tmp;
            for (int i = 0; i < n_layer; i++) {
                if (old_r_l[i] && new_r_l[i]) {
                    size_t bytes = ggml_row_size(old_r_l[i]->type, hparams.n_embd_r()) * n_copy_rows;
                    tmp.resize(bytes);
                    ggml_backend_tensor_get(old_r_l[i], tmp.data(), 0, bytes);
                    ggml_backend_tensor_set(new_r_l[i], tmp.data(), 0, bytes);
                }
                if (old_s_l[i] && new_s_l[i]) {
                    size_t bytes = ggml_row_size(old_s_l[i]->type, hparams.n_embd_s()) * n_copy_rows;
                    tmp.resize(bytes);
                    ggml_backend_tensor_get(old_s_l[i], tmp.data(), 0, bytes);
                    ggml_backend_tensor_set(new_s_l[i], tmp.data(), 0, bytes);
                }
            }
        }

        std::vector<mem_cell> new_cells = cells;
        new_cells.resize(new_mem_size);
        std::vector<uint32_t> new_rs_idx(new_mem_size, 0);
        std::vector<uint32_t> new_rollback_valid_depth(new_mem_size, 0);

        uint32_t new_used = 0;
        for (auto & cell : new_cells) {
            cell.tail = -1;

            for (auto it = cell.seq_id.begin(); it != cell.seq_id.end();) {
                if (*it < 0 || (uint32_t) *it >= new_mem_size) {
                    LLAMA_LOG_WARN("%s: dropping seq_id %d after resize %u -> %u\n",
                            __func__, *it, old_size, new_mem_size);
                    it = cell.seq_id.erase(it);
                } else {
                    ++it;
                }
            }

            if (cell.seq_id.empty()) {
                cell.pos  = -1;
                cell.src  = -1;
                cell.src0 = -1;
                continue;
            }

            if (cell.src >= (int32_t) new_mem_size) {
                LLAMA_LOG_WARN("%s: clearing out-of-range src %d after resize %u -> %u\n",
                        __func__, cell.src, old_size, new_mem_size);
                cell.src = -1;
            }
            if (cell.src0 >= (int32_t) new_mem_size) {
                LLAMA_LOG_WARN("%s: clearing out-of-range src0 %d after resize %u -> %u\n",
                        __func__, cell.src0, old_size, new_mem_size);
                cell.src0 = -1;
            }

            new_used++;
        }

        for (uint32_t i = 0; i < new_mem_size; ++i) {
            for (llama_seq_id seq_id : new_cells[i].seq_id) {
                new_cells[seq_id].tail = i;
            }
        }

        const uint32_t new_head = new_mem_size == 0 ? 0 : std::min(head, new_mem_size - 1);
        const uint32_t new_n = new_mem_size == 0 ? 0 : std::min(n, new_mem_size);
        const int32_t new_rs_z = new_mem_size == 0 || rs_z >= (int32_t) new_mem_size ? -1 : rs_z;

        if (recurrent_resize_test_fail_before_publish()) {
            LLAMA_LOG_ERROR("%s: injected failure before publishing resized rs cache\n", __func__);
            return false;
        }

        // All allocation, copying, and metadata normalization succeeded. These swaps are
        // non-throwing, and the old contexts stay alive in the staged locals until every
        // live tensor pointer has been rebound.
        r_l.swap(new_r_l);
        s_l.swap(new_s_l);
        ctxs_bufs.swap(new_ctxs_bufs);
        cells.swap(new_cells);
        rs_idx.swap(new_rs_idx);
        rollback_valid_depth.swap(new_rollback_valid_depth);
        size = new_mem_size;
        used = new_used;
        head = new_head;
        n = new_n;
        rs_z = new_rs_z;
        bump_tensor_binding_epoch();

        const size_t memory_size_r = size_r_bytes();
        const size_t memory_size_s = size_s_bytes();
        LLAMA_LOG_INFO("%s: resized %u -> %u cells, used=%u, head=%u, n=%u, rs_z=%d, R: %7.2f MiB, S: %7.2f MiB\n", __func__,
                old_size, new_mem_size,
                used, head, n, rs_z,
                (float)memory_size_r / (1024.0f * 1024.0f),
                (float)memory_size_s / (1024.0f * 1024.0f));

        return true;
    } catch (const std::length_error &) {
        LLAMA_LOG_ERROR("%s: host container size is invalid while staging resized rs cache\n", __func__);
        return false;
    } catch (const std::bad_alloc &) {
        LLAMA_LOG_ERROR("%s: host allocation failed while staging resized rs cache\n", __func__);
        return false;
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_recurrent::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());
        if (hparams.no_alloc) {
            // fit probe: the buffer is a 0-size dummy — report what the real allocation costs,
            // or the fit stops billing RS entirely and under-projects by the whole cache
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }
    return ret;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_recurrent::memory_breakdown_fixed() const {
    // the whole RS cache is n_seq_max-sized, not n_ctx-sized
    return memory_breakdown();
}

llama_memory_context_ptr llama_memory_recurrent::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // TODO: non-sequential equal split can be done if using unified KV cache
                //       for simplicity, we always use sequential equal split for now
                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                ubatch = balloc.split_equal(n_ubatch, true, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        if (!prepare(ubatches)) {
            break;
        }

        return std::make_unique<llama_memory_recurrent_context>(this, std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_recurrent::init_full() {
    return std::make_unique<llama_memory_recurrent_context>(this);
}

llama_memory_context_ptr llama_memory_recurrent::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_recurrent::prepare(const std::vector<llama_ubatch> & ubatches) {
    // simply remember the full state because it is very small for this type of cache
    // TODO: optimize
    auto org_cells = cells;
    auto org_used = used;
    auto org_head = head;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        if (!find_slot(ubatch)) {
            success = false;
            break;
        }
    }

    // restore the original state
    cells = std::move(org_cells);
    used = org_used;
    head = org_head;

    return success;
}

bool llama_memory_recurrent::find_slot(const llama_ubatch & ubatch) {
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;
    const uint32_t n_seqs       = ubatch.n_seqs;

    // if we have enough unused cells before the current head ->
    //   better to start searching from the beginning of the cache, hoping to fill it
    if (head > used + 2*n_seqs) {
        head = 0;
    }

    // For recurrent state architectures (like Mamba or RWKV),
    // each cache cell can store the state for a whole sequence.
    // A slot should be always be contiguous.

    // can only process batches with an equal number of new tokens in each sequence
    GGML_ASSERT(ubatch.equal_seqs());

    int32_t min = size - 1;
    int32_t max = 0;

    // everything should fit if all seq_ids are smaller than the max
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens; // first token of sequence set s
        const uint32_t n_seq_id = ubatch.n_seq_id[i];

        for (uint32_t j = 0; j < n_seq_id; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];

            if (seq_id < 0 || (uint32_t) seq_id >= size) {
                // too big seq_id
                // TODO: would it be possible to resize the cache instead?
                LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%u Try using a bigger --parallel value\n", __func__, seq_id, n_seq_max);
                return false;
            }
            if (j > 0) {
                auto & seq = cells[seq_id];
                if (seq.tail >= 0) {
                    auto & cell = cells[seq.tail];
                    // clear cells from seq_ids that become shared
                    // (should not normally happen, but let's handle it anyway)
                    cell.seq_id.erase(seq_id);
                    seq.tail = -1;
                    if (cell.seq_id.empty()) {
                        cell.pos = -1;
                        cell.src = -1;
                        used -= 1;
                    }
                }
            }
        }
    }

#ifndef NDEBUG
    {
        std::vector<int32_t> tails_verif;
        tails_verif.assign(size, -1);
        for (uint32_t i = 0; i < size; ++i) {
            auto & cell = cells[i];
            for (llama_seq_id seq_id : cell.seq_id) {
                if (tails_verif[seq_id] != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tails_verif[seq_id]);
                }
                tails_verif[seq_id] = i;
            }
        }
        for (uint32_t i = 0; i < size; ++i) {
            if (tails_verif[i] != cells[i].tail) {
                LLAMA_LOG_ERROR("%s: wrong tail for seq_id %d, (%d instead of %d)\n", __func__, i, cells[i].tail, tails_verif[i]);
            }
        }
    }
#endif

    // find next empty cell
    uint32_t next_empty_cell = head;

    for (uint32_t i = 0; i < size; ++i) {
        if (next_empty_cell >= size) { next_empty_cell -= size; }
        auto & cell = cells[next_empty_cell];
        if (cell.is_empty()) { break; }
        next_empty_cell += 1;
    }

    // find usable cell range
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        auto & seq_meta = cells[seq_id];
        bool has_cell = false;
        if (seq_meta.tail >= 0) {
            auto & cell = cells[seq_meta.tail];
            GGML_ASSERT(cell.has_seq_id(seq_id));
            // does this seq_id "own" the cell?
            if (cell.seq_id.size() == 1) { has_cell = true; }
        }
        if (!has_cell) {
            auto & empty_cell = cells[next_empty_cell];
            GGML_ASSERT(empty_cell.is_empty());
            // copy old tail into the empty cell
            if (seq_meta.tail >= 0) {
                auto & orig_cell = cells[seq_meta.tail];
                empty_cell.pos = orig_cell.pos;
                empty_cell.src = seq_meta.tail;
                orig_cell.seq_id.erase(seq_id);
                if (orig_cell.is_empty()) {
                    orig_cell.pos = -1;
                    orig_cell.src = -1;
                    used -= 1;
                }
                empty_cell.seq_id.insert(seq_id);
            }
            seq_meta.tail = next_empty_cell;
            // find next empty cell
            if (s + 1 < n_seqs) {
                for (uint32_t j = 0; j < size; ++j) {
                    next_empty_cell += 1;
                    if (next_empty_cell >= size) { next_empty_cell -= size; }
                    auto & cell = cells[next_empty_cell];
                    if (cell.is_empty()) { break; }
                }
            }
        }
        if (min > seq_meta.tail) { min = seq_meta.tail; }
        if (max < seq_meta.tail) { max = seq_meta.tail; }
    }

    // gather and re-order
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const int32_t dst_id = s + min;
        const int32_t src_id = cells[ubatch.seq_id[i][0]].tail;
        if (dst_id != src_id) {
            auto & dst_cell = cells[dst_id];
            auto & src_cell = cells[src_id];

            std::swap(dst_cell.pos, src_cell.pos);
            std::swap(dst_cell.src, src_cell.src);
            std::swap(dst_cell.seq_id, src_cell.seq_id);

            // swap tails
            for (uint32_t j = 0; j < size; ++j) {
                int32_t & tail = cells[j].tail;
                if (tail == src_id) {
                    tail = dst_id;
                } else if (tail == dst_id) {
                    tail = src_id;
                }
            }
        }
    }

    // update the pos of the used seqs
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_pos last_pos = ubatch.pos[i + n_seq_tokens - 1];
        const int32_t cell_id = s + min;
        auto & cell = cells[cell_id];

        if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
            // What should happen when the pos backtracks or skips a value?
            // Clearing the state mid-batch would require special-casing which isn't done.
            LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n",
                __func__, last_pos, cell.pos, ubatch.seq_id[i][0], n_seq_tokens);
        }
        cell.pos = last_pos;
        cell.seq_id.clear();
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];
            cell.seq_id.insert(seq_id);
            cells[seq_id].tail = cell_id;
        }
    }

    // Find first cell without src refs, to use as the zero-ed state
    {
        // TODO: bake-in src refcounts in the cell metadata
        std::vector<int32_t> refcounts(size, 0);
        for (size_t i = 0; i < size; ++i) {
            const int32_t src = cells[i].src;
            if (src >= 0) {
                refcounts[src] += 1;
            }
        }

        rs_z = -1;
        for (int i = min; i <= max; ++i) {
            if (refcounts[i] == 0) {
                rs_z = i;
                break;
            }
        }

        for (int i = min; i <= max; ++i) {
            if (cells[i].src < 0) {
                GGML_ASSERT(rs_z >= 0);
                cells[i].src0 = rs_z;
            } else {
                // Stage the source ids for all used cells to allow correct seq_* behavior
                // and still make these values available when setting the inputs
                cells[i].src0 = cells[i].src;
            }
            cells[i].src = i; // avoid moving or clearing twice
        }
    }

    // allow getting the range of used cells, from head to head + n
    head = min;
    n    = max - min + 1;
    used = std::count_if(cells.begin(), cells.end(),
        [](const mem_cell & cell){ return !cell.is_empty(); });

    // sanity check
    return n >= n_seqs;
}

bool llama_memory_recurrent::get_can_shift() const {
    // shifting the pos is trivial for recurrent models
    return true;
}

size_t llama_memory_recurrent::total_size() const {
    size_t size = 0;
    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_memory_recurrent::size_r_bytes() const {
    size_t size_r_bytes = 0;

    for (const auto & r : r_l) {
        if (r != nullptr) {
            size_r_bytes += ggml_nbytes(r);
        }
    }

    return size_r_bytes;
}

size_t llama_memory_recurrent::size_s_bytes() const {
    size_t size_s_bytes = 0;

    for (const auto & s : s_l) {
        if (s != nullptr) {
            size_s_bytes += ggml_nbytes(s);
        }
    }

    return size_s_bytes;
}

size_t llama_memory_recurrent::size_p_bytes() const {
    size_t size_p_bytes = 0;

    for (const auto & p : p_l) {
        if (p != nullptr) {
            size_p_bytes += ggml_nbytes(p);
        }
    }

    return size_p_bytes;
}

void llama_memory_recurrent::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges_data; // logical source row ranges
    uint32_t cell_count = 0;

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t cell_range_begin = size;
    for (uint32_t i = 0; i < size; ++i) {
        const auto & cell = cells[i];
        // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
        if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
            ++cell_count;
            uint32_t rs_idx_cur = 0;

            if (n_rs_seq != 0) {
                if (seq_id != -1) {
                    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < rs_idx.size());
                    rs_idx_cur = rs_idx[seq_id];
                } else {
                    bool has_rs_idx = false;
                    for (const llama_seq_id cell_seq_id : cell.seq_id) {
                        GGML_ASSERT(cell_seq_id >= 0 && (size_t) cell_seq_id < rs_idx.size());

                        const uint32_t seq_rs_idx = rs_idx[cell_seq_id];
                        if (!has_rs_idx) {
                            rs_idx_cur = seq_rs_idx;
                            has_rs_idx = true;
                        } else if (rs_idx_cur != seq_rs_idx) {
                            GGML_ABORT("cannot write shared recurrent state with different rollback indices");
                        }
                    }
                }
            }

            const uint32_t cell_id = rs_idx_cur * size + (cell.src >= 0 ? cell.src : (int32_t) i);
            if (cell_ranges_data.empty() || cell_ranges_data.back().second != cell_id) {
                cell_ranges_data.emplace_back(cell_id, cell_id + 1);
            } else {
                cell_ranges_data.back().second++;
            }

            if (cell_range_begin == size) {
                cell_range_begin = i;
            }
        } else {
            if (cell_range_begin != size) {
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = size;
            }
        }
    }
    if (cell_range_begin != size) {
        cell_ranges.emplace_back(cell_range_begin, size);
    }

    if ((flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) && cell_ranges.size() > 1) {
        GGML_ABORT("cannot save/load multiple ranges of cells to/from device memory\n");
    }

    // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
    uint32_t cell_count_check = 0;
    for (const auto & range : cell_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    cell_count_check = 0;
    for (const auto & range : cell_ranges_data) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    io.write(&cell_count, sizeof(cell_count));

    state_write_meta(io, cell_ranges, seq_id);
    state_write_data(io, cell_ranges_data);
}

void llama_memory_recurrent::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t cell_count;
    io.read(&cell_count, sizeof(cell_count));

    bool res = true;

    res = res && state_read_meta(io, cell_count, seq_id);

    try {
        res = res && state_read_data(io, cell_count);
    } catch (...) {
        res = false;
    }

    if (!res) {
        // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
        if (seq_id == -1) {
            clear(true);
        } else {
            seq_rm(seq_id, -1, -1);
        }
        throw std::runtime_error("failed to restore kv cache");
    }

    if (seq_id == -1) {
        std::fill(rs_idx.begin(), rs_idx.end(), 0);
        std::fill(rollback_valid_depth.begin(), rollback_valid_depth.end(), 0);
        GGML_ASSERT(std::all_of(rollback_valid_depth.begin(), rollback_valid_depth.end(),
                    [](uint32_t depth) { return depth == 0; }));
    } else {
        reset_rollback_state(seq_id);
        GGML_ASSERT(rollback_valid_depth[seq_id] == 0);
    }
}

void llama_memory_recurrent::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const auto & cell = cells[i];
            const llama_pos pos      = cell.pos;
            const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id) {
                for (auto seq_id : cell.seq_id) {
                    io.write(&seq_id, sizeof(seq_id));
                }
            }
        }
    }
}

void llama_memory_recurrent::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const {
    const uint32_t s_trans = 0;
    const uint32_t n_layer = hparams.n_layer();

    io.write(&s_trans, sizeof(s_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the R tensors first, each row is a cell
    // Get whole range at a time
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
        if (r_l[il] == nullptr) continue;

        // Write R tensor type
        const int32_t r_type_i = (int32_t)r_l[il]->type;
        io.write(&r_type_i, sizeof(r_type_i));

        // Write row size of R tensor
        const uint64_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        io.write(&r_size_row, sizeof(r_size_row));

        // Write each logical cell row range. With pending recurrent rollback,
        // the logical current state may live in a rollback snapshot plane.
        for (const auto & range : cell_ranges) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * r_size_row;
            io.write_tensor(r_l[il], range.first * r_size_row, buf_size);
        }

        // the PLE conv history is a second recurrent row, so it has to travel with the first
        if (p_l[il] != nullptr) {
            const uint64_t p_size_row = ggml_row_size(p_l[il]->type, hparams.ple_conv_state());
            io.write(&p_size_row, sizeof(p_size_row));

            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                io.write_tensor(p_l[il], range.first * p_size_row, range_size * p_size_row);
            }
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write row size of S tensor
            const uint64_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            io.write(&s_size_row, sizeof(s_size_row));

            // Write each logical cell row range. With pending recurrent rollback,
            // the logical current state may live in a rollback snapshot plane.
            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * s_size_row;
                io.write_tensor(s_l[il], range.first * s_size_row, buf_size);
            }
        }
    } else {
        // When S tensor is transposed, we also need the element size and get the element ranges from each row
        const uint32_t mem_size = size;
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write element size
            const uint32_t s_size_el = ggml_type_size(s_l[il]->type);
            io.write(&s_size_el, sizeof(s_size_el));

            // Write GQA embedding size
            io.write(&n_embd_s, sizeof(n_embd_s));

            // For each row, we get the element values of each logical cell
            for (uint32_t j = 0; j < n_embd_s; ++j) {
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * mem_size) * s_size_el;
                    const size_t buf_size = range_size * s_size_el;
                    io.write_tensor(s_l[il], src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_memory_recurrent::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        if (cell_count == 0) {
            return true;
        }

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 0) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            ubatch.pos[i] = pos;
        }
        ubatch.n_seq_id[0] = 1;
        ubatch.seq_id[0] = &dest_seq_id;

        if (!find_slot(ubatch)) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        // DEBUG CHECK: kv.head should be our first cell, kv.head + cell_count - 1 should be our last cell (verify seq_id and pos values)
        // Assume that this is one contiguous block of cells
        GGML_ASSERT(head + cell_count <= size);
        GGML_ASSERT(cells[head].pos == ubatch.pos[0]);
        GGML_ASSERT(cells[head + cell_count - 1].pos == ubatch.pos[cell_count - 1]);
        GGML_ASSERT(cells[head].has_seq_id(dest_seq_id));
        GGML_ASSERT(cells[head + cell_count - 1].has_seq_id(dest_seq_id));
    } else {
        // whole KV cache restore

        if (cell_count > size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            auto & cell = cells[i];

            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cell.pos = pos;

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= this->n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, this->n_seq_max);
                    return false;
                }

                cell.seq_id.insert(seq_id);

                int32_t & tail = cells[seq_id].tail;
                if (tail != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tail);
                    return false;
                }
                tail = i;
            }
        }

        head = 0;
        used = cell_count;
    }

    for (uint32_t i = 0; i < cell_count; ++i) {
        uint32_t cell_id = head + i;
        // make sure the recurrent states will keep their restored state
        cells[cell_id].src = cell_id;
    }

    return true;
}

bool llama_memory_recurrent::state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    uint32_t s_trans;
    uint32_t n_layer;
    io.read(&s_trans, sizeof(s_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != hparams.n_layer()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer());
        return false;
    }
    if (cell_count > size) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, size);
        return false;
    }
    if (false != (bool) s_trans) {
        LLAMA_LOG_ERROR("%s: incompatible s transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers
        if (r_l[il] == nullptr) continue;

        // Read type of key
        int32_t r_type_i_ref;
        io.read(&r_type_i_ref, sizeof(r_type_i_ref));
        const int32_t r_type_i = (int32_t) r_l[il]->type;
        if (r_type_i != r_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r type (%d != %d, layer %d)\n", __func__, r_type_i, r_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t r_size_row_ref;
        io.read(&r_size_row_ref, sizeof(r_size_row_ref));
        const size_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        if (r_size_row != r_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r row size (%zu != %zu, layer %d)\n", __func__, r_size_row, (size_t) r_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // Read and set the keys for the whole cell range
            io.read_tensor(r_l[il], head * r_size_row, cell_count * r_size_row);
        }

        if (p_l[il] != nullptr) {
            uint64_t p_size_row_ref;
            io.read(&p_size_row_ref, sizeof(p_size_row_ref));
            const size_t p_size_row = ggml_row_size(p_l[il]->type, hparams.ple_conv_state());
            if (p_size_row != p_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched ple row size (%zu != %zu, layer %d)\n", __func__, p_size_row, (size_t) p_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                io.read_tensor(p_l[il], head * p_size_row, cell_count * p_size_row);
            }
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;

            if (s_type_i != s_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t s_size_row_ref;
            io.read(&s_size_row_ref, sizeof(s_size_row_ref));
            const size_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            if (s_size_row != s_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s row size (%zu != %zu, layer %d)\n", __func__, s_size_row, (size_t) s_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the values for the whole cell range
                io.read_tensor(s_l[il], head * s_size_row, cell_count * s_size_row);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            if (s_type_i != s_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t s_size_el_ref;
            io.read(&s_size_el_ref, sizeof(s_size_el_ref));
            const size_t s_size_el = ggml_type_size(s_l[il]->type);
            if (s_size_el != s_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s element size (%zu != %zu, layer %d)\n", __func__, s_size_el, (size_t) s_size_el_ref, il);
                return false;
            }

            // Read state embedding size
            uint32_t n_embd_s_ref;
            io.read(&n_embd_s_ref, sizeof(n_embd_s_ref));
            if (n_embd_s != n_embd_s_ref) {
                LLAMA_LOG_ERROR("%s: mismatched s embedding size (%u != %u, layer %d)\n", __func__, n_embd_s, n_embd_s_ref, il);
                return false;
            }

            if (cell_count) {
                // For each row in the transposed matrix, read the values for the whole cell range
                for (uint32_t j = 0; j < n_embd_s; ++j) {
                    const size_t dst_offset = (head + j * size) * s_size_el;
                    io.read_tensor(s_l[il], dst_offset, cell_count * s_size_el);
                }
            }
        }
    }

    return true;
}

//
// llama_memory_recurrent_context
//

llama_memory_recurrent_context::llama_memory_recurrent_context(llama_memory_status status) : status(status) {}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), is_full(true) {
}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), ubatches(std::move(ubatches)) {}

llama_memory_recurrent_context::~llama_memory_recurrent_context() = default;

bool llama_memory_recurrent_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    // graph_compute() succeeded for the current ubatch before the caller advances us
    mem->commit_rollback(ubatches[i_next]);

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_recurrent_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is an update
    if (ubatches.empty()) {
        // recurrent cache never performs updates
        assert(status == LLAMA_MEMORY_STATUS_NO_UPDATE);

        return true;
    }

    // A failed/aborted graph may have partially mutated the device planes. Invalidate
    // before submission, then publish the exact written depth only from next().
    mem->invalidate_rollback(ubatches[i_next]);
    mem->find_slot(ubatches[i_next]);

    return true;
}

llama_memory_status llama_memory_recurrent_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_recurrent_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

uint32_t llama_memory_recurrent_context::get_n_rs() const {
    return is_full ? mem->size : mem->n;
}

uint32_t llama_memory_recurrent_context::get_head() const {
    return is_full ? 0 : mem->head;
}

int32_t llama_memory_recurrent_context::get_rs_z() const {
    return is_full ? 0 : mem->rs_z;
}

uint64_t llama_memory_recurrent_context::get_tensor_binding_epoch() const {
    return mem->tensor_binding_epoch_;
}

uint32_t llama_memory_recurrent_context::get_size() const {
    return mem->size;
}

ggml_tensor * llama_memory_recurrent_context::get_r_l(int32_t il) const {
    return mem->r_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_s_l(int32_t il) const {
    return mem->s_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_p_l(int32_t il) const {
    return mem->p_l[il];
}

int32_t llama_memory_recurrent_context::s_copy(int i) const {
    const uint32_t cell_idx = i + mem->head;
    const int32_t  src0     = mem->cells[cell_idx].src0;

    if (mem->n_rs_seq == 0) {
        return src0;
    }

    uint32_t idx = 0;
    if (!mem->cells[cell_idx].seq_id.empty()) {
        const llama_seq_id seq = *mem->cells[cell_idx].seq_id.begin();
        if (seq >= 0 && (size_t) seq < mem->rs_idx.size()) {
            idx = mem->rs_idx[seq];
            mem->rs_idx[seq] = 0;
        }
    }
    return (int32_t)(idx * mem->size) + src0;
}

bool llama_memory_recurrent_context::states_are_contiguous_identity(uint32_t n_seqs) const {
    // Only the simple, common decode case is optimized: no full-defrag pass, no rollback
    // index remapping, and the gather must cover exactly the active cells (no extra states).
    if (is_full || mem->n_rs_seq != 0) {
        return false;
    }
    // rs_z >= 0 means a cell in range is a fresh/reset sequence that build_rs zeroes via
    // ggml_scale_inplace(state_zero, 0). A direct view would skip that zeroing and leak the
    // previous occupant's state into the new sequence, so fall back to the gather path.
    if (mem->rs_z != -1) {
        return false;
    }
    if (get_n_rs() != n_seqs) {
        return false;
    }
    const uint32_t head = mem->head;
    for (uint32_t i = 0; i < n_seqs; ++i) {
        if (mem->cells[head + i].src0 != (int32_t) (head + i)) {
            return false;
        }
    }
    return true;
}
