#include "spec-defaults.h"

#include "common.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "log.h"
#include "speculative.h"

#include "../src/llama-ext.h" // llama_model_load_metadata

#include <algorithm>
#include <string>

// one row per model family, keyed by general.architecture
static const struct {
    const char * arch;
    const char * drafter; // name used in the log line
} spec_family_defaults[] = {
    // Qwen3.8 (qwen35) ships a trained single MTP head: the production settings are the
    // current flag defaults (fixed draft depth 4, p-min 0), so the default only needs to
    // select the drafter
    { "qwen35", "MTP drafter" },
};

static const char * spec_family_drafter_name(const std::string & arch) {
    for (const auto & row : spec_family_defaults) {
        if (arch == row.arch) {
            return row.drafter;
        }
    }
    return nullptr;
}

// general.architecture and the number of MTP (nextn) layers of a GGUF model, header only
// (no tensor data). n_nextn is 0 when the MTP block is declared but its tensors are not in
// the model files (e.g. a head shipped as a separate sidecar), so the default never asks the
// loader for tensors that are not there
static bool spec_read_arch_nextn(const std::string & path, std::string & arch, uint32_t & n_nextn) {
    gguf_context_ptr ctx(llama_model_load_metadata(path.c_str()));
    if (!ctx) {
        return false;
    }

    const int64_t arch_id = gguf_find_key(ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return false;
    }

    arch = gguf_get_val_str(ctx.get(), arch_id);
    n_nextn = 0;

    const int64_t nextn_id = gguf_find_key(ctx.get(), (arch + ".nextn_predict_layers").c_str());
    if (nextn_id >= 0) {
        switch (gguf_get_kv_type(ctx.get(), nextn_id)) {
            case GGUF_TYPE_UINT32: n_nextn = gguf_get_val_u32(ctx.get(), nextn_id); break;
            case GGUF_TYPE_INT32:  n_nextn = (uint32_t) std::max(0, gguf_get_val_i32(ctx.get(), nextn_id)); break;
            default: break;
        }
    }
    if (n_nextn == 0) {
        return true;
    }

    // the MTP block is the last one: look for the eh_proj of that block
    // note: a split GGUF carries the full tensor metadata in every split, one file is enough
    uint32_t block_count = 0;
    const int64_t block_id = gguf_find_key(ctx.get(), (arch + ".block_count").c_str());
    if (block_id >= 0 && gguf_get_kv_type(ctx.get(), block_id) == GGUF_TYPE_UINT32) {
        block_count = gguf_get_val_u32(ctx.get(), block_id);
    }
    const std::string name = "blk." + std::to_string(block_count > 0 ? block_count - 1 : 0) + ".nextn.eh_proj.weight";
    if (gguf_find_tensor(ctx.get(), name.c_str()) < 0) {
        LOG_INF("speculative: %s declares %u MTP layer(s) but the model file holds no MTP tensors ('%s')\n",
                arch.c_str(), n_nextn, name.c_str());
        n_nextn = 0;
    }

    return true;
}

void common_speculative_apply_model_default(common_params & params) {
    auto & spec = params.speculative;

    // no I/O when the drafting was chosen explicitly (any --spec-type value, including none)
    // or when a draft model is already attached (the sidecar inference path owns that case)
    if (params.model.path.empty() || spec.spec_type_set || spec.has_dft()) {
        return;
    }
    if (spec.types != std::vector<enum common_speculative_type>{ COMMON_SPECULATIVE_TYPE_NONE }) {
        return;
    }

    std::string arch;
    uint32_t n_nextn = 0;
    if (!spec_read_arch_nextn(params.model.path, arch, n_nextn)) {
        return; // an unreadable model file is reported by the model load
    }
    if (n_nextn == 0 || spec_family_drafter_name(arch) == nullptr) {
        return;
    }

    // same list as --spec-type <type> gives: the handler appends to the default { none }
    spec.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);

    LOG_INF("speculative: %s on by default for %s (nextn=%u): draft-mtp, n-max %d; --spec-type none disables\n",
            spec_family_drafter_name(arch), arch.c_str(), n_nextn, spec.draft.n_max);
}
