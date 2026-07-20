#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <set>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Dynamic VBR (S3): degrade tier ladder + the measured price order (generated table).
enum vbr_tier : uint8_t {
    VBR_TIER_T8,
    VBR_TIER_T4,
    VBR_TIER_T3_TCQ,
    VBR_TIER_T2_TCQ,
    VBR_TIER_T1_TCQ,
    VBR_TIER_COUNT,
};

// a type the degrade ladder can move: the five turbo tiers plus F16, which is the dynamic
// entry tier (full-quality until budget pressure; the measured orders' first band is
// fp16->t8). Anything else living in a VMM pool — an explicitly non-vbr side of a mixed
// -ct config (q8_0, bf16) — is PINNED: the transcode dequant has no source support for it,
// so a step touching it must be skipped, never executed (it would GGML_ABORT mid-decode).
static bool vbr_type_is_movable(ggml_type t) {
    return t == GGML_TYPE_F16 ||
           t == GGML_TYPE_TURBO8_0 || t == GGML_TYPE_TURBO4_0 || t == GGML_TYPE_TURBO3_TCQ ||
           t == GGML_TYPE_TURBO2_TCQ || t == GGML_TYPE_TURBO1_TCQ;
}

static ggml_type vbr_tier_type(uint8_t tier) {
    switch (tier) {
        case VBR_TIER_T8:     return GGML_TYPE_TURBO8_0;
        case VBR_TIER_T4:     return GGML_TYPE_TURBO4_0;
        case VBR_TIER_T3_TCQ: return GGML_TYPE_TURBO3_TCQ;
        case VBR_TIER_T2_TCQ: return GGML_TYPE_TURBO2_TCQ;
        case VBR_TIER_T1_TCQ: return GGML_TYPE_TURBO1_TCQ;
        default:              GGML_ABORT("invalid vbr tier %d", (int) tier);
    }
}

#include "llama-vbr-degrade-orders.inc"  // arch-keyed registry (matrix v3, 2026-07-05)

// Turbo TCQ prompt cache safety: compute a fingerprint from the codebook env
// vars so that loading a cache created with a different codebook is detected.
// The fingerprint is a CRC32 of the codebook FILE CONTENTS (not the path),
// so the check is relocatable — only the actual data matters.
static uint32_t turbo_tcq_codebook_crc32(const char * path, size_t n_floats) {
    if (!path || !path[0]) {
        return 0; // no custom codebook → use compiled-in default → hash 0
    }
    uint32_t crc = 0xFFFFFFFF;
    FILE * f = fopen(path, "rb");
    if (!f) { return 0; }
    float buf[512];
    size_t n = fread(buf, sizeof(float), n_floats, f);
    fclose(f);
    if (n != n_floats) { return 0; }
    const uint8_t * data = (const uint8_t *)buf;
    for (size_t i = 0; i < n_floats * sizeof(float); i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
        }
    }
    return crc ^ 0xFFFFFFFF;
}

static uint32_t turbo_tcq_fingerprint(void) {
    const char * cb3 = getenv("TURBO_TCQ_CB");
    const char * cb2 = getenv("TURBO_TCQ_CB2");
    uint32_t h3 = turbo_tcq_codebook_crc32(cb3, 512);
    uint32_t h2 = turbo_tcq_codebook_crc32(cb2, 256);
    return h3 ^ (h2 * 0x9E3779B9); // mix both hashes
}

static bool ggml_type_is_turbo_tcq(enum ggml_type t) {
    return t == GGML_TYPE_TURBO3_TCQ || t == GGML_TYPE_TURBO2_TCQ || t == GGML_TYPE_TURBO1_TCQ;
}

static bool ggml_type_is_turbo(enum ggml_type t) {
    return ggml_is_turbo_kv_type(t);
}

// Resolve the backend's turbo/VBR vtable (ggml-vbr.h) for a KV buffer type. Returns nullptr
// when the owning backend does not export GGML_VBR_BACKEND_IFACE_PROC — i.e. it cannot host
// turbo-typed KV at all (CPU, or a GPU backend without the kernels). libllama never links
// backend symbols for this feature; everything goes through the resolved vtable.
static const ggml_vbr_backend_iface * llama_vbr_backend_iface_for_buft(ggml_backend_buffer_type_t buft) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return nullptr;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }
    const auto get = (ggml_backend_vbr_iface_fn_t) ggml_backend_reg_get_proc_address(reg, GGML_VBR_BACKEND_IFACE_PROC);
    return get != nullptr ? get() : nullptr;
}

// One VBR-capable device behind a KV buffer type: the backend vtable, the vtable's device
// ordinal, and the device's own (simple) buffer type.
struct llama_vbr_dev {
    const ggml_vbr_backend_iface * be     = nullptr;
    int                            device = -1;
    ggml_backend_buffer_type_t     buft   = nullptr;
};

// Resolve every device behind a KV buffer type. A plain device buft resolves to one entry; the
// meta (--split-mode tensor) buft resolves to one entry per simple device underneath — turbo/VBR
// support then means EVERY simple device exports the vtable. Returns empty when any device lacks
// support (same contract as a nullptr from llama_vbr_backend_iface_for_buft).
static std::vector<llama_vbr_dev> llama_vbr_backend_devs_for_buft(ggml_backend_buffer_type_t buft) {
    std::vector<llama_vbr_dev> ret;
    const bool   is_meta = ggml_backend_buft_is_meta(buft);
    const size_t n_devs  = is_meta ? ggml_backend_meta_buft_n_bufts(buft) : 1;
    for (size_t i = 0; i < n_devs; ++i) {
        llama_vbr_dev d;
        d.buft = is_meta ? ggml_backend_meta_buft_simple_buft(buft, i) : buft;
        d.be   = llama_vbr_backend_iface_for_buft(d.buft);
        if (d.be == nullptr) {
            return {};
        }
        for (int j = 0; j < d.be->get_device_count(); ++j) {
            if (d.be->buffer_type(j) == d.buft) {
                d.device = j;
                break;
            }
        }
        ret.push_back(d);
    }
    return ret;
}

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

static std::string turbo_vbr_trim(std::string s) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

static std::string turbo_vbr_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s;
}

static std::vector<std::string> turbo_vbr_split(const std::string & s, char delim) {
    std::vector<std::string> out;
    std::string item;
    std::istringstream iss(s);
    while (std::getline(iss, item, delim)) {
        out.push_back(turbo_vbr_trim(item));
    }
    return out;
}

static const char * turbo_vbr_getenv(const char * name) {
    const char * e = getenv(name);
    return e && e[0] ? e : nullptr;
}

static bool turbo_vbr_env_enabled(const char * name) {
    const char * e = turbo_vbr_getenv(name);
    return e && atoi(e) != 0;
}

static bool turbo_vbr_type_from_str(const std::string & raw, ggml_type & out) {
    std::string s = turbo_vbr_lower(turbo_vbr_trim(raw));
    if (s == "fp16" || s == "f16") {
        out = GGML_TYPE_F16;
        return true;
    } else if (s == "bf16") {
        out = GGML_TYPE_BF16;
        return true;
    } else if (s == "t8" || s == "turbo8" || s == "turbo8_0") {
        out = GGML_TYPE_TURBO8_0;
        return true;
    } else if (s == "t4" || s == "turbo4" || s == "turbo4_0") {
        out = GGML_TYPE_TURBO4_0;
        return true;
    } else if (s == "turbo3_0") {
        out = GGML_TYPE_TURBO3_0; // vanilla PolarQuant tier: explicit _0 spelling only
        return true;
    } else if (s == "turbo2_0") {
        out = GGML_TYPE_TURBO2_0;
        return true;
    } else if (s == "t3" || s == "turbo3" || s == "t3tcq" || s == "turbo3tcq" || s == "turbo3_tcq") {
        // bare tier aliases mean the TCQ codec, matching --vbr-floor and degrade-order files
        out = GGML_TYPE_TURBO3_TCQ;
        return true;
    } else if (s == "t2" || s == "turbo2" || s == "t2tcq" || s == "turbo2tcq" || s == "turbo2_tcq") {
        out = GGML_TYPE_TURBO2_TCQ;
        return true;
    } else if (s == "t1" || s == "turbo1" || s == "t1tcq" || s == "turbo1tcq" || s == "turbo1_tcq") {
        out = GGML_TYPE_TURBO1_TCQ;
        return true;
    }

    const ggml_type types[] = {
        GGML_TYPE_F16,
        GGML_TYPE_BF16,
        GGML_TYPE_Q8_0,
        GGML_TYPE_TURBO2_0,
        GGML_TYPE_TURBO3_0,
        GGML_TYPE_TURBO4_0,
        GGML_TYPE_TURBO8_0,
        GGML_TYPE_TURBO3_TCQ,
        GGML_TYPE_TURBO2_TCQ,
        GGML_TYPE_TURBO1_TCQ,
    };

    for (ggml_type t : types) {
        if (s == ggml_type_name(t)) {
            out = t;
            return true;
        }
    }
    return false;
}

struct turbo_vbr_layer_policy {
    bool enabled = false;
    bool has_turbo = false;
    bool has_turbo_k = false;
    bool has_turbo_v = false;
    int layer_side_bands = 0;
    int ignored_bands = 0;
    int segmented_row_bands = 0;
    int segmented_coord_bands = 0;
    int segmented_layer_specific_bands = 0;
    int segmented_k_bands = 0;
    int segmented_v_bands = 0;
    std::vector<ggml_type> k;
    std::vector<ggml_type> v;
};

static std::string turbo_vbr_segmented_reject_reason(const turbo_vbr_layer_policy & policy) {
    std::vector<std::string> reasons;
    if (policy.segmented_coord_bands > 0) {
        reasons.push_back(format("%d coord/head segmented bands", policy.segmented_coord_bands));
    }
    if (policy.segmented_v_bands > 0) {
        reasons.push_back(format("%d V-side segmented bands (positional/row VBR not supported)", policy.segmented_v_bands));
    }
    if (policy.segmented_k_bands > 0) {
        reasons.push_back(format("%d K-side segmented bands (positional/row VBR not supported)", policy.segmented_k_bands));
    }
    if (reasons.empty()) {
        return "schedule contains malformed or unsupported bands";
    }

    std::ostringstream ss;
    for (size_t i = 0; i < reasons.size(); ++i) {
        if (i > 0) {
            ss << "; ";
        }
        ss << reasons[i];
    }
    return ss.str();
}

static std::string turbo_vbr_read_schedule_env(const char * env, std::string & src) {
    src = "inline";
    if (!env || !env[0]) {
        return {};
    }

    std::string value = turbo_vbr_trim(env);
    std::string path;
    if (!value.empty() && value[0] == '@') {
        path = value.substr(1);
    } else if (value.find('=') == std::string::npos && value.find(';') == std::string::npos) {
        path = value;
    }

    if (!path.empty()) {
        std::ifstream f(path);
        if (!f) {
            LLAMA_LOG_WARN("llama_kv_cache: could not open VBR_LAYER_SCHEDULE file %s\n", path.c_str());
            return {};
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        src = path;
        return ss.str();
    }

    return value;
}

static bool turbo_vbr_layer_strict_enabled(void) {
    return turbo_vbr_env_enabled("VBR_LAYER_STRICT");
}

static int turbo_vbr_schedule_ctx(void) {
    const char * e = turbo_vbr_getenv("VBR_SCHEDULE_CTX");
    if (!e || !e[0]) {
        return 8192;
    }
    const int value = atoi(e);
    return value > 0 ? value : 8192;
}

static bool turbo_vbr_parse_int_range(const std::string & s, int & lo, int & hi) {
    const size_t dash = s.find('-');
    if (dash == std::string::npos) {
        return false;
    }
    try {
        lo = std::stoi(s.substr(0, dash));
        hi = std::stoi(s.substr(dash + 1));
    } catch (...) {
        return false;
    }
    return lo <= hi;
}

static turbo_vbr_layer_policy turbo_vbr_layer_policy_from_env(
        uint32_t  n_layer,
        ggml_type base_k,
        ggml_type base_v,
        uint32_t  kv_size) {
    turbo_vbr_layer_policy policy;
    policy.k.assign(n_layer, base_k);
    policy.v.assign(n_layer, base_v);

    const char * env = turbo_vbr_getenv("VBR_LAYER_SCHEDULE");
    std::string src;
    std::string schedule = turbo_vbr_read_schedule_env(env, src);
    if (schedule.empty()) {
        if (env && env[0] && turbo_vbr_layer_strict_enabled()) {
            throw std::runtime_error("VBR_LAYER_SCHEDULE is empty or could not be read while VBR_LAYER_STRICT=1 is set");
        }
        return policy;
    }

    policy.enabled = true;
    // schedule_ctx = the discovery window the schedule's row coordinates refer to (VBR_SCHEDULE_CTX,
    // default 8192). Only whole-window bands are supported — see covers_whole_active_cache below.
    const int schedule_ctx = turbo_vbr_schedule_ctx();
    GGML_UNUSED(kv_size);

    for (const std::string & item_raw : turbo_vbr_split(schedule, ';')) {
        if (item_raw.empty()) {
            continue;
        }
        if (item_raw.rfind("default=", 0) == 0) {
            ggml_type t;
            if (turbo_vbr_type_from_str(item_raw.substr(strlen("default=")), t)) {
                std::fill(policy.k.begin(), policy.k.end(), t);
                std::fill(policy.v.begin(), policy.v.end(), t);
            } else {
                policy.ignored_bands++;
                LLAMA_LOG_WARN("llama_kv_cache: ignoring unknown VBR default tier '%s'\n", item_raw.c_str());
            }
            continue;
        }
        if (item_raw.rfind("band=", 0) != 0) {
            policy.ignored_bands++;
            continue;
        }

        const std::string body = item_raw.substr(strlen("band="));
        const auto parts = turbo_vbr_split(body, ':');
        if (parts.size() < 2) {
            policy.ignored_bands++;
            continue;
        }

        int row0 = 0;
        int row1 = 0;
        ggml_type tier;
        if (!turbo_vbr_parse_int_range(parts[0], row0, row1) || !turbo_vbr_type_from_str(parts[1], tier)) {
            policy.ignored_bands++;
            continue;
        }

        int layer0 = 0;
        int layer1 = (int) n_layer - 1;
        bool saw_layer = false;
        bool apply_k = false;
        bool apply_v = false;
        bool has_coord = false;

        for (size_t i = 2; i < parts.size(); ++i) {
            const std::string p = turbo_vbr_lower(parts[i]);
            if (p == "k") {
                apply_k = true;
                continue;
            }
            if (p == "v") {
                apply_v = true;
                continue;
            }
            if (!p.empty() && p[0] == 'l') {
                int lo = 0;
                int hi = 0;
                if (turbo_vbr_parse_int_range(p.substr(1), lo, hi)) {
                    layer0 = lo;
                    layer1 = hi;
                    saw_layer = true;
                }
                continue;
            }
            if (!p.empty() && (p[0] == 'c' || p[0] == 'h')) {
                has_coord = true;
            }
        }

        if (!apply_k && !apply_v) {
            apply_k = true;
            apply_v = true;
        }

        // A band that covers the whole measured discovery window is a
        // layer-side rule for Stage 1 production, even when runtime kv_size is
        // larger. The discovery window defaults to 8k and can be overridden for
        // future non-8k schedule artifacts.
        const bool covers_whole_active_cache = row0 == 0 && row1 >= schedule_ctx;
        if (!covers_whole_active_cache || has_coord) {
            if (has_coord) {
                policy.segmented_coord_bands++;
            } else {
                policy.segmented_row_bands++; // apply_k/apply_v forced true above
            }
            if (saw_layer) {
                policy.segmented_layer_specific_bands++;
            }
            if (apply_k) {
                policy.segmented_k_bands++;
            }
            if (apply_v) {
                policy.segmented_v_bands++;
            }
            // Per-side static VBR allocator: only full-cache layer-side tiers are
            // supported. Partial-row / coordinate (Stage2A) bands are no longer
            // implemented — ignore them.
            policy.ignored_bands++;
            continue;
        }

        if (!saw_layer) {
            layer0 = 0;
            layer1 = (int) n_layer - 1;
        }
        layer0 = std::max(layer0, 0);
        layer1 = std::min(layer1, (int) n_layer - 1);
        if (layer0 > layer1) {
            policy.ignored_bands++;
            continue;
        }

        for (int il = layer0; il <= layer1; ++il) {
            if (apply_k) {
                policy.k[il] = tier;
                policy.layer_side_bands++;
            }
            if (apply_v) {
                policy.v[il] = tier;
                policy.layer_side_bands++;
            }
        }
    }

    for (uint32_t il = 0; il < n_layer; ++il) {
        policy.has_turbo_k = policy.has_turbo_k || ggml_type_is_turbo(policy.k[il]);
        policy.has_turbo_v = policy.has_turbo_v || ggml_type_is_turbo(policy.v[il]);
    }
    policy.has_turbo = policy.has_turbo_k || policy.has_turbo_v;


    LLAMA_LOG_INFO("llama_kv_cache: VBR layer schedule enabled from %s: schedule_ctx=%d, applied %d layer-side entries, ignored %d segmented bands, segmented_axes={row:%d, coord:%d, layer_specific:%d, k:%d, v:%d}\n",
            src.c_str(),
            schedule_ctx,
            policy.layer_side_bands,
            policy.ignored_bands,
            policy.segmented_row_bands,
            policy.segmented_coord_bands,
            policy.segmented_layer_specific_bands,
            policy.segmented_k_bands,
            policy.segmented_v_bands);

    return policy;
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

static ggml_tensor * ggml_mul_mat_aux(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * rot) {
    const auto n = rot->ne[0];

    ggml_tensor * res;

    res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    res = ggml_mul_mat   (ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

//
// llama_kv_cache
//

// fresh cells are fully sized here, in the same initializer that decides ownership — the ctor
// body must never resize v_cells: when the cache shares another cache's cells (mem_other,
// [TAG_KV_CACHE_SHARE_CELLS]) the vector aliases the SOURCE cache's live stream layout
static std::shared_ptr<llama_kv_cells_vec> kv_cells_make(uint32_t n_stream, uint32_t kv_size) {
    auto cells = std::make_shared<llama_kv_cells_vec>();
    cells->resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        (*cells)[s].resize(kv_size);
    }
    return cells;
}

// VMM pool wrapping a physical buffer, if any
template <typename POOLS>
static auto kv_vmm_pool_for(POOLS & pools, ggml_backend_buffer_t pb) -> decltype(&pools[0]) {
    for (auto & p : pools) {
        if (p.vmm != nullptr && p.buf == pb) {
            return &p;
        }
    }
    return nullptr;
}

// physical buffers behind one KV buffer: the per-device simple buffers underneath a meta
// buffer (-sm tensor), else the buffer itself — the boundary translation clear() and
// memory_breakdown() share
static std::vector<ggml_backend_buffer_t> kv_phys_buffers(ggml_backend_buffer_t buf) {
    std::vector<ggml_backend_buffer_t> phys;
    if (ggml_backend_buffer_is_meta(buf)) {
        const size_t n = ggml_backend_meta_buffer_n_bufs(buf);
        for (size_t i = 0; i < n; ++i) {
            phys.push_back(ggml_backend_meta_buffer_simple_buffer(buf, i));
        }
    } else {
        phys.push_back(buf);
    }
    return phys;
}

// fixed VA slot of a cache tensor: every extent reserves its F16-max footprint so tier flips
// never move data pointers — this expression is the single encoding of that sizing rule
static size_t vbr_slot_bytes(const ggml_tensor * t) {
    return (size_t) ggml_row_size(GGML_TYPE_F16, t->ne[0]) * t->ne[1] * t->ne[2];
}

// byte spans of one (pool, extent) unit at tier type_B: how much must stay resident (keep),
// how far the live watermark extends it (keep_live), and the page-padded map target (keep_pad).
// One computation shared by the promote hysteresis/map/transcode phases and the degrade path.
struct vbr_span {
    size_t slot, keep, keep_live, keep_pad;
};
static vbr_span vbr_span_of(const ggml_tensor * t, ggml_type type_B, int64_t n_cells,
                            uint32_t wm_next, size_t gran) {
    const size_t rB   = ggml_row_size(type_B, t->ne[0]);
    const size_t slot = vbr_slot_bytes(t);
    const size_t keep      = rB * (size_t) std::max<int64_t>(n_cells, 1);
    const size_t keep_live = std::min(slot, std::max(keep, rB * (size_t) wm_next));
    const size_t keep_pad  = std::min(slot, (size_t) GGML_PAD(keep_live, gran));
    return { slot, keep, keep_live, keep_pad };
}


llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share,
    const llama_memory_vbr_params & vbr) :
    model(model), hparams(hparams), vbr_params_(vbr), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : kv_cells_make(unified ? 1 : n_seq_max, kv_size)),
    v_cells(*v_cells_impl) {

    // A share-linked cache follows the OWNER's dynamic VBR: tier flips mutate the owner's
    // tensors in place (which this cache's layer entries alias) and graph reuse fences on
    // the delegated vbr_tier_epoch() (see llama-kv-cache.h). What a share-linked cache must
    // NOT do is arm its own controller on top — two controllers would double-manage the
    // same pool and the shared v_cells occupancy would confuse the second watermark.
    // llama_context disarms drafter-side VBR at creation (ctx_other), so hitting this is an
    // internal-API misuse, not a user configuration.
    if ((other || share) && (vbr_params_.dynamic || vbr_params_.budget_bytes)) {
        throw std::runtime_error("internal: share-linked KV cache must not arm its own VBR controller");
    }

    // shared cells view the source cache's K/V tensors, so the cell count
    // follows the source allocation: a fitted target can be smaller than the
    // draft default and oversized views would overflow the source tensors
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }
    }

    GGML_ASSERT(kv_size % n_pad == 0);

    const uint32_t n_layer = hparams.n_layer_all;

    turbo_vbr_layer_policy vbr_layer_policy =
        turbo_vbr_layer_policy_from_env(hparams.n_layer_all, type_k, type_v, kv_size);
    if (vbr_layer_policy.enabled && turbo_vbr_layer_strict_enabled() && vbr_layer_policy.ignored_bands > 0) {
        throw std::runtime_error(format(
                "VBR_LAYER_SCHEDULE contains unsupported segmented bands but VBR_LAYER_STRICT=1 was set: %s",
                turbo_vbr_segmented_reject_reason(vbr_layer_policy).c_str()));
    }

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type. Dynamic VBR counts as turbo-managed even when
    // the ENTRY types are f16 — later degrades flip tensors to turbo tiers, which need the
    // rotation matrices, the padded allocs and the VMM/extent machinery from the start.
    const bool is_turbo = ggml_type_is_turbo(type_k) || ggml_type_is_turbo(type_v) ||
                          vbr_layer_policy.has_turbo || vbr_params_.dynamic;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            const size_t n_turbo_extra = is_turbo ? 8 : 0; // rotation matrices + safety margin
            ggml_init_params params = {
                /*.mem_size   =*/ size_t((2u*(1 + n_stream)*n_layer + n_turbo_extra)*ggml_tensor_overhead()),
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

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    // fresh cells were sized by kv_cells_make in the initializer; shared cells keep the
    // source cache's layout — either way this cache's streams must fit inside the vector
    GGML_ASSERT(v_cells.size() >= n_stream);

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                // A tensor-split target's cache is head-sharded across devices in the meta
                // buffer — this drafter's scheduler has no meta backend, so adopting the tensor
                // would hard-abort later at sched reserve ("pre-allocated tensor (cache_k_lN)
                // in a buffer (Meta())"). Refuse here with an actionable message instead.
                if (layer_share.k->buffer != nullptr && ggml_backend_buffer_is_meta(layer_share.k->buffer)) {
                    throw std::runtime_error(
                        "shared-KV drafter cannot read a tensor-split target: the target's KV cache is "
                        "head-sharded across devices under --split-mode tensor — run the target with "
                        "--split-mode layer, or drop the drafter (native in-model MTP heads still work)");
                }

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                layers.push_back(layer_share);
                layers.back().il = il;

                continue;
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (!is_mla) {
            if (n_embd_head_v_all == 0) {
                n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
            } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
                n_embd_head_v_all = -1;
            }
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        const bool cpu_bound_kv = ggml_backend_buft_is_host(buft);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        const bool has_k = true;
        const bool has_v = !is_mla;

        // per-layer types: uniform (-ctk/-ctv) unless a VBR layer schedule overrides.
        // (the pre-VBR TURBO_LAYER_ADAPTIVE 18-mode experiment matrix was retired 2026-07-05 —
        // VBR_LAYER_SCHEDULE expresses all of it and more)
        ggml_type layer_type_k = type_k;
        ggml_type layer_type_v = type_v;
        {
            if (vbr_layer_policy.enabled) {
                layer_type_k = vbr_layer_policy.k[il];
                layer_type_v = vbr_layer_policy.v[il];
            }
            // Turbo types have no CPU vec_dot kernel; partial offload keeps GPU
            // VBR layers native and falls CPU-bound VBR layers back to q8_0.
            if (cpu_bound_kv && (ggml_type_is_turbo(layer_type_k) || ggml_type_is_turbo(layer_type_v))) {
                if (layer_type_k != GGML_TYPE_Q8_0) {
                    layer_type_k = GGML_TYPE_Q8_0;
                }
                if (layer_type_v != GGML_TYPE_Q8_0) {
                    layer_type_v = GGML_TYPE_Q8_0;
                }
                static bool warned = false;
                if (!warned) {
                    LLAMA_LOG_WARN("llama_kv_cache: turbo KV cache falling back to q8_0 for CPU-bound layers (partial offload)\n");
                    warned = true;
                }
            }
        }

        // Turbo FA vec kernel supports head_dim <= 512.
        // Fall back to f16 for layers with larger head dimensions.
        {
            const uint32_t head_k = hparams.n_embd_head_k(il);
            const uint32_t head_v = hparams.n_embd_head_v(il);
            if (ggml_type_is_turbo(layer_type_k) && head_k > 512) {
                layer_type_k = GGML_TYPE_F16;
                static bool logged_k = false;
                if (!logged_k) {
                    LLAMA_LOG_WARN("llama_kv_cache: layer %d head_dim_k=%u > 512, falling back to f16 K (turbo FA limit)\n", il, head_k);
                    logged_k = true;
                }
            }
            if (ggml_type_is_turbo(layer_type_v) && head_v > 512) {
                layer_type_v = GGML_TYPE_F16;
                static bool logged_v = false;
                if (!logged_v) {
                    LLAMA_LOG_WARN("llama_kv_cache: layer %d head_dim_v=%u > 512, falling back to f16 V (turbo FA limit)\n", il, head_v);
                    logged_v = true;
                }
            }
        }

        // Turbo head padding: FWHT requires head_dim % 128 == 0
        // Pad per-head to nearest 128 with zeros (contribute nothing via Parseval's theorem)
        uint32_t n_embd_k_alloc = n_embd_k_gqa;
        uint32_t n_embd_v_alloc = n_embd_v_gqa;
        {
            if (ggml_type_is_turbo(layer_type_k) || vbr_params_.dynamic) {
                uint32_t head_k = hparams.n_embd_head_k(il);
                uint32_t padded = ((head_k + 127) / 128) * 128;
                if (padded > head_k) {
                    n_embd_k_alloc = padded * hparams.n_head_kv(il);
                }
            }
            if ((ggml_type_is_turbo(layer_type_v) || vbr_params_.dynamic) && !v_trans) {
                uint32_t head_v = hparams.n_embd_head_v(il);
                uint32_t padded = ((head_v + 127) / 128) * 128;
                if (padded > head_v) {
                    n_embd_v_alloc = padded * hparams.n_head_kv(il);
                }
            }
            if (n_embd_k_alloc != n_embd_k_gqa || n_embd_v_alloc != n_embd_v_gqa) {
                static bool logged = false;
                if (!logged) {
                    LLAMA_LOG_INFO("llama_kv_cache: turbo head padding: %u -> %u per head\n",
                        hparams.n_embd_head_k(il), ((hparams.n_embd_head_k(il) + 127) / 128) * 128);
                    logged = true;
                }
            }
        }

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, layer_type_k, n_embd_k_alloc, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, layer_type_v, n_embd_v_alloc, kv_size, n_stream) : nullptr;

        has_k && ggml_format_name(k, "cache_k_l%d", il);
        has_v && ggml_format_name(v, "cache_v_l%d", il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_stream, v_stream });

        // TurboQuant: create rotation matrix tensors (once, shared across layers)
        if (turbo_rotation == nullptr &&
            (ggml_type_is_turbo(type_k) || ggml_type_is_turbo(type_v) || vbr_layer_policy.has_turbo ||
             vbr_params_.dynamic)) {
            turbo_rotation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128);
            ggml_format_name(turbo_rotation, "turbo_rotation");  // R (forward)
            turbo_rotation_inv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128);
            ggml_format_name(turbo_rotation_inv, "turbo_rotation_inv");  // R
        }
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // Dynamic VBR S2 (option C): back the KV context with a VMM pool — one VA reservation, each
    // (layer,side) tensor at a fixed page-aligned offset sized for the MAX tier (F16 x kv_size),
    // physical pages mapped on demand as occupancy grows (vbr_vmm_ensure_mapped). Tier degrades
    // then shrink a tensor in place and unmap its tail; freed pages are fungible across tensors,
    // so nothing ever relocates. Driven by cparams.vbr_dynamic (threaded through create_memory);
    // VBR_VMM / VBR_MODE env remain developer overrides in BOTH directions (VBR_VMM=0 forces off).
    bool vbr_dynamic_wanted = vbr_params_.dynamic;
    if (const char * e = getenv("VBR_VMM")) {
        vbr_dynamic_wanted = atoi(e) != 0;
    } else if (const char * m = getenv("VBR_MODE")) {
        vbr_dynamic_wanted = strcmp(m, "dynamic") == 0;
    }
    const bool vbr_vmm_wanted = vbr_dynamic_wanted && !hparams.no_alloc &&
                                (vbr_layer_policy.enabled || is_turbo) && n_stream == 1 && !v_trans;
    if (vbr_dynamic_wanted && !vbr_vmm_wanted && !hparams.no_alloc) {
        // fail loud, not silent: the caller asked for the degrade controller and would otherwise
        // get a static max-tier cache while the logs still advertise dynamic VBR
        LLAMA_LOG_WARN("%s: dynamic VBR requested but the controller cannot arm (%s) — this cache "
                "stays static at its entry tiers\n", __func__,
                !(vbr_layer_policy.enabled || is_turbo) ? "KV is not turbo-typed" :
                n_stream != 1 ? "KV is split per sequence (n_stream > 1; run with --kv-unified)" :
                "the V cache is transposed (flash attention is off)");
    }
    LLAMA_LOG_DEBUG("%s: VBR_VMM gate: dynamic=%d no_alloc=%d policy=%d is_turbo=%d n_stream=%u v_trans=%d -> wanted=%d\n",
            __func__, (int) vbr_dynamic_wanted, (int) hparams.no_alloc, (int) vbr_layer_policy.enabled,
            (int) is_turbo, n_stream, (int) v_trans, (int) vbr_vmm_wanted);

    auto try_vmm_alloc = [&](ggml_context * c, ggml_backend_buffer_type_t bft) -> ggml_backend_buffer_t {
        const std::vector<llama_vbr_dev> devs = llama_vbr_backend_devs_for_buft(bft);
        if (devs.empty()) {
            // Falling back silently would be a trap: the fit pass priced this KV at the floor
            // tier (1.25 bits/value) but a static fallback stays at the f16 entry tier — 12.8x
            // the budgeted VRAM with no degrade possible, an OOM at depth on any fitted config.
            throw std::runtime_error(format(
                    "dynamic VBR (-ctk vbr) requires per-device KV buffers with turbo/VBR backend "
                    "support, but the KV buffer type is %s (or a device underneath it) without "
                    "that support — offload the KV cache to supported GPUs or use a static KV "
                    "type (f16/q8_0).",
                    ggml_backend_buft_name(bft)));
        }
        for (const auto & d : devs) {
            if (d.device < 0 || !d.be->vmm_available(d.device)) {
                LLAMA_LOG_WARN("%s: VBR_VMM requested but %s — falling back to static allocation\n", __func__,
                        d.device < 0 ? "the KV buffer type is not a device-default buffer" : "a device lacks VMM support");
                return nullptr;
            }
        }

        // lay out + place ONE device's tensors into a fresh VMM pool. `cc` holds either the KV
        // context itself (plain device buft) or one device's shard tensors (meta buft under
        // -sm tensor) — the walk is identical: every non-view tensor gets a page-aligned VA
        // slot; cache tensors are sized for the max tier so a later tier change never moves them.
        auto vmm_alloc_ctx = [&](const llama_vbr_dev & d, ggml_context * cc) -> ggml_backend_buffer_t {
            const ggml_vbr_backend_iface * be = d.be;
            const int device = d.device;
            const size_t gran = be->vmm_granularity(device);

            std::vector<std::pair<ggml_tensor *, size_t>> places;
            size_t off = 0;
            for (ggml_tensor * t = ggml_get_first_tensor(cc); t != nullptr; t = ggml_get_next_tensor(cc, t)) {
                if (t->view_src != nullptr) {
                    continue;
                }
                const bool is_cache = strncmp(t->name, "cache_", 6) == 0;
                const size_t slot = is_cache ? vbr_slot_bytes(t) : ggml_nbytes(t);
                // slot sizing assumes F16 is the widest tier any cache tensor can hold
                GGML_ASSERT(!is_cache || ggml_row_size(t->type, t->ne[0]) <= ggml_row_size(GGML_TYPE_F16, t->ne[0]));
                off = GGML_PAD(off, gran);
                places.push_back({ t, off });
                off += slot;
            }
            const size_t va_size = GGML_PAD(off, gran);

            ggml_vbr_vmm_pool * pool = be->vmm_pool_init(device, va_size);
            if (pool == nullptr) {
                LLAMA_LOG_WARN("%s: VBR_VMM: VA reservation of %.2f MiB failed — falling back\n", __func__, va_size/1024.0/1024.0);
                return nullptr;
            }
            char * base = (char *) be->vmm_pool_base(pool);
            ggml_backend_buffer_t b = be->buffer_from_ptr(device, base, va_size);
            if (b == nullptr) {
                be->vmm_pool_free(pool);
                return nullptr;
            }
            for (auto & [t, o] : places) {
                t->buffer = b;
                t->data   = base + o;
                // non-cache tensors (rotation matrices) are small model constants: map them up front
                if (strncmp(t->name, "cache_", 6) != 0 &&
                    !be->vmm_pool_map(pool, o, ggml_nbytes(t))) {
                    ggml_backend_buffer_free(b);
                    be->vmm_pool_free(pool);
                    return nullptr;
                }
            }
            // views: needed on the plain-buft direct call; the meta _ext path would also
            // finalize them (it skips views whose buffer is already set)
            for (ggml_tensor * t = ggml_get_first_tensor(cc); t != nullptr; t = ggml_get_next_tensor(cc, t)) {
                if (t->view_src != nullptr) {
                    t->buffer = b;
                    t->data   = (char *) t->view_src->data + t->view_offs;
                }
            }
            vbr_pool p;
            p.buf         = b;
            p.base        = base;
            p.size        = va_size;
            p.be          = be;
            p.vmm         = pool;
            p.device      = device;
            p.gran        = gran;
            p.mapped_base = be->vmm_pool_mapped(pool);
            vbr_pools_.push_back(std::move(p));
            LLAMA_LOG_INFO("%s: VBR VMM pool #%zu: %.2f MiB VA reserved (device %d, %zu KiB pages), %.2f MiB mapped up front\n",
                    __func__, vbr_pools_.size() - 1, va_size/1024.0/1024.0, device, gran/1024,
                    vbr_pools_.back().mapped_base/1024.0/1024.0);
            return b;
        };

        if (!ggml_backend_buft_is_meta(bft)) {
            return vmm_alloc_ctx(devs[0], c);
        }

        // -sm tensor: the meta backend shards every KV tensor per device (axis-0, head-aligned —
        // see llama_meta_device_get_split_state); allocate one VMM pool per simple device and
        // hand each device's shard context to the same layout routine. The meta buffer wraps the
        // per-device pool buffers so graph building sees ordinary meta tensors.
        const size_t pools_before = vbr_pools_.size();
        // std::function bridges the capturing lambda across the C callback (alive only for this call)
        const std::function<ggml_backend_buffer_t(size_t, ggml_context *)> alloc_one =
            [&](size_t i, ggml_context * sctx) { return vmm_alloc_ctx(devs[i], sctx); };
        ggml_backend_buffer_t buf = ggml_backend_meta_alloc_ctx_tensors_from_buft_ext(c, bft,
            [](size_t i, ggml_backend_buffer_type_t /*simple_buft*/, ggml_context * sctx, void * u) -> ggml_backend_buffer_t {
                return (*(const std::function<ggml_backend_buffer_t(size_t, ggml_context *)> *) u)(i, sctx);
            }, (void *) &alloc_one);
        if (buf == nullptr) {
            // unwind pools created for devices that DID succeed: their buffers were already freed
            // by the meta teardown; the VA reservations are ours to release
            while (vbr_pools_.size() > pools_before) {
                auto & p = vbr_pools_.back();
                if (p.vmm != nullptr) {
                    p.be->vmm_pool_free(p.vmm);
                }
                vbr_pools_.pop_back();
            }
        }
        return buf;
    };

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        // Turbo-typed KV requires a backend exporting the VBR interface (ggml-vbr.h): the codecs
        // have no CPU decode path (CPU set_rows would call a null from_float; CPU attention can't
        // read turbo bits). Refuse at init instead of crashing at the first decode. no_alloc
        // (externally managed KV) is exempt. Under --split-mode tensor the buffer type is the
        // meta buft: turbo KV is fine there as long as every device underneath supports it.
        if (!hparams.no_alloc && llama_vbr_backend_devs_for_buft(buft).empty()) {
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                if (ggml_is_turbo_kv_type(t->type)) {
                    LLAMA_LOG_ERROR("%s: KV cache type %s (tensor %s) needs a backend with TurboQuant support "
                            "(currently: CUDA), but its KV buffer type is %s — offload the KV cache to a "
                            "supported GPU (-ngl on all layers, without --no-kv-offload) or use a standard "
                            "cache type (f16/q8_0)\n",
                            __func__, ggml_type_name(t->type), t->name, ggml_backend_buft_name(buft));
                    throw std::runtime_error("turbo KV cache type on a backend without TurboQuant support");
                }
            }
        }
        ggml_backend_buffer_t buf = nullptr;
        bool is_vmm_buf = false;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else if (vbr_vmm_wanted) {
            // one VMM pool per KV buffer — one per device shard under -sm tensor, else one per
            // device KV context under -sm layer
            buf = try_vmm_alloc(ctx.get(), buft); // nullptr -> fall through to static allocation
            // NOTE: under -sm tensor `buf` is the META buffer while the pools hold the per-device
            // buffers — the flag must come from the allocation path, not a pool.buf match
            is_vmm_buf = buf != nullptr;
        }
        if (buf == nullptr && !hparams.no_alloc) {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB%s\n", __func__, ggml_backend_buffer_name(buf),
                ggml_backend_buffer_get_size(buf)/1024.0/1024.0, is_vmm_buf ? " (VA; physical maps on demand)" : "");

        if (!is_vmm_buf) {
            ggml_backend_buffer_clear(buf, 0); // VMM pages are zeroed at map time instead
        }

        // Fill turbo rotation matrices AFTER buffer clear (clear zeroes everything)
        if (turbo_rotation != nullptr && turbo_rotation->buffer != nullptr && !model.hparams.no_alloc) {
            #include "turbo-rotation-data.h"
            // turbo_rotation holds R (Q forward rotation), turbo_rotation_inv holds R^T (V output
            // un-rotation). The arrays are row-major; through ggml's column-major view plus
            // ggml_mul_mat's transpose, mul_mat(A, x) computes A @ x for a row-major-stored A
            // (verified by test) — so each tensor is stored exactly as named.
            ggml_backend_tensor_set(turbo_rotation, TURBO_ROTATION_R, 0, 128 * 128 * sizeof(float));
            ggml_backend_tensor_set(turbo_rotation_inv, TURBO_ROTATION_RT, 0, 128 * 128 * sizeof(float));
            LLAMA_LOG_INFO("%s: TurboQuant rotation matrices initialized (128x128)\n", __func__);
        }
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    // Dynamic VBR (M2): record per-(layer,side) descriptors over the just-placed KV tensors.
    // Pure bookkeeping — allocation behavior above is unchanged (bit-identical decode). M3 uses
    // these to transcode a tensor down a tier in place and return the freed bytes to the pool.
    if ((vbr_layer_policy.enabled || is_turbo) && !hparams.no_alloc) {
        // Per-pool tensor instances: a tensor placed in a plain device buffer is its own (single)
        // instance; a tensor behind the meta buffer (-sm tensor) has one shard instance per simple
        // device. Instances with no bytes (zero-width shard) are skipped.
        auto tensor_instances = [&](ggml_tensor * t) -> std::vector<ggml_tensor *> {
            std::vector<ggml_tensor *> out;
            if (t == nullptr || t->buffer == nullptr) {
                return out;
            }
            if (ggml_backend_buffer_is_meta(t->buffer)) {
                const size_t n = ggml_backend_meta_buffer_n_bufs(t->buffer);
                for (size_t i = 0; i < n; ++i) {
                    ggml_tensor * shard = ggml_backend_meta_buffer_simple_tensor(t, i);
                    if (shard != nullptr && shard->data != nullptr && ggml_nbytes(shard) > 0) {
                        out.push_back(shard);
                    }
                }
            } else if (t->data != nullptr) {
                out.push_back(t);
            }
            return out;
        };
        // one pool per KV-hosting buffer (per device under -sm layer / -sm tensor). VMM pools were
        // created at allocation time; any buffer without one (static allocation) gets a
        // bookkeeping-only pool.
        for (const auto & L : layers) {
            for (ggml_tensor * t : { L.k, L.v }) {
                for (ggml_tensor * inst : tensor_instances(t)) {
                    if (vbr_pool_of(inst) == nullptr) {
                        vbr_pool p;
                        p.buf  = inst->buffer;
                        p.base = (char *) ggml_backend_buffer_get_base(inst->buffer);
                        p.size = ggml_backend_buffer_get_size(inst->buffer);
                        vbr_pools_.push_back(std::move(p));
                    }
                }
            }
        }
        for (auto & p : vbr_pools_) {
            p.k.assign(layers.size(), {});
            p.v.assign(layers.size(), {});
        }
        auto record = [&](ggml_tensor * t, size_t ikv, bool is_v) {
            for (ggml_tensor * inst : tensor_instances(t)) {
                vbr_pool * p = vbr_pool_of(inst);
                if (p == nullptr) {
                    continue;
                }
                vbr_extent & e = is_v ? p->v[ikv] : p->k[ikv];
                e.t        = inst;
                e.byte_off = (size_t)((char *) inst->data - p->base);
                e.type0    = inst->type; // entry tier — the full-clear reset target
                p->used = std::max(p->used, e.byte_off + ggml_nbytes(inst));
            }
        };
        for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
            record(layers[ikv].k, ikv, false);
            record(layers[ikv].v, ikv, true);
        }
        // Rotation matrices are model constants placed in a KV buffer; include them in the owning
        // pool's high-water so the free region [used, size) never overlaps them (M3/M4 reuse safety).
        for (ggml_tensor * rt : { turbo_rotation, turbo_rotation_inv }) {
            for (ggml_tensor * inst : tensor_instances(rt)) {
                vbr_pool * p = vbr_pool_of(inst);
                if (p != nullptr) {
                    p->used = std::max(p->used, (size_t) ((char *) inst->data - p->base) + ggml_nbytes(inst));
                }
            }
        }
        for (size_t pi = 0; pi < vbr_pools_.size(); ++pi) {
            auto & p = vbr_pools_[pi];
            LLAMA_LOG_INFO("%s: VBR pool #%zu (device %d): %.2f MiB buffer, %.2f MiB used\n",
                    __func__, pi, p.device, p.size/1024.0/1024.0, p.used/1024.0/1024.0);
        }

        // (pool, extent) unit table: which VMM pools hold each (ikv, side) unit is fixed from
        // here on — precompute so the per-boundary degrade/promote walks never allocate. MUST
        // precede vbr_floor_clamp_order below (it consults vbr_unit_pooled), and vbr_pools_
        // must never grow again (the table holds pointers into it).
        vbr_units_tab_.resize(layers.size() * 2);
        for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
            for (int side = 0; side < 2; ++side) {
                auto & slot = vbr_units_tab_[ikv * 2 + side];
                for (auto & p : vbr_pools_) {
                    if (p.vmm == nullptr) {
                        continue;
                    }
                    vbr_extent & e = side ? p.v[ikv] : p.k[ikv];
                    if (e.t != nullptr) {
                        slot.push_back({ &p, &e });
                    }
                }
            }
        }
        // S3/S4: arm the decode-time degrade controller (VMM mode only). Inputs come from
        // cparams (llama_memory_vbr_params, threaded through create_memory): budget_bytes is
        // either the explicit --vbr-vram value or the fit pass's auto budget; min_bits is the
        // --vbr-floor aggregate clamp (see vbr_floor_clamp_order). VBR_BUDGET_MIB / VBR_MIN_BITS
        // env remain developer overrides; VBR_STASH_ROWS / VBR_DEGRADE_ORDER are direct
        // experiment overrides.
        if (vbr_vmm_active()) {
            vbr_load_degrade_order();
            vbr_floor_clamp_order();
            vbr_budget_bytes_    = (size_t) vbr_params_.budget_bytes;
            vbr_budget_explicit_ = vbr_params_.budget_explicit;
            if (const char * env = getenv("VBR_BUDGET_MIB")) {
                vbr_budget_bytes_    = (size_t) strtoull(env, nullptr, 10) * 1024 * 1024;
                vbr_budget_explicit_ = true; // forced-budget instrumentation must never grow
            }
            // growth headroom: env override > fit target (threaded) > 1 GiB default
            vbr_growth_headroom_ = (size_t) vbr_params_.growth_headroom_bytes;
            if (const char * env = getenv("VBR_GROWTH_HEADROOM_MIB")) {
                vbr_growth_headroom_ = (size_t) strtoull(env, nullptr, 10) * 1024 * 1024;
            }
            if (vbr_growth_headroom_ == 0) {
                vbr_growth_headroom_ = 1024ull * 1024 * 1024;
            }
            const bool budget_fit_armed = vbr_budget_bytes_ > 0;
            if (budget_fit_armed) {
                LLAMA_LOG_INFO("%s: VBR budget: %.2f MiB mapped-physical (degrade trigger armed)\n",
                        __func__, vbr_budget_bytes_/1024.0/1024.0);
            }
            // split the global budget across the VMM pools proportional to each pool's VA-size
            // share (single pool -> exact global budget); all mapped-bytes checks are per-pool.
            // WITHOUT a fit-resolved budget (fit disabled, failed, or not implemented —
            // SPLIT_MODE_TENSOR), derive each pool's budget HERE from live per-device free
            // memory, the same formula the boundary re-derivation uses. The ctor runs before
            // compute buffers allocate, so the number over-states reach; that optimism is
            // bounded by vbr_budget_eff's live free-VRAM clamp on every decision and corrected
            // by the periodic re-derivation. The re-derivation FLOOR (budget_base) is the pool's
            // floor-layout share — the minimum that guarantees the advertised context — so the
            // derived value can tighten back down under co-tenants, never below the guarantee.
            {
                size_t total_va = 0;
                size_t n_vmm    = 0;
                for (const auto & p : vbr_pools_) {
                    total_va += p.vmm != nullptr ? p.size : 0;
                    n_vmm    += p.vmm != nullptr;
                }
                size_t derived_total = 0;
                for (size_t pi = 0; pi < vbr_pools_.size(); ++pi) {
                    auto & p = vbr_pools_[pi];
                    if (p.vmm == nullptr || total_va == 0) {
                        continue;
                    }
                    // exact for the single-pool (single-GPU) case; double is plenty for the
                    // multi-pool proportional split (checks are page-granular anyway)
                    const auto share_of = [&](size_t total) {
                        return n_vmm == 1 ? total : (size_t) ((double) total * ((double) p.size / (double) total_va));
                    };
                    if (budget_fit_armed) {
                        p.budget      = share_of(vbr_budget_bytes_);
                        p.budget_base = p.budget; // re-derivation floor: never below the armed value
                    } else {
                        const size_t floor_share = share_of(vbr_floor_cost_bytes_);
                        p.budget      = std::max(vbr_pool_reach(p), floor_share);
                        p.budget_base = floor_share;
                        derived_total += p.budget;
                    }
                    if (n_vmm > 1 || !budget_fit_armed) {
                        LLAMA_LOG_INFO("%s: VBR pool #%zu (device %d) budget: %.2f MiB%s\n",
                                __func__, pi, p.device, p.budget/1024.0/1024.0,
                                budget_fit_armed ? "" : " (auto, from live free device memory)");
                    }
                }
                if (!budget_fit_armed) {
                    vbr_budget_bytes_ = std::max(derived_total, vbr_floor_cost_bytes_);
                    LLAMA_LOG_INFO("%s: VBR budget: %.2f MiB mapped-physical (auto: live free device "
                            "memory at init, floored at the %.2f MiB floor-layout cost; re-derived "
                            "each boundary)\n", __func__, vbr_budget_bytes_/1024.0/1024.0,
                            vbr_floor_cost_bytes_/1024.0/1024.0);
                }
            }
            // f16 sink-stash: DEFAULT ON (128 rows) since the S6 long-decode gate (2026-07-03)
            // — erases sink-row requant accumulation across any hop count for ~8 MiB + µs per
            // degrade. VBR_STASH_ROWS overrides (0 disables).
            const char * stash_env = getenv("VBR_STASH_ROWS");
            vbr_stash_rows_ = stash_env ? (uint32_t) atoi(stash_env) : 128;
            if (vbr_stash_rows_ > 0) {
                LLAMA_LOG_INFO("%s: VBR f16 sink-stash: %u rows per (layer,side)\n",
                        __func__, vbr_stash_rows_);
            }
        }
    }


    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
        if (vbr_layer_policy.enabled) {
            auto type_histogram = [&](bool want_k) {
                std::map<ggml_type, int> counts;
                for (const auto & layer : layers) {
                    ggml_tensor * t = want_k ? layer.k : layer.v;
                    if (t) {
                        counts[t->type]++;
                    }
                }

                std::ostringstream ss;
                bool first = true;
                for (const auto & it : counts) {
                    if (!first) {
                        ss << ", ";
                    }
                    first = false;
                    ss << ggml_type_name(it.first) << ":" << it.second;
                }
                return ss.str();
            };
            LLAMA_LOG_INFO("%s: VBR actual layer types: K {%s}, V {%s}\n", __func__,
                    type_histogram(true).c_str(),
                    type_histogram(false).c_str());
        }
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        attn_rot_k = other->attn_rot_k;
        attn_rot_v = other->attn_rot_v;
    } else {
        const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\n", __func__);
        }

        // turbo types have their own FWHT rotation — skip upstream Hadamard rotation
        const bool is_turbo_k = ggml_type_is_turbo(type_k) || vbr_layer_policy.has_turbo_k;
        const bool is_turbo_v = ggml_type_is_turbo(type_v) || vbr_layer_policy.has_turbo_v;

        attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) && !is_turbo_k &&
            hparams.n_embd_head_k() % 64 == 0;

        // always create Hadamard rotation tensors for DeepSeek lightning indexers
        if ((model.arch == LLM_ARCH_DEEPSEEK32 || model.arch == LLM_ARCH_DEEPSEEK4) &&
                hparams.n_embd_head_k_full == hparams.indexer_head_size) {
            attn_rot_k = true;
        }

        attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) && !is_turbo_v &&
            hparams.n_embd_head_v() % 64 == 0;
    }

    LLAMA_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (attn_rot_k || attn_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

llama_kv_cache::~llama_kv_cache() {
    for (auto & p : vbr_pools_) {
        if (p.backend != nullptr) {
            // S5: a degrade wave may still be in flight on the side stream — it must finish before
            // the stash buffer / VMM VA it touches are torn down. The queued tail unmaps are moot
            // here (vmm_pool_free unmaps every chunk).
            ggml_backend_synchronize(p.backend);
            p.unmap_deferred.clear();
        }
        if (p.stash_buf != nullptr) {
            ggml_backend_buffer_free(p.stash_buf);
            p.stash_buf = nullptr;
        }
        if (p.backend != nullptr) {
            ggml_backend_free(p.backend);
            p.backend = nullptr;
        }
    }
    // free the VMM pools AFTER nothing can touch KV data; the (non-owning) ggml buffers in
    // ctxs_bufs are freed by member destructors afterwards and never dereference the VA
    for (auto & p : vbr_pools_) {
        if (p.vmm != nullptr) {
            p.be->vmm_pool_free(p.vmm);
            p.vmm = nullptr;
        }
    }
}

void llama_kv_cache::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (data) {
        // S5: settle any in-flight degrade wave first — its transcode/scrub writes must not race
        // the memsets below, and the queued tail unmaps must land before pages are re-zeroed
        if (vbr_vmm_active()) {
            vbr_flush_deferred_unmaps();
            for (auto & p : vbr_pools_) {
                if (p.backend != nullptr) {
                    ggml_backend_synchronize(p.backend);
                }
            }
        }
        for (auto & [_, buf] : ctxs_bufs) {
            for (ggml_backend_buffer_t pb : kv_phys_buffers(buf.get())) {
                const vbr_pool * p = kv_vmm_pool_for(vbr_pools_, pb);
                if (p != nullptr) {
                    // a full-buffer clear would memset unmapped VA; zero only the mapped pages
                    p->be->vmm_pool_clear(p->vmm);
                } else {
                    ggml_backend_buffer_clear(pb, 0);
                }
            }
        }

        // Re-initialize turbo rotation matrices after buffer clear (clear zeroes everything)
        if (turbo_rotation != nullptr && turbo_rotation->buffer != nullptr) {
            #include "turbo-rotation-data.h"
            ggml_backend_tensor_set(turbo_rotation, TURBO_ROTATION_R, 0, 128 * 128 * sizeof(float));
            ggml_backend_tensor_set(turbo_rotation_inv, TURBO_ROTATION_RT, 0, 128 * 128 * sizeof(float));
        }
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (i < vbr_stash_rows_) {
                    vbr_stash_dirty_ = true; // a sink cell can now be rewritten by another request
                }
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);
                if (i < vbr_stash_rows_) {
                    vbr_stash_dirty_ = true; // a sink cell can now be rewritten by another request
                }

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

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

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            // for VMM-backed buffers the buffer size is the VA reservation — report the
            // mapped-physical bytes instead (summed across the per-device buffers under a
            // meta buffer; -sm tensor)
            size_t sz      = 0;
            bool   any_vmm = false;
            for (ggml_backend_buffer_t pb : kv_phys_buffers(buf.get())) {
                const vbr_pool * p = kv_vmm_pool_for(vbr_pools_, pb);
                if (p != nullptr) {
                    sz     += p->be->vmm_pool_mapped(p->vmm);
                    any_vmm = true;
                } else {
                    sz += pb != nullptr ? ggml_backend_buffer_get_size(pb) : 0;
                }
            }
            if (!any_vmm) {
                sz = ggml_backend_buffer_get_size(buf.get());
            }
            ret[buft] += sz;
        }
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    // VBR S4 degrade trigger — MUST run at the llama_decode boundary, before any of this batch's
    // cells are positioned. Measured (VBR_MAP_RAWSCAN): mid-batch, apply_ubatch runs ahead of graph
    // execution by up to the whole batch — a mid-batch transcode captures positioned-but-unwritten
    // rows as zeros and races graphs built against the old tier. Here the previous batch's writes
    // are complete/visible and no built-but-unexecuted graphs exist. prepare() is the choke point
    // BOTH paths use (kv_cache::init_batch and llama_memory_hybrid::init_batch call it directly).
    // The check is predictive: fit the WORST-CASE watermark this batch can reach at current tiers
    // so it never overruns the budget mid-flight.
    if (vbr_vmm_active()) {
        // S5: release the tail pages queued by the PREVIOUS wave first — their transcodes are long
        // done (fence-ordered before the previous graph). Must precede this boundary's degrades and
        // ensure_mapped so no later page map can be ripped by a stale queued unmap.
        vbr_flush_deferred_unmaps();
    }
    if (vbr_vmm_active() && vbr_budget_bytes_ > 0) {
        uint32_t n_tokens = 0;
        for (const auto & ub : ubatches) {
            n_tokens += ub.n_tokens;
        }
        // sink-stash staleness: if any sink cell was freed since capture, every stash may hold
        // another request's rows — drop them all (they recapture at the next first degrade)
        if (vbr_stash_dirty_) {
            for (auto & p : vbr_pools_) {
                for (size_t j = 0; j < layers.size(); ++j) {
                    p.k[j].stash_valid = 0;
                    p.v[j].stash_valid = 0;
                }
            }
            vbr_stash_dirty_ = false;
        }
        // -- Stability fast-path: skip per-batch bookkeeping when settled (avoids ~1ms/token) --
        uint32_t used_now = 0;
        for (uint32_t st = 0; st < n_stream; ++st) {
            used_now += v_cells[st].get_used();
        }
        // Stable when: budget fully explored, quiet for N boundaries, occupancy hasn't meaningfully moved.
        static const uint32_t VBR_STABLE_QUICK = 10; // quiet-boundary threshold for fast path
        static const int32_t  VBR_USED_DELTA   = 512; // occupancy delta below which we're stable
        bool vbr_stable = (vbr_degrade_cursor_ >= std::min(vbr_degrade_order_.size(), vbr_degrade_limit_) &&
                           vbr_quiet_boundaries_ >= VBR_STABLE_QUICK &&
                           std::abs((int64_t)used_now - (int64_t)vbr_last_used_) < VBR_USED_DELTA);

        // predicted watermark for THIS boundary; both paths eagerly map to it once, after the if/else.
        uint32_t wm_next = 0;
        if (!vbr_stable) {
            // auto budgets track reality: throttle re-derive from live free VRAM — during steady
            // decode occupancy barely changes, so querying every token is waste. Fire on the first
            // boundary (lazy cuBLAS init), or when a degrades/promotes happen, or every 8th token.
            if (!vbr_budget_explicit_) {
                const bool budget_dirty = vbr_degrade_cursor_ > 0 && vbr_quiet_boundaries_ < VBR_STABLE_QUICK;
                // vbr_boundary_count_ is a free-running per-boundary counter (incremented once per
                // prepare() below) — NOT coupled to whether we actually re-derive, or the throttle
                // could never advance its own gate. count==0 is the first boundary (skipped inside
                // vbr_rederive_budget for lazy cuBLAS); every 8th boundary re-derives thereafter.
                const bool budget_periodic = (vbr_boundary_count_ % 8 == 0);
                if (budget_dirty || budget_periodic) {
                    vbr_rederive_budget();
                }
            }
            // Degrades are one-way and lossy — the two honest recovery levers both live here, at the
            // decode boundary, BEFORE the budget check:
            //  - full-clear reset: the cache is EMPTY, so undoing every degrade is free and lossless;
            //  - container promotion: occupancy dropped (seq_rm) far enough that a higher tier fits
            //    with headroom — old rows keep their degraded quality (re-encoded recon, no
            //    information restored), but FUTURE rows encode at the higher tier.
            if (vbr_degrade_cursor_ > 0 && used_now == 0) {
                vbr_full_reset();
            }
            wm_next = vbr_watermark_cells(n_tokens);
            vbr_shrink_watermark(); // occupancy drops release phantom tail pages first
            static const bool vbr_promote_on = [] {
                const char * e = getenv("VBR_PROMOTE"); // kill switch for experiments; default ON
                return e == nullptr || atoi(e) != 0;
            }();
            // promote pacing: ONE step per boundary, and only after a quiet window (no degrade in
            // the last 4 boundaries). Promotes re-encode aged rows from degraded recon — error
            // compounds per hop — so waves spread out and a clamp-driven degrade vetoes the
            // immediate bounce-back. Boundary counting keeps the cooldown deterministic.
            vbr_quiet_boundaries_++;
            if (vbr_promote_on && vbr_degrade_cursor_ > 0 && vbr_quiet_boundaries_ >= 4) {
                vbr_promote_next(wm_next);
            }
            // budget trigger: degrade while ANY pool exceeds its share. A step only shrinks the pool
            // that owns its tensor, but the cursor is a global price order — advancing it while any
            // pool is over budget is the simplest rule that terminates and preserves the price order.
            while (vbr_over_budget(wm_next)) {
                vbr_quiet_boundaries_ = 0; // degrade pressure this boundary — cool the promote path
                if (!vbr_degrade_next(wm_next)) {
                    if (!vbr_budget_warned_) { // terminal state — one warning, not one per batch
                        vbr_budget_warned_ = true;
                        size_t projected_total = 0;
                        for (const auto & p : vbr_pools_) {
                            projected_total += p.vmm != nullptr ? vbr_vmm_projected_bytes(p, wm_next) : 0;
                        }
                        LLAMA_LOG_WARN("%s: VBR budget %.2f MiB exceeded with the degrade order %s (projected %.2f MiB at %u cells)\n",
                                __func__, vbr_budget_bytes_/1024.0/1024.0,
                                vbr_degrade_limit_ < vbr_degrade_order_.size() ? "clamped at the --vbr-floor" : "exhausted",
                                projected_total/1024.0/1024.0, wm_next);
                    }
                    break;
                }
            }
            // per-pool budget/occupancy trace for multi-GPU verification (visible with -v)
            for (size_t pi = 0; pi < vbr_pools_.size(); ++pi) {
                const auto & p = vbr_pools_[pi];
                if (p.vmm == nullptr) {
                    continue;
                }
                LLAMA_LOG_DEBUG("%s: VBR pool #%zu (device %d): projected %.2f / budget %.2f MiB (mapped %.2f) at %u cells\n",
                        __func__, pi, p.device, vbr_vmm_projected_bytes(p, wm_next)/1024.0/1024.0,
                        p.budget/1024.0/1024.0, p.be->vmm_pool_mapped(p.vmm)/1024.0/1024.0, wm_next);
            }
            // S5: the wave's transcodes/scrubs are queued on each pool's side stream — arm that
            // device's fence so the next graph_compute GPU-waits on them; the host proceeds straight
            // to graph build.
            for (auto & p : vbr_pools_) {
                if (p.wave_pending) {
                    p.be->fence_arm(p.backend);
                    p.wave_pending = false;
                }
            }
        } else {
            // Fast path: settled — skip the budget/degrade bookkeeping. wm_next stays the current
            // watermark; the shared eager map below still covers occupancy that creeps up under the
            // stable threshold.
            wm_next = vbr_watermark_cells(n_tokens);
            vbr_quiet_boundaries_++;
        }
        // Eager physical backing to the predicted watermark, for BOTH paths: map failures surface HERE,
        // where no graphs exist and init_batch fails RECOVERABLY (llama_decode returns an error; the
        // server's decode-failure ladder — idle purge, batch halving — works under VBR instead of a
        // process abort killing every client). try_map is a no-op when the watermark hasn't grown
        // (wm <= wm_cells). Runs after the non-stable path's fence arm so an already-queued transcode
        // wave stays fenced for the NEXT batch's graph. apply_ubatch's ensure_mapped is the mid-batch
        // backstop for placements past the prediction.
        if (!vbr_vmm_try_map(wm_next)) {
            LLAMA_LOG_ERROR("%s: VBR VMM: physical map to %u cells failed (device memory exhausted) — "
                    "failing this batch recoverably\n", __func__, wm_next);
            return {};
        }
        // free-running boundary counter: drives the auto-budget re-derive throttle above and the
        // first-boundary skip inside vbr_rederive_budget(). Advances every boundary regardless of
        // which path ran, so the %8 cadence is real wall-boundary time.
        vbr_boundary_count_++;
        vbr_last_used_ = used_now;
    }

    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d()) {
                llama_kv_cell_ext ext {
                    /*.x =*/ ubatch.pos[i + ubatch.n_tokens*2],
                    /*.y =*/ ubatch.pos[i + ubatch.n_tokens],
                };
                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }

    // VBR VMM: the graph compute that follows reads/writes cells up to the padded watermark —
    // grow the physical backing first (no-op unless the watermark advanced past a page)
    if (vbr_vmm_active()) {
        vbr_vmm_ensure_mapped();
    }

    // Dynamic VBR M3 anchor self-test (env-gated, runs once). Fires from the 2nd apply_ubatch call so
    // a prior attention has identity-init'd the decode-side InnerQ scales the dequant relies on.
    static const bool vbr_test_env = getenv("VBR_TRANSCODE_TEST") != nullptr; // once, not per ubatch
    if (vbr_test_env) {
        static bool vbr_test_armed = false;
        static bool vbr_test_done  = false;
        // apply_ubatch only POSITIONS cells; the K/V write happens in the following graph compute.
        // So arm once >=512 cells are occupied, then fire on the NEXT call (by which point the
        // prior ubatch's cells are written and a prior attention has identity-init'd the decode
        // InnerQ scales).
        if (!vbr_test_done) {
            if (vbr_test_armed) {
                vbr_test_done = true;
                vbr_transcode_anchor_test();
            } else if (v_cells[0].get_used() >= 512) {
                vbr_test_armed = true;
            }
        }
    }
}

// padded cell watermark: the extent get_n_kv derives read views from (256 = the fattn padding
// floor), optionally projected forward by an incoming batch's tokens. prepare()'s predictive
// budget check and ensure_mapped's backing MUST agree on this formula — keep it in one place.
uint32_t llama_kv_cache::vbr_watermark_cells(uint32_t extra_tokens) const {
    const uint32_t n_pad_cur = std::max(n_pad, 256u);
    uint32_t wm = 0;
    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];
        wm = std::max(wm, std::min(cells.size(), GGML_PAD(cells.used_max_p1() + extra_tokens, n_pad_cur)));
    }
    return wm;
}

// multi-pool helpers: any pool VMM-backed / pool owning a tensor (by buffer) / any pool projected
// past its per-pool budget share
bool llama_kv_cache::vbr_vmm_active() const {
    for (const auto & p : vbr_pools_) {
        if (p.vmm != nullptr) {
            return true;
        }
    }
    return false;
}

llama_kv_cache::vbr_pool * llama_kv_cache::vbr_pool_of(const ggml_tensor * t) {
    if (t == nullptr || t->buffer == nullptr) {
        return nullptr;
    }
    for (auto & p : vbr_pools_) {
        if (p.buf == t->buffer) {
            return &p;
        }
    }
    return nullptr;
}

const llama_kv_cache::vbr_pool * llama_kv_cache::vbr_pool_of(const ggml_tensor * t) const {
    return const_cast<llama_kv_cache *>(this)->vbr_pool_of(t);
}

const std::vector<std::pair<llama_kv_cache::vbr_pool *, llama_kv_cache::vbr_extent *>> &
llama_kv_cache::vbr_units_of(size_t ikv, bool is_v) const {
    // non-VBR caches never build the table (no pools) — every unit is unpooled
    static const std::vector<std::pair<vbr_pool *, vbr_extent *>> none;
    return vbr_units_tab_.empty() ? none : vbr_units_tab_[ikv * 2 + (is_v ? 1 : 0)];
}

bool llama_kv_cache::vbr_unit_pooled(size_t ikv, bool is_v) const {
    return !vbr_units_of(ikv, is_v).empty();
}

// The configured budget bounds the POOL, but the card bounds reality: weights + compute
// buffers leave whatever they leave, and the floor-cost fallback budget never consulted free
// VRAM at all. Clamp each pool's target by what its device can actually map right now
// (mapped + free − headroom) so tiers demote EARLY at the decode boundary instead of
// ensure_mapped hitting the hard wall mid-batch (seen: first f16→t8 wave on a 24GB card).
// Shared by the degrade trigger AND the promote hysteresis: gating the two on different
// references (raw budget vs clamped) made every boundary under a co-tenant clamp promote
// then re-degrade — two transcodes plus one extra quantization hop on aged rows per flap.
size_t llama_kv_cache::vbr_budget_eff(const vbr_pool & p) const {
    static const size_t headroom = [] {
        const char * e = getenv("VBR_VRAM_HEADROOM_MIB");
        return (size_t) (e ? strtoull(e, nullptr, 10) : 192) * 1024 * 1024;
    }();
    // memoized per boundary: the degrade loop re-evaluates over_budget per step and the promote
    // hysteresis visits every pool, so an uncached get_device_memory here is a driver round-trip
    // multiplied by wave length x pools — and free VRAM cannot meaningfully move within one
    // boundary (tail unmaps are deferred to the next one)
    if (p.budget_eff_stamp == vbr_boundary_count_) {
        return p.budget_eff_cache;
    }
    size_t budget_eff = p.budget;
    size_t free_b = 0, total_b = 0;
    p.be->get_device_memory(p.device, &free_b, &total_b);
    const size_t mapped_now = p.be->vmm_pool_mapped(p.vmm);
    const size_t cap = mapped_now + (free_b > headroom ? free_b - headroom : 0);
    if (cap < budget_eff) {
        budget_eff = cap;
    }
    p.budget_eff_stamp = vbr_boundary_count_;
    p.budget_eff_cache = budget_eff;
    return budget_eff;
}

// Auto-budget runtime re-derivation: the startup number is a snapshot (fit-armed on whatever
// the box looked like at load, or the bare floor-cost fallback when no fit ran) — a co-tenant
// present at startup or a missing arm otherwise pinned quality low FOREVER on a box with
// gigabytes free. Once per boundary, re-derive each pool's budget from what its device can
// actually give it: share x (mapped + free − growth_headroom), quantized to 64 MiB so driver
// jitter cannot move a tier decision between identical runs, RE-DERIVED not max-ratcheted (a
// sawtooth co-tenant's trough must not be captured as permanent), floored at the init-armed
// value (never stingier than startup), and never touched at all for explicit budgets. Throttled
// by the caller (see prepare()); vbr_boundary_count_ is advanced there, not here.
size_t llama_kv_cache::vbr_pool_reach(const vbr_pool & p) const {
    constexpr size_t quantum = 64ull * 1024 * 1024;
    size_t free_b = 0, total_b = 0;
    p.be->get_device_memory(p.device, &free_b, &total_b);
    const size_t mapped_now = p.be->vmm_pool_mapped(p.vmm);
    const size_t reach_raw  = mapped_now + (free_b > vbr_growth_headroom_ ? free_b - vbr_growth_headroom_ : 0);
    size_t reach = (size_t) ((double) reach_raw * vbr_params_.device_share);
    return reach / quantum * quantum; // 64 MiB quanta = the flap/jitter hysteresis
}

void llama_kv_cache::vbr_rederive_budget() {
    // skip the FIRST boundary: cuBLAS workspaces and CUDA-graph pools allocate lazily during
    // the first graph_compute, so free measured before it overstates reality
    if (vbr_boundary_count_ == 0) {
        return;
    }
    for (size_t pi = 0; pi < vbr_pools_.size(); ++pi) {
        auto & p = vbr_pools_[pi];
        if (p.vmm == nullptr) {
            continue;
        }
        size_t reach = vbr_pool_reach(p);
        reach = std::max(reach, p.budget_base); // the armed/fallback value is the floor
        if (reach != p.budget) {
            LLAMA_LOG_INFO("%s: VBR pool #%zu (device %d) budget re-derived %.2f -> %.2f MiB "
                    "(headroom %.2f, share %.2f)\n",
                    __func__, pi, p.device, p.budget/1024.0/1024.0, reach/1024.0/1024.0,
                    vbr_growth_headroom_/1024.0/1024.0, vbr_params_.device_share);
            p.budget = reach;
        }
    }
}

bool llama_kv_cache::vbr_over_budget(uint32_t wm_cells) const {
    for (const auto & p : vbr_pools_) {
        if (p.vmm == nullptr) {
            continue;
        }
        if (vbr_vmm_projected_bytes(p, wm_cells) > vbr_budget_eff(p)) {
            return true;
        }
    }
    return false;
}

// grow every pool's physical backing to `wm` cells. Returns false on physical exhaustion
// (after reclaiming the previous wave's deferred tail unmaps and retrying once) WITHOUT
// aborting — the caller decides whether its position in the batch lifecycle is recoverable.
// On failure pool.wm_cells stays at its old value; already-mapped delta pages are harmless
// (maps are idempotent, a later retry re-walks them for free).
bool llama_kv_cache::vbr_vmm_try_map(uint32_t wm) {
    for (auto & pool : vbr_pools_) {
        if (pool.vmm == nullptr || wm <= pool.wm_cells) {
            continue;
        }
        // Map only the DELTA [wm_cells, wm): rows below the old watermark stay mapped through degrades
        // (the tail unmap keeps [0, keep)), so re-walking their chunks every growth is pure waste.
        for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
            for (int side = 0; side < 2; ++side) {
                const vbr_extent  & e = side ? pool.v[ikv] : pool.k[ikv];
                const ggml_tensor * t = e.t; // pool-local instance (shard under -sm tensor)
                if (t == nullptr) {
                    continue;
                }
                const size_t row_b = ggml_row_size(t->type, t->ne[0]); // n_stream == 1 (gated at construction)
                const size_t start = row_b * pool.wm_cells;
                const size_t need  = row_b * wm;
                if (!pool.be->vmm_pool_map(pool.vmm, e.byte_off + start, need - start)) {
                    // Physical exhaustion here is usually the FIRST big degrade wave's transient:
                    // the wave's old-tier tail pages are still mapped (their unmap is deferred to
                    // the next decode boundary) while this growth maps the new watermark. Reclaim
                    // them now and retry once before giving up — the flush synchronizes the side
                    // stream, so nothing can still read those pages.
                    LLAMA_LOG_WARN("%s: physical map of %zu bytes failed at offset %zu (watermark %u) — "
                            "flushing deferred unmaps and retrying\n",
                            __func__, need - start, e.byte_off + start, wm);
                    vbr_flush_deferred_unmaps();
                    if (!pool.be->vmm_pool_map(pool.vmm, e.byte_off + start, need - start)) {
                        return false;
                    }
                }
            }
        }
        pool.wm_cells = wm;
    }
    return true;
}

void llama_kv_cache::vbr_vmm_ensure_mapped() {
    // the coming graph's writes land on positioned cells below the watermark; reads pad up to it.
    // NOTE: the S4 degrade trigger deliberately does NOT live here — apply_ubatch runs mid-batch
    // where positioned cells outrun the graph writes (see prepare()). prepare() already mapped to
    // its predicted watermark recoverably, so this fires only when placement outran the
    // prediction (freed low cells + a head above used_max_p1) — with graphs already built,
    // aborting is all that is left. Kept as the backstop, expected unreachable.
    const uint32_t wm = vbr_watermark_cells(0);
    if (!vbr_vmm_try_map(wm)) {
        GGML_ABORT("VBR VMM: out of physical memory mapping to watermark %u cells mid-batch", wm);
    }
}

// mapped-physical bytes needed to back `wm_cells` of ONE pool's extents at the CURRENT per-tensor
// tiers (page-rounded), plus that pool's up-front constants (rotation matrices)
size_t llama_kv_cache::vbr_vmm_projected_bytes(const vbr_pool & p, uint32_t wm_cells) const {
    size_t total = p.mapped_base;
    for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
        for (int side = 0; side < 2; ++side) {
            const vbr_extent  & e = side ? p.v[ikv] : p.k[ikv];
            const ggml_tensor * t = e.t; // pool-local instance (shard under -sm tensor)
            if (t == nullptr) {
                continue;
            }
            const size_t need = (size_t) ggml_row_size(t->type, t->ne[0]) * wm_cells;
            total += GGML_PAD(need, p.gran);
        }
    }
    return total;
}

// S5: unmap tail pages queued by the previous degrade wave. Safe only after the wave's transcodes
// finished (they READ the old tier-A extent, which reaches into these pages) — one side-stream
// sync per pool makes that certain; by the next decode boundary the wave is long done, so this is
// ~free.
void llama_kv_cache::vbr_flush_deferred_unmaps() {
    for (auto & p : vbr_pools_) {
        if (p.unmap_deferred.empty()) {
            continue;
        }
        GGML_ASSERT(p.backend != nullptr); // entries are only queued after async work on it
        ggml_backend_synchronize(p.backend);
        for (const auto & [off, len] : p.unmap_deferred) {
            p.be->vmm_pool_unmap(p.vmm, off, len);
        }
        p.unmap_deferred.clear();
    }
}

// Generic degrade-rank curves for models WITHOUT a baked order (matrix v3, 2026-07-05).
// Derived by averaging the five measured models' cheap-first price orders (q27, qwen35moe,
// g12, g26, g31 — dense, MoE-hybrid and SWA-mixed layouts) in NORMALIZED KV-layer position.
// What generalized: the fp16->t8 band is near-universal (deep-first, front protected, K~V;
// cross-model rank deviation 0.036); below t8 the robust invariants are final-layer V
// maximally protected in EVERY band, front V cheapest at the bottom rungs, K positionally
// flat. Sub-t8 mid-band shapes disagree across models (deviation ~0.2) — the mean is a
// hedge, not a truth; a measured per-model order is always better.
// [band][is_v][grid p=0..1 step 1/16]; lower value = degrade earlier.
static const float vbr_generic_rank[5][2][17] = {
    { // fp16-t8
        { 0.95f, 0.82f, 0.83f, 0.82f, 0.78f, 0.71f, 0.63f, 0.55f, 0.52f, 0.51f, 0.39f, 0.32f, 0.26f, 0.19f, 0.13f, 0.08f, 0.00f },
        { 0.92f, 0.94f, 0.83f, 0.80f, 0.79f, 0.71f, 0.62f, 0.54f, 0.48f, 0.45f, 0.39f, 0.32f, 0.24f, 0.21f, 0.14f, 0.08f, 0.02f },
    },
    { // t8-t4
        { 0.62f, 0.66f, 0.61f, 0.54f, 0.77f, 0.66f, 0.47f, 0.58f, 0.46f, 0.48f, 0.40f, 0.46f, 0.35f, 0.41f, 0.42f, 0.44f, 0.53f },
        { 0.49f, 0.37f, 0.58f, 0.50f, 0.56f, 0.41f, 0.48f, 0.47f, 0.38f, 0.33f, 0.35f, 0.44f, 0.36f, 0.41f, 0.31f, 0.53f, 1.00f },
    },
    { // t4-t3tcq
        { 0.54f, 0.40f, 0.53f, 0.49f, 0.49f, 0.45f, 0.44f, 0.54f, 0.56f, 0.52f, 0.53f, 0.60f, 0.52f, 0.54f, 0.55f, 0.54f, 0.50f },
        { 0.35f, 0.42f, 0.32f, 0.53f, 0.54f, 0.43f, 0.50f, 0.48f, 0.44f, 0.42f, 0.41f, 0.49f, 0.50f, 0.56f, 0.43f, 0.64f, 0.93f },
    },
    { // t3tcq-t2tcq
        { 0.28f, 0.35f, 0.42f, 0.44f, 0.60f, 0.53f, 0.48f, 0.46f, 0.54f, 0.61f, 0.62f, 0.57f, 0.53f, 0.60f, 0.69f, 0.60f, 0.71f },
        { 0.16f, 0.14f, 0.19f, 0.46f, 0.51f, 0.53f, 0.47f, 0.57f, 0.41f, 0.44f, 0.44f, 0.45f, 0.48f, 0.57f, 0.47f, 0.69f, 1.00f },
    },
    { // t2tcq-t1tcq
        { 0.46f, 0.32f, 0.36f, 0.36f, 0.62f, 0.48f, 0.49f, 0.58f, 0.64f, 0.57f, 0.58f, 0.67f, 0.54f, 0.56f, 0.63f, 0.63f, 0.73f },
        { 0.12f, 0.18f, 0.24f, 0.36f, 0.33f, 0.40f, 0.39f, 0.32f, 0.45f, 0.39f, 0.48f, 0.56f, 0.55f, 0.46f, 0.62f, 0.73f, 0.99f },
    },
};

void llama_kv_cache::vbr_load_degrade_order() {
    vbr_degrade_order_.clear();
    // VBR_FORCE_GENERIC=1: skip the file/registry paths — A/B instrument for the generic
    // curves, and exactly the path an unsupported arch takes.
    if (getenv("VBR_FORCE_GENERIC") != nullptr) {
        vbr_synth_generic_order();
        return;
    }
    if (const char * path = getenv("VBR_DEGRADE_ORDER")) {
        std::ifstream f(path);
        std::string tok;
        bool ok = (bool) f;
        // tokens "<il><k|v>:<t8|t4|t3|t2|t1>", whitespace-separated
        while (ok && f >> tok) {
            const size_t colon = tok.find(':');
            ok = colon != std::string::npos && colon >= 2;
            if (!ok) {
                break;
            }
            const char side = tok[colon - 1];
            const std::string tier = tok.substr(colon + 1);
            static const std::map<std::string, uint8_t> tiers = {
                {"t8", VBR_TIER_T8}, {"t4", VBR_TIER_T4}, {"t3", VBR_TIER_T3_TCQ},
                {"t2", VBR_TIER_T2_TCQ}, {"t1", VBR_TIER_T1_TCQ},
            };
            const auto it = tiers.find(tier);
            // the layer id must be the ENTIRE prefix and a valid layer (atoi silently accepted
            // garbage as layer 0 and >255 truncated through the uint8 cast)
            char * endp = nullptr;
            const std::string il_str = tok.substr(0, colon - 1);
            const long il = strtol(il_str.c_str(), &endp, 10);
            ok = (side == 'k' || side == 'v') && it != tiers.end() &&
                 endp != nullptr && *endp == '\0' && !il_str.empty() &&
                 il >= 0 && il < (long) hparams.n_layer_all;
            if (ok) {
                vbr_degrade_order_.push_back({ (uint8_t) il, (uint8_t) (side == 'v'), it->second });
            }
        }
        if (ok && !vbr_degrade_order_.empty()) {
            size_t n_unmatched = 0;
            for (const auto & st : vbr_degrade_order_) {
                n_unmatched += map_layer_ids.find(st.il) == map_layer_ids.end();
            }
            if (n_unmatched > 0) {
                // valid layer ids that hold no KV in THIS cache (e.g. recurrent layers of a
                // hybrid model, or the wrong cache of an iSWA pair) — they no-op at runtime,
                // which silently hides typos
                LLAMA_LOG_WARN("%s: VBR degrade order: %zu of %zu steps reference layers with no KV "
                        "in this cache (they will be skipped)\n",
                        __func__, n_unmatched, vbr_degrade_order_.size());
            }
            LLAMA_LOG_INFO("%s: VBR degrade order: %zu steps from %s\n", __func__, vbr_degrade_order_.size(), path);
            return;
        }
        LLAMA_LOG_WARN("%s: VBR_DEGRADE_ORDER %s unreadable or malformed (near '%s') — using baked order\n",
                __func__, path, tok.c_str());
        vbr_degrade_order_.clear();
    }
    // Arch-keyed baked orders (matrix v3, 2026-07-05): per-model price orders measured under the
    // deployment-true tap config with reliability-gated statistics (bench-validated lens per
    // model; fp16->t8 band from the frac lens). Keyed on (arch, n_layer) so the gemma4 family
    // resolves per MODEL — their price structures are opposite (front-hot vs deep-hot).
    for (const auto & e : vbr_baked_orders) {
        if (e.arch == model.arch && e.n_layer == hparams.n_layer_all) {
            vbr_degrade_order_.assign(e.steps, e.steps + e.n);
            LLAMA_LOG_INFO("%s: VBR degrade order: %zu baked steps (arch-matched, matrix v3)\n",
                    __func__, vbr_degrade_order_.size());
            return;
        }
    }
    // Models ship with and without MTP/nextn predict layers, which append to n_layer while
    // leaving the KV-bearing backbone identical — so fall back to matching on (arch + the
    // EXACT KV-layer-id set). Set equality (not subset) so different-sized same-arch models
    // can never cross-match, and a cache holding layers the table does not cover falls
    // through to the generic order instead of silently never degrading them.
    for (const auto & e : vbr_baked_orders) {
        if (e.arch != model.arch) {
            continue;
        }
        std::set<uint8_t> tbl_ils;
        for (size_t i = 0; i < e.n; ++i) {
            tbl_ils.insert(e.steps[i].il);
        }
        if (tbl_ils.size() != map_layer_ids.size()) {
            continue;
        }
        bool same = true;
        for (const uint8_t il : tbl_ils) {
            if (map_layer_ids.find(il) == map_layer_ids.end()) {
                same = false;
                break;
            }
        }
        if (same) {
            vbr_degrade_order_.assign(e.steps, e.steps + e.n);
            LLAMA_LOG_INFO("%s: VBR degrade order: %zu baked steps (arch + KV-layout matched; "
                    "n_layer %u vs table %u — MTP/nextn-style variant)\n",
                    __func__, vbr_degrade_order_.size(), hparams.n_layer_all, e.n_layer);
            return;
        }
    }
    LLAMA_LOG_WARN("%s: no measured VBR degrade order for this arch/n_layer — "
            "using the generic cross-model order (a measured per-model order is better; "
            "set VBR_DEGRADE_ORDER=<file> to supply one)\n", __func__);
    vbr_synth_generic_order();
}

// Synthesize a degrade order from the generic curves: strictly banded (the whole cache
// reaches tier N before any unit drops below it — the safe monotone default when real
// per-model prices are unknown), and within each band cells sorted cheap-first by the
// curve rank at the layer's normalized position among this model's KV-BEARING layers
// (MoE/hybrid layouts: only layers that hold KV count, matching how the curves were fit).
void llama_kv_cache::vbr_synth_generic_order() {
    std::vector<uint32_t> ils;
    for (const auto & l : layers) {
        ils.push_back(l.il);
    }
    std::sort(ils.begin(), ils.end());
    const size_t n = ils.size();
    if (n == 0) {
        return;
    }
    static const uint8_t band_tier[5] = {
        VBR_TIER_T8, VBR_TIER_T4, VBR_TIER_T3_TCQ, VBR_TIER_T2_TCQ, VBR_TIER_T1_TCQ,
    };
    for (int band = 0; band < 5; ++band) {
        std::vector<std::pair<float, uint16_t>> cells; // rank, (i<<1)|is_v
        cells.reserve(n * 2);
        for (size_t i = 0; i < n; ++i) {
            const float x  = n > 1 ? 16.0f * (float) i / (float) (n - 1) : 0.0f;
            const int   i0 = std::min((int) x, 15);
            const float fr = x - (float) i0;
            for (int is_v = 0; is_v < 2; ++is_v) {
                const float r = vbr_generic_rank[band][is_v][i0] * (1.0f - fr)
                              + vbr_generic_rank[band][is_v][i0 + 1] * fr;
                cells.push_back({ r, (uint16_t) ((i << 1) | is_v) });
            }
        }
        std::stable_sort(cells.begin(), cells.end(),
                [](const auto & a, const auto & b) { return a.first < b.first; });
        for (const auto & c : cells) {
            vbr_degrade_order_.push_back({ (uint8_t) ils[c.second >> 1],
                                           (uint8_t) (c.second & 1), band_tier[band] });
        }
    }
    LLAMA_LOG_INFO("%s: VBR degrade order: %zu generic steps (cross-model curves, %zu KV layers)\n",
            __func__, vbr_degrade_order_.size(), n);
}

bool llama_kv_cache::vbr_unit_movable(ggml_type t, bool is_v) const {
    return vbr_type_is_movable(t) && !vbr_side_pinned(is_v);
}

// resolve a --vbr-floor value: env override, then the bottom-tier default for 0/auto
static double vbr_resolve_floor_bpv(double min_bits) {
    double floor_bpv = min_bits;
    if (const char * env = getenv("VBR_MIN_BITS")) {
        floor_bpv = atof(env); // "auto"/"none" parse to 0 -> the t1 default below
    }
    if (floor_bpv <= 0.0) {
        floor_bpv = 8.0 * ggml_type_size(GGML_TYPE_TURBO1_TCQ) / ggml_blck_size(GGML_TYPE_TURBO1_TCQ);
    }
    return floor_bpv;
}

// Shared floor-walk core (runtime clamp AND fit capacity math): simulate the degrade order over
// the per-unit tiers of the entry layout, stopping before the aggregate would cross floor_bpv.
// PINNED units (non-tier types or a flag-pinned side) stay in the aggregate at their fixed bpv —
// the floor is a literal aggregate — but no step may move them (mirrors vbr_degrade_next).
// pooled_only restricts units to VMM-pooled ones (the runtime); dry-load contexts have no pools
// and pass false. entry_k/entry_v override each side's tensor type (the fit's cparams types are
// price-swapped during fitting; it passes the true entry types) — GGML_TYPE_COUNT = tensor type.
llama_kv_cache::vbr_floor_sim_result llama_kv_cache::vbr_floor_sim(
        double floor_bpv, bool pooled_only, ggml_type entry_k, ggml_type entry_v) const {
    vbr_floor_sim_result res;
    res.end_types.assign(layers.size() * 2, GGML_TYPE_COUNT);
    auto & sim = res.end_types;
    double  sum_bits = 0.0;
    int64_t sum_vals = 0;
    for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
        for (int side = 0; side < 2; ++side) {
            const ggml_tensor * t = side ? layers[ikv].v : layers[ikv].k;
            if (t == nullptr || (pooled_only && !vbr_unit_pooled(ikv, side != 0))) {
                continue; // absent, or (runtime) not VMM-pooled — only pooled units can degrade
            }
            const ggml_type entry = side ? (entry_v != GGML_TYPE_COUNT ? entry_v : t->type)
                                         : (entry_k != GGML_TYPE_COUNT ? entry_k : t->type);
            // aggregate math on the canonical tensor: shard row sizes are additive across pools
            // (blocks never straddle the split), so this is exact under -sm tensor too
            sim[ikv*2 + side] = entry;
            sum_bits += 8.0 * ggml_row_size(entry, t->ne[0]);
            sum_vals += t->ne[0];
            res.n_pinned += !vbr_unit_movable(entry, side != 0);
        }
    }
    res.clamp_step = vbr_degrade_order_.size();
    if (sum_vals == 0) {
        return res;
    }
    for (size_t i = 0; i < vbr_degrade_order_.size(); ++i) {
        const auto & st = vbr_degrade_order_[i];
        const auto it = map_layer_ids.find(st.il);
        if (it == map_layer_ids.end()) {
            continue;
        }
        const size_t slot = (size_t) it->second * 2 + (st.is_v ? 1 : 0);
        const ggml_tensor * t = st.is_v ? layers[it->second].v : layers[it->second].k;
        if (sim[slot] == GGML_TYPE_COUNT || t == nullptr || !vbr_unit_movable(sim[slot], st.is_v != 0)) {
            continue; // absent or pinned (runtime skips these steps identically)
        }
        const ggml_type type_B = vbr_tier_type(st.tier);
        const size_t rA = ggml_row_size(sim[slot], t->ne[0]);
        const size_t rB = ggml_row_size(type_B,    t->ne[0]);
        if (sim[slot] == type_B || rB >= rA) {
            continue; // same no-op rule as vbr_degrade_next
        }
        const double bits_next = sum_bits - 8.0*rA + 8.0*rB;
        if (bits_next / sum_vals < floor_bpv - 1e-9) {
            res.clamp_step = i;
            res.next_bpv   = bits_next / sum_vals;
            break;
        }
        sim[slot] = type_B;
        sum_bits  = bits_next;
    }
    res.bits_per_token = sum_bits; // one row per token per unit
    return res;
}

// per-token KV bits of the layout the floor clamp lands on — the fit pass calls this on its
// dry-load context (llama_vbr_floor_bits_per_token) for floor-true capacity math
double llama_kv_cache::memory_vbr_floor_bits_per_token(ggml_type entry_k, ggml_type entry_v, double floor_bpv) {
    if (vbr_degrade_order_.empty()) {
        vbr_load_degrade_order(); // dry contexts never reach the VMM arming block
    }
    return vbr_floor_sim(vbr_resolve_floor_bpv(floor_bpv), !vbr_pools_.empty(), entry_k, entry_v).bits_per_token;
}

// --vbr-floor (cparams min_bits; env VBR_MIN_BITS override, decimal bits/value): a LITERAL
// aggregate floor. Walk the order against the initial layout and clamp the cursor at the first
// step that would take the aggregate below the floor — e.g. floor 4.25 with t4 = 4.125 bpv stops
// with a few units still a tier higher. Strict-prefix clamp: the aggregate is monotone decreasing
// along the order, and skipping ahead to a cheaper later step would violate the measured price
// order. The default t1 floor (1.25) equals the full order's end point, so nothing clamps.
void llama_kv_cache::vbr_floor_clamp_order() {
    const double floor_bpv = vbr_resolve_floor_bpv(vbr_params_.min_bits);
    const auto res = vbr_floor_sim(floor_bpv, /*pooled_only =*/ true);
    vbr_degrade_limit_ = res.clamp_step;
    if (res.bits_per_token == 0.0) {
        return; // no VMM-pooled units
    }
    if (res.n_pinned > 0) {
        LLAMA_LOG_INFO("%s: VBR: %zu (layer,side) units are PINNED at non-vbr types — degrade steps "
                "touching them are skipped; they stay in the aggregate at their fixed bits/value\n",
                __func__, res.n_pinned);
    }
    if (res.clamp_step < vbr_degrade_order_.size()) {
        LLAMA_LOG_INFO("%s: VBR floor %.4g bits/value: degrade order clamped at %zu/%zu steps "
                "(next step would drop the aggregate to %.4g)\n",
                __func__, floor_bpv, res.clamp_step, vbr_degrade_order_.size(), res.next_bpv);
    }

    // page-exact mapped-physical cost of the FLOOR layout (the sim's end state) at full kv_size —
    // the minimum budget that guarantees the advertised context fits. Used as the fallback budget
    // when dynamic mode reaches us without a fit-resolved one. Summed across pools (page rounding
    // uses each tensor's OWNING pool granularity).
    vbr_floor_cost_bytes_ = 0;
    for (const auto & p : vbr_pools_) {
        vbr_floor_cost_bytes_ += p.mapped_base;
    }
    for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
        for (int side = 0; side < 2; ++side) {
            if (res.end_types[ikv*2 + side] == GGML_TYPE_COUNT) {
                continue;
            }
            // page rounding is per pool instance (per device shard under -sm tensor)
            for (const auto & [p, e] : vbr_units_of(ikv, side != 0)) {
                const size_t need = ggml_row_size(res.end_types[ikv*2 + side], e->t->ne[0]) * (size_t) e->t->ne[1];
                vbr_floor_cost_bytes_ += GGML_PAD(need, p->gran);
            }
        }
    }
}

// flip a cache tensor (and its per-stream views) to a new tier — host metadata consumed at graph
// BUILD time; callers order the GPU against the matching transcode via the S5 fence
static void vbr_set_tensor_type_impl(ggml_tensor * t, const std::vector<ggml_tensor *> & views, ggml_type type) {
    t->type  = type;
    t->nb[0] = ggml_type_size(type);
    t->nb[1] = ggml_row_size(type, t->ne[0]);
    t->nb[2] = t->nb[1]*t->ne[1];
    t->nb[3] = t->nb[2]*t->ne[2];
    for (ggml_tensor * vt : views) {
        if (vt != nullptr) {
            vt->type  = type;
            vt->nb[0] = t->nb[0];
            vt->nb[1] = t->nb[1];
            vt->nb[2] = t->nb[2];
            vt->nb[3] = t->nb[3];
        }
    }
}

static void vbr_set_tensor_type(ggml_tensor * t, std::vector<ggml_tensor *> & views, ggml_type type) {
    vbr_set_tensor_type_impl(t, views, type);
    // -sm tensor: graphs are built from this (meta) tensor, but the per-pool byte math and the
    // transcode kernels operate on the per-device SHARDS — flip them in lockstep. Shard strides
    // derive from the shard's own ne0 (its slice of the head*dim axis), not the meta tensor's.
    if (t->buffer != nullptr && ggml_backend_buffer_is_meta(t->buffer)) {
        const size_t n = ggml_backend_meta_buffer_n_bufs(t->buffer);
        for (size_t i = 0; i < n; ++i) {
            ggml_tensor * shard = ggml_backend_meta_buffer_simple_tensor(t, i);
            if (shard == nullptr) {
                continue;
            }
            std::vector<ggml_tensor *> shard_views;
            shard_views.reserve(views.size());
            for (ggml_tensor * vt : views) {
                shard_views.push_back(vt != nullptr ? ggml_backend_meta_buffer_simple_tensor(vt, i) : nullptr);
            }
            vbr_set_tensor_type_impl(shard, shard_views, type);
        }
    }
}

// lazily size + allocate one pool's f16 sink-stash buffer (one slab per pool on that pool's
// device, per-extent offsets); returns its base. Requires the pool's side-stream backend.
char * llama_kv_cache::vbr_stash_ensure(vbr_pool & p) {
    if (p.stash_buf == nullptr) {
        size_t total = 0;
        for (size_t j = 0; j < layers.size(); ++j) {
            for (int side = 0; side < 2; ++side) {
                vbr_extent        & ex = side ? p.v[j] : p.k[j];
                const ggml_tensor * tt = ex.t; // pool-local instance (shard under -sm tensor)
                if (tt == nullptr) {
                    continue;
                }
                ex.stash_off = total;
                total += (size_t) vbr_stash_rows_ * tt->ne[0] * sizeof(uint16_t);
            }
        }
        GGML_ASSERT(p.backend != nullptr);
        p.stash_buf = ggml_backend_buft_alloc_buffer(
                ggml_backend_get_default_buffer_type(p.backend), total);
        GGML_ASSERT(p.stash_buf != nullptr);
        LLAMA_LOG_INFO("%s: VBR sink-stash buffer (device %d): %.2f MiB\n", __func__, p.device, total/1024.0/1024.0);
    }
    return (char *) ggml_backend_buffer_get_base(p.stash_buf);
}

// The cache is EMPTY: nothing is stored, so undoing every degrade is free and LOSSLESS — unlike
// container promotion this genuinely restores quality, because all future content is new. Flip
// every tensor back to its entry tier, rewind the price cursor, drop the (now stale) sink
// stashes and release every physical page; the next session refills from a clean entry-tier
// start (prefill-direct). Fires lazily from prepare() at the first decode after the cache
// empties, whatever emptied it (clear, seq_rm, server slot recycle).
void llama_kv_cache::vbr_full_reset() {
    // in-flight safety before ripping pages: settle the side streams and the devices once
    // (session-boundary event — the sync cost is irrelevant)
    vbr_flush_deferred_unmaps();
    size_t undone = 0;
    size_t mapped = 0;
    for (auto & pool : vbr_pools_) {
        if (pool.vmm == nullptr) {
            continue;
        }
        if (pool.backend != nullptr) {
            ggml_backend_synchronize(pool.backend);
        }
        pool.be->sync_device(pool.device);

        for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
            for (int side = 0; side < 2; ++side) {
                vbr_extent  & e  = side ? pool.v[ikv]   : pool.k[ikv];
                ggml_tensor * tc = side ? layers[ikv].v : layers[ikv].k; // canonical (graph) tensor
                if (e.t == nullptr || tc == nullptr) {
                    continue;
                }
                if (tc->type != e.type0) {
                    // flips the canonical tensor AND every shard instance (see vbr_set_tensor_type);
                    // under -sm tensor a later pool sees the type already restored and only unmaps
                    vbr_set_tensor_type(tc, side ? layers[ikv].v_stream : layers[ikv].k_stream, e.type0);
                    vbr_tier_epoch_++; // fence graph reuse off the old views
                    undone++;
                }
                e.stash_valid  = 0;
                e.promote_hops = 0; // fresh hop budget — the reset epoch starts clean
                const size_t slot = vbr_slot_bytes(e.t);
                pool.be->vmm_pool_unmap(pool.vmm, e.byte_off, slot);
            }
        }
        pool.wm_cells = 0;
        mapped += pool.be->vmm_pool_mapped(pool.vmm);
    }
    vbr_degrade_cursor_    = 0;
    vbr_budget_warned_     = false;
    vbr_stash_dirty_       = false;
    vbr_quiet_boundaries_  = 0;
    LLAMA_LOG_INFO("%s: VBR full reset: cache empty — %zu tensors back at their entry tier, pools released "
            "(%.2f MiB mapped)\n", __func__, undone, mapped/1024.0/1024.0);
}

// Occupancy DROPPED (seq_rm trimmed a session): pull the mapped watermark back down and release
// the tail pages. They are phantom cost against the budget (mapped but unreadable — attention
// pads only to used+256), and both promotion's hysteresis and its re-encode row count follow
// wm_cells, so leaving it at the old high-water lets a promotion map tiers back up at a stale
// size (the G4b bug: approved at 1536 live rows, executed at 5632). The rows shrunk away keep
// valid current-tier bytes, so no scrub is needed; regrowth remaps zero-filled pages on demand.
// 25% hysteresis so growth/trim jitter cannot thrash map/unmap.
void llama_kv_cache::vbr_shrink_watermark() {
    const uint32_t wm_now = vbr_watermark_cells(0);
    for (auto & pool : vbr_pools_) {
        if (pool.vmm == nullptr || wm_now + wm_now / 4 >= pool.wm_cells) {
            continue;
        }
        // decode boundary + flushed deferred queue; settle the streams once before ripping pages
        if (pool.backend != nullptr) {
            ggml_backend_synchronize(pool.backend);
        }
        pool.be->sync_device(pool.device);
        for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
            for (int side = 0; side < 2; ++side) {
                const vbr_extent  & e = side ? pool.v[ikv] : pool.k[ikv];
                const ggml_tensor * t = e.t; // pool-local instance (shard under -sm tensor)
                if (t == nullptr) {
                    continue;
                }
                const size_t keep = ggml_row_size(t->type, t->ne[0]) * (size_t) wm_now;
                const size_t slot = vbr_slot_bytes(t);
                pool.be->vmm_pool_unmap(pool.vmm, e.byte_off + keep, slot - keep);
            }
        }
        LLAMA_LOG_INFO("%s: VBR watermark shrink (device %d): %u -> %u cells (%.2f MiB mapped)\n",
                __func__, pool.device, pool.wm_cells, wm_now,
                pool.be->vmm_pool_mapped(pool.vmm)/1024.0/1024.0);
        pool.wm_cells = wm_now;
    }
}

// Container promotion: occupancy DROPPED (seq_rm trimmed a long session) and a higher tier now
// fits with headroom. This restores NO information — the live rows are re-encoded from their
// current recon (same quality, bigger container) — but every FUTURE row encodes at the higher
// tier, which the bathtub makes worth real quality. Walks the price cursor BACKWARDS one step
// per call; the in-place transcode runs descending tiles (see vbr-transcode.cu) after the grown
// extent is mapped up front.
bool llama_kv_cache::vbr_promote_next(uint32_t wm_next) {
    while (vbr_degrade_cursor_ > 0) {
        const auto & st = vbr_degrade_order_[vbr_degrade_cursor_ - 1];
        const auto it = map_layer_ids.find(st.il);
        if (it == map_layer_ids.end()) {
            vbr_degrade_cursor_--;
            continue;
        }
        const int32_t ikv = it->second;
        ggml_tensor * t = st.is_v ? layers[ikv].v : layers[ikv].k; // canonical tensor: the type source of truth
        const auto & units = vbr_units_of(ikv, st.is_v != 0); // every VMM pool holding this unit
        if (t == nullptr || units.empty()) {
            vbr_degrade_cursor_--; // this entry's degrade never applied (skipped no-op) — free rewind
            continue;
        }
        if (!vbr_unit_movable(t->type, st.is_v != 0)) {
            // pinned unit: its degrades never applied, so its entries rewind for free. The
            // type-mismatch check below does NOT cover a side pinned AT a tier type (the entry
            // type equals the step tier for its own order entries) — guard explicitly.
            vbr_degrade_cursor_--;
            continue;
        }
        if (t->type != vbr_tier_type(st.tier)) {
            vbr_degrade_cursor_--; // this entry's degrade never applied (skipped no-op) — free rewind
            continue;
        }
        // the tier this unit held BEFORE this step: its previous appearance in the order, else entry
        ggml_type type_B = units[0].second->type0;
        for (size_t j = vbr_degrade_cursor_ - 1; j-- > 0; ) {
            const auto & pj = vbr_degrade_order_[j];
            if (pj.il == st.il && pj.is_v == st.is_v) {
                type_B = vbr_tier_type(pj.tier);
                break;
            }
        }
        if (type_B == GGML_TYPE_F16 || type_B == GGML_TYPE_TURBO8_0) {
            // promotion CAPS below the tap boundary: sources under t8 store mean-subtracted
            // rows (V - mu_V), and neither t8 nor f16 decode restores the means (turbo_tap_mu
            // gates t8 out of the tap; f16 has no add-back) — promoting across the boundary
            // would serve mean-shifted values. Promotion still operates within the tapped
            // tiers (t4 <-> t3 <-> t2 <-> t1); the f16 entry returns losslessly at full reset.
            return false;
        }
        for (const auto & u : units) {
            if (u.second->promote_hops >= 2) {
                // hop cap: each promote with live rows re-encodes the aged rows from their degraded
                // recon — error compounds per hop, and only FUTURE rows gain. Capping per extent
                // between resets bounds the damage; stopping the walk here is required anyway
                // (promotion is LIFO along the price order — skipping past a capped top entry
                // would re-order the ladder). Hops advance in lockstep across pools.
                return false;
            }
        }
        {
            const size_t rA = ggml_row_size(t->type, t->ne[0]);
            const size_t rB = ggml_row_size(type_B,  t->ne[0]);
            if (rB <= rA) {
                vbr_degrade_cursor_--;
                continue;
            }
        }

        // hysteresis: promote only while the promoted layout keeps ~15% headroom of EVERY
        // affected pool's LIVE-CLAMPED budget — churn costs a transcode each way AND an extra
        // quantization hop on the aged rows, and clamping only the degrade side made co-tenant
        // boundaries flap (promote on raw budget, re-degrade on the clamp). Basis =
        // max(projected watermark, mapped watermark): the transcode re-encodes and maps wm_cells
        // rows, so the check must price what will actually be mapped. Under -sm tensor pools
        // shrink/grow together, so the tightest device gates the promotion (mirrors degrade).
        for (const auto & [pp, ep] : units) {
            const int64_t ne0 = ep->t->ne[0];
            const size_t rA = ggml_row_size(ep->t->type, ne0);
            const size_t rB = ggml_row_size(type_B,      ne0);
            const uint32_t wm_eff = std::max(wm_next, pp->wm_cells);
            const size_t projected = vbr_vmm_projected_bytes(*pp, wm_eff)
                                   + GGML_PAD(rB * (size_t) wm_eff, pp->gran)
                                   - GGML_PAD(rA * (size_t) wm_eff, pp->gran);
            const size_t budget_eff = vbr_budget_eff(*pp);
            if (projected > budget_eff - budget_eff / 7) {
                return false;
            }
        }

        // the footprint GROWS: back it in EVERY pool before any transcode writes into it (new
        // pages zero-fill, so [keep, keep_pad) only needs the scrub for previously-mapped stale
        // tier-A bytes). All maps must succeed BEFORE the flip; on a mid-way failure the extra
        // pages of earlier pools are harmless (fungible, reclaimed by the next watermark shrink).
        for (const auto & [pp, ep] : units) {
            const vbr_span sp = vbr_span_of(ep->t, type_B, pp->wm_cells, wm_next, pp->gran);
            if (!pp->be->vmm_pool_map(pp->vmm, ep->byte_off, sp.keep_pad)) {
                return false; // physical memory tight — promotion is optional, just stop
            }
        }

        for (auto & [pp, ep] : units) {
            vbr_extent  & e = *ep;
            const int64_t n_cells = pp->wm_cells;
            const vbr_span sp = vbr_span_of(e.t, type_B, n_cells, wm_next, pp->gran);
            if (n_cells > 0) {
                if (pp->backend == nullptr) {
                    pp->backend = pp->be->backend_init(pp->device);
                    GGML_ASSERT(pp->backend != nullptr);
                }
                if (!pp->wave_pending) {
                    pp->be->sync_device(pp->device);
                }
                // reuse an existing sink stash (captured pristine at the first degrade) so the sink
                // recovers toward single-hop error; do NOT capture here — a promote-time snapshot
                // would lock in the DEGRADED recon as the reference
                const void * stash_ptr  = nullptr;
                int64_t      stash_rows = 0;
                if (vbr_stash_rows_ > 0 && e.stash_valid > 0 && pp->stash_buf != nullptr) {
                    stash_ptr  = (char *) ggml_backend_buffer_get_base(pp->stash_buf) + e.stash_off;
                    stash_rows = e.stash_valid;
                }
                const ggml_vbr_transcode_params tp = {
                    /*.src         =*/ e.t,
                    /*.type_B      =*/ type_B,
                    /*.dst         =*/ e.t->data,
                    /*.pool_buf    =*/ pp->buf,
                    /*.n_cells     =*/ n_cells,
                    /*.is_v        =*/ st.is_v != 0,
                    /*.stash_f16   =*/ stash_ptr,
                    /*.stash_rows  =*/ stash_rows,
                    /*.scrub_bytes =*/ sp.keep_pad - sp.keep,
                };
                pp->be->kv_transcode(pp->backend, &tp);
                pp->wave_pending = true;
                e.promote_hops++; // only live-row re-encodes count — a 0-cell flip is free re-typing
            }
        }
        vbr_set_tensor_type(t, st.is_v ? layers[ikv].v_stream : layers[ikv].k_stream, type_B);
        vbr_tier_epoch_++; // fence graph reuse off the old views (type/strides changed in place)
        vbr_degrade_cursor_--;

        for (const auto & [pp, ep] : units) {
            LLAMA_LOG_INFO("%s: VBR promote #%zu: %s L%d -> %s (%u cells re-encoded on side stream, "
                    "device %d mapped %.2f MiB)\n",
                    __func__, vbr_degrade_cursor_, ep->t->name, (int) st.il, ggml_type_name(type_B),
                    pp->wm_cells, pp->device, pp->be->vmm_pool_mapped(pp->vmm)/1024.0/1024.0);
        }
        return true;
    }
    return false;
}

bool llama_kv_cache::vbr_degrade_next(uint32_t wm_next) {
    while (vbr_degrade_cursor_ < std::min(vbr_degrade_order_.size(), vbr_degrade_limit_)) {
        const auto & st = vbr_degrade_order_[vbr_degrade_cursor_++];

        const auto it = map_layer_ids.find(st.il);
        if (it == map_layer_ids.end()) {
            continue;
        }
        const int32_t ikv = it->second;
        ggml_tensor * t = st.is_v ? layers[ikv].v : layers[ikv].k; // canonical tensor: the type source of truth
        // every VMM pool holding this unit: one under -sm layer, one per device (each with its
        // shard) under -sm tensor. The tier flip is a property of the UNIT — all pools move together.
        const auto & units = vbr_units_of(ikv, st.is_v != 0);
        if (t == nullptr || units.empty()) {
            continue;
        }
        if (!vbr_unit_movable(t->type, st.is_v != 0)) {
            continue; // PINNED unit (explicit non-vbr side): the ladder never touches it
        }
        const ggml_type type_B = vbr_tier_type(st.tier);
        {
            // tier decision on the canonical tensor — relative row sizes are identical on every
            // instance (blocks never straddle the shard split)
            const size_t rA = ggml_row_size(t->type, t->ne[0]);
            const size_t rB = ggml_row_size(type_B,  t->ne[0]);
            if (t->type == type_B || rB >= rA) {
                continue; // not a real degrade from the current tier (e.g. F16 band on a t8 static start)
            }
        }

        for (auto & [pp, ep] : units) {
            vbr_extent  & e   = *ep;
            const int64_t ne0 = e.t->ne[0];
            const size_t rA = ggml_row_size(e.t->type, ne0);

            // every mapped row must become a VALID tier-B row — reads pad n_kv past the used cells, and
            // stale tier-A bytes reinterpreted as B can carry NaN f16 block scales that poison V sums
            const int64_t n_cells = pp->wm_cells;

            // footprint bookkeeping (byte offsets within this tensor's fixed VA slot):
            //   keep      — valid tier-B rows the transcode writes
            //   keep_live — must STAY mapped through this batch: ensure_mapped backs the projected
            //               watermark wm_next at the new tier before the wave's transcode completes
            //   mapped_hi — current mapped high-water for this tensor (tier-A extent, page-rounded);
            //               scrub stops here — pages past it are zero-filled fresh on map
            const vbr_span sp = vbr_span_of(e.t, type_B, n_cells, wm_next, pp->gran);
            const size_t slot      = sp.slot;
            const size_t keep      = sp.keep;
            const size_t keep_live = sp.keep_live;
            const size_t mapped_hi = std::min(slot, (size_t) GGML_PAD(rA * (size_t) n_cells, pp->gran));
            const size_t scrub_end = std::min(mapped_hi, (size_t) GGML_PAD(keep_live, pp->gran));

            if (n_cells > 0) {
                if (pp->backend == nullptr) {
                    pp->backend = pp->be->backend_init(pp->device); // the dedicated per-device side stream
                    GGML_ASSERT(pp->backend != nullptr);
                }
                // first transcode of this wave on this device: make the previous graph's KV writes
                // visible to the side stream — ONE host round-trip per (wave, device); later degrades
                // queue behind it stream-ordered
                if (!pp->wave_pending) {
                    pp->be->sync_device(pp->device);
                }

                // f16 sink-stash: capture rows [0, stash_rows) from the tier-A recon at the FIRST
                // degrade (≈pristine when A is high), then every hop re-encodes those rows from the
                // stash — the sink is the only region both permanently hot and permanently old
                const void * stash_ptr  = nullptr;
                int64_t      stash_rows = 0;
                if (vbr_stash_rows_ > 0) {
                    char * sbase = vbr_stash_ensure(*pp);
                    // the stash is injected into TAPPED-tier encodes with the tap suppressed, so it
                    // must hold tapped-domain rows (V - mu_V). t8 and f16 store FULL-domain — defer
                    // capture until the source is a tapped tier (t4 or lower); the first tapped hop's
                    // recon is the earliest domain-correct snapshot.
                    if (e.stash_valid == 0 && ggml_is_turbo_kv_type(e.t->type) &&
                        e.t->type != GGML_TYPE_TURBO8_0) {
                        e.stash_valid = (uint32_t) std::min<int64_t>(vbr_stash_rows_, n_cells);
                        pp->be->kv_stash_capture(pp->backend, e.t, sbase + e.stash_off,
                                                           e.stash_valid, st.is_v != 0);
                    }
                    stash_ptr  = sbase + e.stash_off;
                    stash_rows = e.stash_valid;
                }

                // S5: transcode + scrub run ASYNC on the side stream; the fence armed at end-of-wave
                // (prepare()) makes the next decode graph GPU-wait on them. The scrub zeroes stale
                // tier-A bytes on kept mapped pages past the new extent — attention pads reads up to
                // 256 rows past the used cells BEFORE those rows are rewritten, and old bytes read as
                // tier B can carry NaN f16 block scales that poison V sums (0*NaN=NaN survives the
                // softmax mask). Zero rows decode benign, matching a static cache.
                const ggml_vbr_transcode_params tp = {
                    /*.src         =*/ e.t,
                    /*.type_B      =*/ type_B,
                    /*.dst         =*/ e.t->data,
                    /*.pool_buf    =*/ pp->buf,
                    /*.n_cells     =*/ n_cells,
                    /*.is_v        =*/ st.is_v != 0,
                    /*.stash_f16   =*/ stash_ptr,
                    /*.stash_rows  =*/ stash_rows,
                    /*.scrub_bytes =*/ scrub_end - keep,
                };
                pp->be->kv_transcode(pp->backend, &tp);
                pp->wave_pending = true;
            }
            // queue the tail release: pages wholly past keep_live return to the pool at the NEXT decode
            // boundary — the in-flight transcode still READS the tier-A extent, which reaches into them
            if (n_cells > 0 && slot > keep_live) {
                pp->unmap_deferred.push_back({ e.byte_off + keep_live, slot - keep_live });
            }

            LLAMA_LOG_INFO("%s: VBR degrade #%zu: %s L%d -> %s (%lld cells transcoding on side stream, "
                    "device %d mapped %.2f MiB pre-release)\n",
                    __func__, vbr_degrade_cursor_, e.t->name, (int) st.il, ggml_type_name(type_B),
                    (long long) n_cells, pp->device, pp->be->vmm_pool_mapped(pp->vmm)/1024.0/1024.0);
        }
        // flip metadata now (host state, consumed at graph BUILD time; shards flip in lockstep);
        // data ptr = fixed VA. The fence guarantees the built graph never RUNS before the bytes
        // are tier B.
        vbr_set_tensor_type(t, st.is_v ? layers[ikv].v_stream : layers[ikv].k_stream, type_B);
        vbr_tier_epoch_++; // fence graph reuse off the old views (type/strides changed in place)
        return true;
    }
    return false;
}

// Permanent transcode oracle (env VBR_TRANSCODE_TEST, armed from apply_ubatch): SELF-CONTAINED —
// synthesize valid turbo8 by encoding a known f32 pattern, then (a) transcode A->A and byte-compare
// the round-trip, (b) transcode A->B twice (separate-dst vs in-place) and require identical bytes
// (the in-place trailing invariant). No live-KV dependency: on this hybrid arch the kv_cache
// instance seen here is not the one the active graph writes. Run with TURBO_MEANSUB_OFF=1 for a
// clean tap-off comparison; VBR_TRANSCODE_TEST_N scales the row count (issues past row 4096 only
// reproduce at scale). fprintf(stderr) so results show regardless of log verbosity.
void llama_kv_cache::vbr_transcode_anchor_test() {
    if (vbr_pools_.empty()) {
        fprintf(stderr, "VBR anchor: no VBR pools, skipping\n");
        return;
    }
    // run once per distinct pool device (multi-GPU: exercise every device's transcode path)
    std::vector<std::pair<int, const ggml_vbr_backend_iface *>> devices;
    for (const auto & p : vbr_pools_) {
        if (p.be == nullptr) {
            continue; // bookkeeping-only pool (static allocation), no backend vtable
        }
        const int dev = p.device >= 0 ? p.device : 0;
        if (std::find_if(devices.begin(), devices.end(),
                [dev](const auto & d) { return d.first == dev; }) == devices.end()) {
            devices.push_back({ dev, p.be });
        }
    }
    for (const auto & [dev, be] : devices) {
    fprintf(stderr, "VBR anchor: device %d\n", dev);
    ggml_backend_t bk = be->backend_init(dev);
    if (!bk) {
        fprintf(stderr, "VBR anchor: backend_init failed\n");
        continue;
    }
    {
        const ggml_type t8 = GGML_TYPE_TURBO8_0;
        int64_t ne0 = 1024;
        for (size_t i = 0; i < layers.size(); ++i) if (layers[i].k && layers[i].k->type == t8) { ne0 = layers[i].k->ne[0]; break; }
        const char *  nenv = getenv("VBR_TRANSCODE_TEST_N");
        const int64_t N  = nenv ? atoll(nenv) : 256;
        const size_t  r8 = ggml_row_size(t8, ne0);

        ggml_init_params ip = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), nullptr, true };
        ggml_context * gctx = ggml_init(ip);
        ggml_tensor * src_f32 = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, ne0, N);
        ggml_tensor * idx     = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, N);
        ggml_tensor * tq      = ggml_new_tensor_2d(gctx, t8, ne0, N);
        ggml_set_name(tq, "cache_k_l3");  // cache_k_ prefix -> K codebook in the encoder
        ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, bk);

        std::vector<float>   hp((size_t) ne0 * N);
        for (size_t i = 0; i < hp.size(); ++i) {
            uint32_t r = (uint32_t) i * 1103515245u + 12345u; hp[i] = (float)(r & 0xFFFF) / 32768.0f - 1.0f; // deterministic non-zero
        }
        std::vector<int32_t> hi(N);
        for (int64_t i = 0; i < N; ++i) hi[i] = (int32_t) i;
        ggml_backend_tensor_set(src_f32, hp.data(), 0, ggml_nbytes(src_f32));
        ggml_backend_tensor_set(idx,     hi.data(), 0, ggml_nbytes(idx));

        ggml_tensor * enc = ggml_set_rows(gctx, tq, src_f32, idx);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, enc);
        ggml_backend_graph_compute(bk, gf);
        ggml_backend_synchronize(bk);

        const size_t bytes = (size_t) N * r8;
        ggml_backend_buffer_t dbuf = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(bk), bytes);
        void * dst_dev = ggml_backend_buffer_get_base(dbuf);
        const ggml_vbr_transcode_params tp_aa = {
            tq, t8, dst_dev, dbuf, N, /*is_v=*/false, nullptr, 0, /*scrub_bytes=*/0,
        };
        be->kv_transcode(bk, &tp_aa);
        ggml_backend_synchronize(bk);

        std::vector<uint8_t> a(bytes), b(bytes);
        ggml_backend_tensor_get(tq, a.data(), 0, bytes);
        { ggml_init_params ip2 = { 2*ggml_tensor_overhead(), nullptr, true }; ggml_context * tc = ggml_init(ip2);
          ggml_tensor * dt = ggml_new_tensor_1d(tc, GGML_TYPE_I8, (int64_t) bytes); dt->data = dst_dev; dt->buffer = dbuf;
          ggml_backend_tensor_get(dt, b.data(), 0, bytes); ggml_free(tc); }
        size_t nz = 0, match = 0; for (size_t i = 0; i < bytes; ++i) if (a[i]) { nz++; if (a[i]==b[i]) match++; }
        fprintf(stderr, "VBR SELFTEST turbo8->turbo8 (synthetic) N=%lld: enc-nonzero %.3f%% identical (%zu/%zu); enc-bytes-nz %zu/%zu\n",
                (long long) N, nz?100.0*(double)match/(double)nz:0.0, match, nz, nz, bytes);

        // A->B DEGRADE + IN-PLACE trailing: turbo8 -> turbo4, separate-dst vs in-place must be IDENTICAL.
        {
            const ggml_type t4 = GGML_TYPE_TURBO4_0;
            const size_t bytesB = (size_t) N * ggml_row_size(t4, ne0);
            ggml_backend_buffer_t sepbuf = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(bk), bytesB);
            void * sep = ggml_backend_buffer_get_base(sepbuf);
            const ggml_vbr_transcode_params tp_sep = {
                tq, t4, sep, sepbuf, N, /*is_v=*/false, nullptr, 0, /*scrub_bytes=*/0,
            };
            be->kv_transcode(bk, &tp_sep);
            ggml_backend_synchronize(bk);

            ggml_init_params ipw = { 2*ggml_tensor_overhead(), nullptr, true };
            ggml_context * wc = ggml_init(ipw);
            ggml_tensor * work = ggml_new_tensor_2d(wc, t8, ne0, N); ggml_set_name(work, "cache_k_l3");
            ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wc, bk);
            ggml_backend_tensor_set(work, a.data(), 0, bytes);  // work := copy of tq (turbo8)
            const ggml_vbr_transcode_params tp_inp = {
                work, t4, work->data, wbuf, N, /*is_v=*/false, nullptr, 0, /*scrub_bytes=*/0,
            };
            be->kv_transcode(bk, &tp_inp); // in-place
            ggml_backend_synchronize(bk);

            std::vector<uint8_t> sb(bytesB), wb(bytesB);
            { ggml_init_params ip3 = { 2*ggml_tensor_overhead(), nullptr, true }; ggml_context * tc = ggml_init(ip3);
              ggml_tensor * dt = ggml_new_tensor_1d(tc, GGML_TYPE_I8, (int64_t) bytesB); dt->data = sep; dt->buffer = sepbuf;
              ggml_backend_tensor_get(dt, sb.data(), 0, bytesB); ggml_free(tc); }
            ggml_backend_tensor_get(work, wb.data(), 0, bytesB);
            size_t same = 0; for (size_t i = 0; i < bytesB; ++i) if (sb[i] == wb[i]) same++;
            fprintf(stderr, "VBR SELFTEST turbo8->turbo4 in-place==separate: %.3f%% (%zu/%zu)\n",
                    100.0*(double)same/(double)bytesB, same, bytesB);
            ggml_backend_buffer_free(sepbuf); ggml_free(wc); if (wbuf) ggml_backend_buffer_free(wbuf);
        }

        // C) PROMOTE (grow, in-place DESCENDING tiles): the degrade cases above never exercise
        //    rB > rA. Degrade the synthetic t8 to t1_tcq (validated direction), then walk the
        //    promote ladder t1 -> t2 -> t3 -> t4 — every hop run twice, separate-dst vs in-place,
        //    which must produce IDENTICAL bytes. K and V variants: separate codebooks, and the V
        //    dequant carries the decode-alpha epilogue. Each hop's in-place result feeds the next,
        //    so later hops double as the multi-hop chain from the live promote-burst repro.
        for (int ivar = 0; ivar < 2; ++ivar) {
            const char * nm = ivar ? "cache_v_l3" : "cache_k_l3";
            const ggml_type ladder[4] = { GGML_TYPE_TURBO1_TCQ, GGML_TYPE_TURBO2_TCQ,
                                          GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO4_0 };
            const size_t r_max = ggml_row_size(ladder[3], ne0);

            // t8 source re-encoded under this variant's codebook name
            ggml_init_params ips = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            ggml_context * sc = ggml_init(ips);
            ggml_tensor * s_f32 = ggml_new_tensor_2d(sc, GGML_TYPE_F32, ne0, N);
            ggml_tensor * s_idx = ggml_new_tensor_1d(sc, GGML_TYPE_I32, N);
            ggml_tensor * s_t8  = ggml_new_tensor_2d(sc, t8, ne0, N);
            ggml_set_name(s_t8, nm);
            ggml_backend_buffer_t sbuf = ggml_backend_alloc_ctx_tensors(sc, bk);
            ggml_backend_tensor_set(s_f32, hp.data(), 0, ggml_nbytes(s_f32));
            ggml_backend_tensor_set(s_idx, hi.data(), 0, ggml_nbytes(s_idx));
            ggml_tensor * s_enc = ggml_set_rows(sc, s_t8, s_f32, s_idx);
            ggml_cgraph * sg = ggml_new_graph(sc);
            ggml_build_forward_expand(sg, s_enc);
            ggml_backend_graph_compute(bk, sg);
            ggml_backend_synchronize(bk);

            // slab plays the VMM slot: sized for the largest tier so in-place grows stay in bounds
            ggml_backend_buffer_t slabbuf = ggml_backend_buft_alloc_buffer(
                    ggml_backend_get_default_buffer_type(bk), r_max * (size_t) N);
            void * slab = ggml_backend_buffer_get_base(slabbuf);
            const ggml_vbr_transcode_params tp_dn = {
                s_t8, ladder[0], slab, slabbuf, N, /*is_v=*/ivar != 0, nullptr, 0, /*scrub_bytes=*/0,
            };
            be->kv_transcode(bk, &tp_dn);
            ggml_backend_synchronize(bk);

            // header contexts: transcode reads type/ne[0]/nb[1]/data/name of src only
            ggml_init_params iph = { 48*ggml_tensor_overhead(), nullptr, true };
            ggml_context * hc = ggml_init(iph);
            auto alias = [&](ggml_type tt) {
                ggml_tensor * x = ggml_new_tensor_2d(hc, tt, ne0, N);
                x->data = slab; x->buffer = slabbuf;
                ggml_set_name(x, nm);
                return x;
            };
            // raw device->host byte copy via a throwaway I8 header (tensor_get needs one)
            auto download = [&](void * base, ggml_backend_buffer_t b, void * dst, size_t nbytes) {
                ggml_tensor * d = ggml_new_tensor_1d(hc, GGML_TYPE_I8, (int64_t) nbytes);
                d->data = base; d->buffer = b;
                ggml_backend_tensor_get(d, dst, 0, nbytes);
            };
            for (int h = 0; h + 1 < 4; ++h) {
                const ggml_type tto  = ladder[h + 1];
                const size_t bytesTo = ggml_row_size(tto, ne0) * (size_t) N;

                ggml_backend_buffer_t hsep = ggml_backend_buft_alloc_buffer(
                        ggml_backend_get_default_buffer_type(bk), bytesTo);
                const ggml_vbr_transcode_params tp_hs = {
                    alias(ladder[h]), tto, ggml_backend_buffer_get_base(hsep), hsep,
                    N, /*is_v=*/ivar != 0, nullptr, 0, /*scrub_bytes=*/0,
                };
                be->kv_transcode(bk, &tp_hs);
                ggml_backend_synchronize(bk);

                const ggml_vbr_transcode_params tp_hi = {
                    alias(ladder[h]), tto, slab, slabbuf,   // dst == src->data -> reverse tiles
                    N, /*is_v=*/ivar != 0, nullptr, 0, /*scrub_bytes=*/0,
                };
                be->kv_transcode(bk, &tp_hi);
                ggml_backend_synchronize(bk);

                std::vector<uint8_t> hb(bytesTo), ib(bytesTo);
                download(ggml_backend_buffer_get_base(hsep), hsep, hb.data(), bytesTo);
                download(slab, slabbuf, ib.data(), bytesTo);
                size_t same = 0, first_bad = bytesTo;
                for (size_t i = 0; i < bytesTo; ++i) {
                    if (hb[i] == ib[i]) { same++; } else if (first_bad == bytesTo) { first_bad = i; }
                }
                fprintf(stderr, "VBR SELFTEST PROMOTE %s %s->%s in-place==separate: %.3f%% (%zu/%zu)%s first-diff byte %zd (row %lld)\n",
                        nm, ggml_type_name(ladder[h]), ggml_type_name(tto),
                        100.0*(double)same/(double)bytesTo, same, bytesTo,
                        same == bytesTo ? "" : " byte-MISMATCH", same == bytesTo ? (ssize_t) -1 : (ssize_t) first_bad,
                        same == bytesTo ? -1LL : (long long)(first_bad / ggml_row_size(tto, ne0)));
                if (same != bytesTo) {
                    // TCQ trellis blocks carry trailing don't-care bits the decode never reads, so a
                    // byte diff is not yet corruption — adjudicate on DEQUANTIZED values instead
                    const size_t fb = (size_t) N * ne0 * sizeof(uint16_t);
                    ggml_backend_buffer_t f1 = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(bk), fb);
                    ggml_backend_buffer_t f2 = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(bk), fb);
                    ggml_tensor * asep = ggml_new_tensor_2d(hc, tto, ne0, N);
                    asep->data = ggml_backend_buffer_get_base(hsep); asep->buffer = hsep;
                    ggml_set_name(asep, nm);
                    be->kv_stash_capture(bk, asep,      ggml_backend_buffer_get_base(f1), N, ivar != 0);
                    be->kv_stash_capture(bk, alias(tto), ggml_backend_buffer_get_base(f2), N, ivar != 0);
                    ggml_backend_synchronize(bk);
                    std::vector<uint16_t> v1(fb/2), v2(fb/2);
                    download(ggml_backend_buffer_get_base(f1), f1, v1.data(), fb);
                    download(ggml_backend_buffer_get_base(f2), f2, v2.data(), fb);
                    size_t vbad = 0; int64_t first_row = -1;
                    for (size_t i = 0; i < v1.size(); ++i) {
                        if (v1[i] != v2[i]) { vbad++; if (first_row < 0) { first_row = (int64_t) (i / (size_t) ne0); } }
                    }
                    fprintf(stderr, "VBR SELFTEST PROMOTE %s %s->%s DEQUANT compare: %s (%zu/%zu f16 values differ, first row %lld)\n",
                            nm, ggml_type_name(ladder[h]), ggml_type_name(tto),
                            vbad == 0 ? "IDENTICAL (slack bits only)" : "VALUE MISMATCH",
                            vbad, v1.size(), (long long) first_row);
                    ggml_backend_buffer_free(f1); ggml_backend_buffer_free(f2);
                }
                // continue the chain from the in-place result (== separate when the hop passes)
                ggml_backend_buffer_free(hsep);
            }
            ggml_free(hc);
            ggml_backend_buffer_free(slabbuf);
            ggml_free(sc);
            if (sbuf) ggml_backend_buffer_free(sbuf);
        }

        ggml_backend_buffer_free(dbuf);
        ggml_free(gctx);
        if (gbuf) ggml_backend_buffer_free(gbuf);
    }
    ggml_backend_free(bk);
    }
}

void llama_kv_cache::kv_bpv_accum(double & bits, double & vals) const {
    // cells are uniform within one cache, so weight each tensor by ne0 x cells; the per-cache
    // ratio reduces to sum(row_bits)/sum(ne0), but the totals let iSWA combine two caches of
    // different sizes correctly
    const double cells = (double) get_size();
    for (const auto & l : layers) {
        for (const ggml_tensor * t : { l.k, l.v }) {
            if (t == nullptr) {
                continue;
            }
            bits += 8.0 * (double) ggml_row_size(t->type, t->ne[0]) * cells;
            vals += (double) t->ne[0] * cells;
        }
    }
}

double llama_kv_cache::kv_bpv() const {
    double bits = 0.0;
    double vals = 0.0;
    kv_bpv_accum(bits, vals);
    return vals > 0.0 ? bits / vals : -1.0;
}

llama_memory_vbr_state_data llama_kv_cache::memory_vbr_state(llama_seq_id seq_id, uint32_t n_tokens_extra) const {
    llama_memory_vbr_state_data st = {};

    // full-reset feasibility: used cells the asking seq does not exclusively own. Cells above
    // used_max_p1 are empty by definition, so the scan is bounded by live occupancy.
    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];
        const uint32_t top = cells.used_max_p1();
        for (uint32_t i = 0; i < top; ++i) {
            if (cells.is_empty(i)) {
                continue;
            }
            if (seq_id < 0 || !(cells.seq_count(i) == 1 && cells.seq_has(i, seq_id))) {
                st.used_cells_other++;
            }
        }
    }

    if (!vbr_vmm_active() || vbr_budget_bytes_ == 0) {
        return st; // no controller: zeros besides the occupancy count
    }
    st.cursor = (int32_t) vbr_degrade_cursor_;

    const uint32_t wm_next = vbr_watermark_cells(n_tokens_extra);

    // deficits: max over pools, exactly like the degrade trigger. raw = configured budget only
    // (page-exact, deterministic — the policy input); clamped = the live budget_eff (telemetry).
    int64_t deficit_raw     = INT64_MIN;
    int64_t deficit_clamped = INT64_MIN;
    std::vector<int64_t> pool_proj(vbr_pools_.size(), 0);
    for (size_t pi = 0; pi < vbr_pools_.size(); ++pi) {
        const auto & p = vbr_pools_[pi];
        if (p.vmm == nullptr) {
            continue;
        }
        pool_proj[pi] = (int64_t) vbr_vmm_projected_bytes(p, wm_next);
        deficit_raw     = std::max(deficit_raw,     pool_proj[pi] - (int64_t) p.budget);
        deficit_clamped = std::max(deficit_clamped, pool_proj[pi] - (int64_t) vbr_budget_eff(p));
    }
    if (deficit_raw == INT64_MIN) {
        return st; // no VMM pools — controller effectively inert
    }
    st.deficit_raw     = deficit_raw;
    st.deficit_clamped = deficit_clamped;

    // bpv_if_degraded: walk the ladder from the CURRENT cursor with the same skip rules as
    // vbr_degrade_next until every pool's RAW projection fits (or the floor clamp stops it) —
    // the aggregate the controller would land at if the deficit were paid by tiers alone.
    // Mirrors the vbr_floor_clamp_order simulation; aggregate basis = VMM-pooled units.
    std::vector<ggml_type> sim(layers.size() * 2, GGML_TYPE_COUNT);
    double  sum_bits = 0.0;
    int64_t sum_vals = 0;
    for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
        for (int side = 0; side < 2; ++side) {
            const ggml_tensor * t = side ? layers[ikv].v : layers[ikv].k;
            if (t == nullptr || !vbr_unit_pooled(ikv, side != 0)) {
                continue;
            }
            sim[ikv*2 + side] = t->type;
            sum_bits += 8.0 * (double) ggml_row_size(t->type, t->ne[0]);
            sum_vals += t->ne[0];
        }
    }
    auto pools_fit = [&]() {
        for (size_t pi = 0; pi < vbr_pools_.size(); ++pi) {
            if (vbr_pools_[pi].vmm != nullptr && pool_proj[pi] > (int64_t) vbr_pools_[pi].budget) {
                return false;
            }
        }
        return true;
    };
    for (size_t i = vbr_degrade_cursor_;
         i < std::min(vbr_degrade_order_.size(), vbr_degrade_limit_) && !pools_fit(); ++i) {
        const auto & stp = vbr_degrade_order_[i];
        const auto it = map_layer_ids.find(stp.il);
        if (it == map_layer_ids.end()) {
            continue;
        }
        const size_t slot = (size_t) it->second * 2 + (stp.is_v ? 1 : 0);
        const ggml_tensor * t = stp.is_v ? layers[it->second].v : layers[it->second].k;
        if (sim[slot] == GGML_TYPE_COUNT || t == nullptr || !vbr_unit_movable(sim[slot], stp.is_v != 0)) {
            continue;
        }
        const ggml_type type_B = vbr_tier_type(stp.tier);
        const size_t rA = ggml_row_size(sim[slot], t->ne[0]);
        const size_t rB = ggml_row_size(type_B,    t->ne[0]);
        if (sim[slot] == type_B || rB >= rA) {
            continue;
        }
        // projection deltas land in EVERY pool holding the unit, priced at each pool's shard width
        for (size_t pi = 0; pi < vbr_pools_.size(); ++pi) {
            const auto & p = vbr_pools_[pi];
            if (p.vmm == nullptr) {
                continue;
            }
            const vbr_extent & e = stp.is_v ? p.v[it->second] : p.k[it->second];
            if (e.t == nullptr) {
                continue;
            }
            const size_t rA_p = ggml_row_size(sim[slot], e.t->ne[0]);
            const size_t rB_p = ggml_row_size(type_B,    e.t->ne[0]);
            pool_proj[pi] += (int64_t) GGML_PAD(rB_p * (size_t) wm_next, p.gran)
                           - (int64_t) GGML_PAD(rA_p * (size_t) wm_next, p.gran);
        }
        sum_bits += 8.0 * ((double) rB - (double) rA);
        sim[slot] = type_B;
    }
    st.bpv_if_degraded = sum_vals > 0 ? sum_bits / (double) sum_vals : 0.0;

    return st;
}

bool llama_kv_cache::get_can_shift() const {
    // VBR VMM v1: build_graph_shift views the FULL kv_size cells — executing it would touch
    // unmapped VA. TODO(S6+): bound the shift views to the mapped watermark instead.
    if (vbr_vmm_active()) {
        return false;
    }
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

std::vector<uint32_t> llama_kv_cache::get_layer_ids() const {
    std::vector<uint32_t> res;
    res.reserve(layers.size());

    for (const auto & layer : layers) {
        res.push_back(layer.il);
    }

    return res;
}

ggml_tensor * llama_kv_cache::get_k_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].k;
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }

    return result;
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    // may be padded for turbo FWHT alignment
    assert(n_embd_k_gqa >= hparams.n_embd_k_gqa(il));

    const uint32_t n_head_kv     = hparams.n_head_kv(il);
    const uint32_t n_embd_head_k = n_embd_k_gqa / n_head_kv;

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            n_embd_head_k, n_head_kv, n_kv, ns,
            ggml_row_size(k->type, n_embd_head_k),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}


ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // use padded head_dim from cache tensor (may be padded for turbo FWHT)
        const uint32_t n_head_kv     = hparams.n_head_kv(il);
        const uint32_t n_embd_head_v = n_embd_v_gqa / n_head_kv;

        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                n_embd_head_v, n_head_kv, n_kv, ns,
                ggml_row_size(v->type, n_embd_head_v),          // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}


ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    // cache head_dim may be padded for turbo FWHT alignment
    const int64_t cache_head = k->ne[0] / n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    // pad per-head to match cache (zeros contribute nothing via Parseval's theorem)
    if (n_embd_head < cache_head) {
        k_cur = ggml_pad(ctx, k_cur, cache_head - n_embd_head, 0, 0, 0);
    }

    const int64_t n_embd_gqa = cache_head * n_head;

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}


ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        // pad per-head for turbo FWHT alignment
        const int64_t cache_head = v->ne[0] / n_head;
        if (n_embd_head < cache_head) {
            v_cur = ggml_pad(ctx, v_cur, cache_head - n_embd_head, 0, 0, 0);
        }
        const int64_t n_embd_gqa_cache = cache_head * n_head;

        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa_cache, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa_cache == v->ne[0]);
            assert(kv_size          == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa_cache, kv_size*n_stream);
        }

        return ggml_set_rows(ctx, v, v_cur, v_idxs);
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}


ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_k) {
        int nrot = 64;

        // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
        do {
            nrot *= 2;
        } while (n_embd_head_k_all % nrot == 0);
        nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_v) {
        int nrot = 64;
        // using smaller rotation matrices for V seems beneficial
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
        //do {
        //    nrot *= 2;
        //} while (hparams.n_embd_head_v() % nrot == 0);
        //nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}


void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
};

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    data[idst + j] = mask_keep;
                }

                continue;
skip:
                data[idst + j] = mask_drop;
            }
        }
    }
}

template<typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<T, causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, is_2d, false>(args, data);
    }
}

template<typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<T, causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, false>(args, data);
    }
}

template<typename T, bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<T, causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, false>(args, data);
    }
}

template<typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data, bool causal_attn) {
    if (causal_attn) {
        set_input_kq_mask_impl<T, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    //const int64_t t_start = ggml_time_us();

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
    };

    if (dst->type == GGML_TYPE_F16) {
        set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) dst->data, causal_attn);
    } else {
        set_input_kq_mask_impl<float>(args, (float *) dst->data, causal_attn);
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        // rotate back
        tmp = ggml_mul_mat_aux(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = ggml_mul_mat_aux(ctx, tmp, rot);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_rot) {
        kv_self->set_input_k_rot(k_rot);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    inp->k_rot = build_input_k_rot(ctx);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, get_size()*n_stream,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, inp->k_rot, rope_factors, freq_base_l, freq_scale_l, il);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    if (vbr_vmm_active()) {
        // settle in-flight degrade waves: the S5 fence only orders graph_compute, not the
        // tensor_get path io.write_tensor uses — an unsettled wave would serialize torn bytes
        // under the already-flipped type
        for (const auto & p : vbr_pools_) {
            if (p.backend != nullptr) {
                ggml_backend_synchronize(p.backend);
            }
            if (p.vmm != nullptr && p.be != nullptr) {
                p.be->sync_device(p.device);
            }
        }
        // a degraded-tier snapshot can never restore (state_read requires the fresh context's
        // entry tiers) — refuse at SAVE time instead of failing the user at load time
        for (const auto & p : vbr_pools_) {
            if (p.vmm == nullptr) {
                continue;
            }
            for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
                for (int side = 0; side < 2; ++side) {
                    const vbr_extent  & e = side ? p.v[ikv] : p.k[ikv];
                    const ggml_tensor * t = side ? layers[ikv].v : layers[ikv].k;
                    if (e.t != nullptr && t != nullptr && t->type != e.type0) {
                        throw std::runtime_error(
                            "cannot serialize a dynamic-VBR KV cache after tier degrades — the "
                            "snapshot could never restore; save before the budget triggers, or "
                            "run without dynamic VBR");
                    }
                }
            }
        }
    }

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id);
        if (res && seq_id == -1 && vbr_vmm_active()) {
            // the whole-cache branch positions cells directly (no apply_ubatch), so nothing has
            // grown the VMM physical backing yet — state_read_data would write into unmapped VA
            vbr_vmm_ensure_mapped();
        }
        res = res && state_read_data(io, strm, cell_count, sinfo);

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (hparams.n_pos_per_embd() > 1) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[cr.strm];

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }

    // Turbo TCQ safety footer: embed codebook fingerprint so loading a cache
    // saved with a different TURBO_TCQ_CB/CB2 is detected at load time.
    bool has_tcq = false;
    for (const auto & layer : layers) {
        if (ggml_type_is_turbo_tcq(layer.k_stream[cr.strm]->type)) { has_tcq = true; break; }
        auto * v = layer.v_stream[cr.strm];
        if (v && ggml_type_is_turbo_tcq(v->type)) { has_tcq = true; break; }
    }
    if (has_tcq) {
        const uint32_t magic = 0x54514346; // "TQCF" — TurboQuant Cache Fingerprint
        const uint32_t fp    = turbo_tcq_fingerprint();
        io.write(&magic, sizeof(magic));
        io.write(&fp,    sizeof(fp));
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        sinfo = find_slot(ubatch, false);
        if (sinfo.empty()) {
            LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
            return false;
        }

        // TODO: we cannot yet restore llama_kv_cell_ext as the apply_ubatch() does not support it yet
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    auto & cells = v_cells[strm];

    uint32_t v_trans;
    uint32_t n_layer;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            if (sinfo.is_contiguous()) {
                // Fast path: contiguous cells, single memcpy
                io.read_tensor(k, sinfo.head() * k_size_row, cell_count * k_size_row);
            } else {
                // Slow path: scatter to non-contiguous positions
                for (uint32_t i = 0; i < cell_count; ++i) {
                    const size_t dst_offset = sinfo.idxs[0][i] * k_size_row;
                    io.read_tensor(k, dst_offset, k_size_row);
                }
            }
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells, single memcpy
                    io.read_tensor(v, sinfo.head() * v_size_row, cell_count * v_size_row);
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        const size_t dst_offset = sinfo.idxs[0][i] * v_size_row;
                        io.read_tensor(v, dst_offset, v_size_row);
                    }
                }
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells
                    const uint32_t h = sinfo.head();
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const size_t dst_offset = (h + j * cells.size()) * v_size_el;
                        io.read_tensor(v, dst_offset, cell_count * v_size_el);
                    }
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        for (uint32_t i = 0; i < cell_count; ++i) {
                            const size_t dst_offset = (sinfo.idxs[0][i] + j * cells.size()) * v_size_el;
                            io.read_tensor(v, dst_offset, v_size_el);
                        }
                    }
                }
            }
        }
    }

    // Turbo TCQ safety: verify codebook fingerprint matches current process.
    bool has_tcq = false;
    for (const auto & layer : layers) {
        if (ggml_type_is_turbo_tcq(layer.k_stream[strm]->type)) { has_tcq = true; break; }
        auto * v = layer.v_stream[strm];
        if (v && ggml_type_is_turbo_tcq(v->type)) { has_tcq = true; break; }
    }
    if (has_tcq) {
        uint32_t magic_ref = 0;
        io.read(&magic_ref, sizeof(magic_ref));
        if (magic_ref != 0x54514346) { // "TQCF"
            LLAMA_LOG_ERROR("%s: turbo TCQ cache file missing codebook fingerprint — "
                            "file may have been saved by an older build without TCQ safety checks\n", __func__);
            return false;
        }
        uint32_t fp_ref = 0;
        io.read(&fp_ref, sizeof(fp_ref));
        const uint32_t fp_now = turbo_tcq_fingerprint();
        if (fp_ref != fp_now) {
            LLAMA_LOG_ERROR("%s: turbo TCQ codebook mismatch — cache was saved with fingerprint "
                            "0x%08X but current TURBO_TCQ_CB/CB2 gives 0x%08X. "
                            "Set the same codebook env vars as when the cache was created.\n",
                            __func__, fp_ref, fp_now);
            return false;
        }
        LLAMA_LOG_INFO("%s: turbo TCQ codebook fingerprint verified (0x%08X)\n", __func__, fp_ref);
    }

    return true;
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    n_kv = kv->get_n_kv(sinfos[i_cur]);

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint64_t llama_kv_cache_context::get_vbr_epoch() const {
    return kv->vbr_tier_epoch();
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}


ggml_tensor * llama_kv_cache_context::get_turbo_rotation() const {
    return kv->get_turbo_rotation();
}

ggml_tensor * llama_kv_cache_context::get_turbo_rotation_inv() const {
    return kv->get_turbo_rotation_inv();
}

ggml_tensor * llama_kv_cache_context::get_turbo_rot_forward() const {
    return kv->get_turbo_rotation();
}

ggml_tensor * llama_kv_cache_context::get_turbo_rot_inverse() const {
    return kv->get_turbo_rotation_inv();
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}


ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}


ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}


void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}
