#include "llama-kv-cache.h"

#include "llama-vbr-artifact-capture.h"
#include "llama-vbr-explicit-capture.h"
#include "llama-vbr-artifact-validate.h"
#include "llama-vbr-config.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-sha256.h"
#include "llama-vbr-identity-digest.h"
#include "llama-vbr-physical.h"
#include "llama-vram-demand.h"
#include "llama-vram-ledger.h"
#include "ggml-turbo-meansub.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <set>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Dynamic-VBR degrade tier ladder and measured price order (generated table).
enum vbr_tier : uint8_t {
    VBR_TIER_T8,
    VBR_TIER_T4,
    VBR_TIER_T3_TCQ,
    VBR_TIER_T2_TCQ,
    VBR_TIER_T1_TCQ,
    VBR_TIER_COUNT,
};
bool llama_kv_cache::vbr_hard_seal_classify(
        vbr_hard_seal_classification & out) const noexcept {
    return vbr_classify_hard_seal(
        vbr_degrade_order_, VBR_TIER_T4, out);
}

void llama_kv_cache::vbr_hard_seal_guard_set(vbr_hard_seal_guard guard) {
    vbr_hard_seal_guard_ = std::move(guard);
    vbr_hard_seal_blocked_ = false;
    vbr_hard_seal_evidence_.clear();
    if (vbr_hard_seal_guard_) {
        vbr_hard_seal_deferred_.reserve(vbr_degrade_order_.size());
        vbr_hard_seal_attempted_.assign(vbr_degrade_order_.size(), 0);
    } else {
        vbr_hard_seal_deferred_.clear();
        vbr_hard_seal_attempted_.clear();
    }
}

bool llama_kv_cache::vbr_hard_seal_blocked_take(bool decode_failed) {
    return vbr_hard_seal_take_decode_terminal(
        decode_failed, vbr_hard_seal_blocked_);
}

void llama_kv_cache::vbr_hard_seal_evidence_take(
        std::vector<vbr_hard_seal_subject> & out) {
    out.insert(out.end(), vbr_hard_seal_evidence_.begin(),
            vbr_hard_seal_evidence_.end());
    vbr_hard_seal_evidence_.clear();
}

void llama_kv_cache::vbr_hard_seal_evidence_record(size_t order_ordinal) {
    if (order_ordinal >= vbr_degrade_order_.size()) {
        return;
    }
    const auto & step = vbr_degrade_order_[order_ordinal];
    const vbr_hard_seal_subject evidence {
        step.il, step.is_v != 0, order_ordinal,
    };
    if (std::find(vbr_hard_seal_evidence_.begin(),
            vbr_hard_seal_evidence_.end(), evidence) ==
            vbr_hard_seal_evidence_.end()) {
        vbr_hard_seal_evidence_.push_back(evidence);
    }
}

bool llama_kv_cache::vbr_hard_seal_step_blocked(
        size_t order_ordinal,
        vbr_hard_seal_consult_session & session) const {
    if (!vbr_hard_seal_guard_) {
        return false;
    }

    // The common lifecycle-on/no-hard-lease case stops before classifying the
    // ladder, walking cells, or hashing live identities.
    if (!session.any_hard_sampled) {
        session.any_hard_sampled = true;
        session.any_hard = vbr_hard_seal_guard_.any_hard_lease();
    }
    if (!session.any_hard) {
        return false;
    }
    if (!session.classified) {
        session.classified = true;
        if (!vbr_hard_seal_classify(session.classification)) {
            session.classification_failed = true;
            session.verdict_sampled = true;
            session.verdict = vbr_hard_seal_guard_result::hard_lease_blocked;
            return true;
        }
    }
    if (session.classification_failed) {
        return true;
    }
    const auto * protected_step = vbr_hard_seal_subject_for_step(
        session.classification, order_ordinal);
    if (protected_step == nullptr) {
        return false;
    }

    if (!session.ranges_built) {
        session.ranges_built = true;
        std::map<llama_seq_id, std::pair<llama_pos, llama_pos>> occupied;
        for (const auto & cells : v_cells) {
            for (uint32_t cell = 0; cell < cells.size(); ++cell) {
                if (cells.is_empty(cell)) {
                    continue;
                }
                const llama_pos pos = cells.pos_get(cell);
                cells.seq_for_each(cell, [&](llama_seq_id sequence) {
                    auto [it, inserted] = occupied.emplace(
                        sequence, std::make_pair(pos, pos));
                    if (!inserted) {
                        it->second.first = std::min(it->second.first, pos);
                        it->second.second = std::max(it->second.second, pos);
                    }
                });
            }
        }
        session.ranges.reserve(occupied.size());
        for (const auto & [sequence, extent] : occupied) {
            if (extent.first < 0 || extent.second < extent.first ||
                static_cast<uint64_t>(extent.second - extent.first) >= UINT32_MAX) {
                // Fail closed if live geometry cannot be represented by the
                // range-qualified lease door.
                session.verdict_sampled = true;
                session.verdict =
                    vbr_hard_seal_guard_result::hard_lease_blocked;
                return true;
            }
            session.ranges.push_back({ sequence, static_cast<uint32_t>(extent.first),
                static_cast<uint32_t>(extent.second - extent.first + 1) });
        }
    }
    const auto verdict = vbr_hard_seal_guard_.inspect(
        *protected_step, session.ranges);
    // The installed guard evaluates occupied lease ranges, not layer/side.
    // Subject-uniformity is the contract that makes policy filtering and the
    // transaction-boundary recheck consistent across custom ladder orders.
    if (session.verdict_sampled) {
        GGML_ASSERT(session.verdict == verdict);
    } else {
        session.verdict_sampled = true;
        session.verdict = verdict;
    }
    return verdict == vbr_hard_seal_guard_result::hard_lease_blocked;
}

// a type the degrade ladder can move: the five turbo tiers plus F16, which is the default dynamic
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

struct llama_kv_cache::vbr_shared_scratch_registry {
    static constexpr size_t no_slot = (size_t) -1;

    struct layer_slots {
        size_t k = no_slot;
        size_t v = no_slot;
    };

    struct consumer {
        uint64_t id = 0;
        const ggml_vbr_backend_iface * be = nullptr;
        ggml_backend_t compute_backend = nullptr;
        int device = -1;
        std::vector<layer_slots> layers;
    };

    std::mutex mutex;
    uint64_t next_id = 1;
    std::vector<consumer> consumers;
};

llama_kv_cache::vbr_shared_scratch_registration::vbr_shared_scratch_registration(
        const std::shared_ptr<vbr_shared_scratch_registry> & registry,
        uint64_t id) : registry(registry), id(id) {
}

llama_kv_cache::vbr_shared_scratch_registration::~vbr_shared_scratch_registration() {
    reset();
}

llama_kv_cache::vbr_shared_scratch_registration::vbr_shared_scratch_registration(
        vbr_shared_scratch_registration && other) noexcept :
    registry(std::move(other.registry)), id(other.id) {
    other.id = 0;
}

llama_kv_cache::vbr_shared_scratch_registration &
llama_kv_cache::vbr_shared_scratch_registration::operator=(
        vbr_shared_scratch_registration && other) noexcept {
    if (this != &other) {
        reset();
        registry = std::move(other.registry);
        id = other.id;
        other.id = 0;
    }
    return *this;
}

void llama_kv_cache::vbr_shared_scratch_registration::reset() {
    if (id == 0) {
        return;
    }
    if (const auto owner = registry.lock()) {
        std::lock_guard<std::mutex> lock(owner->mutex);
        const auto it = std::find_if(owner->consumers.begin(), owner->consumers.end(),
                [&](const vbr_shared_scratch_registry::consumer & c) { return c.id == id; });
        if (it != owner->consumers.end()) {
            owner->consumers.erase(it);
        }
    }
    registry.reset();
    id = 0;
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
    FILE * f = fopen(path, "rb");
    if (!f) { return 0; }
    float buf[512];
    size_t n = fread(buf, sizeof(float), n_floats, f);
    fclose(f);
    if (n != n_floats) { return 0; }
    return llama_crc32((const uint8_t *) buf, n_floats * sizeof(float));
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

static const ggml_vbr_cross_domain_iface_v1 * llama_vbr_cross_domain_iface_for_buft(
        ggml_backend_buffer_type_t buft) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return nullptr;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }
    const auto get = (ggml_backend_vbr_cross_domain_iface_v1_fn_t)
        ggml_backend_reg_get_proc_address(reg, GGML_VBR_CROSS_DOMAIN_IFACE_V1_PROC);
    const auto * iface = get != nullptr ? get() : nullptr;
    if (iface == nullptr ||
        iface->abi_version != GGML_VBR_CROSS_DOMAIN_IFACE_V1_VERSION ||
        iface->struct_size != sizeof(ggml_vbr_cross_domain_iface_v1)) {
        return nullptr;
    }
    return iface;
}

// One VBR-capable device behind a KV buffer type: the backend vtable, the vtable's device
// ordinal, and the device's own (simple) buffer type.
struct llama_vbr_dev {
    const ggml_vbr_backend_iface * be     = nullptr;
    const ggml_vbr_cross_domain_iface_v1 * cross_be = nullptr;
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
        d.cross_be = llama_vbr_cross_domain_iface_for_buft(d.buft);
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

bool llama_vbr_resolve_layer_schedule(
        const char * env,
        std::string & schedule,
        std::string & source) {
    schedule.clear();
    source = "inline";
    if (!env || !env[0]) {
        return true;
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
            source = path;
            return false;
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        schedule = ss.str();
        source = path;
        return true;
    }

    schedule = std::move(value);
    return true;
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
    std::string schedule;
    llama_vbr_resolve_layer_schedule(env, schedule, src);
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

// Logical bytes in a cache tensor's fixed VA slot. Physical mapping and tail release must use
// the page-padded span: the VMM backend maps intersecting chunks but unmaps only fully-contained
// chunks, so an interval ending at this raw length cannot release a terminal partial page.
static size_t vbr_slot_bytes(const ggml_tensor * t) {
    return (size_t) ggml_row_size(GGML_TYPE_F16, t->ne[0]) * t->ne[1] * t->ne[2];
}

static size_t vbr_slot_span(const ggml_tensor * t, size_t gran) {
    return GGML_PAD(vbr_slot_bytes(t), gran);
}

// byte spans of one (pool, extent) unit at tier type_B: how much must stay resident (keep),
// how far the live watermark extends it (keep_live), and the page-padded map target (keep_pad).
// One computation shared by the promote hysteresis/map/transcode phases and the degrade path.
struct vbr_span {
    size_t slot, keep, keep_live, keep_pad;
};
static vbr_span vbr_span_of(const ggml_tensor * t, ggml_type type_B, int64_t n_cells,
                            uint32_t wm_next, size_t gran) {
    const size_t rB           = ggml_row_size(type_B, t->ne[0]);
    const size_t logical_slot = vbr_slot_bytes(t);
    const size_t slot         = vbr_slot_span(t, gran);
    const size_t keep      = rB * (size_t) std::max<int64_t>(n_cells, 1);
    const size_t keep_live = std::min(logical_slot, std::max(keep, rB * (size_t) wm_next));
    const size_t keep_pad  = std::min(slot, (size_t) GGML_PAD(keep_live, gran));
    return { slot, keep, keep_live, keep_pad };
}

struct vbr_scrub_span {
    size_t keep;
    size_t scrub_end;
};

// One page-tail scrub calculation shared by live degrade, atomic tree shed,
// and downward import. The callers retain their local failure policy: live
// controller paths assert an impossible geometry, while artifact import
// rejects malformed/overflowing evidence recoverably.
static bool vbr_scrub_span_of(
        const ggml_tensor * t,
                  ggml_type type_A,
                    int64_t n_cells,
           const vbr_span & span,
                     size_t gran,
           vbr_scrub_span & result) {
    if (t == nullptr || n_cells < 0 || gran == 0) {
        return false;
    }
    const size_t row_a = ggml_row_size(type_A, t->ne[0]);
    const size_t cells = size_t(n_cells);
    if ((row_a != 0 && cells > SIZE_MAX/row_a) ||
        row_a*cells > SIZE_MAX-(gran - 1) ||
        span.keep_live > SIZE_MAX-(gran - 1)) {
        return false;
    }
    const size_t mapped_hi = std::min(
        span.slot, (size_t) GGML_PAD(row_a*cells, gran));
    const size_t scrub_end = std::min(
        mapped_hi, (size_t) GGML_PAD(span.keep_live, gran));
    if (scrub_end < span.keep) {
        return false;
    }
    result = { span.keep, scrub_end };
    return true;
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
    const llama_memory_vbr_params & vbr,
             const char *   name_tag) :
    model(model), hparams(hparams), vbr_params_(vbr), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : kv_cells_make(unified ? 1 : n_seq_max, kv_size)),
    v_cells(*v_cells_impl) {

    name_tag = name_tag ? name_tag : "";

    // Construct eagerly so multiple share-linked contexts can register without racing a lazy
    // owner-side pointer initialization. The registry itself stays empty for ordinary caches.
    vbr_shared_scratch_registry_ = std::make_shared<vbr_shared_scratch_registry>();

    // A constructor failure does not run ~llama_kv_cache(), while VMM pools are raw backend
    // resources owned outside the ggml buffers. Keep the normal teardown armed until every
    // initialization step (including model-aware floor validation) has succeeded.
    auto construction_rollback = std::unique_ptr<llama_kv_cache, std::function<void(llama_kv_cache *)>>(
        this, [](llama_kv_cache * cache) { cache->vbr_release_resources(); });

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
    // Kept parallel to layers[] so scratch registration can select only tensors actually
    // aliased from mem_other. Local f16 tensors must remain zero-cost.
    std::vector<bool> layer_is_shared;

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
                layer_is_shared.push_back(true);

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
            // Turbo types have no CPU vec_dot kernel. A movable dynamic-VBR side also cannot
            // degrade on the host because host buffers have no VMM pool. Pin only those sides
            // at q8_0; preserve an explicitly pinned f16/bf16/q8_0 side in a mixed -ct config.
            // The no-alloc fit construction takes this same path, so its host-memory price and
            // the real allocation stay identical.
            if (cpu_bound_kv) {
                const auto cpu_type = [&](ggml_type type, bool pinned) {
                    const bool needs_q8 = ggml_type_is_turbo(type) ||
                        (vbr_params_.dynamic && !pinned && type == GGML_TYPE_F16);
                    return needs_q8 ? GGML_TYPE_Q8_0 : type;
                };
                const ggml_type cpu_type_k = cpu_type(layer_type_k, vbr_params_.pin_k);
                const ggml_type cpu_type_v = cpu_type(layer_type_v, vbr_params_.pin_v);
                const bool pinned_to_q8 = cpu_type_k != layer_type_k || cpu_type_v != layer_type_v;
                layer_type_k = cpu_type_k;
                layer_type_v = cpu_type_v;
                if (pinned_to_q8) {
                    static bool warned = false;
                    if (!warned) {
                        LLAMA_LOG_WARN("llama_kv_cache: CPU-bound movable KV sides pinned at q8_0 "
                                "(partial offload; excluded from the VBR degrade ladder)\n");
                        warned = true;
                    }
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

        has_k && ggml_format_name(k, "cache_%sk_l%d_ms%d", name_tag, il, hparams.turbo_meansub_id);
        has_v && ggml_format_name(v, "cache_%sv_l%d_ms%d", name_tag, il, hparams.turbo_meansub_id);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_stream, v_stream,
                { hparams.turbo_meansub_id, (int) il } });
        layer_is_shared.push_back(false);

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

    // Back the dynamic-VBR KV context with a VMM pool: one VA reservation, each
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
            // Host KV from partial offload is legal under dynamic VBR: movable sides were
            // pinned at q8_0 by the CPU fallback above, while explicitly pinned sides kept
            // their requested type. They live in system memory outside the fit's VRAM budget,
            // and the degrade walk skips units with no pool — static allocation is correct.
            if (ggml_backend_buft_is_host(bft)) {
                LLAMA_LOG_WARN("%s: dynamic VBR with partial offload: CPU-bound KV layers stay "
                        "static (movable sides use q8_0; explicit side pins keep their type); "
                        "only GPU-resident layers degrade\n", __func__);
                return nullptr;
            }
            // For device KV without VBR backend support (the meta tensor-parallel buft),
            // falling back silently would be a trap: the fit pass priced this KV at the floor
            // tier (1.25 bits/value) but a static fallback stays at the entry tier — up to
            // 12.8x the budgeted VRAM with no degrade possible, an OOM at depth on any
            // fitted config.
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

        // Resolve every simple KV device to this context's already-existing compute backend
        // before entering the meta allocator's C callback. A missing binding is a construction
        // error, but throwing it through that callback would bypass the meta allocator's NULL
        // failure path and its rollback of already-installed device buffers.
        std::vector<ggml_backend_t> compute_backends(devs.size(), nullptr);
        for (size_t i = 0; i < devs.size(); ++i) {
            compute_backends[i] = vbr_params_.compute_backend_for_buft
                    ? vbr_params_.compute_backend_for_buft(devs[i].buft)
                    : nullptr;
            if (compute_backends[i] == nullptr) {
                throw std::runtime_error(format(
                        "VBR could not bind KV buffer type %s to this context's compute backend",
                        ggml_backend_buft_name(devs[i].buft)));
            }
        }
        // Keep all potentially-throwing vector growth outside the C callback and before any
        // device pool/buffer has been acquired.
        vbr_pools_.reserve(vbr_pools_.size() + devs.size());

        // lay out + place ONE device's tensors into a fresh VMM pool. `cc` holds either the KV
        // context itself (plain device buft) or one device's shard tensors (meta buft under
        // -sm tensor) — the walk is identical: every non-view tensor gets a page-aligned VA
        // slot; cache tensors are sized for the max tier so a later tier change never moves them.
        auto vmm_alloc_ctx = [&](size_t idev, ggml_context * cc) -> ggml_backend_buffer_t {
            const llama_vbr_dev & d = devs[idev];
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
                off += GGML_PAD(slot, gran);
            }
            const size_t va_size = GGML_PAD(off, gran);

            // Resolve/copy metadata while no device resource needs exception cleanup.
            std::string busid = "-";
            if (ggml_backend_dev_t bdev = ggml_backend_buft_get_device(d.buft)) {
                ggml_backend_dev_props bprops;
                ggml_backend_dev_get_props(bdev, &bprops);
                if (bprops.device_id != nullptr) {
                    busid = bprops.device_id;
                }
            }

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
            p.cross_be    = d.cross_be;
            p.compute_backend = compute_backends[idev];
            p.vmm         = pool;
            p.device      = device;
            p.gran        = gran;
            p.mapped_base = be->vmm_pool_mapped(pool);
            // co-tenancy: resolve the PCI bus id eagerly — p.backend stays null until the
            // first degrade wave arms the side stream, far too late for marker publication
            p.busid = std::move(busid);
            vbr_pools_.push_back(std::move(p));
            LLAMA_LOG_INFO("%s: VBR VMM pool #%zu: %.2f MiB VA reserved (device %d, %zu KiB pages), %.2f MiB mapped up front\n",
                    __func__, vbr_pools_.size() - 1, va_size/1024.0/1024.0, device, gran/1024,
                    vbr_pools_.back().mapped_base/1024.0/1024.0);
            return b;
        };

        if (!ggml_backend_buft_is_meta(bft)) {
            return vmm_alloc_ctx(0, c);
        }

        // -sm tensor: the meta backend shards every KV tensor per device (axis-0, head-aligned —
        // see llama_meta_device_get_split_state); allocate one VMM pool per simple device and
        // hand each device's shard context to the same layout routine. The meta buffer wraps the
        // per-device pool buffers so graph building sees ordinary meta tensors.
        const size_t pools_before = vbr_pools_.size();
        // std::function bridges the capturing lambda across the C callback (alive only for this call)
        std::function<ggml_backend_buffer_t(size_t, ggml_context *)> alloc_one =
            [&](size_t i, ggml_context * sctx) { return vmm_alloc_ctx(i, sctx); };
        ggml_backend_buffer_t buf = ggml_backend_meta_alloc_ctx_tensors_from_buft_ext(c, bft,
            [](size_t i, ggml_backend_buffer_type_t /*simple_buft*/, ggml_context * sctx, void * u) -> ggml_backend_buffer_t {
                // Never unwind C-style allocator frames with a C++ exception. Returning NULL
                // makes the meta allocator free all simple buffers installed by earlier calls;
                // the VMM-pool rollback immediately below then releases their VA reservations.
                try {
                    return (*(std::function<ggml_backend_buffer_t(size_t, ggml_context *)> *) u)(i, sctx);
                } catch (...) {
                    return nullptr;
                }
            }, &alloc_one);
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
            // co-tenancy hold-aware alloc (a failing ask lands as the claim's one allowed
            // est_partial upward revision)
            buf = llama_vram_hold_alloc_ctx_tensors(ctx.get(), buft);
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

    GGML_ASSERT(layer_is_shared.size() == layers.size());

    // A shared-KV cache executes attention using this context's backend objects while its
    // layer entries alias tensors owned (and tier-mutated) by another context. Register those
    // aliases as scratch-only bindings. This restores the boundary-reserve invariant without
    // creating a drafter-side VMM pool/controller/ledger participant.
    if (other && !hparams.no_alloc) {
        auto find_binding = [&](ggml_backend_buffer_t buf) -> vbr_shared_scratch_binding * {
            for (auto & b : vbr_shared_scratch_bindings_) {
                if (b.buf == buf) {
                    return &b;
                }
            }
            return nullptr;
        };

        for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
            if (!layer_is_shared[ikv]) {
                continue;
            }
            for (int side = 0; side < 2; ++side) {
                ggml_tensor * t = side ? layers[ikv].v : layers[ikv].k;
                if (t == nullptr || t->buffer == nullptr || t->data == nullptr) {
                    continue;
                }
                // Tensor-split shared KV is rejected above, so aliases are simple device
                // buffers. Keep this assertion beside the registration contract so future
                // relaxation cannot silently bind a meta tensor to the wrong backend.
                GGML_ASSERT(!ggml_backend_buffer_is_meta(t->buffer));

                vbr_shared_scratch_binding * b = find_binding(t->buffer);
                if (b == nullptr) {
                    const auto devs = llama_vbr_backend_devs_for_buft(
                            ggml_backend_buffer_get_type(t->buffer));
                    if (devs.empty()) {
                        // CPU/non-Turbo aliases have no dequant scratch backend and cannot
                        // activate the CUDA TurboQuant materialization path.
                        continue;
                    }
                    GGML_ASSERT(devs.size() == 1);
                    vbr_shared_scratch_binding fresh;
                    fresh.buf    = t->buffer;
                    fresh.be     = devs[0].be;
                    fresh.device = devs[0].device;
                    fresh.compute_backend = vbr_params_.compute_backend_for_buft
                            ? vbr_params_.compute_backend_for_buft(
                                    ggml_backend_buffer_get_type(t->buffer))
                            : nullptr;
                    if (fresh.compute_backend == nullptr) {
                        throw std::runtime_error(format(
                                "shared-KV scratch could not bind tensor buffer type %s to this context's compute backend",
                                ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer))));
                    }
                    fresh.k.assign(layers.size(), nullptr);
                    fresh.v.assign(layers.size(), nullptr);
                    vbr_shared_scratch_bindings_.push_back(std::move(fresh));
                    b = &vbr_shared_scratch_bindings_.back();
                }
                (side ? b->v : b->k)[ikv] = t;
            }
        }

        for (size_t i = 0; i < vbr_shared_scratch_bindings_.size(); ++i) {
            const auto & b = vbr_shared_scratch_bindings_[i];
            LLAMA_LOG_INFO("%s: shared-KV scratch binding #%zu (device %d, backend %s)\n",
                    __func__, i, b.device, ggml_backend_name(b.compute_backend));
            vbr_shared_scratch_registrations_.push_back(other->vbr_shared_scratch_register(b));
        }
    }

    // Record per-(layer,side) dynamic-VBR descriptors over the just-placed KV tensors.
    // Pure bookkeeping: allocation behavior above is unchanged. The runtime controller uses
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
                        // Even bookkeeping-only (static, non-VMM) pools need the backend
                        // vtable + device ordinal for the boundary-time dequant-scratch reserve.
                        // inst is always a simple (non-meta) buffer here, so the resolver
                        // returns 0 or 1 entries; empty (no turbo support) leaves be null and
                        // the reserve loop skips the pool — its types can never be turbo anyway.
                        const auto pdevs = llama_vbr_backend_devs_for_buft(ggml_backend_buffer_get_type(inst->buffer));
                        if (!pdevs.empty()) {
                            p.be     = pdevs[0].be;
                            p.cross_be = pdevs[0].cross_be;
                            p.device = pdevs[0].device;
                            p.compute_backend = vbr_params_.compute_backend_for_buft
                                    ? vbr_params_.compute_backend_for_buft(
                                            ggml_backend_buffer_get_type(inst->buffer))
                                    : nullptr;
                            if (p.compute_backend == nullptr) {
                                throw std::runtime_error(format(
                                        "VBR could not bind KV buffer type %s to this context's compute backend",
                                        ggml_backend_buft_name(ggml_backend_buffer_get_type(inst->buffer))));
                            }
                        }
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
        // pool's high-water so the free region [used, size) never overlaps them.
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
        // Arm the decode-time degrade controller (VMM mode only). Inputs come from
        // cparams (llama_memory_vbr_params, threaded through create_memory): budget_bytes is
        // either the explicit --vbr-vram value or the fit pass's auto budget; min_bits is the
        // --vbr-floor aggregate clamp (see vbr_floor_clamp_order). VBR_BUDGET_MIB / VBR_MIN_BITS
        // env remain developer overrides; VBR_STASH_ROWS / VBR_DEGRADE_ORDER are direct
        // experiment overrides.
        if (vbr_vmm_active()) {
            vbr_load_degrade_order();
            vbr_capture_retier_deferred_.reserve(vbr_degrade_order_.size());
            vbr_capture_retier_attempted_.assign(
                vbr_degrade_order_.size(), 0);
            vbr_capture_unit_attempt_boundary_.assign(
                layers.size() * 2, UINT64_MAX);
            // co-tenancy band cap: demand-driven sheds may only spend the leading f16->t8
            // band of the price order — the one cheap AND domain-reversible rung (sub-t8
            // sheds imprint irreversible re-encode error into existing tokens). A custom
            // VBR_DEGRADE_ORDER carries no band guarantee, so it disables demand shedding.
            t8_band_end_ = 0;
            if (getenv("VBR_DEGRADE_ORDER") == nullptr) {
                while (t8_band_end_ < vbr_degrade_order_.size() &&
                       vbr_degrade_order_[t8_band_end_].tier == VBR_TIER_T8) {
                    t8_band_end_++;
                }
            }
            // consent comes ONLY from the typed flag (or its documented LLAMA_ARG env,
            // which sets min_bits_explicit through the arg handler) — the raw VBR_MIN_BITS
            // developer override still moves the floor VALUE but never grants peer-yield
            // consent (bare presence of a debug env must not consent to sub-t8 loss)
            vbr_floor_typed_ = vbr_params_.min_bits_explicit;
            LLAMA_LOG_INFO("%s: co-tenancy: f16->t8 band = %zu of %zu order steps%s%s\n",
                    __func__, t8_band_end_, vbr_degrade_order_.size(),
                    t8_band_end_ == 0 ? " (demand shedding disabled)" : "",
                    t8_band_end_ != 0 && vbr_floor_typed_
                        ? " — explicit floor: peer yield consented to the floor" : "");
            vbr_floor_clamp_order();
            vbr_budget_bytes_    = (size_t) vbr_params_.budget_bytes;
            vbr_budget_explicit_ = vbr_params_.budget_explicit;
            if (const char * env = getenv("VBR_BUDGET_MIB")) {
                vbr_budget_bytes_    = (size_t) strtoull(env, nullptr, 10) * 1024 * 1024;
                vbr_budget_explicit_ = true; // forced-budget instrumentation must never grow
            }
            // Deterministic freeze is test/gating only (see header). Read after the budget
            // is final so the explicit-budget precondition is decided. Skips the vbr_budget_eff live
            // clamp + the ledger; a fixed budget + no clamp makes degrade waves a pure function of
            // occupancy (deterministic "scripted" waves). No effect on any production run.
            vbr_freeze_ = turbo_vbr_env_enabled("VBR_FREEZE");
            const bool preserve_empty_tiers =
                turbo_vbr_env_enabled("VBR_FREEZE_PRESERVE_EMPTY_TIERS");
            vbr_freeze_preserve_empty_tiers_ = vbr_freeze_ && preserve_empty_tiers;
            if (vbr_freeze_) {
                LLAMA_LOG_INFO("%s: VBR_FREEZE active — live-VRAM clamp + ledger disabled "
                        "(deterministic tier schedule; test/gating only)\n", __func__);
                if (!vbr_budget_explicit_) {
                    LLAMA_LOG_WARN("%s: VBR_FREEZE without an explicit VBR_BUDGET_MIB — the auto-budget "
                            "re-derivation reads live free VRAM and is NOT frozen; set VBR_BUDGET_MIB\n",
                            __func__);
                }
                if (vbr_freeze_preserve_empty_tiers_) {
                    LLAMA_LOG_WARN("%s: VBR_FREEZE_PRESERVE_EMPTY_TIERS active — "
                            "empty boundaries retain the current tier vector "
                            "(deterministic freeze testing only)\n", __func__);
                }
            }
            // Test-only schedule trace recorder: open the sink if requested.
            if (const char * tp = turbo_vbr_getenv("VBR_TRACE")) {
                // Per-child suffix: iSWA base/SWA caches must not both truncate the same
                // path. Standalone caches (trace_label == nullptr) keep the bare path unchanged.
                std::string trace_path = tp;
                if (vbr_params_.trace_label != nullptr) {
                    trace_path += '.';
                    trace_path += vbr_params_.trace_label;
                }
                vbr_trace_fp_.reset(fopen(trace_path.c_str(), "w"));
                if (vbr_trace_fp_) {
                    fprintf(vbr_trace_fp_.get(), "# phase\tboundary\tcursor\ttier_fnv\twm\tused\tmapped_bytes\n");
                    fflush(vbr_trace_fp_.get());
                    LLAMA_LOG_INFO("%s: VBR_TRACE -> %s (per-boundary tier-schedule trace)\n", __func__, trace_path.c_str());
                } else {
                    LLAMA_LOG_WARN("%s: VBR_TRACE=%s could not be opened for writing\n", __func__, trace_path.c_str());
                }
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
            vbr_budget_from_scalar_ = budget_fit_armed;
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
                    const size_t floor_share = share_of(vbr_floor_cost_bytes_);
                    if (budget_fit_armed) {
                        p.budget = share_of(vbr_budget_bytes_);
                        // An explicit scalar is a hard per-pool cap after proportional splitting.
                        // An auto scalar only proves aggregate capacity: fit sums device-local
                        // allowances, while VA-proportional splitting can move that capacity onto
                        // another device. Let live per-device re-derivation correct that split, but
                        // never below the pool's share of the advertised floor-layout cost.
                        p.budget_base = vbr_budget_explicit_ ? p.budget : floor_share;
                    } else {
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
                // Assign stable offsets now, while the pool/extent topology is immutable. The
                // reservation itself stays lazy: cuMemAddressReserve can fail at runtime, and that
                // failure must surface at the pre-mutation reserve boundary rather than model load.
                for (auto & p : vbr_pools_) {
                    if (p.vmm == nullptr) {
                        continue;
                    }
                    size_t total = 0;
                    for (size_t j = 0; j < layers.size(); ++j) {
                        for (int side = 0; side < 2; ++side) {
                            vbr_extent & e = side ? p.v[j] : p.k[j];
                            if (e.t == nullptr) {
                                continue;
                            }
                            e.stash_off = total;
                            const size_t ne0 = (size_t) e.t->ne[0];
                            if (ne0 > SIZE_MAX / sizeof(uint16_t) ||
                                (size_t) vbr_stash_rows_ > SIZE_MAX / (ne0 * sizeof(uint16_t))) {
                                throw std::runtime_error("VBR sink-stash size overflow");
                            }
                            const size_t bytes = (size_t) vbr_stash_rows_ * ne0 * sizeof(uint16_t);
                            if (total > SIZE_MAX - bytes) {
                                throw std::runtime_error("VBR sink-stash size overflow");
                            }
                            total += bytes;
                        }
                    }
                    if (total > SIZE_MAX - (p.gran - 1)) {
                        throw std::runtime_error("VBR sink-stash page padding overflow");
                    }
                    p.stash_size = GGML_PAD(total, p.gran);
                }
            }

            // Generation-tracker construction-final arming. This is shadow-only dual-write storage:
            // current checkpoint selection still reads the legacy epoch exclusively. Allocate
            // only after the effective budget is final, so an unarmed/static cache is byte-for-
            // byte and allocation-for-allocation unchanged.
            if (vbr_budget_bytes_ > 0) {
                vbr_generation_ = std::make_unique<vbr_generation_tracker>(
                        n_stream, kv_size, static_cast<uint32_t>(layers.size() * 2));
                vbr_capture_unit_leases_.resize(layers.size() * 2);
                // Keep construction fail-closed; unavailable durable lineage
                // is routed through the typed import-refusal path.
                GGML_ASSERT(vbr_generation_->active());
                // Dual-view ownership index: physical masks for enumeration plus
                // per-active-seq logical-position order statistic for scan-free exact rank.
                vbr_ownership_ = std::make_unique<vbr_ownership_index>(
                        n_stream, static_cast<uint32_t>(seq_to_stream.size()), kv_size);
                for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
                    for (uint32_t side = 0; side < 2; ++side) {
                        const ggml_tensor * tensor = side != 0 ? layers[ikv].v : layers[ikv].k;
                        const int32_t type = tensor != nullptr ? static_cast<int32_t>(tensor->type) : -1;
                        const vbr_repr_domain domain =
                                tensor != nullptr && ggml_is_turbo_kv_type(tensor->type) &&
                                                tensor->type != GGML_TYPE_TURBO8_0
                                        ? vbr_repr_domain::tapped
                                        : vbr_repr_domain::full;
                        GGML_ASSERT(vbr_generation_->initialize_unit(
                                static_cast<uint32_t>(ikv * 2 + side), type, domain));
                    }
                }
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
        if ((model.arch == LLM_ARCH_DEEPSEEK32 || model.arch == LLM_ARCH_DEEPSEEK4 ||
                model.arch == LLM_ARCH_GLM_DSA || model.arch == LLM_ARCH_DOTS3NOTE) &&
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
    construction_rollback.release();
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
             const char *   name_tag) :
    llama_kv_cache(model, hparams, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type, mem_other,
            filter, reuse, share, llama_memory_vbr_params{}, name_tag) {
}

llama_kv_cache::~llama_kv_cache() {
    vbr_release_resources();
}

void llama_kv_cache::vbr_release_resources() {
    // vbr_trace_fp_ closes itself through its RAII unique_ptr.
    // Normally the enclosing llama_context's detach guard clears these while this context's
    // compute backends are still alive. Keep RAII cleanup for direct construction and for a
    // cache constructor that throws after registering with its owner.
    vbr_shared_scratch_detach();

    for (auto & p : vbr_pools_) {
        if (p.backend != nullptr) {
            // A degrade wave may still be in flight on the side stream; it must finish before
            // the stash buffer / VMM VA it touches are torn down. The queued tail unmaps are moot
            // here (vmm_pool_free unmaps every chunk).
            ggml_backend_synchronize(p.backend);
            p.unmap_deferred.clear();
        }
        if (p.stash_vmm != nullptr) {
            // The side-stream synchronize above retires capture/transcode readers before the
            // stash VA disappears. vmm_pool_free adds a device-wide fence as a final backend guard.
            p.be->vmm_pool_free(p.stash_vmm);
            p.stash_vmm = nullptr;
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
    // Retire the imported-live accounting receipt only after the mapped
    // target ranges are no longer live. Never create a release-first window.
    vbr_import_receipts_release();
}

llama_kv_cache::vbr_shared_scratch_registration llama_kv_cache::vbr_shared_scratch_register(
        const vbr_shared_scratch_binding & binding) {
    if (binding.be == nullptr || binding.compute_backend == nullptr || binding.device < 0) {
        throw std::runtime_error("internal: incomplete shared-KV scratch consumer registration");
    }

    vbr_shared_scratch_registry::consumer consumer;
    consumer.be = binding.be;
    consumer.compute_backend = binding.compute_backend;
    consumer.device = binding.device;

    const auto tensor_slot = [&](const ggml_tensor * tensor, bool is_v) -> size_t {
        if (tensor == nullptr) {
            return vbr_shared_scratch_registry::no_slot;
        }
        for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
            if ((is_v ? layers[ikv].v : layers[ikv].k) == tensor) {
                return ikv*2 + (is_v ? 1 : 0);
            }
        }
        throw std::runtime_error("internal: shared-KV scratch alias is absent from its target owner");
    };

    GGML_ASSERT(binding.k.size() == binding.v.size());
    for (size_t ikv = 0; ikv < binding.k.size(); ++ikv) {
        vbr_shared_scratch_registry::layer_slots slots;
        slots.k = tensor_slot(binding.k[ikv], false);
        slots.v = tensor_slot(binding.v[ikv], true);
        if (slots.k != vbr_shared_scratch_registry::no_slot ||
            slots.v != vbr_shared_scratch_registry::no_slot) {
            consumer.layers.push_back(slots);
        }
    }
    if (consumer.layers.empty()) {
        throw std::runtime_error("internal: empty shared-KV scratch consumer registration");
    }

    const auto registry = vbr_shared_scratch_registry_;
    std::lock_guard<std::mutex> lock(registry->mutex);
    consumer.id = registry->next_id++;
    const uint64_t id = consumer.id;
    registry->consumers.push_back(std::move(consumer));
    return vbr_shared_scratch_registration(registry, id);
}

void llama_kv_cache::vbr_shared_scratch_visit(
        const std::vector<ggml_type> & terminal_types,
        const vbr_device_watermarks & terminal_watermarks,
        const vbr_shared_scratch_visitor & visitor) const {
    if (!visitor) {
        return;
    }
    if (!terminal_types.empty() && terminal_types.size() != layers.size()*2) {
        throw std::invalid_argument("shared-KV scratch terminal type vector has the wrong size");
    }
    const auto registry = vbr_shared_scratch_registry_;
    if (!registry) {
        return;
    }

    std::lock_guard<std::mutex> lock(registry->mutex);
    std::vector<vbr_shared_scratch_plan> plans;
    plans.reserve(registry->consumers.size());

    const auto tensor_and_type = [&](size_t slot) {
        const size_t ikv = slot/2;
        const bool is_v = (slot & 1) != 0;
        const ggml_tensor * tensor = is_v ? layers[ikv].v : layers[ikv].k;
        GGML_ASSERT(tensor != nullptr);
        ggml_type type = tensor->type;
        if (!terminal_types.empty() && terminal_types[slot] != GGML_TYPE_COUNT) {
            type = terminal_types[slot];
        }
        return std::make_pair(tensor, type);
    };

    for (const auto & consumer : registry->consumers) {
        const auto wm = terminal_watermarks.find(consumer.device);
        if (wm == terminal_watermarks.end()) {
            throw std::invalid_argument(format(
                    "shared-KV scratch terminal watermark is missing device %d",
                    consumer.device));
        }
        auto it = std::find_if(plans.begin(), plans.end(), [&](const vbr_shared_scratch_plan & p) {
            return p.compute_backend == consumer.compute_backend;
        });
        if (it == plans.end()) {
            plans.push_back({ consumer.be, consumer.compute_backend, consumer.device, 0, 0 });
            it = std::prev(plans.end());
        } else if (it->be != consumer.be || it->device != consumer.device) {
            throw std::runtime_error("internal: one shared-KV compute backend has inconsistent device metadata");
        }

        size_t k_row = 0;
        size_t v_row = 0;
        for (const auto & slots : consumer.layers) {
            const ggml_tensor * tk = nullptr;
            const ggml_tensor * tv = nullptr;
            ggml_type type_k = GGML_TYPE_F16;
            ggml_type type_v = GGML_TYPE_F16;
            if (slots.k != vbr_shared_scratch_registry::no_slot) {
                std::tie(tk, type_k) = tensor_and_type(slots.k);
            }
            if (slots.v != vbr_shared_scratch_registry::no_slot) {
                std::tie(tv, type_v) = tensor_and_type(slots.v);
            }
            bool need_k = false;
            bool need_v = false;
            ggml_vbr_kv_dequant_sides(type_k, type_v, &need_k, &need_v);
            if (need_k && tk != nullptr) {
                k_row = std::max(k_row, ggml_row_size(GGML_TYPE_F16, tk->ne[0]));
            }
            if (need_v && tv != nullptr) {
                v_row = std::max(v_row, ggml_row_size(GGML_TYPE_F16, tv->ne[0]));
            }
        }
        const auto scale_row = [&](size_t row) {
            if (wm->second != 0 && row > SIZE_MAX/(size_t) wm->second) {
                throw std::overflow_error(format(
                        "shared-KV scratch terminal size overflows on device %d",
                        consumer.device));
            }
            return row*(size_t) wm->second;
        };
        it->k_bytes = std::max(it->k_bytes, scale_row(k_row));
        it->v_bytes = std::max(it->v_bytes, scale_row(v_row));
    }

    // Keep the registry locked through the callback: llama_context's detach guard acquires the
    // same lock before it allows the consumer backend-owning members to be destroyed.
    for (const auto & plan : plans) {
        visitor(plan);
    }
}

void llama_kv_cache::vbr_shared_scratch_detach() {
    vbr_shared_scratch_registrations_.clear();
}

void llama_kv_cache::clear(bool data) {
    vbr_mutation_op mutation_op(this, vbr_operation_kind::sequence_edit,
            vbr_operation_class::state_api, -1, 0, std::numeric_limits<llama_pos>::max());
    const vbr_mutation_op::success_on_return mutation_ok(mutation_op);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (data) {
        // Settle any in-flight degrade wave first; its transcode/scrub writes must not race
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
        if (turbo_rotation != nullptr && turbo_rotation->buffer != nullptr && !hparams.no_alloc) {
            #include "turbo-rotation-data.h"
            ggml_backend_tensor_set(turbo_rotation, TURBO_ROTATION_R, 0, 128 * 128 * sizeof(float));
            ggml_backend_tensor_set(turbo_rotation_inv, TURBO_ROTATION_RT, 0, 128 * 128 * sizeof(float));
        }
    }

    // Clearing severs every checkpoint's attention lineage, even when only metadata is
    // requested. Keep this separate from vbr_full_reset(): clear may leave the tier cursor in
    // place until the next empty-cache boundary, and both mutations must remain observable.
    vbr_attention_content_changed();
    vbr_generation_global(vbr_mutation_registrant::clear, vbr_operation_class::state_api);
    if (vbr_ownership_) {
        vbr_ownership_->clear_all();
    }
    // The target is now logically empty (and, for data clears, byte-cleared),
    // so its conservative import receipt can finally be released.
    vbr_import_receipts_release();
}

void llama_kv_cache::vbr_import_receipts_release() noexcept {
    vbr_import_receipt_.reset();
}

void llama_kv_cache::vbr_import_receipts_release_if_empty() noexcept {
    if (!vbr_import_receipt_) {
        return;
    }
    const bool empty = std::all_of(
        v_cells.begin(), v_cells.end(),
        [](const llama_kv_cells & cells) { return cells.get_used() == 0; });
    if (empty) {
        // seq_rm is the ordinary server erase path. Once the imported image has
        // no live cell membership, retaining its conservative adoption claims
        // would make a later import fail prepare_receipts despite the target
        // being construction-empty. Every attention child owns one shared_ptr;
        // the receipt group releases the ledger operations only after the last
        // child reaches this boundary.
        vbr_import_receipts_release();
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    return seq_rm_impl(seq_id, p0, p1, seq_rm_mode::public_commit);
}

bool llama_kv_cache::seq_rm_transient(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    return seq_rm_impl(seq_id, p0, p1, seq_rm_mode::nested_commit);
}

bool llama_kv_cache::seq_rm_attn_transient(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    return seq_rm_transient(seq_id, p0, p1);
}

bool llama_kv_cache::seq_rm_impl(
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1,
        seq_rm_mode mode) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // VBR mutation scope: authenticated (sequence_edit, seq, [p0,p1)). Generic seq_rm remains
    // membership-only state_api (not provenance-bearing); the destructive §7.5 classes arrive
    // with the classed server paths in the mutation coordinator commit.
    const bool commit = mode != seq_rm_mode::dry_run;
    vbr_mutation_op mutation_op(commit ? this : nullptr, vbr_operation_kind::sequence_edit,
            vbr_operation_class::state_api, seq_id, p0, p1);
    const vbr_mutation_op::success_on_return mutation_ok(mutation_op);
    bool changed = false;

    if (seq_id >= 0) {
        const uint32_t stream = seq_to_stream[seq_id];
        auto & cells = v_cells[stream];
        auto & head  = v_heads[stream];
        auto generation_event = commit
                ? vbr_generation_begin(
                    vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api, stream,
                    vbr_generation_stamp_kind::membership)
                : vbr_generation_event{};

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id)) {
                changed = true;
                const llama_pos rm_pos = cells.pos_get(i);
                const bool became_empty = cells.seq_rm(i, seq_id);
                if (commit && vbr_ownership_) {
                    vbr_ownership_->remove_cell(stream, seq_id, i, rm_pos);
                }
                if (commit) {
                    vbr_stamp(mutation_op, generation_event, i, seq_id, rm_pos);
                }
                if (!became_empty) {
                    continue;
                }
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
        // Gap fix: a fully removed sequence releases its index view (frees the Fenwick).
        if (commit && vbr_ownership_ && cells.seq_pos_min(seq_id) < 0) {
            vbr_ownership_->clear_seq(stream, seq_id);
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];
            auto generation_event = commit
                    ? vbr_generation_begin(
                        vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api, s,
                        vbr_generation_stamp_kind::membership)
                    : vbr_generation_event{};

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                const bool had_membership = !cells.is_empty(i);
                const llama_pos any_rm_pos = had_membership ? cells.pos_get(i) : -1;
                if (had_membership) {
                    changed = true;
                    if (commit) {
                        vbr_ownership_update_all_seqs(s, i, any_rm_pos, /*add=*/false);
                    }
                }
                cells.rm(i);
                if (commit && had_membership) {
                    vbr_stamp(mutation_op, generation_event, i, -1, any_rm_pos);
                }
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

    if (mode == seq_rm_mode::public_commit && changed) {
        if (seq_id >= 0) {
            vbr_attention_content_changed(seq_id);
        } else {
            vbr_attention_content_changed();
        }
    }
    if (commit) {
        vbr_import_receipts_release_if_empty();
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    seq_cp_impl(seq_id_src, seq_id_dst, p0, p1, true);
}

bool llama_kv_cache::try_seq_cp_transient(
        llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    seq_cp_impl(seq_id_src, seq_id_dst, p0, p1, false);
    return true;
}

void llama_kv_cache::seq_cp_impl(
        llama_seq_id seq_id_src, llama_seq_id seq_id_dst,
        llama_pos p0, llama_pos p1, bool publish_lineage) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    vbr_normalize_edit_range(p0, p1);  // Canonical range before the scope.

    // VBR mutation scope. Cross-stream copies additionally reserve recovery and carry the
    // source-stability token; that wiring rides the sc_info pending owner below.
    vbr_mutation_op mutation_op(this, vbr_operation_kind::sequence_edit,
            vbr_operation_class::state_api, seq_id_dst, p0, p1);
    const vbr_mutation_op::success_on_return mutation_ok(mutation_op);

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];
        auto generation_event = vbr_generation_begin(
                vbr_mutation_registrant::seq_cp, vbr_operation_class::state_api, s0,
                vbr_generation_stamp_kind::membership);

        if (seq_id_src == seq_id_dst) {
                return;
        }

        bool changed = false;
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src) && !cells.seq_has(i, seq_id_dst)) {
                changed = true;
                cells.seq_add(i, seq_id_dst);
                const llama_pos cp_pos = cells.pos_get(i);
                if (vbr_ownership_) {
                    vbr_ownership_->add_cell(s0, seq_id_dst, i, cp_pos);
                }
                vbr_stamp(mutation_op, generation_event, i, seq_id_dst, cp_pos);
            }
        }

        if (publish_lineage && changed) {
            vbr_attention_content_changed(seq_id_dst);
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    // Deferred-copy fence (pending owner through stream_copy_info, commit at byte-copy
    // completion) is NOT implemented — it is structurally unreachable because armed VBR
    // requires n_stream == 1. Fail loudly if that invariant ever breaks rather than let a
    // cross-stream copy close its operation before bytes land (design finding R3-4).
    GGML_ASSERT(vbr_generation_tracker_mut() == nullptr &&
                "cross-stream seq_cp under armed VBR requires the pending-owner fence");

    bool is_full = true;

    if (p0 > 0 && p0 < (int) get_size() - 1) {
        is_full = false;
    }

    if (p1 > 0 && p1 < (int) get_size() - 1) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    bool changed = false;
    auto destination_reset_event = vbr_generation_begin(
            vbr_mutation_registrant::seq_cp, vbr_operation_class::state_api, s1,
            vbr_generation_stamp_kind::membership);
    for (uint32_t i = 0; i < v_cells[s1].size(); ++i) {
        if (!v_cells[s1].is_empty(i)) {
            changed = true;
            vbr_stamp(mutation_op, destination_reset_event, i, -1);
        }
    }
    v_cells[s1].reset();
    auto generation_event = vbr_generation_begin(
            vbr_mutation_registrant::seq_cp, vbr_operation_class::state_api, s1,
            vbr_generation_stamp_kind::dependency, true);
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            changed = true;
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
            vbr_stamp(mutation_op, generation_event, i, seq_id_dst);
        }
    }

    v_heads[s1] = v_heads[s0];

    if (publish_lineage && changed) {
        vbr_attention_content_changed(seq_id_dst);
    }

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // VBR mutation scope: whole-range membership edit. The manifest declares the sequence.
    // Wildcard: keep removes membership from every sequence except the kept one, so
    // per-cell stamps carry whichever sequence remains (or none) — a single-seq claim would
    // be a false declaration.
    vbr_mutation_op mutation_op(this, vbr_operation_kind::sequence_edit,
            vbr_operation_class::state_api, -1, 0, std::numeric_limits<llama_pos>::max());
    const vbr_mutation_op::success_on_return mutation_ok(mutation_op);

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const uint32_t stream = seq_to_stream[seq_id];
    auto & cells = v_cells[stream];
    auto & head  = v_heads[stream];
    auto generation_event = vbr_generation_begin(
            vbr_mutation_registrant::seq_keep, vbr_operation_class::state_api, stream,
            vbr_generation_stamp_kind::membership);

    uint32_t new_head = cells.size();
    bool changed_any = false;
    std::array<bool, LLAMA_MAX_SEQ> changed_sequences = {};

    for (uint32_t i = 0; i < cells.size(); ++i) {
        const bool changed = !cells.is_empty(i) && (!cells.seq_has(i, seq_id) || cells.seq_count(i) > 1);
        changed_any = changed_any || changed;
        if (changed) {
            for (llama_seq_id removed = 0; removed < LLAMA_MAX_SEQ; ++removed) {
                if (removed != seq_id && cells.seq_has(i, removed)) {
                    changed_sequences[size_t(removed)] = true;
                }
            }
            vbr_ownership_update_all_seqs(stream, i, cells.pos_get(i), /*add=*/false,
                                          /*exclude_seq=*/seq_id);
        }
        if (cells.seq_keep(i, seq_id)) {
            if (i < vbr_stash_rows_) {
                vbr_stash_dirty_ = true; // a sink cell can now be rewritten by another request
            }
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
        if (changed) {
            vbr_stamp(mutation_op, generation_event, i, cells.is_empty(i) ? -1 : seq_id,
                      cells.is_empty(i) ? -1 : cells.pos_get(i));
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
    if (changed_any) {
        vbr_attention_content_changed(changed_sequences);
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    seq_add_impl(seq_id, p0, p1, shift, false);
}

void llama_kv_cache::seq_add_raw_mrope(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_ASSERT(supports_qwen4_text_mrope_shift());
    seq_add_impl(seq_id, p0, p1, shift, true);
}

void llama_kv_cache::seq_add_impl(
        llama_seq_id seq_id,
           llama_pos p0,
           llama_pos p1,
           llama_pos shift,
                bool raw_keys) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    const bool text_mrope = supports_qwen4_text_mrope_shift();
    GGML_ASSERT((hparams.n_pos_per_embd() == 1 || text_mrope) &&
            "seq_add() only supports one-dimensional or Qwen4 text-broadcast positions");
    GGML_ASSERT((!raw_keys || text_mrope) &&
            "metadata-only shifting is restricted to Qwen4 text M-RoPE raw-key caches");
    GGML_ASSERT((!text_mrope || can_shift_qwen4_text_range(seq_id, p0, p1)) &&
            "Qwen4 M-RoPE shifting is restricted to broadcast text positions");

    const uint32_t stream = seq_to_stream[seq_id];
    auto & cells = v_cells[stream];
    auto & head  = v_heads[stream];
    vbr_normalize_edit_range(p0, p1);  // Canonical range before the scope.
    // VBR mutation scope: position shift is a dependency-changing edit over [p0,p1).
    vbr_mutation_op mutation_op(this, vbr_operation_kind::sequence_edit,
            vbr_operation_class::state_api, seq_id, p0, p1);
    const vbr_mutation_op::success_on_return mutation_ok(mutation_op);
    auto generation_event = vbr_generation_begin(
            vbr_mutation_registrant::seq_add, vbr_operation_class::state_api, stream,
            vbr_generation_stamp_kind::dependency, true);

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();
    bool changed = false;

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            changed = true;
            const llama_pos old_pos = cells.pos_get(i);
            const bool removed = text_mrope
                ? (raw_keys
                    ? cells.pos_add_mrope_text_raw(i, shift)
                    : cells.pos_add_mrope_text(i, shift))
                : cells.pos_add(i, shift);
            if (vbr_ownership_) {
                if (removed) {
                    vbr_ownership_->remove_cell(stream, seq_id, i, old_pos);
                } else {
                    vbr_ownership_->move_cell(stream, seq_id, i, old_pos, old_pos + shift);
                }
            }
            if (removed) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
            vbr_stamp(mutation_op, generation_event, i, seq_id, old_pos);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
    if (changed) {
        vbr_attention_content_changed(seq_id);
    }
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    const uint32_t stream = seq_to_stream[seq_id];
    auto & cells = v_cells[stream];
    vbr_normalize_edit_range(p0, p1);  // Canonical range before the scope.
    // VBR mutation scope: position division is a dependency-changing edit over [p0,p1).
    vbr_mutation_op mutation_op(this, vbr_operation_kind::sequence_edit,
            vbr_operation_class::state_api, seq_id, p0, p1);
    const vbr_mutation_op::success_on_return mutation_ok(mutation_op);
    auto generation_event = vbr_generation_begin(
            vbr_mutation_registrant::seq_div, vbr_operation_class::state_api, stream,
            vbr_generation_stamp_kind::dependency, true);

    if (d == 1) {
        return;
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    bool changed = false;
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            const llama_pos old_pos = cells.pos_get(i);
            cells.pos_div(i, d);
            changed = changed || cells.pos_get(i) != old_pos;
            if (vbr_ownership_) {
                vbr_ownership_->move_cell(stream, seq_id, i, old_pos, cells.pos_get(i));
            }
            vbr_stamp(mutation_op, generation_event, i, seq_id, old_pos);
        }
    }
    if (changed) {
        vbr_attention_content_changed(seq_id);
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

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown_vbr_managed() const {
    // A plain KV cache has no auxiliary context-linear state: its entire allocation follows the
    // selected K/V representation, even when the selection is a fixed Turbo tier rather than a
    // live dynamic controller.
    return memory_breakdown();
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
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true, 0);

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
    auto sinfos = plan_slots(ubatches);
    if (sinfos.empty()) {
        // plan_slots() can refuse before prepare_with_slots() reaches its
        // boundary reset. Do not let a prior donor-side seal refusal retype a
        // later unrelated cell-exhaustion terminal.
        llama_kv_cache * root = vbr_tree_root();
        root->vbr_hard_seal_blocked_ = false;
        root->vbr_hard_seal_evidence_.clear();
        if (root->vbr_ledger_sibling_ != nullptr) {
            root->vbr_ledger_sibling_->vbr_hard_seal_blocked_ = false;
            root->vbr_ledger_sibling_->vbr_hard_seal_evidence_.clear();
        }
        return {};
    }

    return prepare_with_slots(ubatches, std::move(sinfos));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare_with_slots(
        const std::vector<llama_ubatch> & ubatches,
        slot_info_vec_t                   sinfos) {
    GGML_ASSERT(sinfos.size() == ubatches.size());

    vbr_runtime_was_over_ = false;
    // The VBR degrade trigger must run at the llama_decode boundary, before any of this batch's
    // cells are committed. Measured (VBR_MAP_RAWSCAN): mid-batch, apply_ubatch runs ahead of graph
    // execution by up to the whole batch — a mid-batch transcode captures positioned-but-unwritten
    // rows as zeros and races graphs built against the old tier. Here slot admission has completed,
    // the previous batch's writes are visible, and no built-but-unexecuted graphs exist.
    // prepare_with_slots() is the post-admission choke point both ordinary and iSWA paths use. The
    // check is predictive: fit the WORST-CASE watermark this batch can reach at current tiers so it
    // never overruns the budget mid-flight.
    if (vbr_vmm_active()) {
        // Reserve failures are boundary-local. The exact step remains at the cursor and may be
        // retried after another process releases capacity.
        llama_kv_cache * root = vbr_tree_root();
        root->vbr_reserve_failed_ = false;
        root->vbr_hard_seal_blocked_ = false;
        root->vbr_hard_seal_evidence_.clear();
        if (root->vbr_ledger_sibling_ != nullptr) {
            root->vbr_ledger_sibling_->vbr_reserve_failed_ = false;
            root->vbr_ledger_sibling_->vbr_hard_seal_blocked_ = false;
            root->vbr_ledger_sibling_->vbr_hard_seal_evidence_.clear();
        }
        // Release the tail pages queued by the previous wave first; their transcodes are long
        // done (fence-ordered before the previous graph). Must precede this boundary's degrades and
        // ensure_mapped so no later page map can be ripped by a stale queued unmap.
        vbr_flush_deferred_unmaps();
        // A boundary is decode activity for idleness; ticks never write this
        vbr_last_prepare_ns_ = llama_vram_ledger_now_ns();
    }
    // one ubatch token sum serves the budget block and the unified scratch projection below;
    // non-unified scratch uses the planned physical views because their stream span also matters
    uint32_t n_tokens = 0;
    for (const auto & ub : ubatches) {
        n_tokens += ub.n_tokens;
    }
    vbr_runtime_wm_ = vbr_watermark_cells(n_tokens);
    if (vbr_vmm_active() && vbr_budget_bytes_ > 0) {
        // sink-stash staleness: if any sink cell was freed since capture, every stash may hold
        // another request's rows — drop them all (they recapture at the next first degrade)
        vbr_invalidate_dirty_stash();
        // -- Stability fast-path: skip per-batch bookkeeping when settled (avoids ~1ms/token) --
        uint32_t used_now = 0;
        for (uint32_t st = 0; st < n_stream; ++st) {
            used_now += v_cells[st].get_used();
        }
        // Stable when: budget fully explored, quiet for N boundaries, occupancy hasn't meaningfully moved.
        static const uint32_t VBR_STABLE_QUICK = 10; // quiet-boundary threshold for fast path
        static const int32_t  VBR_USED_DELTA   = 512; // occupancy delta below which we're stable
        // co-tenancy: the ledger pre-check runs EVERY boundary, outside the stable gate —
        // a peer's rename (new claim, offer change) forces the full controller path.
        // Freeze mode disables the ledger for deterministic tests, so the schedule ignores co-tenant
        // state; the tree force stays clear and the scan below is skipped too.
        if (!vbr_freeze_) {
            vbr_ledger_precheck();
        }
        const bool vbr_reconcile_now = vbr_retier_take_reconcile("prepare");
        // Compare predicted padded watermarks, not just total occupancy. A high-tail trim can make
        // shrink or promotion work possible while changing fewer than VBR_USED_DELTA cells (or no
        // cells at all after sequence redistribution). Consume each decrease exactly once: using a
        // pool's grow-only wm_cells here would keep the full path hot until the 25% shrink threshold.
        const uint32_t wm_next = vbr_watermark_cells(n_tokens);
        const bool wm_receded = wm_next < vbr_last_wm_;
        const bool reset_due = vbr_degrade_cursor_ > 0 && used_now == 0;
        bool vbr_stable = (vbr_retier_freeze_depth_ == 0 && !vbr_reconcile_now &&
                           vbr_degrade_cursor_ >= std::min(vbr_degrade_order_.size(), vbr_degrade_limit_) &&
                           vbr_quiet_boundaries_ >= VBR_STABLE_QUICK &&
                           std::abs((int64_t)used_now - (int64_t)vbr_last_used_) < VBR_USED_DELTA &&
                           !wm_receded && !reset_due &&
                           !vbr_tree_forced());

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
            if (vbr_degrade_cursor_ > 0 && used_now == 0 &&
                !vbr_freeze_preserve_empty_tiers_) {
                vbr_full_reset();
            }
            vbr_shrink_watermark(); // occupancy drops release phantom tail pages first

            // promote pacing: ONE step per boundary, and only after a quiet window (no degrade in
            // the last 4 boundaries). Promotes re-encode aged rows from degraded recon — error
            // compounds per hop — so waves spread out and a clamp-driven degrade vetoes the
            // immediate bounce-back. Boundary counting keeps the cooldown deterministic.
            vbr_quiet_boundaries_++;
            // co-tenancy: promotes freeze while any unamortized grant remains, and around
            // presence changes (gates live in vbr_maybe_promote)
            vbr_maybe_promote(wm_next);
            // budget trigger: degrade while ANY pool exceeds its share. A step only shrinks the pool
            // that owns its tensor, but the cursor is a global price order — advancing it while any
            // pool is over budget is the simplest rule that terminates and preserves the price order.
            // Pre-loop pressure snapshot: the runtime demand's honest ask is what
            // this boundary was short BEFORE the own ladder's sacrifice resolved it
            const bool vbr_was_over = vbr_over_budget(wm_next);
            vbr_runtime_was_over_ = vbr_was_over;
            if (vbr_was_over) {
                vbr_pre_deficit_.assign(vbr_pools_.size(), 0);
                for (size_t pi = 0; pi < vbr_pools_.size(); ++pi) {
                    const auto & pp = vbr_pools_[pi];
                    if (pp.vmm != nullptr) {
                        const size_t proj = vbr_vmm_projected_bytes(pp, wm_next);
                        const size_t be   = vbr_budget_eff(pp);
                        vbr_pre_deficit_[pi] = proj > be ? proj - be : 0;
                    }
                }
            }
            if (vbr_retier_freeze_depth_ > 0 && vbr_over_budget(wm_next)) {
                // The preflight should make this unreachable for a correctly bounded consumer,
                // but keep decode/bookkeeping live and remember the pressure if a caller grows
                // past its declared bound. Never interpret "frozen" as ladder exhaustion.
                vbr_retier_defer("degrade_pressure");
            } else {
                while (vbr_over_budget(wm_next)) {
                    vbr_quiet_boundaries_ = 0; // degrade pressure this boundary — cool the promote path
                    const vbr_degrade_result degrade = vbr_degrade_next(wm_next);
                    if (degrade == vbr_degrade_result::reserve_failed) {
                        LLAMA_LOG_ERROR("%s: VBR component reserve failed before tier mutation — "
                                "failing this batch recoverably\n", __func__);
                        // Earlier successful steps in this boundary remain committed. Fence their
                        // side-stream waves before returning without a graph to carry the normal wait.
                        vbr_tree_force();
                        vbr_arm_wave_fences();
                        return {};
                    }
                    if (degrade == vbr_degrade_result::hard_lease_blocked) {
                        LLAMA_LOG_WARN("%s: VBR pressure reached the hard-lease seal; "
                                "failing this batch recoverably\n", __func__);
                        vbr_tree_force();
                        vbr_arm_wave_fences();
                        return {};
                    }
                    if (degrade == vbr_degrade_result::capture_lease_blocked) {
                        LLAMA_LOG_DEBUG(
                            "%s: VBR pressure reached an active per-unit capture "
                            "lease; retrying from fresh controller state\n",
                            __func__);
                        vbr_tree_force();
                        vbr_arm_wave_fences();
                        return {};
                    }
                    if (degrade == vbr_degrade_result::exhausted) {
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
            // Runtime-growth demand: the trigger is band-spent and under pressure this
            // boundary (pre-own-loop snapshot) — the own loop's exit makes post-loop
            // projected <= budget_eff whenever the sub-band ladder still works, which must
            // not hide the demand (the band is what peers owe; the sub-band walk is the
            // demander's own sacrifice)
            // co-tenancy: full ledger pass on a pre-check hit, or unconditionally every 8th
            // boundary once ≥1s has passed since the last full scan (bounds the miss window
            // when our own rename baseline-swallowed a peer's concurrent rename)
            if (!vbr_freeze_ && (vbr_tree_forced() ||
                (vbr_boundary_count_ % 8 == 0 &&
                 llama_vram_ledger_now_ns() - vbr_last_scan_ns_ >= 1000000000ull))) {
                vbr_ledger_scan_service(n_tokens);
                if (vbr_tree_root()->vbr_reserve_failed_) {
                    LLAMA_LOG_ERROR("%s: VBR demand-shed reserve failed before its tier mutation — "
                            "failing this batch recoverably\n", __func__);
                    vbr_tree_root()->vbr_arm_wave_fences();
                    return {};
                }
            }
            // The wave's transcodes/scrubs are queued on each pool's side stream; arm that
            // device's fence so the next graph_compute GPU-waits on them; the host proceeds straight
            // to graph build.
            vbr_arm_wave_fences();
        } else {
            // Fast path: settled — skip the budget/degrade bookkeeping. wm_next stays the current
            // watermark; the shared eager map below still covers occupancy that creeps up under the
            // stable threshold.
            vbr_quiet_boundaries_++;
        }
        // Every child records at the common post-policy point. Non-roots leave their sample
        // for the last-running root, which aggregates even when its own path was stable.
        if (!vbr_freeze_) {
            vbr_runtime_demand_update(wm_next, vbr_runtime_was_over_);
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
            // co-tenancy (spec: try_map-failed runtime-demand disjunct): route the NEXT
            // boundary to the full controller path — a stable-path resident squeezed by a
            // rename-free co-tenant would otherwise never publish its runtime demand and
            // livelock on failing batches
            vbr_tree_force();
            // This boundary's degrade may already have flipped tiers, so trace it (distinct
            // phase, counter not advanced) instead of silently dropping it under OOM/fault validation.
            vbr_trace_emit("prepare_mapfail", wm_next, used_now);
            return {};
        }
        // free-running boundary counter: drives the auto-budget re-derive throttle above and the
        // first-boundary skip inside vbr_rederive_budget(). Advances every boundary regardless of
        // which path ran, so the %8 cadence is real wall-boundary time.
        vbr_boundary_count_++;
        vbr_last_used_ = used_now;
        vbr_last_wm_ = wm_next;
        vbr_trace_emit("prepare", wm_next, used_now);
    }

    // Grow flash-attention f16 dequant scratch to this batch's watermark outside the graphs, for
    // the sides that are dequant-active after the wave above — see vbr_scratch_reserve. Runs for
    // every turbo-typed cache (bookkeeping pools exist even without the dynamic controller);
    // non-turbo caches have no pools and skip in O(1).
    if (!vbr_pools_.empty() || !vbr_shared_scratch_bindings_.empty()) {
        size_t scratch_cells = vbr_watermark_cells(n_tokens);
        if (n_stream > 1) {
            // A non-unified graph views K/V as [head_dim, heads, n_kv, stream_span], and the
            // CUDA materializer flattens all four dimensions into one shared f16 scratch. Predict
            // the largest flattened view in this batch from the already-planned physical slots.
            // Keep prior ubatch inserts in the prediction; ignoring a later purge can only make
            // this an upper bound. Dynamic VBR is forced unified and retains the original path.
            std::array<uint32_t, LLAMA_MAX_SEQ> used_max_p1 = {};
            for (uint32_t s = 0; s < n_stream; ++s) {
                used_max_p1[s] = v_cells[s].used_max_p1();
            }
            scratch_cells = 0;
            const uint32_t n_pad_cur = std::max(n_pad, 256u);
            for (const auto & sinfo : sinfos) {
                uint32_t n_kv = 0;
                for (size_t s = 0; s < sinfo.n_stream(); ++s) {
                    const uint32_t stream = sinfo.strm[s];
                    for (const uint32_t idx : sinfo.idxs[s]) {
                        used_max_p1[stream] = std::max(used_max_p1[stream], idx + 1);
                    }
                    n_kv = std::max(n_kv,
                            std::min(v_cells[stream].size(), GGML_PAD(used_max_p1[stream], n_pad_cur)));
                }
                const size_t stream_span = (size_t) sinfo.s1 - sinfo.s0 + 1;
                scratch_cells = std::max(scratch_cells, (size_t) n_kv * stream_span);
            }
        }
        if (!vbr_scratch_reserve(scratch_cells)) {
            LLAMA_LOG_ERROR("%s: f16 dequant scratch reserve failed (device memory exhausted) — "
                    "failing this batch recoverably\n", __func__);
            vbr_tree_force(); // same routing as the try_map failure above
            return {};
        }
    }

    return sinfos;
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::plan_slots(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    // Slot selection for later ubatches must observe the metadata changes made by
    // earlier ones (notably SWA eviction and its contiguous-prefix purge).  Apply
    // those changes speculatively, journal every cell they can touch, then restore
    // the exact original metadata before returning the plan.  commit=false keeps
    // VMM growth and the optional transcode self-test out of this transaction.
    struct state_t {
        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch
        std::vector<uint32_t> streams;
        std::vector<std::vector<uint32_t>> idxs;
        std::vector<llama_kv_cells> cells;
    };

    std::vector<state_t> states;
    const bool stash_dirty_old = vbr_stash_dirty_;

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

        state_t state;
        state.v_heads_old = v_heads;

        // Start with the selected cells themselves.
        std::map<uint32_t, std::vector<uint32_t>> touched;
        llama_pos seq_pos_max_rm[LLAMA_MAX_SEQ];
        std::fill(std::begin(seq_pos_max_rm), std::end(seq_pos_max_rm), -1);

        for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
            const uint32_t stream = sinfo_new.strm[s];
            auto & idxs = touched[stream];
            idxs.insert(idxs.end(), sinfo_new.idxs[s].begin(), sinfo_new.idxs[s].end());

            const auto & cells = v_cells[stream];
            for (const uint32_t idx : sinfo_new.idxs[s]) {
                if (!cells.is_empty(idx)) {
                    GGML_ASSERT(cells.seq_count(idx) == 1);
                    const llama_seq_id seq_id = cells.seq_get(idx);
                    seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], cells.pos_get(idx));
                }
            }
        }

        // Slot planning mutates then restores cell metadata. Generation stamps belong only to
        // the later committed apply() call; a failed/dry prepare must leave no shadow mutation.
        // apply_ubatch() also purges an evicted SWA sequence's older prefix.
        // Include those cells in the journal even when they are outside sinfo.
        for (uint32_t seq_id = 0; seq_id < LLAMA_MAX_SEQ; ++seq_id) {
            if (seq_pos_max_rm[seq_id] < 0) {
                continue;
            }

            GGML_ASSERT(seq_id < seq_to_stream.size());
            const uint32_t stream = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream];
            auto & idxs = touched[stream];
            for (uint32_t idx = 0; idx < cells.size(); ++idx) {
                if (cells.pos_in(idx, 0, seq_pos_max_rm[seq_id] + 1) && cells.seq_has(idx, seq_id)) {
                    idxs.push_back(idx);
                }
            }
        }

        for (auto & entry : touched) {
            auto & idxs = entry.second;
            std::sort(idxs.begin(), idxs.end());
            idxs.erase(std::unique(idxs.begin(), idxs.end()), idxs.end());

            state.streams.push_back(entry.first);
            state.idxs.push_back(idxs);
            state.cells.push_back(v_cells[entry.first].cp(idxs));
        }

        states.push_back(std::move(state));
        apply_ubatch(sinfo_new, ubatch, false);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        for (size_t i = 0; i < it->streams.size(); ++i) {
            v_cells[it->streams[i]].set(it->idxs[i], it->cells[i]);
        }
        v_heads = it->v_heads_old;
    }
    vbr_stash_dirty_ = stash_dirty_old;

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // This tracker's cache services its pending quarantines at the decode boundary
    // — perform the invalidation FIRST, then ack with the token; only the ack reclaims the
    // ring slot. Failures without capabilities resolve here too, keeping the ring live.
    if (auto * tracker = vbr_generation_tracker_mut()) {
        const vbr_controller_instance_id instance = tracker->runtime_instance();
        vbr_recovery_advance_recorded(instance);  // Recorded failures enter quarantine.
        for (;;) {
            auto work = vbr_recovery_take_quarantine(instance);
            if (!work.token) {
                break;
            }
            if (!tracker->global_invalidate_and_reset_extents(
                        vbr_mutation_registrant::authenticated_recovery,
                        vbr_operation_class::controller)) {
                // Invalidation impossible right now (mid-mutation): latch unavailable,
                // RELEASE the take so the record stays serviceable, and retry next boundary
                // — never ack unperformed work.
                tracker->set_shadow_unavailable();
                GGML_ASSERT(vbr_recovery_untake_quarantine(work.token, instance));
                break;
            }
            GGML_ASSERT(vbr_recovery_ack_quarantine(work.token, instance));
        }
        // Monotone re-arm: the tracker owns the whole
        // availability-creating transition (ring + capacity proof, sanctioned invalidation,
        // monotone clear). Never re-arms in the same pass that latched: the latch path above
        // breaks with an un-acked record that the ring proof still sees.
        tracker->try_rearm();
    }

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

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch, bool commit) {
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

    // One decode-kind operation scope per apply_ubatch commit. Recovery is eager
    // (reserved in the ctor — the wrap class is provenance-bearing); the extents and the
    // reuse EVENT are LAZY — reserved per SELECTED target at the first destructive stamp, so
    // a wrap-free decode pays neither extent traffic nor a destructive event.
    // The ubatch target scan is armed-only, so non-VBR decode never pays it, and
    // skipped entirely when a composite wrapper already supplied the manifest (adoption).
    const bool decode_armed = commit && vbr_generation_tracker_get() != nullptr;
    const bool wrap_provenance = decode_armed && n_swa != 0;
    const vbr_operation_class decode_class =
            wrap_provenance ? vbr_operation_class::swa_wrap : vbr_operation_class::ordinary_decode;
    const uint16_t extent_stream =
            static_cast<uint16_t>(sinfo.n_stream() == 1 ? sinfo.strm[0] : VBR_STREAM_ANY);
    const vbr_controller_instance_id decode_instance =
            decode_armed ? vbr_instance_id() : vbr_controller_instance_id{};
    // Build a sequence/range-bound manifest from the actual ubatch; ceiling overflow yields a
    // zero-target binding, which the registry REFUSES -> fail-closed shadow-unavailable.
    vbr_operation_binding decode_manifest;
    decode_manifest.kind        = vbr_operation_kind::decode;
    decode_manifest.child_phase = vbr_operation_phase::mutate;
    if (decode_armed && !vbr_adopted_operation_ && !vbr_adopted_refused_) {
        (void) vbr_decode_targets_from_ubatch(decode_manifest, decode_instance,
                                              wrap_provenance, extent_stream, ubatch);
    }
    // Every asynchronous decode operation reserves
    // recovery eagerly; only the tombstone EXTENTS stay wrap-gated and lazy.
    vbr_mutation_op mutation_op(
            decode_armed ? this : nullptr,
            decode_manifest,
            /*provenance_bearing=*/decode_armed);
    if (decode_armed) {
        // Pre-reserve pending capacity before any mutation so allocation never follows
        // apply; geometric growth keeps the ramp O(k), not O(k^2). The
        // awaiting list must absorb EVERY accumulated pending record without allocating —
        // terminal transfer runs inside the decode transaction's noexcept destructor.
        if (vbr_pending_decode_ops_.capacity() == vbr_pending_decode_ops_.size()) {
            vbr_pending_decode_ops_.reserve(
                    std::max<size_t>(8, 2 * vbr_pending_decode_ops_.size() + 1));
        }
        const size_t awaiting_needed =
                vbr_awaiting_commit_.size() + vbr_pending_decode_ops_.size() + 1;
        if (vbr_awaiting_commit_.capacity() < awaiting_needed) {
            vbr_awaiting_commit_.reserve(
                    std::max(awaiting_needed, 2 * vbr_awaiting_commit_.capacity() + 8));
        }
    }

    bool reused_occupied_cell = false;
    std::array<bool, LLAMA_MAX_SEQ> reused_sequences = {};
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const uint32_t stream = sinfo.strm[s];
        auto append_event = decode_armed
                ? vbr_generation_begin(
                    vbr_mutation_registrant::apply_ubatch_append,
                    vbr_operation_class::ordinary_decode,
                    stream,
                    vbr_generation_stamp_kind::dependency)
                : vbr_generation_event{};
        std::optional<vbr_generation_event> reuse_event;  // Lazily minted at first reuse.
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            const bool occupied = !cells.is_empty(idx);
            llama_pos  prior_pos = -1;
            if (occupied) {
                if (decode_armed && !reuse_event.has_value()) {
                    // No eager extent: the destructive stamps below reserve
                    // per-SELECTED-target extents through the event's trampoline.
                    reuse_event.emplace(vbr_generation_begin(
                            vbr_mutation_registrant::apply_ubatch_occupied_reuse,
                            decode_class, stream, vbr_generation_stamp_kind::dependency, true));
                }
                reused_occupied_cell = true;
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);
                prior_pos = pos;

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);
                reused_sequences[size_t(seq_id)] = true;

                if (decode_armed && vbr_ownership_) {
                    vbr_ownership_->remove_cell(stream, seq_id, idx, pos);
                }
                if (idx < vbr_stash_rows_) {
                    vbr_stash_dirty_ = true; // the SWA slot is about to hold a different token
                }
                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d() || ubatch.token || hparams.ple_n_heads > 0) {
                llama_kv_cell_ext ext;

                if (ubatch.is_pos_2d()) {
                    ext.x = ubatch.pos[i + ubatch.n_tokens*2];
                    ext.y = ubatch.pos[i + ubatch.n_tokens];
                }

                if (ubatch.token) {
                    ext.tok = ubatch.token[i];
                } else if (hparams.ple_n_heads > 0) {
                    // embd batch (multimodal input) has no token ids, need to pad it with the correct ID for PLE layers
                    // TODO @ngxson : check if we can do the same as gemma 3n / gemma 4
                    ext.tok = hparams.ple_image_token_id != 0
                        ? (llama_token) hparams.ple_image_token_id
                        : (llama_token) hparams.ple_eos_token_id;
                }

                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
                if (occupied) {
                    reused_sequences[size_t(ubatch.seq_id[i][s])] = true;
                }
            }

            if (decode_armed && vbr_ownership_) {
                for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                    vbr_ownership_->add_cell(stream, ubatch.seq_id[i][s], idx, ubatch.pos[i]);
                }
            }

            vbr_generation_event * generation_event =
                    occupied ? (reuse_event.has_value() && *reuse_event ? &*reuse_event : nullptr)
                             : (append_event ? &append_event : nullptr);
            if (generation_event != nullptr) {
                // The stamp proves a covering target for every member of the
                // token's sequence set at its pre-mutation position; a refusal poisons the
                // event, latches the shadow, and fails the operation (via vbr_stamp) — the
                // decode proceeds untracked.
                vbr_stamp(mutation_op, *generation_event, idx,
                          ubatch.seq_id[i], ubatch.n_seq_id[i],
                          occupied ? prior_pos : ubatch.pos[i]);
            }
        }
    }

    // Appending into empty cells does not alter a checkpoint's attention prefix. Reusing an
    // occupied SWA/ring cell does: the following graph overwrites bytes a checkpoint may still
    // reference, so make that ownership/representation change visible before graph execution.
    if (commit && reused_occupied_cell) {
        vbr_attention_content_changed(reused_sequences);
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

            seq_rm_impl(
                    s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1,
                    commit ? seq_rm_mode::nested_commit : seq_rm_mode::dry_run);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }

    // VBR VMM: the graph compute that follows reads/writes cells up to the padded watermark —
    // grow the physical backing first (no-op unless the watermark advanced past a page)
    if (commit && vbr_vmm_active()) {
        vbr_vmm_ensure_mapped();
    }

    // Dynamic-VBR anchor self-test (environment-gated, runs once). It fires from the second apply_ubatch call so
    // a prior attention has identity-init'd the decode-side InnerQ scales the dequant relies on.
    static const bool vbr_test_env = getenv("VBR_TRANSCODE_TEST") != nullptr; // once, not per ubatch
    if (commit && vbr_test_env) {
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

    // The decode operation outlives apply_ubatch; its outcome is unknown until
    // the graph result. Transfer op + extent + recovery reservation to the pending list;
    // vbr_decode_ops_finish(ok) resolves them at the decode boundary (extent -> submitted on
    // success, promoted to committed at the synchronize fence; everything fails closed on a
    // decode error).
    if (auto pending = mutation_op.detach_deferred()) {
        vbr_pending_decode_ops_.push_back(std::move(*pending));
    }
}

// Defined with the mutation-op implementation below.
static void vbr_fail_extent_set(
        vbr_generation_tracker * tracker,
        std::array<vbr_extent_handle, vbr_operation_binding::MAX_TARGETS> & extents);

void llama_kv_cache::vbr_commit_submitted() {
    if (other) {
        other->vbr_commit_submitted();
        return;
    }
    if (vbr_awaiting_commit_.empty()) {
        return;
    }
    // The sync fence delivers the terminal result to each pending owner. Commit
    // success closes the operation committed and releases recovery; a failed commit (slab
    // reset / obsolete handle) is counted, latches availability, and closes failed — never a
    // silent vanish. Every per-target handle commits; one failure fails the
    // owner and every remaining handle.
    auto * tracker = vbr_generation_tracker_mut();
    for (auto & pending : vbr_awaiting_commit_) {
        bool committed = true;
        if (tracker != nullptr) {
            for (auto & extent : pending.extents) {
                if (extent && committed && !tracker->extent_store().commit(extent)) {
                    committed = false;
                    ++vbr_pending_commit_failures_;
                    tracker->set_shadow_unavailable();
                }
            }
            if (!committed) {
                // The op terminal-fails: EVERY handle fails, including any committed before
                // the failing one — failed operations leave no admissible evidence.
                vbr_fail_extent_set(tracker, pending.extents);
            }
        }
        if (pending.recovery_index >= 0 && committed) {
            vbr_recovery_release_unused(pending.recovery_index, pending.operation_id);
        }
        if (pending.composite) {
            // The sealed aggregate owns the root close: this slot's terminal
            // result feeds it, failure dominating.
            pending.composite->report_terminal(committed);
        } else if (pending.owns_close) {
            vbr_operation_registry_close(pending.operation_id,
                    committed ? vbr_operation_outcome::committed : vbr_operation_outcome::failed);
        }
    }
    vbr_awaiting_commit_.clear();
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

llama_kv_cache * llama_kv_cache::vbr_tree_root() {
    return vbr_ledger_root_ != nullptr ? vbr_ledger_root_ : this;
}

const llama_kv_cache * llama_kv_cache::vbr_tree_root() const {
    return vbr_ledger_root_ != nullptr ? vbr_ledger_root_ : this;
}

void llama_kv_cache::vbr_attach_ledger_tree(
        llama_kv_cache * root, llama_kv_cache * peer, double device_share) {
    GGML_ASSERT(root != nullptr);
    vbr_ledger_root_ = root;
    vbr_ledger_sibling_ = peer;
    vbr_ledger_owner_ = root == this;
    vbr_tree_device_share_ = device_share;
}

void llama_kv_cache::vbr_finalize_ledger_tree() {
    GGML_ASSERT(vbr_ledger_owner_ && vbr_tree_root() == this);

    // Constructor-time live budgets were derived before the peer backlink existed. Repair
    // only that fallback path; fit-provided and explicit scalar budgets retain their split.
    auto finalize_child = [](llama_kv_cache * child) {
        if (child == nullptr || child->vbr_budget_from_scalar_) {
            return;
        }
        size_t derived_total = 0;
        for (auto & p : child->vbr_pools_) {
            if (p.vmm == nullptr) {
                continue;
            }
            p.budget = std::max(child->vbr_pool_reach(p), p.budget_base);
            p.budget_eff_stamp = ~0ull;
            derived_total += p.budget;
        }
        if (derived_total > 0) {
            child->vbr_budget_bytes_ = std::max(derived_total, child->vbr_floor_cost_bytes_);
        }
    };
    finalize_child(this);
    finalize_child(vbr_ledger_sibling_);
}

void llama_kv_cache::vbr_finalize_failed_child(uint32_t n_tokens, bool root_ran) {
    llama_kv_cache * root = vbr_tree_root();
    if (root != this) {
        root->vbr_finalize_failed_child(n_tokens, root_ran);
        return;
    }

    // The root may not have run when the earlier child failed. Do not aggregate its stale
    // pressure sample. If the root itself failed, its current sample was already recorded.
    if (!root_ran) {
        root->vbr_runtime_was_over_ = false;
        root->vbr_runtime_wm_ = root->vbr_watermark_cells(n_tokens);
        root->vbr_runtime_demand_update(root->vbr_runtime_wm_, false);
    }

    // A child rejected a component reserve before its tier mutation.  Do not let finalization
    // consume the forced scan edge or service another donation from either child while this batch
    // is already returning failure.  Preserve the retry route and fence any earlier successful
    // steps from this boundary; the next boundary will re-run the full tree after memory changes.
    if (root->vbr_reserve_failed_ ||
        (root->vbr_ledger_sibling_ != nullptr && root->vbr_ledger_sibling_->vbr_reserve_failed_)) {
        root->vbr_tree_force();
        root->vbr_arm_wave_fences();
        return;
    }

    if (root->vbr_tree_forced() ||
        (root->vbr_boundary_count_ % 8 == 0 &&
         llama_vram_ledger_now_ns() - root->vbr_last_scan_ns_ >= 1000000000ull)) {
        root->vbr_ledger_scan_service(n_tokens);
    }
    // Service can queue waves on either child even though the parent is returning failure.
    root->vbr_arm_wave_fences();
}

bool llama_kv_cache::vbr_tree_forced() const {
    return vbr_tree_root()->vbr_ledger_force_;
}

void llama_kv_cache::vbr_tree_force() {
    vbr_tree_root()->vbr_ledger_force_ = true;
}

bool llama_kv_cache::vbr_tree_budget_explicit() const {
    const llama_kv_cache * root = vbr_tree_root();
    return root->vbr_budget_explicit_ ||
           (root->vbr_ledger_sibling_ != nullptr && root->vbr_ledger_sibling_->vbr_budget_explicit_);
}

size_t llama_kv_cache::vbr_tree_total_grant_decrement() const {
    const llama_kv_cache * root = vbr_tree_root();
    size_t total = root->vbr_total_grant_decrement();
    if (root->vbr_ledger_sibling_ != nullptr) {
        total += root->vbr_ledger_sibling_->vbr_total_grant_decrement();
    }
    return total;
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
// Auto budgets already preserve vbr_growth_headroom_ when periodically re-derived. Explicit
// budgets never re-derive—the typed value is a hard cap—so their live clamp must preserve that
// same fit headroom instead of only the smaller co-tenancy reserve. This tightens the effective
// per-device decision limit without changing or growing the user's configured cap.
// Shared by the degrade trigger AND the promote hysteresis: gating the two on different
// references (raw budget vs clamped) made every boundary under a co-tenant clamp promote
// then re-degrade — two transcodes plus one extra quantization hop on aged rows per flap.
size_t llama_kv_cache::vbr_budget_eff(const vbr_pool & p) const {
    // memoized per boundary: the degrade loop re-evaluates over_budget per step and the promote
    // hysteresis visits every pool, so an uncached get_device_memory here is a driver round-trip
    // multiplied by wave length x pools — and free VRAM cannot meaningfully move within one
    // boundary (tail unmaps are deferred to the next one)
    if (p.budget_eff_stamp == vbr_boundary_count_) {
        return p.budget_eff_cache;
    }
    const size_t budget_eff = vbr_budget_eff_uncached(p);
    p.budget_eff_stamp = vbr_boundary_count_;
    p.budget_eff_cache = budget_eff;
    return budget_eff;
}

size_t llama_kv_cache::vbr_budget_eff_uncached(const vbr_pool & p) const {
    size_t budget_eff = p.budget;
    const size_t mapped_now = p.be->vmm_pool_mapped(p.vmm); // deterministic (pool mapped bytes)
    // Freeze mode skips every live-VRAM/co-tenancy input, including the free-VRAM clamp
    // (cudaMemGetInfo), the N_live>1 fair-share relax, and the grant-decrement subtraction — so
    // budget_eff stays the fixed armed budget (still floored at mapped_now below). This clamp is the
    // ONLY live-VRAM path that fires under an explicit budget, so neutralizing it makes both the pp
    // one-shot fit and the tg degrade trajectory a pure function of the fixed budget + occupancy.
    // With freezing off this block runs verbatim, preserving the original behavior.
    if (!vbr_freeze_) {
        const size_t ledger_headroom = llama_vram_headroom_bytes();
        size_t free_b = 0, total_b = 0;
        p.be->get_device_memory(p.device, &free_b, &total_b);
        // Fairness: reserve ordinary ledger headroom for every live peer. An explicit
        // budget's own process instead reserves the (usually larger) fit growth target because it
        // has no auto re-derivation path. At N_live > 1 the budget base also relaxes to a fair
        // share of its own spare — mapped-anchored and 64 MiB-quantized.
        const uint32_t n_live = vbr_pool_n_live(p);
        // Explicit budgets reserve the fit growth target, but never more than the pool could
        // still MAP (budget - mapped): the fit margin prices consumers that have already
        // materialized by decode time (the spec drafter's model+context+compute raise
        // fit_params_target before the fit, then allocate at load) — free VRAM already
        // reflects them, so re-reserving their full size double-counts and strangles a
        // comfortably-fitting pool to the --vbr-floor. Measured 2026-08-10 (27B + DFlash
        // drafter at inherited -cd, --vbr-vram-budget 1024M): margin 3309 MiB vs 1937 MiB
        // free -> budget_eff collapsed to mapped (4 MiB) against a 68 MiB need and the first
        // boundary cascaded all 71 clamped steps to the floor.
        const size_t grow_room = p.budget > mapped_now ? p.budget - mapped_now : 0;
        const size_t own_headroom = vbr_budget_explicit_
                ? std::max(std::min(vbr_growth_headroom_, grow_room), ledger_headroom)
                : ledger_headroom;
        const size_t headroom_eff = own_headroom + ledger_headroom * (n_live - 1);
        if (n_live > 1) {
            constexpr size_t quantum = 64ull * 1024 * 1024;
            size_t va_total = 0;
            for (const auto & q : vbr_pools_) {
                va_total += q.vmm != nullptr ? q.size : 0;
            }
            const size_t floor_share = va_total > 0
                ? (size_t) ((double) vbr_floor_cost_bytes_ * (double) p.size / (double) va_total) : 0;
            const size_t spare = budget_eff > mapped_now ? budget_eff - mapped_now : 0;
            budget_eff = std::max(floor_share, mapped_now + spare / n_live / quantum * quantum);
        }
        const size_t cap = mapped_now + (free_b > headroom_eff ? free_b - headroom_eff : 0);
        if (cap < budget_eff) {
            budget_eff = cap;
        }
        // co-tenancy (mapped-anchored invariant): subtract this pool's unamortized grant
        // decrements, then floor at mapped — the floor binds precisely in the shed->flush
        // window so no consumer (including our own degrade loop) ever sees a budget below
        // what is physically mapped. budget_eff = max(mapped, min(budget, live_cap) - Σdecr).
        budget_eff = budget_eff > p.grant_decrement ? budget_eff - p.grant_decrement : 0;
    }
    if (budget_eff < mapped_now) {
        budget_eff = mapped_now;
    }
#ifndef NDEBUG
    GGML_ASSERT(budget_eff >= mapped_now);
#endif
    return budget_eff;
}

// Auto-budget runtime re-derivation: the startup number is a snapshot (fit-armed on whatever
// the box looked like at load, or the bare floor-cost fallback when no fit ran) — a co-tenant
// present at startup or a missing arm otherwise pinned quality low FOREVER on a box with
// gigabytes free. Once per boundary, re-derive each pool's budget from what its device can
// actually give it: share x (mapped + free − growth_headroom), quantized to 64 MiB so driver
// jitter cannot move a tier decision between identical runs, RE-DERIVED not max-ratcheted (a
// sawtooth co-tenant's trough must not be captured as permanent), floored at the pool's
// floor-layout cost for auto budgets (or the init-armed value for explicit budgets), and never
// touched at all for explicit budgets. Throttled
// by the caller (see prepare()); vbr_boundary_count_ is advanced there, not here.
size_t llama_kv_cache::vbr_pool_reach(const vbr_pool & p) const {
    constexpr size_t quantum = 64ull * 1024 * 1024;
    size_t free_b = 0, total_b = 0;
    p.be->get_device_memory(p.device, &free_b, &total_b);
    const size_t mapped_now = p.be->vmm_pool_mapped(p.vmm);
    const size_t spare = free_b > vbr_growth_headroom_ ? free_b - vbr_growth_headroom_ : 0;

    double share = vbr_params_.device_share;
    if (vbr_ledger_root_ != nullptr) {
        const llama_kv_cache * root = vbr_tree_root();
        size_t denom = 0;
        auto add_child = [&](const llama_kv_cache * child) {
            if (child == nullptr) {
                return;
            }
            for (const auto & q : child->vbr_pools_) {
                // Child constructors eagerly assign device ordinals before tree attachment.
                // Do not couple constructor-time normalization to marker-key resolution.
                if (q.vmm != nullptr && q.device == p.device) {
                    denom += q.size;
                }
            }
        };
        add_child(root);
        add_child(root->vbr_ledger_sibling_);
        share = denom > 0
                ? root->vbr_tree_device_share_ * (double) p.size / (double) denom
                : root->vbr_tree_device_share_;
    }

    // A child owns its mapped pages outright; only currently free device memory is divided.
    const size_t reach_raw = mapped_now + (size_t) ((double) spare * share);
    return std::max(mapped_now, reach_raw / quantum * quantum);
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
            p.budget = reach;
        }
    }
}

// Schedule-trace recorder (test/gating only). One line per boundary. The tier vector is
// the live type of every populated (layer,side) extent across all pools, in a stable pool->ikv->k,v
// order, folded into an FNV-1a digest; two runs whose (boundary,cursor,digest,wm,used) columns match
// at every boundary took the identical tier schedule. No effect on the schedule itself.
void llama_kv_cache::vbr_trace_emit(const char * phase, uint32_t wm, uint32_t used) {
    if (!vbr_trace_fp_) {
        return;
    }
    uint64_t fnv    = 1469598103934665603ull;
    size_t   mapped = 0;
    for (const auto & p : vbr_pools_) {
        for (size_t ikv = 0; ikv < p.k.size(); ++ikv) {
            const uint8_t kt = p.k[ikv].t != nullptr ? (uint8_t) p.k[ikv].t->type : 0xFF;
            const uint8_t vt = p.v[ikv].t != nullptr ? (uint8_t) p.v[ikv].t->type : 0xFF;
            fnv = (fnv ^ kt) * 1099511628211ull;
            fnv = (fnv ^ vt) * 1099511628211ull;
        }
        if (p.vmm != nullptr) {
            mapped += p.be->vmm_pool_mapped(p.vmm);
        }
    }
    fprintf(vbr_trace_fp_.get(), "%s\t%llu\t%zu\t%016llx\t%u\t%u\t%zu\n",
            phase, (unsigned long long) vbr_boundary_count_, vbr_degrade_cursor_,
            (unsigned long long) fnv, wm, used, mapped);
    // Traces are test evidence for fail-closed and injected-abort arms. Do not leave the latest
    // completed boundary stranded in stdio buffering when a later assertion terminates the run.
    fflush(vbr_trace_fp_.get());
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

// Boundary-time f16 dequant scratch reserve. The flash-attention prefill/materialize paths grow a
// per-(device, side) f16 scratch to the flattened attended view implicitly, mid-graph: a
// context-linear consumer for unified KV and n_kv*stream_span for non-unified KV. The budget
// does not own it, and it can JUMP from zero to watermark width in a
// single graph when a degrade wave first takes a side off f16 (a 217 MiB grow
// with only the 192 MiB live headroom left at wave time). Growing it HERE — sized for the sides
// that are dequant-active AFTER this boundary's wave — keeps every grow in an eager pass where
// exhaustion fails the batch recoverably. Sides that never leave f16 never reserve a byte, so
// symmetric-vbr sessions under no memory pressure are byte-identical to before. Covers static
// turbo pools too (bookkeeping-only pools resolve their vtable at init).
bool llama_kv_cache::vbr_scratch_reserve(size_t flat_cells) {
    for (auto & p : vbr_pools_) {
        if (p.be == nullptr || p.device < 0) {
            continue;
        }
        GGML_ASSERT(p.compute_backend != nullptr);
        // active-side row maxima change only on a tier flip — memoize on the tier epoch so the
        // per-boundary cost is two multiplies (static caches compute this exactly once)
        if (p.scratch_rows_epoch != vbr_tier_epoch_) {
            p.scratch_k_row = 0;
            p.scratch_v_row = 0;
            for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
                const ggml_tensor * tk = p.k[ikv].t;
                const ggml_tensor * tv = p.v[ikv].t;
                bool need_k = false;
                bool need_v = false;
                ggml_vbr_kv_dequant_sides(tk ? tk->type : GGML_TYPE_F16,
                                          tv ? tv->type : GGML_TYPE_F16, &need_k, &need_v);
                if (need_k && tk) {
                    p.scratch_k_row = std::max(p.scratch_k_row, ggml_row_size(GGML_TYPE_F16, tk->ne[0]));
                }
                if (need_v && tv) {
                    p.scratch_v_row = std::max(p.scratch_v_row, ggml_row_size(GGML_TYPE_F16, tv->ne[0]));
                }
            }
            p.scratch_rows_epoch = vbr_tier_epoch_;
        }
        const size_t k_bytes = p.scratch_k_row * flat_cells;
        const size_t v_bytes = p.scratch_v_row * flat_cells;
        if (k_bytes == 0 && v_bytes == 0) {
            continue;
        }
        if (!p.be->kv_dequant_scratch_reserve(p.compute_backend, k_bytes, v_bytes)) {
            // First-activation transient: the wave that just took this side off f16 queued its
            // freed tier-A tail pages as deferred unmaps (released at the NEXT boundary), so the
            // bytes the wave freed are physically unavailable to the very reserve it triggered.
            // Reclaim them now and retry once — mirrors vbr_vmm_try_map below.
            LLAMA_LOG_WARN("%s: f16 dequant scratch reserve of %.1f + %.1f MiB failed on device %d — "
                    "flushing deferred unmaps and retrying\n",
                    __func__, k_bytes/1048576.0, v_bytes/1048576.0, p.device);
            vbr_flush_deferred_unmaps();
            if (!p.be->kv_dequant_scratch_reserve(p.compute_backend, k_bytes, v_bytes)) {
                return false;
            }
        }
        p.scratch_k_reserved = std::max(p.scratch_k_reserved, k_bytes);
        p.scratch_v_reserved = std::max(p.scratch_v_reserved, v_bytes);
    }
    // Shared aliases are deliberately absent from vbr_pools_. Their live tensor types follow
    // the owner, so use the delegated epoch and reserve against this context's compute backend.
    // Multiple iSWA children call this serially on the same backend; the grow-only backend
    // scratch therefore lands at the per-side maximum rather than the sum.
    const uint64_t owner_epoch = vbr_tier_epoch();
    for (auto & b : vbr_shared_scratch_bindings_) {
        GGML_ASSERT(b.be != nullptr && b.compute_backend != nullptr && b.device >= 0);
        if (b.rows_epoch != owner_epoch) {
            b.k_row = 0;
            b.v_row = 0;
            for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
                const ggml_tensor * tk = b.k[ikv];
                const ggml_tensor * tv = b.v[ikv];
                bool need_k = false;
                bool need_v = false;
                ggml_vbr_kv_dequant_sides(tk ? tk->type : GGML_TYPE_F16,
                                          tv ? tv->type : GGML_TYPE_F16, &need_k, &need_v);
                if (need_k && tk) {
                    b.k_row = std::max(b.k_row, ggml_row_size(GGML_TYPE_F16, tk->ne[0]));
                }
                if (need_v && tv) {
                    b.v_row = std::max(b.v_row, ggml_row_size(GGML_TYPE_F16, tv->ne[0]));
                }
            }
            b.rows_epoch = owner_epoch;
        }
        const size_t k_bytes = b.k_row * flat_cells;
        const size_t v_bytes = b.v_row * flat_cells;
        if ((k_bytes != 0 || v_bytes != 0) &&
            !b.be->kv_dequant_scratch_reserve(b.compute_backend, k_bytes, v_bytes)) {
            // The owner's tier flip may still have old-tier tails queued for release. They
            // belong to the aliased tensors and are safe to reclaim with the same synchronized
            // flush used by the owner's own reserve path. Retry once before failing this draft
            // boundary recoverably.
            LLAMA_LOG_WARN("%s: shared-KV f16 dequant scratch reserve of %.1f + %.1f MiB failed "
                    "on device %d — flushing owner deferred unmaps and retrying\n",
                    __func__, k_bytes/1048576.0, v_bytes/1048576.0, b.device);
            other->vbr_flush_deferred_unmaps();
            if (!b.be->kv_dequant_scratch_reserve(b.compute_backend, k_bytes, v_bytes)) {
                return false;
            }
        }
    }
    return true;
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
        vbr_capture_watermark_publish(pool, wm);
    }
    return true;
}

void llama_kv_cache::vbr_vmm_ensure_mapped() {
    // the coming graph's writes land on positioned cells below the watermark; reads pad up to it.
    // The degrade trigger deliberately does not live here: apply_ubatch runs mid-batch
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
            total += std::min(vbr_slot_span(t, p.gran), (size_t) GGML_PAD(need, p.gran));
        }
    }
    return total;
}

// idle-time maintenance, decode-thread only (llama_memory_breathe). The co-tenancy tick:
// an idle resident runs no decode boundaries, so without this it is deaf to demands and
// its deferred unmaps never flush. Order is normative (design v3.8): flush FIRST (the
// grant math and mapped-floor argument assume freed physical lands before any budget
// evaluation), then stash-clear, budget rederive (throttled to every 8th tick, mirroring
// the boundary throttle), full-reset-if-empty, watermark shrink, ledger scan + demand
// service, promote step, fence-arm (MANDATORY — the next decode graph races the wave
// otherwise), boundary count++ (budget memo + promote pacing depend on it).
void llama_kv_cache::breathe() {
    vbr_runtime_was_over_ = false;
    const size_t flushed = vbr_flush_deferred_unmaps();
    if (flushed > 0) {
        LLAMA_LOG_DEBUG("%s: flushed %zu deferred VBR unmaps at idle\n", __func__, flushed);
    }
    if (!vbr_vmm_active() || vbr_budget_bytes_ == 0) {
        return;
    }
    vbr_retier_take_reconcile("breathe");
    llama_kv_cache * root = vbr_tree_root();
    root->vbr_reserve_failed_ = false;
    if (root->vbr_ledger_sibling_ != nullptr) {
        root->vbr_ledger_sibling_->vbr_reserve_failed_ = false;
    }
    vbr_invalidate_dirty_stash();
    if (!vbr_budget_explicit_ && vbr_boundary_count_ % 8 == 0) {
        vbr_rederive_budget();
    }
    uint32_t used_now = 0;
    for (uint32_t st = 0; st < n_stream; ++st) {
        used_now += v_cells[st].get_used();
    }
    if (vbr_degrade_cursor_ > 0 && used_now == 0 &&
        !vbr_freeze_preserve_empty_tiers_) {
        vbr_full_reset();
    }
    const uint32_t wm_next = vbr_watermark_cells(0);
    vbr_runtime_wm_ = wm_next;
    vbr_shrink_watermark();
    // Freeze mode disables the ledger for deterministic tests; skip precheck and idle scan for a deterministic
    // idle path. OFF => both run verbatim (bit-identical).
    if (!vbr_freeze_) {
        vbr_ledger_precheck();
        if (vbr_tree_forced() ||
            llama_vram_ledger_now_ns() - vbr_last_scan_ns_ >= 1000000000ull) {
            vbr_ledger_scan_service(0); // demand waves run band-capped inside
        }
        if (root->vbr_reserve_failed_) {
            // Idle ticks have no error return. Keep the rejected step at its exact cursor and retry
            // later; arm any earlier committed wave before leaving this tick.
            root->vbr_arm_wave_fences();
            return;
        }
        vbr_runtime_demand_update(wm_next, /*was_over=*/false); // tick: CLEAR path only
    }
    // no spontaneous degrade pressure at idle: budget_eff floors at mapped and nothing
    // grows here, so the general over-budget loop cannot fire — only demand decrements
    // (band-capped, in the scan) shed. Promotes get their idle chance under the same
    // gates as the boundary path.
    vbr_quiet_boundaries_++;
    vbr_maybe_promote(wm_next);
    vbr_arm_wave_fences();
    vbr_boundary_count_++;
    vbr_trace_emit("breathe", wm_next, used_now);
}

// Unmap tail pages queued by the previous degrade wave. Safe only after the wave's transcodes
// finished (they READ the old tier-A extent, which reaches into these pages) — one side-stream
// sync per pool makes that certain; by the next decode boundary the wave is long done, so this is
// ~free.
bool llama_kv_cache::vbr_retire_pending_before_unmap(const std::string & busid) {
    auto pending_it = vbr_grant_pending_.find(busid);
    if (pending_it == vbr_grant_pending_.end() || pending_it->second == 0) {
        return true;
    }

    // A pending grant bridges the interval between committing a shed and making its deferred
    // pages visible in cudaMemGetInfo(). Retire the bridge while those pages are still mapped;
    // doing this after the unmap lets a peer count the same bytes once as free and once as pending.
    // If publication fails, retain both the pages and the local liability for a forced retry.
    if (!llama_vram_ledger_armed() || busid == "-") {
        pending_it->second = 0;
        return true;
    }

    llama_kv_cache * root = vbr_tree_root();
    uint64_t remaining = 0;
    const auto add_child = [&](const llama_kv_cache * child) {
        if (child == nullptr || child == this) {
            return;
        }
        const auto it = child->vbr_grant_pending_.find(busid);
        if (it != child->vbr_grant_pending_.end()) {
            GGML_ASSERT(it->second <= UINT64_MAX - remaining);
            remaining += it->second;
        }
    };
    add_child(root);
    add_child(root->vbr_ledger_sibling_);

    llama_vram_marker_fields f = {};
    f.vbr            = 1;
    f.serviced       = llama_vram_marker_serviced_flag() ? 1u : 0u;
    f.shed_available = 0;
    f.grant_pending  = remaining;
    uint64_t created_ts = 0;
    if (!llama_vram_marker_publish(busid, f, &created_ts)) {
        // A rename may have succeeded before reopening the replacement failed and unlinked it.
        // Forget the cached publication so the forced retry recreates a truthful marker rather
        // than heartbeat-writing the old, now unlinked descriptor.
        root->vbr_marker_pub_.erase(busid);
        root->vbr_tx_suppress(busid);
        return false;
    }

    pending_it->second = 0;
    root->vbr_marker_pub_[busid] = { 0, remaining };
    root->vbr_marker_created_ts_[busid] = created_ts;
    root->vbr_ledger_mtime_ = llama_vram_ledger_dir_mtime_ns();
    return true;
}

size_t llama_kv_cache::vbr_flush_deferred_unmaps() {
    size_t flushed = 0;

    // Synchronize every affected side stream before withdrawing any bridge. Once a marker says
    // the bytes are no longer pending, keep the publication-to-unmap interval host-local and
    // contain no further GPU waits.
    for (auto & p : vbr_pools_) {
        if (p.unmap_deferred.empty()) {
            continue;
        }
        GGML_ASSERT(p.backend != nullptr); // entries are only queued after async work on it
        ggml_backend_synchronize(p.backend);
    }

    std::set<std::string> blocked;
    for (auto & p : vbr_pools_) {
        if (!p.unmap_deferred.empty() && blocked.count(vbr_pool_busid(p)) == 0 &&
            !vbr_retire_pending_before_unmap(vbr_pool_busid(p))) {
            blocked.insert(vbr_pool_busid(p));
        }
    }

    for (auto & p : vbr_pools_) {
        if (p.unmap_deferred.empty() || blocked.count(vbr_pool_busid(p)) != 0) {
            continue;
        }
        for (const auto & [off, len] : p.unmap_deferred) {
            p.be->vmm_pool_unmap(p.vmm, off, len);
        }
        flushed += p.unmap_deferred.size();
        p.unmap_deferred.clear();
    }
    return flushed;
}

void llama_kv_cache::vbr_invalidate_dirty_stash() {
    if (!vbr_stash_dirty_) {
        return;
    }
    for (auto & pool : vbr_pools_) {
        for (size_t i = 0; i < layers.size(); ++i) {
            pool.k[i].stash_valid = 0;
            pool.v[i].stash_valid = 0;
        }
    }
    vbr_stash_dirty_ = false;
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
// Shared degrade-ladder simulation core. Every consumer of the price order — the floor
// clamp/capacity sim, the pressure telemetry's bpv-if-degraded walk, and the co-tenancy
// offer — walks the same rules or the ADVERTS LIE: seed a per-(layer, side) type view,
// then per step apply exactly vbr_degrade_next's skip rules. Callers own their loop
// bounds, stop conditions and accounting; the rules live here once.
void llama_kv_cache::vbr_sim_seed(std::vector<ggml_type> & sim, bool pooled_only,
                                  ggml_type entry_k, ggml_type entry_v,
                                  double * sum_bits, int64_t * sum_vals, size_t * n_pinned) const {
    sim.assign(layers.size() * 2, GGML_TYPE_COUNT);
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
            if (sum_bits != nullptr) {
                *sum_bits += 8.0 * ggml_row_size(entry, t->ne[0]);
            }
            if (sum_vals != nullptr) {
                *sum_vals += t->ne[0];
            }
            if (n_pinned != nullptr) {
                *n_pinned += !vbr_unit_movable(entry, side != 0);
            }
        }
    }
}

// step i applicable? (same skip rules as vbr_degrade_next: unknown layer, absent/pinned
// unit, same-type or no-gain no-ops). On true, fills the slot/tensor/target for the
// caller's accounting; the caller applies sim[slot] = type_B when it accepts the step.
bool llama_kv_cache::vbr_sim_step(const std::vector<ggml_type> & sim, size_t i,
                                  size_t & slot, const ggml_tensor *& t, ggml_type & type_B) const {
    const auto & st = vbr_degrade_order_[i];
    const auto it = map_layer_ids.find(st.il);
    if (it == map_layer_ids.end()) {
        return false;
    }
    slot = (size_t) it->second * 2 + (st.is_v ? 1 : 0);
    t    = st.is_v ? layers[it->second].v : layers[it->second].k;
    if (sim[slot] == GGML_TYPE_COUNT || t == nullptr || !vbr_unit_movable(sim[slot], st.is_v != 0)) {
        return false; // absent or pinned (runtime skips these steps identically)
    }
    type_B = vbr_tier_type(st.tier);
    if (sim[slot] == type_B ||
        ggml_row_size(type_B, t->ne[0]) >= ggml_row_size(sim[slot], t->ne[0])) {
        return false; // same no-op rule as vbr_degrade_next
    }
    return true;
}

// Page-padded LOGICAL endpoint bytes for policy ordering.  This intentionally has no VMM pool or
// residency query: partial mappings and allocator history are physical-pricing inputs, never
// permission to change which measured layer loses quality next.
static bool vbr_policy_endpoint_bytes_checked(
        const ggml_tensor * t, ggml_type type, uint32_t wm,
        size_t gran, uint64_t & result) {
    if (t == nullptr || gran == 0) {
        return false;
    }
    const uint64_t slot = vbr_slot_span(t, gran);
    const uint64_t row  = ggml_row_size(type, t->ne[0]);
    return llama_vbr_policy::logical_endpoint_bytes(
        row, wm, slot, gran, result);
}

static uint64_t vbr_policy_endpoint_bytes(
        const ggml_tensor * t, ggml_type type, uint32_t wm, size_t gran) {
    uint64_t result = 0;
    if (!vbr_policy_endpoint_bytes_checked(t, type, wm, gran, result)) {
        GGML_ABORT("VBR policy endpoint overflow or invalid geometry");
    }
    return result;
}

bool llama_kv_cache::vbr_policy_priced_steps(
        std::vector<ggml_type> & sim, size_t start_cursor,
        int demanded_device, uint32_t watermark, bool fixed_watermark,
        bool fail_closed, llama_vbr_policy::child & out,
        vbr_hard_seal_consult_session * seal_session) const {
    int64_t terminal = out.initial_progress;
    for (size_t i = start_cursor; i < vbr_demand_limit(); ++i) {
        size_t slot = 0;
        const ggml_tensor * canonical = nullptr;
        ggml_type type_b = GGML_TYPE_COUNT;
        if (!vbr_sim_step(sim, i, slot, canonical, type_b)) {
            continue;
        }
        const auto & order = vbr_degrade_order_[i];
        const size_t ikv = slot/2;
        int64_t gain = 0;
        bool valid = true;
        for (const auto & pool : vbr_pools_) {
            if (!pool.vmm ||
                (demanded_device >= 0 && pool.device != demanded_device)) {
                continue;
            }
            const auto & extent = order.is_v ? pool.v[ikv] : pool.k[ikv];
            if (!extent.t) {
                continue;
            }
            const uint32_t wm = fixed_watermark
                ? watermark : std::max(pool.wm_cells, watermark);
            uint64_t before = 0;
            uint64_t after = 0;
            int64_t next = 0;
            valid = vbr_policy_endpoint_bytes_checked(
                        extent.t, sim[slot], wm, pool.gran, before) &&
                    vbr_policy_endpoint_bytes_checked(
                        extent.t, type_b, wm, pool.gran, after) &&
                    before >= after && before-after <= uint64_t(INT64_MAX) &&
                    llama_vbr_policy::checked_add(
                        gain, int64_t(before-after), next);
            if (!valid) {
                break;
            }
            gain = next;
        }
        int64_t next_terminal = 0;
        valid = valid && llama_vbr_policy::checked_add(
            terminal, gain, next_terminal);
        if (!valid) {
            if (fail_closed) {
                return false;
            }
            GGML_ABORT("VBR policy progress overflow or invalid geometry");
        }
        if (seal_session != nullptr &&
            vbr_hard_seal_step_blocked(i, *seal_session)) {
            out.blocked_order_indices.push_back(i);
            continue;
        }
        if (!vbr_capture_unit_write_plan_available(
                static_cast<uint32_t>(slot))) {
            out.capture_blocked_order_indices.push_back(i);
            continue;
        }
        out.steps.push_back({
            i, slot, int32_t(sim[slot]), int32_t(type_b), gain,
        });
        terminal = next_terminal;
        sim[slot] = type_b;
    }
    out.terminal_progress = terminal;
    return true;
}

llama_vbr_policy::child llama_kv_cache::vbr_policy_child_stream(
        int demanded_device, uint32_t wm_next) const {
    llama_vbr_policy::child out;
    std::vector<ggml_type> sim;
    vbr_sim_seed(sim, /*pooled_only=*/true, GGML_TYPE_COUNT, GGML_TYPE_COUNT,
                 nullptr, nullptr, nullptr);

    // Incoming watermark growth is part of the logical baseline-to-prefix delta.  It can make
    // initial progress negative; the interleaver clamps only for its ratio comparison.
    for (const auto & p : vbr_pools_) {
        if (p.vmm == nullptr || p.device != demanded_device) {
            continue;
        }
        const uint32_t terminal_wm = std::max(p.wm_cells, wm_next);
        for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
            for (int side = 0; side < 2; ++side) {
                const vbr_extent & e = side ? p.v[ikv] : p.k[ikv];
                if (e.t == nullptr) {
                    continue;
                }
                const size_t slot = ikv*2 + (side ? 1 : 0);
                const ggml_type type = sim[slot] != GGML_TYPE_COUNT ? sim[slot] : e.t->type;
                const uint64_t before = vbr_policy_endpoint_bytes(e.t, type, p.wm_cells, p.gran);
                const uint64_t after  = vbr_policy_endpoint_bytes(e.t, type, terminal_wm, p.gran);
                GGML_ASSERT(before <= (uint64_t) INT64_MAX && after <= (uint64_t) INT64_MAX);
                int64_t next = 0;
                if (!llama_vbr_policy::checked_add(
                        out.initial_progress,
                        (int64_t) before - (int64_t) after, next)) {
                    GGML_ABORT("VBR policy progress overflow");
                }
                out.initial_progress = next;
            }
        }
    }

    vbr_hard_seal_consult_session seal_session;
    GGML_ASSERT(vbr_policy_priced_steps(
        sim, vbr_degrade_cursor_, demanded_device, wm_next,
        false, false, out,
        vbr_hard_seal_guard_ ? &seal_session : nullptr));
    return out;
}

llama_kv_cache::vbr_floor_sim_result llama_kv_cache::vbr_floor_sim(
        double floor_bpv, bool pooled_only, ggml_type entry_k, ggml_type entry_v) const {
    vbr_floor_sim_result res;
    auto & sim = res.end_types;
    double  sum_bits = 0.0;
    int64_t sum_vals = 0;
    vbr_sim_seed(sim, pooled_only, entry_k, entry_v, &sum_bits, &sum_vals, &res.n_pinned);
    res.clamp_step = vbr_degrade_order_.size();
    if (sum_vals == 0) {
        return res;
    }
    res.initial_bits_per_token = sum_bits;
    res.initial_bpv = sum_bits / sum_vals;
    res.floor_reachable = vbr_floor_reachable(res.initial_bpv, floor_bpv);
    if (!res.floor_reachable) {
        res.bits_per_token = sum_bits;
        return res;
    }
    for (size_t i = 0; i < vbr_degrade_order_.size(); ++i) {
        size_t slot; const ggml_tensor * t; ggml_type type_B;
        if (!vbr_sim_step(sim, i, slot, t, type_B)) {
            continue;
        }
        const size_t rA = ggml_row_size(sim[slot], t->ne[0]);
        const size_t rB = ggml_row_size(type_B,    t->ne[0]);
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
    const auto res = vbr_floor_sim(
        vbr_resolve_floor_bpv(floor_bpv), !vbr_pools_.empty(), entry_k, entry_v);
    return res.floor_reachable ? res.bits_per_token : -1.0;
}

double llama_kv_cache::memory_vbr_entry_bits_per_token(ggml_type entry_k, ggml_type entry_v) {
    if (vbr_degrade_order_.empty()) {
        vbr_load_degrade_order();
    }
    return vbr_floor_sim(0.0, !vbr_pools_.empty(), entry_k, entry_v).initial_bits_per_token;
}

// Per-token bytes of flash-attention f16 dequant scratch at the settled deep-fill state,
// summed over KV-hosting devices. The scratch is one f16-width buffer per (device, side),
// shared across layers — its per-token cost is the widest layer's f16 row, NOT a per-layer
// sum. A side contributes when its settled state needs dequant: static/pinned turbo entry
// types always; movable (unpinned f16) sides in dynamic mode whenever the floor is below f16
// (they leave f16 under pressure at exactly the depths where this cost matters); q8_0/bf16
// pinned sides only next to an active partner. The fit charges this in the total-VRAM wall
// constraint (vbr_growth_reachable_ctx_cap) ONLY — the auto/explicit KV budget solves must
// not carry it (the scratch draws from the fit margin / free VRAM, not from the KV budget).
double llama_kv_cache::memory_vbr_scratch_bytes_per_token(ggml_type entry_k, ggml_type entry_v, double floor_bpv) {
    if (layers.empty()) {
        return 0.0;
    }
    // Project each side to its SETTLED (deep-fill) type, then ask the one authoritative
    // materialize predicate (ggml-vbr.h): the only genuine difference between "settled active"
    // and "currently active" is that an unpinned dynamic f16 side will leave f16 under pressure
    // — represent it by any turbo tier and let the predicate own the pairing rules.
    const double floor_eff = vbr_resolve_floor_bpv(floor_bpv);
    auto settled_type = [&](ggml_type t0, bool pinned) -> ggml_type {
        if (t0 == GGML_TYPE_F16 && !pinned && vbr_params_.dynamic && floor_eff < 16.0 - 1e-9) {
            return GGML_TYPE_TURBO8_0; // representative: degrades off f16 under pressure
        }
        return t0;
    };
    const ggml_type ek = entry_k != GGML_TYPE_COUNT ? entry_k
                       : (layers[0].k ? layers[0].k->type : GGML_TYPE_F16);
    const ggml_type ev = entry_v != GGML_TYPE_COUNT ? entry_v
                       : (layers[0].v ? layers[0].v->type : GGML_TYPE_F16);
    bool ak = false;
    bool av = false;
    ggml_vbr_kv_dequant_sides(settled_type(ek, vbr_params_.pin_k),
                              settled_type(ev, vbr_params_.pin_v), &ak, &av);
    if (!ak && !av) {
        return 0.0;
    }
    // Widest f16 row per active side over the canonical layer tensors — a SINGLE-DEVICE basis:
    // the only caller is the fit's no_alloc dry-load context, where pools are never built and
    // the dry load is single-device. (A live multi-device caller would need per-pool scratch
    // sums; deliberately not built for a dead path.)
    size_t k_row = 0;
    size_t v_row = 0;
    for (const auto & L : layers) {
        if (L.k != nullptr) {
            k_row = std::max(k_row, ggml_row_size(GGML_TYPE_F16, L.k->ne[0]));
        }
        if (L.v != nullptr) {
            v_row = std::max(v_row, ggml_row_size(GGML_TYPE_F16, L.v->ne[0]));
        }
    }
    return (ak ? (double) k_row : 0.0) + (av ? (double) v_row : 0.0);
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
    if (!res.floor_reachable) {
        throw std::runtime_error("VBR aggregate floor " + std::to_string(floor_bpv) +
            " bits/value exceeds the mixed K/V entry layout (" +
            std::to_string(res.initial_bpv) + " bits/value)");
    }
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
// build time; callers order the GPU against the matching transcode via the wave fence
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

// Transaction apply has crossed its recoverable-preflight boundary.  Update the same canonical,
// stream-view and tensor-split metadata without constructing the temporary shard-view vector used
// by the ordinary path.
static void vbr_set_tensor_type_noalloc(
        ggml_tensor * t, std::vector<ggml_tensor *> & views, ggml_type type) {
    vbr_set_tensor_type_impl(t, views, type);
    if (t->buffer == nullptr || !ggml_backend_buffer_is_meta(t->buffer)) {
        return;
    }
    const size_t n = ggml_backend_meta_buffer_n_bufs(t->buffer);
    for (size_t i = 0; i < n; ++i) {
        ggml_tensor * shard = ggml_backend_meta_buffer_simple_tensor(t, i);
        if (shard == nullptr) {
            continue;
        }
        shard->type  = type;
        shard->nb[0] = ggml_type_size(type);
        shard->nb[1] = ggml_row_size(type, shard->ne[0]);
        shard->nb[2] = shard->nb[1]*shard->ne[1];
        shard->nb[3] = shard->nb[2]*shard->ne[2];
        for (ggml_tensor * view : views) {
            ggml_tensor * shard_view = view != nullptr
                    ? ggml_backend_meta_buffer_simple_tensor(view, i) : nullptr;
            if (shard_view != nullptr) {
                shard_view->type  = type;
                shard_view->nb[0] = shard->nb[0];
                shard_view->nb[1] = shard->nb[1];
                shard_view->nb[2] = shard->nb[2];
                shard_view->nb[3] = shard->nb[3];
            }
        }
    }
}

bool llama_kv_cache::vbr_stash_requests_valid(
        const vbr_pool & p, const std::vector<vbr_stash_request> & requests,
        uint32_t stash_rows, bool ownership_authenticated,
        bool & needs_mapping, uint64_t * request_checks) {
    needs_mapping = false;
    if (request_checks) {
        *request_checks = 0;
    }
    if (requests.empty()) {
        return true;
    }
    if (p.stash_size == 0 || p.gran == 0) {
        return false;
    }
    for (const auto & request : requests) {
        if (request_checks) {
            ++*request_checks;
        }
        const vbr_extent * e = request.extent;
        if (e == nullptr || e->t == nullptr || request.rows > stash_rows) {
            return false;
        }
        if (!ownership_authenticated &&
            !std::any_of(p.k.begin(), p.k.end(),
                [&](const vbr_extent & candidate) { return &candidate == e; }) &&
            !std::any_of(p.v.begin(), p.v.end(),
                [&](const vbr_extent & candidate) { return &candidate == e; })) {
            return false;
        }
        const size_t ne0 = (size_t) e->t->ne[0];
        if (ne0 > SIZE_MAX / sizeof(uint16_t) ||
            (size_t) request.rows > SIZE_MAX / (ne0 * sizeof(uint16_t))) {
            return false;
        }
        const size_t bytes = (size_t) request.rows * ne0 * sizeof(uint16_t);
        if (bytes == 0) {
            continue;
        }
        needs_mapping = true;
        if (e->stash_off > p.stash_size || bytes > p.stash_size - e->stash_off) {
            return false;
        }
    }
    return true;
}

bool llama_kv_cache::vbr_stash_memory_impl(
        const vbr_pool & p, const std::vector<vbr_stash_request> & requests,
        bool ownership_authenticated,
        size_t & physical_now, size_t & physical_if_reserved,
        uint64_t * request_checks) const {
    physical_now = p.stash_vmm != nullptr ? p.be->vmm_pool_mapped(p.stash_vmm) : 0;
    physical_if_reserved = physical_now;
    bool needs_mapping = false;
    if (!vbr_stash_requests_valid(
            p, requests, vbr_stash_rows_, ownership_authenticated,
            needs_mapping, request_checks)) {
        return false;
    }
    // One capture pins the slab for the cache lifetime, just as the prior cudaMalloc did. Keeping
    // this a complete page-padded endpoint makes a transaction's stash price independent of which
    // extent happens to be the first capture and avoids page-union arithmetic in the policy layer.
    if (needs_mapping) {
        physical_if_reserved = std::max(physical_now, p.stash_size);
    }
    return true;
}

bool llama_kv_cache::vbr_stash_memory(
        const vbr_pool & p, const std::vector<vbr_stash_request> & requests,
        size_t & physical_now, size_t & physical_if_reserved) const {
    return vbr_stash_memory_impl(
        p, requests, false, physical_now, physical_if_reserved);
}

bool llama_kv_cache::vbr_stash_memory_trusted(
        const vbr_pool & p, const std::vector<vbr_stash_request> & requests,
        size_t & physical_now, size_t & physical_if_reserved) const {
    return vbr_stash_memory_impl(
        p, requests, true, physical_now, physical_if_reserved);
}

bool llama_kv_cache::vbr_stash_reserve(
        vbr_pool & p, const std::vector<vbr_stash_request> & requests) {
    return vbr_stash_reserve_impl(p, requests, false);
}

bool llama_kv_cache::vbr_stash_reserve_trusted(
        vbr_pool & p, const std::vector<vbr_stash_request> & requests) {
    return vbr_stash_reserve_impl(p, requests, true);
}

bool llama_kv_cache::vbr_stash_reserve_impl(
        vbr_pool & p, const std::vector<vbr_stash_request> & requests,
        bool ownership_authenticated) {
    size_t physical_now = 0;
    size_t physical_if_reserved = 0;
    if (!vbr_stash_memory_impl(
            p, requests, ownership_authenticated,
            physical_now, physical_if_reserved)) {
        return false;
    }
    if (physical_if_reserved == physical_now) {
        return true;
    }
    if (p.stash_vmm == nullptr) {
        p.stash_vmm = p.be->vmm_pool_init(p.device, p.stash_size);
        if (p.stash_vmm == nullptr) {
            return false;
        }
    }

    // The complete slab is one deterministic transaction endpoint. vmm_pool_map is idempotent and
    // retains a settled prefix on failure, so retry grows only the missing pages.
    if (!p.be->vmm_pool_map(p.stash_vmm, 0, p.stash_size)) {
        const size_t physical_after = p.be->vmm_pool_mapped(p.stash_vmm);
        LLAMA_LOG_WARN("%s: VBR sink-stash reserve failed on device %d after mapping %.2f "
                "MiB / projected %.2f MiB (%.2f MiB VA); retry is idempotent\n",
                __func__, p.device, physical_after/1048576.0,
                physical_if_reserved/1048576.0, p.stash_size/1048576.0);
        return false;
    }
    const size_t physical_after = p.be->vmm_pool_mapped(p.stash_vmm);
    GGML_ASSERT(physical_after >= physical_if_reserved);
    LLAMA_LOG_INFO("%s: VBR sink-stash reserve (device %d): %.2f MiB mapped / %.2f MiB VA\n",
            __func__, p.device, physical_after/1048576.0, p.stash_size/1048576.0);
    return true;
}

bool llama_kv_cache::vbr_import_transform_reserve(
        const std::vector<const vbr_validated_child_plan *> & plans,
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget,
        vbr_downward_stage_reservation & output) noexcept {
    output = {};
    struct stash_context {
        llama_kv_cache * cache = nullptr;
        vbr_pool * pool = nullptr;
        std::vector<vbr_stash_request> requests;
        std::vector<uint64_t> unit_ids;
        bool required = false;
    };
    struct pool_projection {
        vbr_pool * pool = nullptr;
        vbr_downward_workspace_endpoint workspace;
        stash_context stash;
        llama_cache_acct_resource_domain domain;
        bool have_domain = false;
    };
    try {
        if (plans.empty()) {
            output.status = vbr_downward_reserve_status::projection_unavailable;
            return false;
        }
        std::unordered_map<const void *, size_t> pool_indices;
        pool_indices.reserve(vbr_pools_.size());
        std::vector<pool_projection> projections(vbr_pools_.size());
        for (size_t i = 0; i < vbr_pools_.size(); ++i) {
            auto & pool = vbr_pools_[i];
            if (!pool_indices.emplace(&pool, i).second) {
                output.status =
                    vbr_downward_reserve_status::projection_unavailable;
                return false;
            }
            auto & projection = projections[i];
            projection.pool = &pool;
            projection.workspace.owner = &pool;
            projection.workspace.iface = pool.be;
            projection.workspace.cross_iface = pool.cross_be;
            projection.workspace.backend = pool.backend;
            projection.workspace.device = pool.device;
            projection.stash.cache = this;
            projection.stash.pool = &pool;
        }
        output.stashless_units.reserve(plans.size());
        output.status = vbr_downward_reserve_status::reserved;
        bool have_resource = false;
        for (const auto * plan : plans) {
            if (!plan) {
                continue;
            }
            const bool stash_only =
                plan->transform_kind == vbr_import_transform_kind::none &&
                plan->stash_action ==
                    vbr_validated_stash_action::restore_exact;
            if (plan->transform_kind == vbr_import_transform_kind::none &&
                !stash_only) {
                continue;
            }
            if (!stash_only && plan->transform_kind !=
                    vbr_import_transform_kind::downward &&
                plan->transform_kind !=
                    vbr_import_transform_kind::upward_same_domain &&
                plan->transform_kind !=
                    vbr_import_transform_kind::upward_cross_domain) {
                output.status =
                    vbr_downward_reserve_status::projection_unavailable;
                return false;
            }
            have_resource = true;
            const size_t ikv = plan->logical_unit_id/2;
            const bool is_v = (plan->logical_unit_id & 1u) != 0;
            if (ikv >= layers.size()) {
                output.status =
                    vbr_downward_reserve_status::projection_unavailable;
                return false;
            }
            const auto & units = vbr_units_of(ikv, is_v);
            if (units.size() != plan->shards.size()) {
                output.status =
                    vbr_downward_reserve_status::projection_unavailable;
                return false;
            }
            uint32_t plan_stash_rows = 0;
            bool required_stash = false;
            if (plan->transform_kind ==
                    vbr_import_transform_kind::downward &&
                vbr_downward_recipe_needs_stash(
                    plan->transcode_recipe) && vbr_stash_rows_ > 0) {
                plan_stash_rows = std::min<uint64_t>(
                    vbr_stash_rows_, plan->descriptor.wm_cells);
            } else if ((plan->stash_action ==
                            vbr_validated_stash_action::restore_exact ||
                        plan->stash_action ==
                            vbr_validated_stash_action::consume_exact_then_drop) &&
                       (stash_only ||
                        (plan->transform_kind ==
                             vbr_import_transform_kind::upward_same_domain &&
                         plan->source_domain == vbr_repr_domain::tapped) ||
                        plan->transform_kind ==
                            vbr_import_transform_kind::upward_cross_domain)) {
                const auto & stash = plan->descriptor.clean_stash;
                if (plan->descriptor.clean_stash_state !=
                        vbr_artifact_clean_stash_state::present ||
                    stash.domain != vbr_repr_domain::tapped ||
                    stash.valid_rows == 0 ||
                    stash.valid_rows > plan->descriptor.wm_cells ||
                    stash.valid_rows > vbr_stash_rows_ ||
                    stash.valid_rows > UINT32_MAX) {
                    output.status =
                        vbr_downward_reserve_status::projection_unavailable;
                    return false;
                }
                plan_stash_rows = uint32_t(stash.valid_rows);
                required_stash = true;
            }
            for (const auto & shard : plan->shards) {
                const auto pool_index = pool_indices.find(
                    shard.target_pool_cookie);
                if (pool_index == pool_indices.end()) {
                    output.status =
                        vbr_downward_reserve_status::projection_unavailable;
                    return false;
                }
                auto & projection = projections[pool_index->second];
                auto & pool = *projection.pool;
                const bool cross_domain = plan->transform_kind ==
                    vbr_import_transform_kind::upward_cross_domain;
                if (shard.shard_index >= units.size() ||
                    units[shard.shard_index].first != &pool ||
                    !units[shard.shard_index].second ||
                    !units[shard.shard_index].second->t ||
                    pool.be == nullptr ||
                    (!stash_only &&
                     (pool.be->kv_transcode_workspace_memory == nullptr ||
                      pool.be->kv_transcode_workspace_reserve == nullptr)) ||
                    (cross_domain &&
                     (pool.cross_be == nullptr ||
                      pool.cross_be->kv_cross_domain_reconstruct == nullptr ||
                      pool.cross_be->kv_transcode_workspace_memory_v2 == nullptr ||
                      pool.cross_be->kv_transcode_workspace_reserve_v2 == nullptr))) {
                    output.status =
                        vbr_downward_reserve_status::projection_unavailable;
                    return false;
                }
                if (!projection.have_domain) {
                    projection.domain = shard.domain;
                    projection.workspace.domain = shard.domain;
                    projection.have_domain = true;
                } else if (projection.domain != shard.domain) {
                    output.status =
                        vbr_downward_reserve_status::projection_unavailable;
                    return false;
                }
                if (!stash_only) {
                    projection.workspace.requests.push_back({
                        int64_t(plan->descriptor.wm_cells),
                        units[shard.shard_index].second->t->ne[0],
                        int64_t(plan_stash_rows),
                        cross_domain,
                    });
                }
                if (plan_stash_rows > 0) {
                    projection.stash.requests.push_back({
                        units[shard.shard_index].second, plan_stash_rows,
                    });
                    projection.stash.unit_ids.push_back(
                        vbr_downward_unit_key(
                            plan->child_id, plan->logical_unit_id));
                    projection.stash.required =
                        projection.stash.required || required_stash;
                }
            }
        }
        if (!have_resource) {
            output.status =
                vbr_downward_reserve_status::projection_unavailable;
            return false;
        }
        for (auto & projection : projections) {
            if (projection.workspace.requests.empty() &&
                projection.stash.requests.empty()) {
                continue;
            }
            auto & pool = *projection.pool;
            if (!projection.workspace.requests.empty() &&
                pool.backend == nullptr) {
                pool.backend = pool.be->backend_init(pool.device);
                if (pool.backend == nullptr) {
                    output.status =
                        vbr_downward_reserve_status::workspace_reserve_failed;
                    return true;
                }
                projection.workspace.backend = pool.backend;
            }
            std::vector<vbr_downward_workspace_endpoint> workspaces;
            std::vector<vbr_downward_stash_endpoint> stashes;
            if (!projection.workspace.requests.empty()) {
                workspaces.push_back(std::move(projection.workspace));
            }
            if (!projection.stash.requests.empty()) {
                vbr_downward_stash_endpoint endpoint;
                endpoint.owner = &pool;
                endpoint.unit_ids = std::move(projection.stash.unit_ids);
                endpoint.required = projection.stash.required;
                endpoint.domain = projection.domain;
                endpoint.context = &projection.stash;
                endpoint.memory = [](void * opaque, uint64_t & now,
                                     uint64_t & reserved) {
                    auto & value = *static_cast<stash_context *>(opaque);
                    size_t a = 0;
                    size_t b = 0;
                    if (!value.cache->vbr_stash_memory_trusted(
                            *value.pool, value.requests, a, b)) {
                        return false;
                    }
                    now = a;
                    reserved = b;
                    return true;
                };
                endpoint.reserve = [](void * opaque) {
                    auto & value = *static_cast<stash_context *>(opaque);
                    return value.cache->vbr_stash_reserve_trusted(
                        *value.pool, value.requests);
                };
                stashes.push_back(std::move(endpoint));
            }
            if (!pool.transform_receipts) {
                pool.transform_receipts =
                    std::make_unique<vbr_downward_resource_receipts>(ledger);
            }
            const auto reserved = pool.transform_receipts->reserve_resources(
                budget, workspaces, stashes);
            if (reserved.status != vbr_downward_reserve_status::reserved &&
                reserved.status !=
                    vbr_downward_reserve_status::reserved_stashless) {
                output.status = reserved.status;
                return true;
            }
            output.stashless_units.insert(
                output.stashless_units.end(),
                reserved.stashless_units.begin(),
                reserved.stashless_units.end());
            if (!reserved.stashless_units.empty()) {
                output.status =
                    vbr_downward_reserve_status::reserved_stashless;
            }
        }
        return true;
    } catch (...) {
        output.status = vbr_downward_reserve_status::internal_error;
        return false;
    }
}

bool llama_kv_cache::vbr_downward_policy_input(
        const std::vector<ggml_type> & source_types,
        uint64_t source_cursor,
        uint32_t projected_wm_cells,
        int demanded_device,
        vbr_downward_policy_child & output) const noexcept {
    output = {};
    try {
        if (source_types.size() != layers.size()*2 ||
            projected_wm_cells == 0 ||
            source_cursor > vbr_demand_limit()) {
            return false;
        }
        output.initial_types = source_types;
        output.target_types.resize(source_types.size(), GGML_TYPE_COUNT);
        output.initial_cursor = source_cursor;
        for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
            output.target_types[ikv*2] =
                layers[ikv].k ? layers[ikv].k->type : GGML_TYPE_COUNT;
            output.target_types[ikv*2 + 1] =
                layers[ikv].v ? layers[ikv].v->type : GGML_TYPE_COUNT;
        }
        auto sim = source_types;
        if (!vbr_policy_priced_steps(
                sim, size_t(source_cursor), demanded_device,
                projected_wm_cells, true, true, output.policy)) {
            return false;
        }
        return output.policy.terminal_progress > 0 ||
            output.initial_types == output.target_types;
    } catch (...) {
        output = {};
        return false;
    }
}

bool llama_kv_cache::vbr_import_destination_input(
        uint32_t projected_wm_cells,
        vbr_import_destination_child & output) const noexcept {
    output = {};
    try {
        if (other) {
            return other->vbr_import_destination_input(
                projected_wm_cells, output);
        }
        if (!vbr_vmm_active() || projected_wm_cells == 0) {
            return false;
        }
        output.watermark_cells = projected_wm_cells;
        output.initial_cursor = vbr_degrade_cursor_;
        std::vector<ggml_type> policy_types;
        vbr_sim_seed(policy_types, /* pooled_only = */ true,
                     GGML_TYPE_COUNT, GGML_TYPE_COUNT,
                     nullptr, nullptr, nullptr);
        auto sim = policy_types;
        // Destination negotiation is tree-wide and may span several devices.
        // Summing logical gain across every pool preserves the controller's
        // canonical per-child ladder while giving the tree interleaver one
        // honest progress denominator.
        vbr_hard_seal_consult_session seal_session;
        if (!vbr_policy_priced_steps(
            sim, vbr_degrade_cursor_, /* demanded_device = */ -1,
            projected_wm_cells, true, true, output.policy,
            vbr_hard_seal_guard_ ? &seal_session : nullptr)) {
            return false;
        }
        output.initial_types = std::move(policy_types);
        // The policy stream contains only VMM-pooled movable units, but the
        // negotiated identity must cover the complete controller vector.
        // Fill non-pooled slots from the live canonical tensors only after
        // the stream has been minted so they can never become policy steps.
        for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
            if (output.initial_types[ikv*2] == GGML_TYPE_COUNT &&
                layers[ikv].k) {
                output.initial_types[ikv*2] = layers[ikv].k->type;
            }
            if (output.initial_types[ikv*2 + 1] == GGML_TYPE_COUNT &&
                layers[ikv].v) {
                output.initial_types[ikv*2 + 1] = layers[ikv].v->type;
            }
        }
        return true;
    } catch (...) {
        output = {};
        return false;
    }
}

bool llama_kv_cache::vbr_import_bind_target_unit(
        const vbr_artifact_unit_descriptor & source,
        ggml_type target_type,
        const vbr_upward_representation_identity & selected_source_identity,
        const vbr_upward_representation_identity & selected_target_identity,
        const vbr_downward_policy_projection & projection,
        uint32_t projection_child,
        vbr_target_unit_snapshot & output) const noexcept {
    try {
        const auto source_type = static_cast<ggml_type>(source.current_type);
        const size_t ikv = output.logical_unit_id/2;
        const bool is_v = (output.logical_unit_id & 1u) != 0;
        if (ikv >= layers.size() || projection.status !=
                vbr_downward_policy_status::coherent ||
            projection_child >= projection.final_types.size() ||
            projection_child >= projection.child_type_digests.size() ||
            output.logical_unit_id >=
                projection.final_types[projection_child].size()) {
            return false;
        }
        const ggml_tensor * canonical = is_v ? layers[ikv].v : layers[ikv].k;
        if (!canonical || target_type == GGML_TYPE_COUNT ||
            projection.final_types[projection_child]
                [output.logical_unit_id] != target_type) {
            return false;
        }
        vbr_capture_stability_token policy;
        if (!vbr_capture_policy_snapshot(policy)) {
            return false;
        }
        const bool movable = vbr_unit_movable(source_type, is_v);
        vbr_downward_recipe recipe;
        const auto relation = vbr_downward_resolve_recipe(
            source_type, target_type,
            static_cast<ggml_type>(policy.floor_type), movable, recipe);
        if (relation != vbr_downward_recipe_status::resolved &&
            relation != vbr_downward_recipe_status::equal_tier &&
            relation != vbr_downward_recipe_status::upward_forbidden) {
            return false;
        }
        const bool downward =
            relation == vbr_downward_recipe_status::resolved;
        vbr_upward_recipe upward_recipe;
        const bool upward =
            relation == vbr_downward_recipe_status::upward_forbidden &&
            vbr_upward_resolve_recipe(
                source_type, target_type, upward_recipe) ==
                    vbr_upward_recipe_status::resolved;
        const bool cross_domain = upward &&
            vbr_downward_tier_domain(source_type) !=
                vbr_downward_tier_domain(target_type);
        const bool transformed = downward || upward;
        const auto & units = vbr_units_of(ikv, is_v);
        if (units.empty() || units.size() != output.shards.size()) {
            return false;
        }
        uint64_t mapped = 0;
        uint64_t transfer = 0;
        uint64_t workspace = 0;
        const int64_t stash_rows = vbr_downward_recipe_needs_stash(recipe)
            ? std::min<uint64_t>(vbr_stash_rows_, output.wm_cells)
            : 0;
        for (size_t i = 0; i < units.size(); ++i) {
            const auto * pool = units[i].first;
            const auto * extent = units[i].second;
            if (!pool || !extent || !extent->t || !pool->be ||
                (transformed &&
                 !pool->be->kv_transcode_workspace_memory) ||
                (cross_domain &&
                 (pool->cross_be == nullptr ||
                  pool->cross_be->kv_cross_domain_reconstruct == nullptr ||
                  pool->cross_be->kv_transcode_workspace_memory_v2 == nullptr ||
                  pool->cross_be->kv_transcode_workspace_reserve_v2 == nullptr))) {
                return false;
            }
            const uint64_t target_row = ggml_row_size(
                target_type, extent->t->ne[0]);
            const uint64_t source_row = ggml_row_size(
                source_type, extent->t->ne[0]);
            if (target_row == 0 || source_row == 0 ||
                output.wm_cells > uint64_t(INT64_MAX) ||
                output.wm_cells > UINT64_MAX/target_row ||
                output.wm_cells > UINT64_MAX/source_row) {
                return false;
            }
            const uint64_t target_bytes = output.wm_cells*target_row;
            const uint64_t source_bytes = output.wm_cells*source_row;
            if (mapped > UINT64_MAX-target_bytes ||
                transfer > UINT64_MAX-source_bytes) {
                return false;
            }
            mapped += target_bytes;
            transfer += source_bytes;
            if (transformed) {
                size_t now = 0;
                size_t endpoint = 0;
                const ggml_vbr_transcode_workspace_params_v2 request = {
                    int64_t(output.wm_cells), extent->t->ne[0],
                    stash_rows, cross_domain,
                };
                const bool quoted = cross_domain
                    ? pool->cross_be->kv_transcode_workspace_memory_v2(
                          pool->backend, pool->device, &request, &now, &endpoint)
                    : pool->be->kv_transcode_workspace_memory(
                          pool->backend, pool->device,
                          request.n_cells, request.ne0, request.stash_rows,
                          &now, &endpoint);
                if (!quoted ||
                    endpoint > UINT64_MAX-workspace) {
                    return false;
                }
                workspace += endpoint;
            }
            output.shards[i].row_bytes = target_row;
            output.shards[i].mapped_bytes = target_bytes;
        }
        output.current_type = target_type;
        output.current_domain = vbr_downward_tier_domain(target_type);
        if (!transformed) {
            return true;
        }
        if (downward) {
            output.downward_supported = true;
            output.downward_movable = movable;
            output.controller_floor_type = policy.floor_type;
            output.downward_type = target_type;
            output.downward_domain = vbr_downward_tier_domain(target_type);
            output.downward_recipe_id = VBR_DOWNWARD_RECIPE_ID;
            output.downward_recipe_version = VBR_DOWNWARD_RECIPE_VERSION;
            output.downward_recipe = recipe;
            output.downward_meansub_model_id = hparams.turbo_meansub_id;
            output.downward_row_bytes = output.shards.front().row_bytes;
            output.downward_mapped_bytes = mapped;
            output.downward_transfer_bytes = transfer;
            output.downward_codec_workspace_bytes = workspace;
            output.downward_build_identity_digest = vbr_downward_build_identity(
                recipe, output.downward_meansub_model_id,
                output.meansub_digest,
                projection.child_type_digests[projection_child],
                projection.tree_digest);
            return vbr_digest_nonzero(output.downward_build_identity_digest);
        }
        output.upward_supported = true;
        output.upward_type = target_type;
        output.upward_domain = vbr_downward_tier_domain(target_type);
        output.upward_recipe_id = VBR_UPWARD_RECIPE_ID;
        output.upward_recipe_version = VBR_UPWARD_RECIPE_VERSION;
        output.upward_recipe = upward_recipe;
        const auto meansub_ref = layers[ikv].turbo_meansub_ref;
        output.upward_source_identity = selected_source_identity;
        output.upward_target_identity = selected_target_identity;
        output.upward_meansub_model_id =
            output.upward_source_identity.meansub_model_id;
        if (cross_domain) {
            int max_l = 0;
            int max_c = 0;
            int live = 0;
            if (!output.upward_source_identity.meansub_baked ||
                !output.upward_target_identity.meansub_baked ||
                output.upward_source_identity.meansub_model_id <= 0 ||
                output.upward_source_identity.meansub_model_id !=
                    output.upward_target_identity.meansub_model_id ||
                output.upward_source_identity.meansub_layer < 0 ||
                output.upward_source_identity.meansub_layer !=
                    output.upward_target_identity.meansub_layer ||
                meansub_ref.model_id !=
                    output.upward_target_identity.meansub_model_id ||
                meansub_ref.layer !=
                    output.upward_target_identity.meansub_layer ||
                output.upward_source_identity.meansub_digest !=
                    output.upward_target_identity.meansub_digest ||
                ggml_turbo_meansub_table(
                    output.upward_source_identity.meansub_model_id,
                    is_v ? 1 : 0, &max_l, &max_c, &live) == nullptr ||
                live <= 0 || output.upward_source_identity.meansub_layer >= max_l) {
                return false;
            }
            for (size_t i = 0; i < units.size(); ++i) {
                const uint64_t columns = uint64_t(units[i].second->t->ne[0]);
                if (output.shards[i].logical_offset > uint64_t(max_c) ||
                    columns > uint64_t(max_c)-output.shards[i].logical_offset) {
                    return false;
                }
            }
        }
        output.upward_row_bytes = output.shards.front().row_bytes;
        output.upward_mapped_bytes = mapped;
        output.upward_transfer_bytes = transfer;
        output.upward_codec_workspace_bytes = workspace;
        output.upward_build_identity_digest = vbr_upward_build_identity(
            upward_recipe, output.upward_source_identity,
            output.upward_target_identity,
            projection.child_type_digests[projection_child],
            projection.tree_digest);
        return vbr_digest_nonzero(output.upward_build_identity_digest);
    } catch (...) {
        return false;
    }
}

vbr_downward_transform_status llama_kv_cache::vbr_downward_transform_import(
        const vbr_validated_child_plan & plan,
        bool stashless,
        uint32_t & stash_valid,
        uint32_t & edge_reached) noexcept {
    stash_valid = 0;
    edge_reached = UINT32_MAX;
    try {
        if (plan.transform_kind != vbr_import_transform_kind::downward ||
            plan.logical_unit_id/2 >= layers.size() ||
            plan.transcode_recipe.n_edges == 0) {
            return vbr_downward_transform_status::invalid_recipe;
        }
        const size_t ikv = plan.logical_unit_id/2;
        const bool is_v = (plan.logical_unit_id & 1u) != 0;
        const auto & units = vbr_units_of(ikv, is_v);
        if (units.size() != plan.shards.size()) {
            return vbr_downward_transform_status::invalid_recipe;
        }
        for (const auto & shard : plan.shards) {
            auto * pool = static_cast<vbr_pool *>(
                const_cast<void *>(shard.target_pool_cookie));
            if (!pool || shard.shard_index >= units.size() ||
                units[shard.shard_index].first != pool ||
                !units[shard.shard_index].second ||
                !units[shard.shard_index].second->t ||
                !pool->be || !pool->backend ||
                pool->be->kv_transcode == nullptr ||
                pool->be->kv_stash_capture == nullptr) {
                return vbr_downward_transform_status::transform_failed;
            }
            auto & extent = *units[shard.shard_index].second;
            ggml_tensor source;
            if (!vbr_import_source_alias(
                    *extent.t, plan.transcode_recipe.source_type, source)) {
                return vbr_downward_transform_status::transform_failed;
            }

            struct live_state {
                vbr_pool * pool = nullptr;
                vbr_extent * extent = nullptr;
                ggml_tensor * source = nullptr;
                bool is_v = false;
                bool stashless = false;
                uint32_t requested_stash = 0;
                uint32_t captured_stash = 0;
                uint32_t wm_cells = 0;
            } state;
            state.pool = pool;
            state.extent = &extent;
            state.source = &source;
            state.is_v = is_v;
            state.stashless = stashless;
            state.wm_cells = plan.descriptor.wm_cells;
            state.requested_stash = std::min<uint64_t>(
                vbr_stash_rows_, plan.descriptor.wm_cells);

            vbr_downward_edge_driver driver;
            driver.context = &state;
            driver.stash_available = [](void * opaque) noexcept {
                const auto & value = *static_cast<live_state *>(opaque);
                return value.stashless || value.captured_stash != 0;
            };
            driver.capture_stash = [](void * opaque,
                                      const vbr_downward_edge &) noexcept {
                auto & value = *static_cast<live_state *>(opaque);
                if (value.requested_stash == 0 ||
                    value.pool->stash_vmm == nullptr) {
                    return false;
                }
                char * base = static_cast<char *>(
                    value.pool->be->vmm_pool_base(value.pool->stash_vmm));
                value.pool->be->kv_stash_capture(
                    value.pool->backend, value.source,
                    base + value.extent->stash_off,
                    value.requested_stash, value.is_v);
                value.captured_stash = value.requested_stash;
                return true;
            };
            driver.transcode = [](void * opaque,
                                  const vbr_downward_edge & edge) noexcept {
                auto & value = *static_cast<live_state *>(opaque);
                const void * stash = nullptr;
                int64_t stash_rows = 0;
                if (!value.stashless && value.captured_stash > 0 &&
                    value.pool->stash_vmm != nullptr) {
                    stash = static_cast<char *>(
                        value.pool->be->vmm_pool_base(value.pool->stash_vmm)) +
                        value.extent->stash_off;
                    stash_rows = value.captured_stash;
                }
                // Match the live degrade path's page-tail discipline exactly.
                // The target representation is smaller than its source, but
                // VMM maps page-granular ranges.  Bytes after the target's
                // logical prefix on the terminal mapped page must be scrubbed:
                // a later padded attention read can otherwise reinterpret
                // source-tier tail bytes as turbo block scales and propagate
                // NaNs through an otherwise masked row.
                const vbr_span span = vbr_span_of(
                    value.extent->t, edge.target_type,
                    int64_t(value.wm_cells), value.wm_cells,
                    value.pool->gran);
                vbr_scrub_span scrub;
                if (!vbr_scrub_span_of(
                        value.extent->t, edge.source_type,
                        value.wm_cells, span, value.pool->gran, scrub)) {
                    return false;
                }
                const ggml_vbr_transcode_params params = {
                    value.source, edge.target_type, value.extent->t->data,
                    value.pool->buf, int64_t(value.wm_cells), value.is_v,
                    stash, stash_rows, scrub.scrub_end - scrub.keep,
                };
                value.pool->be->kv_transcode(value.pool->backend, &params);
                const std::vector<ggml_tensor *> no_views;
                vbr_set_tensor_type_impl(
                    value.source, no_views, edge.target_type);
                return true;
            };
            bool regenerated = false;
            const auto status = vbr_downward_execute_edges(
                plan.transcode_recipe, driver, regenerated, &edge_reached);
            if (status != vbr_downward_transform_status::transformed) {
                return status;
            }
            if (regenerated) {
                stash_valid = state.captured_stash;
            }
        }
        return vbr_downward_transform_status::transformed;
    } catch (...) {
        return vbr_downward_transform_status::internal_error;
    }
}

bool llama_kv_cache::vbr_upward_transform_import(
        const vbr_validated_child_plan & plan) noexcept {
    try {
        const bool cross_domain = plan.transform_kind ==
            vbr_import_transform_kind::upward_cross_domain;
        if ((!cross_domain && plan.transform_kind !=
                vbr_import_transform_kind::upward_same_domain) ||
            plan.logical_unit_id/2 >= layers.size() ||
            plan.upward_recipe.n_edges != 1 ||
            plan.descriptor.current_type !=
                int32_t(plan.upward_recipe.source_type) ||
            plan.selected_target_type !=
                int32_t(plan.upward_recipe.target_type)) {
            return false;
        }
        vbr_upward_recipe resolved;
        if (vbr_upward_resolve_recipe(
                plan.upward_recipe.source_type,
                plan.upward_recipe.target_type, resolved) !=
                    vbr_upward_recipe_status::resolved ||
            !(resolved == plan.upward_recipe)) {
            return false;
        }
        const auto source_domain = vbr_downward_tier_domain(
            plan.upward_recipe.source_type);
        const auto target_domain = vbr_downward_tier_domain(
            plan.upward_recipe.target_type);
        if (plan.source_domain != source_domain ||
            plan.selected_target_domain != target_domain ||
            (cross_domain ?
                 (source_domain != vbr_repr_domain::tapped ||
                  target_domain != vbr_repr_domain::full ||
                  plan.upward_recipe.edges[0].mean_action !=
                      vbr_upward_mean_action::add_baked_source_mean) :
                 (source_domain != target_domain ||
                  plan.upward_recipe.edges[0].mean_action !=
                      vbr_upward_mean_action::none))) {
            return false;
        }
        if (source_domain == vbr_repr_domain::tapped) {
            if (plan.descriptor.promote_hops >= 2 ||
                plan.target_promote_hops !=
                    uint8_t(plan.descriptor.promote_hops + 1) ||
                plan.target_last_source_type !=
                    plan.descriptor.current_type) {
                return false;
            }
        } else if (source_domain != vbr_repr_domain::full ||
                   plan.target_promote_hops != 0 ||
                   plan.target_last_source_type !=
                       plan.selected_target_type) {
            return false;
        }
        const size_t ikv = plan.logical_unit_id/2;
        const bool is_v = (plan.logical_unit_id & 1u) != 0;
        int mean_max_l = 0;
        int mean_max_c = 0;
        int mean_live = 0;
        if (cross_domain) {
            const auto meansub_ref = layers[ikv].turbo_meansub_ref;
            if (!plan.transcode_source_identity.meansub_baked ||
                !plan.transcode_target_identity.meansub_baked ||
                plan.transcode_source_identity.meansub_model_id <= 0 ||
                plan.transcode_source_identity.meansub_model_id !=
                    plan.transcode_target_identity.meansub_model_id ||
                plan.transcode_source_identity.meansub_layer < 0 ||
                plan.transcode_source_identity.meansub_layer !=
                    plan.transcode_target_identity.meansub_layer ||
                plan.transcode_source_identity.meansub_digest !=
                    plan.transcode_target_identity.meansub_digest ||
                meansub_ref.model_id !=
                    plan.transcode_source_identity.meansub_model_id ||
                meansub_ref.layer !=
                    plan.transcode_source_identity.meansub_layer ||
                ggml_turbo_meansub_table(
                    meansub_ref.model_id, is_v ? 1 : 0,
                    &mean_max_l, &mean_max_c, &mean_live) == nullptr ||
                mean_live <= 0 || meansub_ref.layer >= mean_max_l) {
                return false;
            }
        }
        const auto & units = vbr_units_of(ikv, is_v);
        if (units.size() != plan.shards.size()) {
            return false;
        }
        const bool restore_stash =
            source_domain == vbr_repr_domain::tapped &&
            (plan.stash_action == vbr_validated_stash_action::restore_exact ||
             plan.stash_action ==
                 vbr_validated_stash_action::consume_exact_then_drop);
        const uint64_t stash_rows = restore_stash
            ? plan.descriptor.clean_stash.valid_rows : 0;
        if (restore_stash &&
             (plan.descriptor.clean_stash_state !=
                  vbr_artifact_clean_stash_state::present ||
              plan.descriptor.clean_stash.domain !=
                  vbr_repr_domain::tapped ||
              stash_rows == 0 || stash_rows > plan.descriptor.wm_cells ||
              stash_rows > vbr_stash_rows_ ||
              plan.descriptor.clean_stash.shards.size() !=
                  plan.shards.size())) {
            return false;
        }
        for (const auto & shard : plan.shards) {
            auto * pool = static_cast<vbr_pool *>(
                const_cast<void *>(shard.target_pool_cookie));
            if (!pool || shard.shard_index >= units.size() ||
                units[shard.shard_index].first != pool ||
                !units[shard.shard_index].second ||
                !units[shard.shard_index].second->t ||
                units[shard.shard_index].second->t->type !=
                    plan.upward_recipe.target_type ||
                !pool->be || !pool->backend ||
                (cross_domain
                    ? pool->cross_be == nullptr ||
                      pool->cross_be->kv_cross_domain_reconstruct == nullptr
                    : pool->be->kv_transcode == nullptr)) {
                return false;
            }
            ggml_tensor source;
            if (!vbr_import_source_alias(
                    *units[shard.shard_index].second->t,
                    plan.upward_recipe.source_type, source)) {
                return false;
            }
            if (restore_stash) {
                const auto * extent =
                    units[shard.shard_index].second;
                const auto & stash_shard =
                    plan.descriptor.clean_stash.shards[shard.shard_index];
                const uint64_t ne0 = uint64_t(extent->t->ne[0]);
                if (ne0 == 0 || ne0 > UINT64_MAX/sizeof(uint16_t)) {
                    return false;
                }
                const uint64_t row_bytes = ne0*sizeof(uint16_t);
                if (stash_rows > UINT64_MAX/row_bytes ||
                    stash_shard.shard_index != shard.shard_index ||
                    stash_shard.row_count != stash_rows ||
                    stash_shard.row_bytes != row_bytes ||
                    stash_shard.payload_bytes != stash_rows*row_bytes ||
                    !pool->stash_vmm ||
                    extent->stash_off > pool->stash_size ||
                    stash_shard.payload_bytes >
                        pool->stash_size-extent->stash_off ||
                    pool->be->vmm_pool_base(pool->stash_vmm) == nullptr) {
                    return false;
                }
            }
            const uint64_t columns = uint64_t(
                units[shard.shard_index].second->t->ne[0]);
            if (cross_domain &&
                (shard.logical_offset > uint64_t(mean_max_c) ||
                 columns > uint64_t(mean_max_c)-shard.logical_offset)) {
                return false;
            }
        }
        // Validate the complete shard set before the first asynchronous
        // submission. After this point the loop is deliberately no-fail;
        // the caller synchronizes all touched backends before publication.
        for (const auto & shard : plan.shards) {
            auto * pool = static_cast<vbr_pool *>(
                const_cast<void *>(shard.target_pool_cookie));
            auto * extent = units[shard.shard_index].second;
            ggml_tensor source = *extent->t;
            const std::vector<ggml_tensor *> no_views;
            vbr_set_tensor_type_impl(
                &source, no_views, plan.upward_recipe.source_type);
            source.data = extent->t->data;
            const void * stash = restore_stash
                ? static_cast<const char *>(
                      pool->be->vmm_pool_base(pool->stash_vmm)) +
                      extent->stash_off
                : nullptr;
            const ggml_vbr_transcode_params params = {
                &source, plan.upward_recipe.target_type,
                extent->t->data, pool->buf,
                int64_t(plan.descriptor.wm_cells), is_v,
                stash, int64_t(stash_rows), 0,
            };
            if (cross_domain) {
                const ggml_vbr_cross_domain_reconstruct_params reconstruct = {
                    params,
                    plan.transcode_source_identity.meansub_model_id,
                    plan.transcode_source_identity.meansub_layer,
                    shard.logical_offset,
                };
                if (!pool->cross_be->kv_cross_domain_reconstruct(
                        pool->backend, &reconstruct)) {
                    return false;
                }
            } else {
                pool->be->kv_transcode(pool->backend, &params);
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool llama_kv_cache::vbr_import_source_alias(
        const ggml_tensor & destination,
        ggml_type source_type,
        ggml_tensor & output) const noexcept {
    if (source_type == GGML_TYPE_COUNT) {
        return false;
    }
    output = destination;
    const std::vector<ggml_tensor *> no_views;
    vbr_set_tensor_type_impl(&output, no_views, source_type);
    output.data = destination.data;
    return true;
}

void llama_kv_cache::vbr_import_set_unit_type_noalloc(
        uint32_t logical_unit, ggml_type type) noexcept {
    const size_t ikv = logical_unit/2;
    const bool is_v = (logical_unit & 1u) != 0;
    GGML_ASSERT(ikv < layers.size() && type != GGML_TYPE_COUNT);
    ggml_tensor * canonical = is_v ? layers[ikv].v : layers[ikv].k;
    GGML_ASSERT(canonical != nullptr);
    vbr_set_tensor_type_noalloc(
        canonical, is_v ? layers[ikv].v_stream : layers[ikv].k_stream,
        type);
}

void llama_kv_cache::vbr_representation_changed() {
    if (vbr_vmm_active() && vbr_budget_bytes_ > 0) {
        GGML_ASSERT(vbr_representation_epoch_ != UINT64_MAX);
        vbr_representation_epoch_++;
    }
}

void llama_kv_cache::vbr_attention_content_changed() {
    if (vbr_vmm_active() && vbr_budget_bytes_ > 0) {
        GGML_ASSERT(vbr_representation_epoch_ != UINT64_MAX);
        GGML_ASSERT(vbr_checkpoint_epoch_ != UINT64_MAX);
        vbr_representation_epoch_++;
        vbr_checkpoint_epoch_++;
        vbr_checkpoint_seq_epochs_.fill(vbr_checkpoint_epoch_);
    }
}

void llama_kv_cache::vbr_attention_content_changed(llama_seq_id seq_id) {
    GGML_ASSERT(seq_id >= 0 && seq_id < LLAMA_MAX_SEQ);
    if (vbr_vmm_active() && vbr_budget_bytes_ > 0) {
        GGML_ASSERT(vbr_representation_epoch_ != UINT64_MAX);
        GGML_ASSERT(vbr_checkpoint_epoch_ != UINT64_MAX);
        vbr_representation_epoch_++;
        vbr_checkpoint_epoch_++;
        vbr_checkpoint_seq_epochs_[size_t(seq_id)] = vbr_checkpoint_epoch_;
    }
}

void llama_kv_cache::vbr_attention_content_changed(
        const std::array<bool, LLAMA_MAX_SEQ> & affected) {
    if (vbr_vmm_active() && vbr_budget_bytes_ > 0) {
        GGML_ASSERT(vbr_representation_epoch_ != UINT64_MAX);
        GGML_ASSERT(vbr_checkpoint_epoch_ != UINT64_MAX);
        vbr_representation_epoch_++;
        vbr_checkpoint_epoch_++;
        for (size_t seq_id = 0; seq_id < affected.size(); ++seq_id) {
            if (affected[seq_id]) {
                vbr_checkpoint_seq_epochs_[seq_id] = vbr_checkpoint_epoch_;
            }
        }
    }
}

vbr_generation_tracker * llama_kv_cache::vbr_generation_tracker_mut() {
    return other != nullptr ? other->vbr_generation_tracker_mut() : vbr_generation_.get();
}

const vbr_generation_tracker * llama_kv_cache::vbr_generation_tracker_get() const {
    return other != nullptr ? other->vbr_generation_tracker_get() : vbr_generation_.get();
}

bool llama_kv_cache::vbr_generation_cell_has_seq_cb(
        const void * context, uint32_t stream, uint32_t cell, llama_seq_id seq_id) {
    const auto * cache = static_cast<const llama_kv_cache *>(context);
    if (cache == nullptr || seq_id < 0 || seq_id >= LLAMA_MAX_SEQ ||
            stream >= cache->v_cells.size() || cell >= cache->v_cells[stream].size()) {
        return false;
    }
    return !cache->v_cells[stream].is_empty(cell) && cache->v_cells[stream].seq_has(cell, seq_id);
}

llama_pos llama_kv_cache::vbr_generation_cell_pos_cb(
        const void * context, uint32_t stream, uint32_t cell) {
    const auto * cache = static_cast<const llama_kv_cache *>(context);
    if (cache == nullptr || stream >= cache->v_cells.size() ||
            cell >= cache->v_cells[stream].size() || cache->v_cells[stream].is_empty(cell)) {
        return -1;
    }
    return cache->v_cells[stream].pos_get(cell);
}

bool llama_kv_cache::vbr_generation_capture_live_guarded(
        uint32_t child_id,
        llama_seq_id seq_id,
        llama_pos computation_frontier,
        vbr_checkpoint_generation_controller & output,
        vbr_artifact_stream_placement * placement,
        vbr_explicit_generation_failure * failure) const {
    if (failure != nullptr) {
        *failure = vbr_explicit_generation_failure::none;
    }
    const auto fail = [&](vbr_explicit_generation_failure why) {
        output = {};
        if (failure != nullptr) {
            *failure = why;
        }
        return false;
    };
    if (other != nullptr) {
        return other->vbr_generation_capture_live_guarded(
                child_id, seq_id, computation_frontier, output,
                placement, failure);
    }

    const auto * tracker = vbr_generation_tracker_get();
    if (tracker == nullptr) {
        return fail(vbr_explicit_generation_failure::tracker_missing);
    }
    if (tracker->shadow_unavailable()) {
        return fail(
            vbr_explicit_generation_failure::tracker_shadow_unavailable);
    }
    if (!tracker->stable()) {
        return fail(vbr_explicit_generation_failure::tracker_unstable);
    }
    if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ ||
            computation_frontier < 0 || static_cast<size_t>(seq_id) >= seq_to_stream.size()) {
        return fail(
            vbr_explicit_generation_failure::invalid_sequence_or_frontier);
    }

    const uint32_t stream = seq_to_stream[seq_id];
    if (stream >= v_cells.size()) {
        return fail(vbr_explicit_generation_failure::invalid_stream);
    }

    // Gap fix (review): capture enumerates owned cells from the ownership-index masks in page
    // order — never a legacy cell scan. The scan remains only as the env-gated oracle check.
    const auto & cells = v_cells[stream];
    if (vbr_ownership_ == nullptr) {
        return fail(
            vbr_explicit_generation_failure::ownership_index_missing);
    }
    if (!vbr_ownership_->initialized(stream, seq_id)) {
        return fail(
            vbr_explicit_generation_failure::ownership_view_missing);
    }
    if (!vbr_ownership_->available(stream, seq_id)) {
        return fail(
            vbr_explicit_generation_failure::ownership_view_unavailable);
    }
    uint32_t expected_rank = 0;
    if (!vbr_ownership_->rank_below(stream, seq_id, computation_frontier, expected_rank)) {
        return fail(
            vbr_explicit_generation_failure::ownership_rank_failed);
    }
    std::vector<uint32_t> owned_cells;
    owned_cells.reserve(expected_rank);
    if (!vbr_ownership_->enumerate_owned(stream, seq_id, owned_cells)) {
        return fail(
            vbr_explicit_generation_failure::ownership_enumeration_failed);
    }
    std::vector<uint32_t> dependency_cells;
    dependency_cells.reserve(expected_rank);

    vbr_artifact_stream_placement captured_placement;
    captured_placement.child_id = child_id;
    captured_placement.stream_index = stream;
    captured_placement.source_sequence = seq_id;
    captured_placement.computation_frontier = computation_frontier;
    captured_placement.cells.reserve(expected_rank);
    for (uint32_t cell : owned_cells) {
        const auto position = cells.pos_get(cell);
        if (position >= computation_frontier) {
            continue;
        }
        // Shift is intentionally absent from artifact v2. A shifted source
        // cannot be represented exactly and therefore fails capture closed.
        if (cells.get_shift(cell) != 0) {
            return fail(
                vbr_explicit_generation_failure::stream_capture_failed);
        }
        const auto & ext = cells.ext_get(cell);
        dependency_cells.push_back(cell);
        captured_placement.cells.push_back({
            cell, position, ext.x, ext.y,
        });
    }
    if (dependency_cells.size() != expected_rank) {
        return fail(
            vbr_explicit_generation_failure::ownership_cardinality_mismatch);
    }

    vbr_checkpoint_generation_stream captured_stream;
    if (!vbr_generation_capture_stream(
                *tracker, stream, seq_id, computation_frontier, dependency_cells, captured_stream)) {
        return fail(vbr_explicit_generation_failure::stream_capture_failed);
    }

    if (!vbr_generation_capture_controller(
                *tracker,
                child_id,
                checkpoint_child_dependency_mode::live_guarded,
                {std::move(captured_stream)},
                output)) {
        return fail(
            vbr_explicit_generation_failure::controller_capture_failed);
    }
    if (placement != nullptr) {
        *placement = std::move(captured_placement);
    }
    return true;
}

// VBR_EXPLICIT_CAPTURE_STABILITY_REGION_BEGIN
// Reviewed capture read authority: these private hooks snapshot and re-read live
// generations to prove a byte capture stayed exact. They never perform
// checkpoint admission; the isolation gate strips only this bounded region.
bool llama_kv_cache::vbr_capture_policy_snapshot(
        vbr_capture_stability_token & output) const noexcept {
    const auto * tracker = vbr_generation_tracker_get();
    if (tracker == nullptr || !tracker->stable()) {
        return false;
    }
    output.lineage_uuid = tracker->lineage_identity();
    output.instance_id = tracker->runtime_instance();
    output.controller_generation = tracker->controller_generation();
    output.mutation_serial = tracker->mutation_serial();

    llama_sha256_writer order_hash;
    static constexpr char ORDER_DOMAIN[] =
        "buun.vbr.capture/degrade-order";
    order_hash.string(ORDER_DOMAIN, sizeof(ORDER_DOMAIN) - 1);
    order_hash.u64(vbr_degrade_order_.size());
    for (const auto & step : vbr_degrade_order_) {
        order_hash.u32(step.il);
        order_hash.u32(step.is_v);
        order_hash.u32(step.tier);
    }
    output.degrade_order_digest = order_hash.finish();
    output.degrade_cursor = vbr_degrade_cursor_;
    const size_t floor_index =
        std::min(vbr_degrade_limit_, vbr_degrade_order_.size());
    output.floor_type = floor_index < vbr_degrade_order_.size()
        ? int32_t(vbr_tier_type(vbr_degrade_order_[floor_index].tier))
        : int32_t(GGML_TYPE_TURBO1_TCQ);
    output.pressure_independent_settings =
        (uint64_t(vbr_params_.dynamic) << 0) |
        (uint64_t(vbr_params_.min_bits_explicit) << 1) |
        (uint64_t(vbr_params_.budget_explicit) << 2) |
        (uint64_t(vbr_params_.pin_k) << 3) |
        (uint64_t(vbr_params_.pin_v) << 4);
    output.completed_wave = std::all_of(
        vbr_pools_.begin(), vbr_pools_.end(),
        [](const auto & pool) {
            return !pool.wave_pending &&
                   pool.unmap_deferred.empty();
        });

    llama_sha256_writer policy_hash;
    static constexpr char POLICY_DOMAIN[] =
        "buun.vbr.capture/controller-policy";
    policy_hash.string(POLICY_DOMAIN, sizeof(POLICY_DOMAIN) - 1);
    policy_hash.bytes(
        output.degrade_order_digest.data(),
        output.degrade_order_digest.size());
    policy_hash.u64(output.degrade_cursor);
    policy_hash.u32(uint32_t(output.floor_type));
    policy_hash.u64(output.pressure_independent_settings);
    policy_hash.u32(n_stream);
    policy_hash.u32(n_stream == 1);
    policy_hash.u64(vbr_pools_.empty()
        ? 0 : vbr_pools_.front().wm_cells);
    output.policy_digest = policy_hash.finish();
    return tracker->stable() &&
           tracker->mutation_serial() == output.mutation_serial;
}

bool llama_kv_cache::vbr_capture_settle() noexcept {
    if (other != nullptr) {
        return other->vbr_capture_settle();
    }
    if (!vbr_operation_armed() || vbr_generation_tracker_get() == nullptr) {
        return false;
    }
    try {
        vbr_flush_deferred_unmaps();
        vbr_arm_wave_fences();
        for (auto & pool : vbr_pools_) {
            // The dedicated side stream is normally created by the first
            // degrade/promote. A fresh F16 cache has mapped KV bytes but has
            // never needed that stream. Explicit capture is its first D2H
            // consumer, so initialize the stream here while the slot is idle.
            // This creates capture infrastructure only; it changes no tier,
            // watermark, ownership, generation, or KV byte.
            if (pool.backend == nullptr) {
                if (pool.be == nullptr || pool.device < 0) {
                    return false;
                }
                pool.backend = pool.be->backend_init(pool.device);
                if (pool.backend == nullptr) {
                    return false;
                }
            }
            if (pool.backend != nullptr) {
                ggml_backend_synchronize(pool.backend);
            }
            if (pool.vmm != nullptr && pool.be != nullptr) {
                pool.be->sync_device(pool.device);
            }
        }
        // The sole intentional pre-quiescence source-side mutation: discard
        // already-dirty stash metadata. This idempotent housekeeping changes
        // no KV bytes, ownership, generation, or cursor and therefore needs
        // no §9 rollback if the later capture fails.
        vbr_invalidate_dirty_stash();
        for (const auto & pool : vbr_pools_) {
            if (pool.wave_pending || !pool.unmap_deferred.empty()) {
                return false;
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool llama_kv_cache::vbr_capture_size_pass(
        const vbr_capture_unit_request & request,
        std::vector<vbr_capture_unit_plan> & output,
        vbr_capture_stability_token & stability,
        vbr_explicit_size_failure * failure) const noexcept {
    output.clear();
    stability = {};
    if (failure != nullptr) {
        *failure = vbr_explicit_size_failure::none;
    }
    const auto fail = [&](vbr_explicit_size_failure why) {
        if (failure != nullptr) {
            *failure = why;
        }
        output.clear();
        stability = {};
        return false;
    };
    if (other != nullptr) {
        return other->vbr_capture_size_pass(
            request, output, stability, failure);
    }
    const auto * tracker = vbr_generation_tracker_get();
    if (!vbr_operation_armed()) {
        return fail(vbr_explicit_size_failure::not_armed);
    }
    if (tracker == nullptr) {
        return fail(vbr_explicit_size_failure::tracker_missing);
    }
    if (!tracker->stable()) {
        return fail(vbr_explicit_size_failure::tracker_unstable);
    }
    if (request.bindings == nullptr) {
        return fail(vbr_explicit_size_failure::bindings_missing);
    }
    if (n_stream != 1 || v_trans) {
        return fail(vbr_explicit_size_failure::stream_layout);
    }
    try {
        const auto & bindings = *static_cast<
            const std::vector<vbr_explicit_capture_pool_binding> *>(
                request.bindings);
        if (!vbr_capture_policy_snapshot(stability)) {
            return fail(vbr_explicit_size_failure::policy_snapshot);
        }
        const uint32_t count = tracker->unit_count();
        output.reserve(count);
        stability.units.reserve(count);
        for (uint32_t unit = 0; unit < count; ++unit) {
            const size_t ikv = unit / 2;
            const bool is_v = (unit & 1u) != 0;
            if (ikv >= layers.size()) {
                return fail(vbr_explicit_size_failure::unit_index);
            }
            const auto & extents = vbr_units_of(ikv, is_v);
            if (extents.empty()) {
                return fail(vbr_explicit_size_failure::extents_empty);
            }
            vbr_capture_unit_plan plan;
            plan.child_id = request.child_id;
            plan.logical_unit = unit;
            plan.is_v = is_v;
            plan.meansub_ref = layers[ikv].turbo_meansub_ref;
            plan.generation = tracker->unit_generation(unit);
            plan.n_stream = n_stream;
            plan.unified = n_stream == 1;

            vbr_capture_stability_token::geometry geometry;
            geometry.logical_unit = unit;
            geometry.generation = plan.generation;
            uint32_t watermark = 0;
            bool first = true;
            bool stash_present = false;
            uint32_t stash_rows = 0;
            uint32_t topology_index = UINT32_MAX;
            uint16_t prior_ordinal = 0;
            for (const auto & [pool, extent] : extents) {
                if (pool == nullptr || extent == nullptr ||
                    extent->t == nullptr) {
                    return fail(
                        vbr_explicit_size_failure::extent_missing);
                }
                if (pool->vmm == nullptr) {
                    return fail(vbr_explicit_size_failure::vmm_missing);
                }
                if (pool->backend == nullptr) {
                    return fail(
                        vbr_explicit_size_failure::backend_unavailable);
                }
                const auto generation_failure =
                    vbr_explicit_capture_validate_extent_generation(
                        pool->wm_cells,
                        static_cast<int32_t>(extent->t->type),
                        extent->promote_hops, plan.generation);
                if (generation_failure !=
                        vbr_explicit_size_failure::none) {
                    return fail(generation_failure);
                }
                if (first) {
                    watermark = pool->wm_cells;
                    stash_present = extent->stash_valid != 0;
                    stash_rows = extent->stash_valid;
                    first = false;
                } else if (watermark != pool->wm_cells ||
                           stash_present != (extent->stash_valid != 0) ||
                           stash_rows != extent->stash_valid) {
                    return fail(
                        vbr_explicit_size_failure::shard_disagreement);
                }
                const auto binding = std::find_if(
                    bindings.begin(), bindings.end(),
                    [&](const vbr_explicit_capture_pool_binding & candidate) {
                        return candidate.instance_id == stability.instance_id &&
                               candidate.device == pool->device;
                    });
                if (binding == bindings.end() ||
                    binding->topology_index == UINT32_MAX ||
                    binding->device_ordinal == UINT16_MAX ||
                    binding->lane == UINT32_MAX) {
                    return fail(
                        vbr_explicit_size_failure::binding_missing);
                }
                if (topology_index == UINT32_MAX) {
                    topology_index = binding->topology_index;
                    prior_ordinal = binding->device_ordinal;
                } else if (binding->topology_index != topology_index ||
                           binding->device_ordinal <= prior_ordinal) {
                    return fail(
                        vbr_explicit_size_failure::topology_order);
                } else {
                    prior_ordinal = binding->device_ordinal;
                }
                const uint64_t row_bytes =
                    ggml_row_size(extent->t->type, extent->t->ne[0]);
                if (row_bytes == 0 ||
                    watermark > UINT64_MAX / row_bytes) {
                    return fail(vbr_explicit_size_failure::bounds);
                }
                const uint64_t bytes = uint64_t(watermark) * row_bytes;
                if (bytes == 0 || bytes > ggml_nbytes(extent->t) ||
                    bytes > vbr_slot_bytes(extent->t) ||
                    extent->byte_off > pool->size ||
                    bytes > pool->size - extent->byte_off) {
                    return fail(vbr_explicit_size_failure::bounds);
                }
                uint64_t stash_bytes = 0;
                if (stash_present) {
                    if (plan.generation.domain != vbr_repr_domain::tapped ||
                        pool->stash_vmm == nullptr ||
                        extent->t->ne[0] >
                            int64_t(UINT64_MAX / sizeof(uint16_t))) {
                        return fail(
                            vbr_explicit_size_failure::stash_bounds);
                    }
                    const uint64_t stash_row_bytes =
                        uint64_t(extent->t->ne[0]) * sizeof(uint16_t);
                    if (stash_rows > UINT64_MAX / stash_row_bytes) {
                        return fail(
                            vbr_explicit_size_failure::stash_bounds);
                    }
                    stash_bytes = uint64_t(stash_rows) * stash_row_bytes;
                    const size_t stash_size = pool->stash_size;
                    if (extent->stash_off > stash_size ||
                        stash_bytes > stash_size - extent->stash_off) {
                        return fail(
                            vbr_explicit_size_failure::stash_bounds);
                    }
                }
                plan.shards.push_back({
                    pool, extent, uint32_t(plan.shards.size()),
                    binding->topology_index, binding->device_ordinal,
                    binding->lane, bytes, row_bytes,
                    uint64_t(extent->t->ne[0]), stash_bytes,
                });
                geometry.tensors.push_back(extent->t);
                geometry.byte_offsets.push_back(extent->byte_off);
                geometry.stash_valid.push_back(extent->stash_valid);
                geometry.stash_offsets.push_back(extent->stash_off);
            }
            plan.wm_cells = watermark;
            geometry.wm_cells = watermark;
            output.push_back(std::move(plan));
            stability.units.push_back(std::move(geometry));
        }
        if (output.empty() || !tracker->stable() ||
            tracker->mutation_serial() != stability.mutation_serial) {
            return fail(
                vbr_explicit_size_failure::stability_reread);
        }
        return true;
    } catch (...) {
        return fail(vbr_explicit_size_failure::internal_error);
    }
}

bool llama_kv_cache::vbr_capture_stream_unit(
        const vbr_capture_unit_plan & plan,
        vbr_unit_build & sink,
        vbr_pinned_chunk_ring & ring,
        vbr_capture_stream_stats & stats,
        void * continue_context,
        bool (*continue_transfer)(void * context) noexcept) const noexcept {
    if (other != nullptr) {
        return other->vbr_capture_stream_unit(
            plan, sink, ring, stats, continue_context, continue_transfer);
    }
    try {
        std::vector<vbr_capture_projected_shard_source> sources;
        if (!vbr_capture_projected_sources(plan, sources) ||
            sources.size() != plan.shards.size()) {
            return false;
        }
        for (size_t shard_index = 0;
             shard_index < plan.shards.size(); ++shard_index) {
            const auto & shard = plan.shards[shard_index];
            auto * pool = static_cast<vbr_pool *>(shard.pool);
            auto * extent = static_cast<vbr_extent *>(shard.extent);
            if (pool == nullptr || extent == nullptr) {
                return false;
            }
            auto chain = std::make_shared<artifact_segment_chain>();
            vbr_capture_stream_stats one;
            sources[shard_index].source.continue_context = continue_context;
            sources[shard_index].source.continue_transfer = continue_transfer;
            const auto status = ring.stream(
                sources[shard_index].source, *chain, one);
            if (status != vbr_capture_stream_status::ok) {
                return false;
            }
            vbr_verified_segment segment;
            segment.unit_index = plan.capture_index;
            segment.shard_index = shard.shard_index;
            segment.bytes = chain;
            segment.streaming_digest = one.streaming_digest;
            if (sink.accept_verified_segment(segment) !=
                    vbr_capture_stream_status::ok) {
                return false;
            }
            stats.bytes += one.bytes;
            stats.chunks += one.chunks;
            stats.backpressure_waits += one.backpressure_waits;
            stats.event_completions += one.event_completions;
            stats.synchronous_fallbacks += one.synchronous_fallbacks;
            stats.max_segment_size =
                std::max(stats.max_segment_size, one.max_segment_size);

            if (shard.stash_bytes != 0) {
                ggml_init_params params = {
                    2*ggml_tensor_overhead(), nullptr, true,
                };
                ggml_context_ptr context { ggml_init(params) };
                if (!context) {
                    return false;
                }
                ggml_tensor * alias = ggml_new_tensor_1d(
                    context.get(), GGML_TYPE_I8,
                    int64_t(shard.stash_bytes));
                // The f16 sink stash is a raw VMM slab (no ggml buffer wrapper since the
                // recoverable-stash rework). Borrow the extent's pool buffer for the CUDA
                // same-device buft assert; the D2H copy reads alias->data directly.
                alias->buffer = extent->t->buffer;
                alias->data = static_cast<char *>(
                    pool->be->vmm_pool_base(pool->stash_vmm)) +
                    extent->stash_off;
                vbr_capture_stream_source stash_source;
                stash_source.lane = shard.lane;
                stash_source.size = shard.stash_bytes;
                stash_source.backend = pool->backend;
                stash_source.device =
                    ggml_backend_get_device(pool->backend);
                stash_source.tensor = alias;
                stash_source.continue_context = continue_context;
                stash_source.continue_transfer = continue_transfer;
                auto stash_chain =
                    std::make_shared<artifact_segment_chain>();
                vbr_capture_stream_stats stash_stats;
                const auto stash_status =
                    ring.stream(stash_source, *stash_chain, stash_stats);
                if (stash_status != vbr_capture_stream_status::ok) {
                    return false;
                }
                vbr_verified_segment stash_segment;
                stash_segment.unit_index = plan.capture_index;
                stash_segment.shard_index = shard.shard_index;
                stash_segment.clean_stash = true;
                stash_segment.bytes = stash_chain;
                stash_segment.streaming_digest =
                    stash_stats.streaming_digest;
                if (sink.accept_verified_segment(stash_segment) !=
                        vbr_capture_stream_status::ok) {
                    return false;
                }
                stats.bytes += stash_stats.bytes;
                stats.chunks += stash_stats.chunks;
                stats.backpressure_waits +=
                    stash_stats.backpressure_waits;
                stats.event_completions +=
                    stash_stats.event_completions;
                stats.synchronous_fallbacks +=
                    stash_stats.synchronous_fallbacks;
                stats.max_segment_size = std::max(
                    stats.max_segment_size,
                    stash_stats.max_segment_size);
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

namespace {

bool vbr_capture_generation_equal(
        const vbr_unit_generation & lhs,
        const vbr_unit_generation & rhs) noexcept {
    return lhs.repr_gen == rhs.repr_gen &&
           lhs.publish_seq == rhs.publish_seq &&
           lhs.current_type == rhs.current_type &&
           lhs.last_source_type == rhs.last_source_type &&
           lhs.domain == rhs.domain &&
           lhs.promote_hops == rhs.promote_hops &&
           lhs.last_transition == rhs.last_transition &&
           lhs.flags == rhs.flags;
}

uint64_t vbr_capture_shard_source_identity(
        vbr_controller_instance_id instance,
        uint32_t child_id,
        uint32_t logical_unit,
        uint32_t shard_index,
        uint32_t topology_index,
        uint16_t device_ordinal,
        uint32_t lane,
        const void * pool,
        const void * extent) noexcept {
    try {
        llama_sha256_writer hash;
        static constexpr char source_identity_domain[] =
            "buun.vbr.capture/live-shard-source/v1";
        hash.string(source_identity_domain, sizeof(source_identity_domain) - 1);
        hash.u64(instance.hi);
        hash.u64(instance.lo);
        hash.u32(child_id);
        hash.u32(logical_unit);
        hash.u32(shard_index);
        hash.u32(topology_index);
        hash.u32(device_ordinal);
        hash.u32(lane);
        hash.bytes(&pool, sizeof(pool));
        hash.bytes(&extent, sizeof(extent));
        const auto digest = hash.finish();
        uint64_t identity = 0;
        std::memcpy(&identity, digest.data(), sizeof(identity));
        return identity == 0 ? 1 : identity;
    } catch (...) {
        return 0;
    }
}

} // namespace

bool llama_kv_cache::vbr_capture_projected_sources(
        const vbr_capture_unit_plan & plan,
        std::vector<vbr_capture_projected_shard_source> & output) const noexcept {
    return vbr_capture_projected_sources_impl(plan, &output, nullptr, false);
}

bool llama_kv_cache::vbr_capture_projected_sources_leased(
        const vbr_capture_unit_plan & plan,
        const std::vector<vbr_capture_projected_shard_source> & expected,
        const vbr_capture_snapshot_session & session) const noexcept {
    uint64_t mutation_serial = 0;
    if (!session.active || session.cache != this || session.plan != &plan ||
        expected.empty() ||
        !vbr_capture_unit_read_serial(plan.logical_unit, mutation_serial)) {
        return false;
    }
    return vbr_capture_projected_sources_impl(
        plan, nullptr, &expected, true);
}

bool llama_kv_cache::vbr_capture_projected_sources_impl(
        const vbr_capture_unit_plan & plan,
        std::vector<vbr_capture_projected_shard_source> * output,
        const std::vector<vbr_capture_projected_shard_source> * expected,
        bool unit_leased) const noexcept {
    if ((output == nullptr) == (expected == nullptr) ||
        unit_leased != (expected != nullptr)) {
        return false;
    }
    if (output != nullptr) {
        output->clear();
    }
    if (other != nullptr) {
        return other->vbr_capture_projected_sources_impl(
            plan, output, expected, unit_leased);
    }
    try {
        const auto * tracker = vbr_generation_tracker_get();
        const auto instance = vbr_instance_id();
        if (!vbr_operation_armed() || tracker == nullptr ||
            !tracker->active() || (!unit_leased && !tracker->stable()) ||
            !vbr_controller_instance_id_is_set(instance) ||
            plan.child_id == UINT32_MAX ||
            plan.logical_unit >= tracker->unit_count() ||
            plan.logical_unit/2 >= layers.size() ||
            plan.is_v != ((plan.logical_unit & 1u) != 0) ||
            plan.meansub_ref.model_id !=
                layers[plan.logical_unit/2].turbo_meansub_ref.model_id ||
            plan.meansub_ref.layer !=
                layers[plan.logical_unit/2].turbo_meansub_ref.layer ||
            plan.n_stream != n_stream || plan.unified != (n_stream == 1) ||
            plan.shards.empty() || plan.wm_cells == 0 ||
            (expected != nullptr && expected->size() != plan.shards.size()) ||
            !vbr_capture_generation_equal(
                tracker->unit_generation(plan.logical_unit),
                plan.generation)) {
            return false;
        }
        const auto & extents = vbr_units_of(
            plan.logical_unit/2, (plan.logical_unit & 1u) != 0);
        if (extents.size() != plan.shards.size()) {
            return false;
        }
        std::vector<vbr_capture_projected_shard_source> prepared;
        if (output != nullptr) {
            prepared.reserve(plan.shards.size());
        }
        for (size_t i = 0; i < plan.shards.size(); ++i) {
            const auto & shard = plan.shards[i];
            auto * pool = static_cast<vbr_pool *>(shard.pool);
            auto * extent = static_cast<vbr_extent *>(shard.extent);
            if (pool == nullptr || extent == nullptr ||
                extents[i].first != pool || extents[i].second != extent ||
                shard.shard_index != i || shard.lane == UINT32_MAX ||
                shard.topology_index == UINT32_MAX ||
                shard.device_ordinal == UINT16_MAX ||
                shard.payload_bytes == 0 || shard.row_bytes == 0 ||
                !vbr_capture_watermark_contains(*pool, plan.wm_cells) ||
                pool->backend == nullptr || extent->t == nullptr ||
                ggml_backend_get_device(pool->backend) == nullptr ||
                extent->t->type != plan.generation.current_type ||
                extent->promote_hops != plan.generation.promote_hops ||
                extent->t->ne[0] <= 0 ||
                uint64_t(extent->t->ne[0]) != shard.columns ||
                ggml_row_size(extent->t->type, extent->t->ne[0]) !=
                    shard.row_bytes ||
                plan.wm_cells > UINT64_MAX/shard.row_bytes ||
                uint64_t(plan.wm_cells)*shard.row_bytes !=
                    shard.payload_bytes ||
                shard.payload_bytes > ggml_nbytes(extent->t)) {
                return false;
            }
            vbr_capture_projected_shard_source source;
            source.shard_index = shard.shard_index;
            source.row_count = plan.wm_cells;
            source.row_bytes = shard.row_bytes;
            source.source_identity = vbr_capture_shard_source_identity(
                instance, plan.child_id, plan.logical_unit,
                shard.shard_index, shard.topology_index,
                shard.device_ordinal, shard.lane, pool, extent);
            source.source.lane = shard.lane;
            source.source.size = shard.payload_bytes;
            source.source.backend = pool->backend;
            source.source.device =
                ggml_backend_get_device(pool->backend);
            source.source.tensor = extent->t;
            if (source.source_identity == 0) {
                return false;
            }
            if (expected != nullptr) {
                const auto & canonical = (*expected)[i];
                if (canonical.shard_index != source.shard_index ||
                    canonical.row_count != source.row_count ||
                    canonical.row_bytes != source.row_bytes ||
                    canonical.source_identity != source.source_identity ||
                    canonical.source.lane != source.source.lane ||
                    canonical.source.size != source.source.size ||
                    canonical.source.backend != source.source.backend ||
                    canonical.source.device != source.source.device ||
                    canonical.source.tensor != source.source.tensor ||
                    canonical.source.tensor_offset !=
                        source.source.tensor_offset ||
                    canonical.source.context != source.source.context ||
                    canonical.source.read != source.source.read) {
                    return false;
                }
            } else {
                prepared.push_back(source);
            }
        }
        if (output != nullptr) {
            uint32_t shard_count = 0;
            std::array<uint8_t, 32> topology = {};
            if (!vbr_capture_projected_shard_topology(
                    prepared, shard_count, topology) ||
                shard_count != prepared.size()) {
                return false;
            }
        }
        if ((!unit_leased && !tracker->stable()) ||
            !vbr_capture_generation_equal(
                tracker->unit_generation(plan.logical_unit),
                plan.generation)) {
            return false;
        }
        if (output != nullptr) {
            *output = std::move(prepared);
        }
        return true;
    } catch (...) {
        if (output != nullptr) {
            output->clear();
        }
        return false;
    }
}

llama_kv_cache::vbr_capture_snapshot_session::~vbr_capture_snapshot_session() {
    if (active && cache != nullptr && plan != nullptr) {
        cache->vbr_capture_unit_read_end(plan->logical_unit);
    }
}

vbr_capture_unit_snapshot_provider
llama_kv_cache::vbr_capture_snapshot_session::provider() noexcept {
    return {
        this,
        &llama_kv_cache::vbr_capture_snapshot_acquire,
        &llama_kv_cache::vbr_capture_snapshot_recheck,
        &llama_kv_cache::vbr_capture_snapshot_release,
    };
}

bool llama_kv_cache::vbr_capture_snapshot_bind(
        const vbr_capture_unit_plan & plan,
        const std::vector<vbr_capture_projected_shard_source> & sources,
        uint64_t source_namespace,
        vbr_capture_snapshot_session & output) const noexcept {
    if (output.active || output.cache != nullptr || source_namespace == 0 ||
        plan.child_id == UINT32_MAX || sources.empty()) {
        return false;
    }
    try {
        uint32_t shard_count = 0;
        std::array<uint8_t, 32> topology = {};
        if (!vbr_capture_projected_shard_topology(
                sources, shard_count, topology) ||
            shard_count != sources.size()) {
            return false;
        }
        output.sources = sources;
        output.cache = this;
        output.plan = &plan;
        output.source_namespace = source_namespace;
        output.shard_count = shard_count;
        output.shard_topology_digest = topology;
        return true;
    } catch (...) {
        output.sources.clear();
        return false;
    }
}

bool llama_kv_cache::vbr_capture_snapshot_acquire(
        void * context, uint64_t source_namespace, uint32_t child_id,
        uint32_t logical_unit_id, vbr_capture_unit_snapshot & output) noexcept {
    output = {};
    auto * session = static_cast<vbr_capture_snapshot_session *>(context);
    if (session == nullptr || session->cache == nullptr || session->plan == nullptr ||
        session->active || source_namespace == 0 ||
        source_namespace != session->source_namespace ||
        child_id != session->plan->child_id ||
        logical_unit_id != session->plan->logical_unit ||
        !session->cache->vbr_capture_unit_read_begin(logical_unit_id)) {
        return false;
    }
    session->active = true;
    const auto fail = [&]() {
        session->cache->vbr_capture_unit_read_end(logical_unit_id);
        session->active = false;
        output = {};
        return false;
    };
    try {
        const auto * tracker = session->cache->vbr_generation_tracker_get();
        uint64_t unit_mutation_serial = 0;
        if (tracker == nullptr || !tracker->active() ||
            tracker->runtime_instance() != session->cache->vbr_instance_id() ||
            logical_unit_id >= tracker->unit_count() ||
            !session->cache->vbr_capture_projected_sources_leased(
                *session->plan, session->sources, *session) ||
            !session->cache->vbr_capture_unit_read_serial(
                logical_unit_id, unit_mutation_serial) ||
            session->sources.size() != session->shard_count) {
            return fail();
        }
        output.source_namespace = source_namespace;
        output.child_id = child_id;
        output.logical_unit_id = logical_unit_id;
        output.lineage_uuid = tracker->lineage_identity();
        output.controller_generation = tracker->controller_generation();
        output.mutation_serial = unit_mutation_serial;
        output.generation = tracker->unit_generation(logical_unit_id);
        output.shard_count = session->shard_count;
        output.shard_topology_digest = session->shard_topology_digest;
        if (!vbr_capture_generation_equal(
                output.generation,
                tracker->unit_generation(logical_unit_id))) {
            return fail();
        }
        return true;
    } catch (...) {
        return fail();
    }
}

bool llama_kv_cache::vbr_capture_snapshot_recheck(
        void * context,
        const vbr_capture_unit_snapshot & expected) noexcept {
    auto * session = static_cast<vbr_capture_snapshot_session *>(context);
    if (session == nullptr || session->cache == nullptr ||
        session->plan == nullptr || !session->active ||
        expected.source_namespace != session->source_namespace ||
        expected.child_id != session->plan->child_id ||
        expected.logical_unit_id != session->plan->logical_unit ||
        expected.shard_count != session->shard_count ||
        expected.shard_topology_digest != session->shard_topology_digest) {
        return false;
    }
    try {
        const auto * tracker = session->cache->vbr_generation_tracker_get();
        uint64_t unit_mutation_serial = 0;
        return tracker != nullptr && tracker->active() &&
            tracker->lineage_identity() == expected.lineage_uuid &&
            tracker->controller_generation() == expected.controller_generation &&
            session->cache->vbr_capture_unit_read_serial(
                expected.logical_unit_id, unit_mutation_serial) &&
            unit_mutation_serial == expected.mutation_serial &&
            vbr_capture_generation_equal(
                expected.generation,
                tracker->unit_generation(expected.logical_unit_id)) &&
            session->cache->vbr_capture_projected_sources_leased(
                *session->plan, session->sources, *session);
    } catch (...) {
        return false;
    }
}

void llama_kv_cache::vbr_capture_snapshot_release(
        void * context,
        const vbr_capture_unit_snapshot & snapshot) noexcept {
    auto * session = static_cast<vbr_capture_snapshot_session *>(context);
    if (session == nullptr || session->cache == nullptr ||
        session->plan == nullptr || !session->active) {
        return;
    }
    GGML_ASSERT(snapshot.logical_unit_id == session->plan->logical_unit);
    session->cache->vbr_capture_unit_read_end(session->plan->logical_unit);
    session->active = false;
}

bool llama_kv_cache::vbr_capture_stability_matches(
        const vbr_capture_stability_token & token) const noexcept {
    if (other != nullptr) {
        return other->vbr_capture_stability_matches(token);
    }
    const auto * tracker = vbr_generation_tracker_get();
    vbr_capture_stability_token policy;
    if (tracker == nullptr || !vbr_capture_policy_snapshot(policy) ||
        policy.lineage_uuid != token.lineage_uuid ||
        policy.instance_id != token.instance_id ||
        policy.controller_generation != token.controller_generation ||
        policy.mutation_serial != token.mutation_serial ||
        policy.degrade_order_digest != token.degrade_order_digest ||
        policy.policy_digest != token.policy_digest ||
        policy.degrade_cursor != token.degrade_cursor ||
        policy.floor_type != token.floor_type ||
        policy.pressure_independent_settings !=
            token.pressure_independent_settings ||
        policy.completed_wave != token.completed_wave ||
        tracker->unit_count() != token.units.size()) {
        return false;
    }
    try {
        for (const auto & expected : token.units) {
            if (tracker->unit_generation(expected.logical_unit).publish_seq !=
                    expected.generation.publish_seq ||
                !(tracker->unit_generation(expected.logical_unit).repr_gen ==
                      expected.generation.repr_gen &&
                  tracker->unit_generation(expected.logical_unit).current_type ==
                      expected.generation.current_type &&
                  tracker->unit_generation(expected.logical_unit).last_source_type ==
                      expected.generation.last_source_type &&
                  tracker->unit_generation(expected.logical_unit).domain ==
                      expected.generation.domain &&
                  tracker->unit_generation(expected.logical_unit).promote_hops ==
                      expected.generation.promote_hops &&
                  tracker->unit_generation(expected.logical_unit).last_transition ==
                      expected.generation.last_transition)) {
                return false;
            }
            const auto & extents = vbr_units_of(
                expected.logical_unit / 2,
                (expected.logical_unit & 1u) != 0);
            if (extents.size() != expected.tensors.size()) {
                return false;
            }
            for (size_t i = 0; i < extents.size(); ++i) {
                if (extents[i].first->wm_cells != expected.wm_cells ||
                    extents[i].second->t != expected.tensors[i] ||
                    extents[i].second->byte_off != expected.byte_offsets[i] ||
                    extents[i].second->stash_valid != expected.stash_valid[i] ||
                    extents[i].second->stash_off != expected.stash_offsets[i]) {
                    return false;
                }
            }
        }
        return tracker->stable() &&
               tracker->mutation_serial() == token.mutation_serial;
    } catch (...) {
        return false;
    }
}

bool llama_kv_cache::vbr_capture_generation_record(
        uint32_t child_id,
        checkpoint_child_dependency_mode dependency_mode,
        llama_seq_id sequence,
        llama_pos frontier,
        vbr_checkpoint_generation_controller & output,
        vbr_artifact_stream_placement * placement,
        vbr_explicit_generation_failure * failure) const noexcept {
    output = {};
    if (placement != nullptr) {
        *placement = {};
    }
    if (failure != nullptr) {
        *failure = vbr_explicit_generation_failure::none;
    }
    try {
        if (other != nullptr) {
            return other->vbr_capture_generation_record(
                child_id, dependency_mode, sequence, frontier, output,
                placement, failure);
        }
        if (dependency_mode ==
                checkpoint_child_dependency_mode::payload_complete) {
            const auto * tracker = vbr_generation_tracker_get();
            if (tracker == nullptr || n_stream != 1 || v_cells.size() != 1) {
                if (failure != nullptr) {
                    *failure =
                        vbr_explicit_generation_failure::tracker_missing;
                }
                return false;
            }
            if (!tracker->stable()) {
                if (failure != nullptr) {
                    *failure =
                        vbr_explicit_generation_failure::tracker_unstable;
                }
                return false;
            }
            // A payload-complete child cannot be projected independently,
            // but exact host restore still needs its physical ownership and
            // generation image. Admit only a sole owner: serializing shared
            // iSWA bytes into one sequence's artifact would otherwise grant
            // that manifest authority over an unrelated live sequence.
            const auto & cells = v_cells.front();
            std::vector<uint32_t> dependency_cells;
            vbr_artifact_stream_placement captured_placement;
            captured_placement.child_id = child_id;
            captured_placement.stream_index = 0;
            captured_placement.source_sequence = sequence;
            captured_placement.computation_frontier = frontier;
            dependency_cells.reserve(cells.get_used());
            captured_placement.cells.reserve(cells.get_used());
            for (uint32_t cell = 0; cell < cells.size(); ++cell) {
                if (cells.is_empty(cell)) {
                    continue;
                }
                if (cells.seq_count(cell) != 1 ||
                    !cells.seq_has(cell, sequence) ||
                    cells.pos_get(cell) >= frontier ||
                    cells.get_shift(cell) != 0) {
                    if (failure != nullptr) {
                        *failure = vbr_explicit_generation_failure::
                            ownership_cardinality_mismatch;
                    }
                    return false;
                }
                const auto & ext = cells.ext_get(cell);
                dependency_cells.push_back(cell);
                captured_placement.cells.push_back({
                    cell, cells.pos_get(cell), ext.x, ext.y,
                });
            }
            if (dependency_cells.empty()) {
                if (failure != nullptr) {
                    *failure = vbr_explicit_generation_failure::
                        ownership_cardinality_mismatch;
                }
                return false;
            }
            vbr_checkpoint_generation_stream captured_stream;
            if (!vbr_generation_capture_stream(
                    *tracker, 0, sequence, frontier, dependency_cells,
                    captured_stream)) {
                if (failure != nullptr) {
                    *failure = vbr_explicit_generation_failure::
                        stream_capture_failed;
                }
                return false;
            }
            output.child_id = child_id;
            output.dependency_mode = dependency_mode;
            output.lineage_uuid = tracker->lineage_identity();
            output.global_generation = tracker->controller_generation();
            output.units.reserve(tracker->unit_count());
            for (uint32_t unit = 0; unit < tracker->unit_count(); ++unit) {
                const auto generation = tracker->unit_generation(unit);
                output.units.push_back({
                    generation.repr_gen,
                    generation.current_type,
                    generation.last_source_type,
                    generation.domain,
                    generation.promote_hops,
                    generation.last_transition,
                });
            }
            output.streams.push_back(std::move(captured_stream));
            if (placement != nullptr) {
                *placement = std::move(captured_placement);
            }
            return tracker->stable() &&
                   vbr_lineage_uuid_is_set(output.lineage_uuid);
        }
        if (dependency_mode !=
                checkpoint_child_dependency_mode::live_guarded) {
            if (failure != nullptr) {
                *failure = vbr_explicit_generation_failure::
                    invalid_sequence_or_frontier;
            }
            return false;
        }
        if (!vbr_generation_capture_live_guarded(
                child_id, sequence, frontier, output, placement, failure)) {
            return false;
        }
        output.dependency_mode = dependency_mode;
        return true;
    } catch (...) {
        output = {};
        if (failure != nullptr) {
            *failure =
                vbr_explicit_generation_failure::internal_error;
        }
        return false;
    }
}
// VBR_EXPLICIT_CAPTURE_STABILITY_REGION_END

// Keep the sequence-state writer's SWA visibility predicate in one place.
bool llama_kv_cache::state_write_includes_cell(
        const llama_kv_cells & cells,
        uint32_t cell,
        llama_seq_id seq_id) const {
    if (cells.is_empty(cell) || (seq_id != -1 && !cells.seq_has(cell, seq_id))) {
        return false;
    }
    if (seq_id == -1) {
        return true;
    }
    return !llama_hparams::is_masked_swa(
            n_swa, swa_type, cells.pos_get(cell), cells.seq_pos_max(seq_id));
}

bool llama_kv_cache::vbr_decode_targets_from_ubatch(vbr_operation_binding & binding,
                                                    vbr_controller_instance_id instance,
                                                    bool wrap_possible, uint16_t stream,
                                                    const llama_ubatch & ubatch) {
    // one (seq -> [min,max+1)) range per touched sequence
    llama_seq_id seqs[vbr_operation_binding::MAX_TARGETS];
    llama_pos    lo [vbr_operation_binding::MAX_TARGETS];
    llama_pos    hi [vbr_operation_binding::MAX_TARGETS];
    uint8_t      n_seqs = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            const llama_seq_id seq = ubatch.seq_id[i][s];
            uint8_t j = 0;
            while (j < n_seqs && seqs[j] != seq) {
                ++j;
            }
            if (j == n_seqs) {
                if (n_seqs == vbr_operation_binding::MAX_TARGETS) {
                    binding.n_targets = 0;  // Transactional: complete or nothing.
                    return false;
                }
                seqs[n_seqs] = seq;
                lo[n_seqs]   = ubatch.pos[i];
                hi[n_seqs]   = ubatch.pos[i] + 1;
                ++n_seqs;
            } else {
                lo[j] = std::min(lo[j], ubatch.pos[i]);
                hi[j] = std::max(hi[j], ubatch.pos[i] + 1);
            }
        }
    }
    const uint8_t per_seq = wrap_possible ? 2 : 1;
    const uint8_t flat    = wrap_possible ? 1 : 0;
    if (binding.n_targets + n_seqs * per_seq + flat > vbr_operation_binding::MAX_TARGETS) {
        binding.n_targets = 0;  // Transactional: complete or nothing.
        return false;
    }
    for (uint8_t j = 0; j < n_seqs; ++j) {
        binding.targets[binding.n_targets++] = vbr_make_target(
                vbr_operation_kind::decode, vbr_operation_class::ordinary_decode,
                instance, stream, seqs[j], lo[j], hi[j]);
        if (wrap_possible) {
            // Unified-SWA slot selection may reuse a masked cell of a different
            // sequence whose position is unbounded by this batch (find_slot checks only the
            // old owner's own SWA mask), so the destructive claim is the incoming sequence's
            // whole range — the SELECTED target still binds the exact seq's damage extent.
            binding.targets[binding.n_targets++] = vbr_make_target(
                    vbr_operation_kind::decode, vbr_operation_class::swa_wrap,
                    instance, stream, seqs[j], 0,
                    std::numeric_limits<llama_pos>::max());
        }
    }
    if (wrap_possible) {
        // Wrap overwrites force a prefix purge of the OVERWRITTEN cells' owners
        // (apply_ubatch's nested seq_rm, the §7.3 composite purge). Those owners are chosen
        // by slot selection INSIDE apply — unknowable at manifest build (the composite mints
        // before its children run) — so the purge is DECLARED as one seq-wildcard whole-range
        // target: still instance-exact, operation-bound, and confined to the generic seq_rm
        // trim class; wildcards match only where the manifest declared them.
        binding.targets[binding.n_targets++] = vbr_make_target(
                vbr_operation_kind::decode, vbr_operation_class::state_api,
                instance, stream, -1, 0, std::numeric_limits<llama_pos>::max());
    }
    return true;
}

// The innermost open scope's id for cited global/unit publications (empty when none).
#define vbr_cited_op() \
    (vbr_current_mutation_ != nullptr && vbr_current_mutation_->active() \
             ? vbr_current_mutation_->operation_id_ : vbr_operation_id{})

// One spelling of "fail every per-target handle", shared by the scope and the
// pending records, and the commit fence. A failed operation leaves NO admissible evidence,
// including any handle that already reached committed before a later one failed.
static void vbr_fail_extent_set(
        vbr_generation_tracker * tracker,
        std::array<vbr_extent_handle, vbr_operation_binding::MAX_TARGETS> & extents) {
    if (tracker == nullptr) {
        return;
    }
    for (auto & extent : extents) {
        if (extent) {
            tracker->extent_store().fail(extent);
            extent = {};
        }
    }
}

llama_kv_cache::vbr_mutation_op::vbr_mutation_op(llama_kv_cache *    cache,
                                                 vbr_operation_kind  kind,
                                                 vbr_operation_class operation_class,
                                                 llama_seq_id        seq_id,
                                                 llama_pos           p0,
                                                 llama_pos           p1,
                                                 bool                provenance_bearing,
                                                 uint16_t            extent_stream)
    : vbr_mutation_op(cache,
                      cache != nullptr && cache->vbr_generation_tracker_mut() != nullptr
                          ? vbr_mutation_binding(
                                kind, seq_id, p0, p1, operation_class,
                                cache->vbr_generation_tracker_mut()->runtime_instance(),
                                extent_stream)
                          : vbr_operation_binding{},
                      provenance_bearing) {}

llama_kv_cache::vbr_mutation_op::vbr_mutation_op(llama_kv_cache *              cache,
                                                 const vbr_operation_binding & manifest,
                                                 bool                          provenance_bearing) {
    if (cache == nullptr || cache->vbr_generation_tracker_mut() == nullptr) {
        return;  // inert: unarmed caches open no operations (armed-VBR-only scope)
    }
    cache_         = cache;
    auto * tracker = cache_->vbr_generation_tracker_mut();

    kind_         = manifest.kind;
    extent_owner_ = this;
    // Composite participation. A nested scope (library-internal composite,
    // e.g. apply_ubatch's purge trims) borrows the outer identity fully. An ADOPTED scope
    // (wrapper-forwarded, e.g. iSWA children under one decode id) shares the id but owns its
    // tracker-local reservations (v2 finding 3).
    if (cache_->vbr_current_mutation_ != nullptr) {
        // Any outer scope determines nested identity. An active outer is joined;
        // a refused/poisoned/inactive outer makes this nested scope inert under the same
        // ABSENT identity — it never falls through to mint (operation registry one-id, including refusal).
        joined_       = true;
        operation_id_ = cache_->vbr_current_mutation_->operation_id_;  // empty when inactive
        extent_owner_ = cache_->vbr_current_mutation_->extent_owner_;
    } else if (cache_->vbr_adopted_refused_) {
        // The composite root's mint was refused, so this child fails closed under
        // the one (absent) identity instead of minting independently (operation registry one-id in refusal).
        adopted_ = true;
        abort_to_shadow_unavailable();
    } else if (cache_->vbr_adopted_operation_) {
        adopted_      = true;
        operation_id_ = cache_->vbr_adopted_operation_;
        composite_    = cache_->vbr_adopted_composite_;
        if (composite_ != nullptr) {
            // Claim the parent-declared participant slot before any mutation.
            composite_->claim();
        }
        if (!vbr_operation_registry_binding(operation_id_, manifest_)) {
            abort_to_shadow_unavailable();
        }
    } else {
        owned_op_.emplace(manifest);
        operation_id_ = owned_op_->id();
        if (!operation_id_) {
            // Registry refusal is fail-closed: the shadow must not survive an
            // untracked mutation. Invalidate before the legacy mutation proceeds, then FALL
            // THROUGH to scope registration: the inert scope must still be current so event
            // sites no-op instead of asserting.
            abort_to_shadow_unavailable();
        }
    }

    if (provenance_bearing && !joined_ && operation_id_) {
        // v3.1 amendment 1: recovery reservation is EAGER — before any mutation (Rev 4 rule
        // uncompromised). The provenance EXTENTS are lazy: reserved per SELECTED target by
        // ensure_extent_for() at the first destructive stamp, so a wrap-free SWA decode pays
        // no extent traffic.
        const vbr_controller_instance_id owner_instance = tracker->runtime_instance();
        recovery_index_ = owned_op_
                ? vbr_recovery_reserve(owned_op_->binding(), owner_instance)
                : vbr_recovery_reserve(operation_id_, owner_instance);
        if (recovery_index_ < 0) {
            abort_to_shadow_unavailable();
        }
    }
    outer_                        = cache_->vbr_current_mutation_;
    cache_->vbr_current_mutation_ = this;
}

llama_kv_cache::vbr_mutation_op::~vbr_mutation_op() {
    if (cache_ == nullptr) {
        return;
    }
    cache_->vbr_current_mutation_ = outer_;
    if (joined_) {
        return;  // borrowed identity: the owner resolves extent/recovery/close
    }
    if (adopted_) {
        // An adopted participant that never transferred its token is terminal
        // here — setup refusal, poison, and exceptions before detach all report FAILED
        // exactly once. Its evidence fails; the shared root's failed close autorecords this
        // child's recovery reservation for the quarantine drain.
        if (composite_ != nullptr && !detached_) {
            composite_->report_terminal(false);
        }
        if (!detached_ && operation_id_) {
            fail_extents();
        }
        return;
    }
    if (detached_) {
        return;  // everything transferred to the pending record
    }
    auto * tracker = cache_->vbr_generation_tracker_mut();
    if (!succeeded_) {
        fail_extents();
    } else if (tracker != nullptr) {
        for (auto & extent : extents_) {
            if (extent) {
                tracker->extent_store().commit(extent);
                extent = {};
            }
        }
    }
    if (recovery_index_ >= 0 && succeeded_) {
        vbr_recovery_release_unused(recovery_index_, operation_id_);
    }
    if (owned_op_) {
        owned_op_->close(succeeded_ ? vbr_operation_outcome::committed
                                    : vbr_operation_outcome::failed);
    }
}

vbr_extent_handle llama_kv_cache::vbr_mutation_op::ensure_extent_for(uint8_t target_index) {
    if (cache_ == nullptr || !active() || joined_ || target_index >= scope_manifest().n_targets) {
        return {};
    }
    if (extents_[target_index]) {
        return extents_[target_index];
    }
    auto * tracker = cache_->vbr_generation_tracker_mut();
    if (tracker == nullptr) {
        return {};
    }
    // The extent copies the selected covering target exactly:
    // the tracker chose it per stamp by (seq, pre-mutation position), so the damage evidence
    // is target-exact by construction.
    const auto & target = scope_manifest().targets[target_index];
    extents_[target_index] = tracker->extent_store().reserve(
            vbr_operation_kind_family(kind_, target.operation_class), target.operation_class,
            target.stream, target.seq_id, target.range.p0, target.range.p1);
    if (!extents_[target_index]) {
        abort_to_shadow_unavailable();
    }
    return extents_[target_index];
}

vbr_extent_handle llama_kv_cache::vbr_mutation_op::extent_trampoline(void * ctx, uint8_t target_index) {
    return static_cast<vbr_mutation_op *>(ctx)->ensure_extent_for(target_index);
}

void llama_kv_cache::vbr_mutation_op::fail_extents() {
    vbr_fail_extent_set(cache_ != nullptr ? cache_->vbr_generation_tracker_mut() : nullptr,
                        extents_);
}

// The one spelling of the shadow-unavailable transition (altitude review): fail whatever
// evidence this scope reserved, invalidate the shadow BEFORE the legacy mutation proceeds,
// and leave the scope inert-but-open (its close stays clean). The recovery reservation is
// deliberately kept: the operation closes failed, that close autorecords the slot,
// and the boundary quarantine drain performs the sanctioned invalidation — the slot must
// survive to that drain.
void llama_kv_cache::vbr_mutation_op::abort_to_shadow_unavailable() {
    auto * tracker = cache_ != nullptr ? cache_->vbr_generation_tracker_mut() : nullptr;
    if (tracker == nullptr) {
        return;
    }
    poisoned_ = true;
    fail_extents();
    (void) tracker->global_invalidate_and_reset_extents(
            vbr_mutation_registrant::authenticated_recovery, vbr_operation_class::controller);
    // Always latch, even when the invalidation above succeeded. The legacy
    // mutation that follows this scope is UNTRACKED; no capture may slip in between it and
    // the next sanctioned transition, which try_clear now proves happened strictly later.
    tracker->set_shadow_unavailable();
    // The scope stays OPEN and inert (operation_id_ empty => events refuse politely); event
    // sites see a current scope and no-op instead of asserting.
    operation_id_ = {};
    owned_op_.reset();
}

// Ownership transfer for deferred mutation families: everything
// the destructor would resolve moves to the pending record instead. Joined scopes own nothing.
std::optional<llama_kv_cache::vbr_pending_decode_op> llama_kv_cache::vbr_mutation_op::detach_deferred() {
    if (joined_) {
        return std::nullopt;
    }
    if (!active() || poisoned_) {
        // A poisoned or refused decode operation never transfers; the destructor reports
        // its terminal FAILURE (aggregate report for composite children, failed close for
        // owned scopes), and the failed close autorecords its recovery reservation.
        return std::nullopt;
    }
    vbr_pending_decode_op pending;
    pending.extents        = extents_;
    extents_               = {};
    pending.recovery_index = recovery_index_;
    recovery_index_        = -1;
    if (owned_op_) {
        pending.operation_id = owned_op_->release();
        pending.owns_close   = true;
    } else {
        // Detach transfers the still-open participant token to the pending owner
        // — never a terminal report; the sealed aggregate closes the root when every
        // declared slot has terminated.
        pending.operation_id = operation_id_;
        pending.owns_close   = false;
        pending.composite    = composite_;
    }
    detached_ = true;
    if (!pending.operation_id) {
        return std::nullopt;
    }
    return pending;
}

void llama_kv_cache::vbr_adopt_operation(vbr_operation_id operation_id) {
    if (other) {
        other->vbr_adopt_operation(operation_id);
        return;
    }
    vbr_adopted_operation_ = operation_id;
}

void llama_kv_cache::vbr_adopt_composite(std::shared_ptr<vbr_composite_outcome> composite) {
    if (other) {
        other->vbr_adopt_composite(std::move(composite));
        return;
    }
    vbr_adopted_composite_ = std::move(composite);
}

void llama_kv_cache::vbr_adopt_refused() {
    if (other) {
        other->vbr_adopt_refused();
        return;
    }
    vbr_adopted_refused_ = true;
}

void llama_kv_cache::vbr_release_adopted() {
    if (other) {
        other->vbr_release_adopted();
        return;
    }
    vbr_adopted_operation_ = {};
    vbr_adopted_composite_.reset();
    vbr_adopted_refused_   = false;
}

// Fixed-participant aggregate. Defined here so the registry close stays in
// the kv-cache trust domain; the iSWA wrapper only constructs/seals it.
void llama_kv_cache::vbr_composite_outcome::claim() {
    ++claimed;
}

void llama_kv_cache::vbr_composite_outcome::report_terminal(bool ok) {
    failed = failed || !ok;
    ++terminal;
    try_close();
}

void llama_kv_cache::vbr_composite_outcome::seal(bool wrapper_ok) {
    failed = failed || !wrapper_ok;
    // Declared participants that never claimed their slot (an exception before a child's
    // scope opened, or an armed child that never applied) are terminal failures by
    // construction — every declared slot reports exactly once.
    if (claimed < declared) {
        failed    = true;
        terminal += declared - claimed;
        claimed   = declared;
    }
    sealed = true;
    try_close();
}

void llama_kv_cache::vbr_composite_outcome::try_close() {
    // Reports may accumulate before seal but can never close; the closed flag makes a late
    // (over-declared) report inert instead of double-closing.
    if (!sealed || closed || terminal < declared) {
        return;
    }
    closed = true;
    if (operation_id) {
        vbr_operation_registry_close(operation_id,
                failed ? vbr_operation_outcome::failed : vbr_operation_outcome::committed);
    }
}

void llama_kv_cache::vbr_decode_ops_finish(bool ok) {
    if (other) {
        other->vbr_decode_ops_finish(ok);
        return;
    }
    // ok=false conservatively fails only this decode's
    // operations — awaiting records are PRIOR submitted decodes whose graphs already ran;
    // only the scheduler fence may decide them. ok=true moves every per-target extent
    // prepared -> submitted; operations + recovery reservations stay OPEN until the
    // synchronize fence delivers the terminal result (Rev 5.1). A submit failure (obsolete
    // post-reset handle) terminal-fails that owner now.
    auto * tracker = vbr_generation_tracker_mut();
    for (auto & pending : vbr_pending_decode_ops_) {
        bool submitted = ok;
        if (ok && tracker != nullptr) {
            for (auto & extent : pending.extents) {
                if (extent && !tracker->extent_store().submit(extent)) {
                    ++vbr_pending_commit_failures_;
                    submitted = false;
                    break;
                }
            }
        }
        if (submitted) {
            vbr_awaiting_commit_.push_back(std::move(pending));
            continue;
        }
        vbr_fail_extent_set(tracker, pending.extents);
        if (pending.composite) {
            // Terminal failure feeds the sealed aggregate.
            pending.composite->report_terminal(false);
        } else if (pending.owns_close) {
            vbr_operation_registry_close(pending.operation_id, vbr_operation_outcome::failed);
        }
    }
    vbr_pending_decode_ops_.clear();
}

vbr_generation_event llama_kv_cache::vbr_generation_begin(
        vbr_mutation_registrant registrant,
        vbr_operation_class operation_class,
        uint32_t stream,
        vbr_generation_stamp_kind stamp_kind,
        bool destructive,
        bool imported) {
    auto * tracker = vbr_generation_tracker_mut();
    if (tracker == nullptr) {
        return {};
    }
    // Events cite the innermost open mutation scope. A mutation
    // reaching here without a scope is a wiring bug — refuse the event loudly rather than mint
    // an uncited one. A scope that opened shadow-unavailable (registry refusal) yields inert
    // events, matching the legacy-proceeds contract — and once a poison latched the shadow
    // mid-decode, later events in the same decode are inert the same way.
    GGML_ASSERT(vbr_current_mutation_ != nullptr && "mutation without an operation scope");
    if (!vbr_current_mutation_->active() || tracker->shadow_unavailable()) {
        return {};  // inert scope / latched shadow: legacy proceeds untracked
    }
    // The per-target extent supplier attaches only to provenance-relevant
    // (destructive/imported) events: apply_ubatch mints an append event and a destructive
    // reuse event under ONE scope, and only the latter's stamps reserve/cite the selected
    // target's extent (§5.5 row 2). Joined scopes route to their root's extents.
    const bool provenance = destructive || imported;
    auto event = tracker->begin_event(
            registrant, operation_class, stream, stamp_kind,
            vbr_current_mutation_->operation_id_,
            provenance ? &vbr_mutation_op::extent_trampoline : nullptr,
            provenance ? static_cast<void *>(vbr_current_mutation_->extent_owner_) : nullptr,
            destructive, imported);
    GGML_ASSERT(event);
    return event;
}

void llama_kv_cache::vbr_stamp(vbr_mutation_op & op, vbr_generation_event & event, uint32_t cell,
                               llama_seq_id membership_seq, llama_pos pre_mutation_pos) {
    vbr_stamp(op, event, cell, &membership_seq, 1, pre_mutation_pos);
}

void llama_kv_cache::vbr_stamp(vbr_mutation_op & op, vbr_generation_event & event, uint32_t cell,
                               const llama_seq_id * seqs, int32_t n_seqs,
                               llama_pos pre_mutation_pos) {
    if (event &&
        !vbr_generation_tracker_mut()->stamp_cell(event, cell, seqs, n_seqs, pre_mutation_pos)) {
        op.poison();
    }
}

void llama_kv_cache::vbr_generation_global(
        vbr_mutation_registrant registrant, vbr_operation_class operation_class) {
    auto * tracker = vbr_generation_tracker_mut();
    if (tracker == nullptr) {
        return;
    }
    // #11 (v2 verdict): a global invalidation is ONE operation — pending decode owners resolve
    // (conservatively failed) first, then the transition also obsoletes every stored extent
    // reference and resets the slab, so old cells can never pin slab capacity into the next
    // lineage epoch.
    vbr_decode_ops_finish(false);
    const vbr_operation_id citing =
            vbr_current_mutation_ != nullptr && vbr_current_mutation_->active()
                    ? vbr_current_mutation_->operation_id_ : vbr_operation_id{};
    GGML_ASSERT(tracker->global_invalidate_and_reset_extents(registrant, operation_class, citing));
}

bool llama_kv_cache::vbr_retier_defer(const char * decision) {
    if (vbr_retier_freeze_depth_ == 0) {
        return false;
    }
    vbr_retier_deferred_decisions_++;
    vbr_retier_reconcile_pending_ = true;
    LLAMA_LOG_INFO("VBR_RETIER_FREEZE event=defer controller=%s decision=%s depth=%u deferred_total=%llu\n",
            vbr_params_.trace_label != nullptr ? vbr_params_.trace_label : "single",
            decision, vbr_retier_freeze_depth_,
            (unsigned long long) vbr_retier_deferred_decisions_);
    return true;
}

bool llama_kv_cache::vbr_capture_unit_read_begin(
        uint32_t logical_unit) const noexcept {
    if (other != nullptr) {
        return other->vbr_capture_unit_read_begin(logical_unit);
    }
    std::lock_guard<std::mutex> lock(vbr_capture_unit_leases_mutex_);
    if (logical_unit >= vbr_capture_unit_leases_.size() ||
        vbr_capture_controller_writer_) {
        return false;
    }
    auto & state = vbr_capture_unit_leases_[logical_unit];
    if (state.writer || state.readers == UINT32_MAX) {
        return false;
    }
    ++state.readers;
    return true;
}

bool llama_kv_cache::vbr_capture_unit_read_serial(
        uint32_t logical_unit, uint64_t & output) const noexcept {
    if (other != nullptr) {
        return other->vbr_capture_unit_read_serial(logical_unit, output);
    }
    std::lock_guard<std::mutex> lock(vbr_capture_unit_leases_mutex_);
    if (logical_unit >= vbr_capture_unit_leases_.size()) {
        return false;
    }
    const auto & state = vbr_capture_unit_leases_[logical_unit];
    if (state.readers == 0 || state.writer ||
        (state.mutation_serial & 1u) != 0) {
        return false;
    }
    output = state.mutation_serial;
    return true;
}

void llama_kv_cache::vbr_capture_unit_read_end(
        uint32_t logical_unit) const noexcept {
    if (other != nullptr) {
        other->vbr_capture_unit_read_end(logical_unit);
        return;
    }
    bool reconcile = false;
    {
        std::lock_guard<std::mutex> lock(vbr_capture_unit_leases_mutex_);
        GGML_ASSERT(logical_unit < vbr_capture_unit_leases_.size());
        auto & state = vbr_capture_unit_leases_[logical_unit];
        GGML_ASSERT(state.readers > 0 && !state.writer);
        --state.readers;
        if (state.readers == 0 && state.mutation_deferred) {
            state.mutation_deferred = false;
            reconcile = true;
        }
    }
    if (reconcile) {
        vbr_capture_reconcile_pending_.store(true, std::memory_order_release);
    }
}

bool llama_kv_cache::vbr_capture_watermark_contains(
        const vbr_pool & pool, uint32_t planned) const noexcept {
    if (other != nullptr) {
        return other->vbr_capture_watermark_contains(pool, planned);
    }
    std::lock_guard<std::mutex> lock(vbr_capture_unit_leases_mutex_);
    return pool.wm_cells >= planned;
}

void llama_kv_cache::vbr_capture_watermark_publish(
        vbr_pool & pool, uint32_t value) noexcept {
    if (other != nullptr) {
        other->vbr_capture_watermark_publish(pool, value);
        return;
    }
    std::lock_guard<std::mutex> lock(vbr_capture_unit_leases_mutex_);
    pool.wm_cells = value;
}

bool llama_kv_cache::vbr_capture_unit_write_begin(
        uint32_t logical_unit) noexcept {
    if (other != nullptr) {
        return other->vbr_capture_unit_write_begin(logical_unit);
    }
    std::lock_guard<std::mutex> lock(vbr_capture_unit_leases_mutex_);
    if (logical_unit >= vbr_capture_unit_leases_.size() ||
        vbr_capture_controller_writer_) {
        return false;
    }
    auto & state = vbr_capture_unit_leases_[logical_unit];
    if (state.writer || state.readers != 0 ||
        state.mutation_serial > UINT64_MAX - 2) {
        if (state.readers != 0) {
            state.mutation_deferred = true;
        }
        return false;
    }
    state.writer = true;
    ++state.mutation_serial;
    return true;
}

bool llama_kv_cache::vbr_capture_unit_write_plan_available(
        uint32_t logical_unit) const noexcept {
    if (other != nullptr) {
        return other->vbr_capture_unit_write_plan_available(logical_unit);
    }
    std::lock_guard<std::mutex> lock(vbr_capture_unit_leases_mutex_);
    if (logical_unit >= vbr_capture_unit_leases_.size() ||
        vbr_capture_controller_writer_) {
        return false;
    }
    auto & state = vbr_capture_unit_leases_[logical_unit];
    if (state.writer || state.readers != 0 ||
        state.mutation_serial > UINT64_MAX - 2) {
        if (state.readers != 0) {
            state.mutation_deferred = true;
        }
        return false;
    }
    return true;
}

void llama_kv_cache::vbr_capture_unit_write_end(
        uint32_t logical_unit) noexcept {
    if (other != nullptr) {
        other->vbr_capture_unit_write_end(logical_unit);
        return;
    }
    std::lock_guard<std::mutex> lock(vbr_capture_unit_leases_mutex_);
    GGML_ASSERT(logical_unit < vbr_capture_unit_leases_.size());
    auto & state = vbr_capture_unit_leases_[logical_unit];
    GGML_ASSERT(state.writer && state.readers == 0 &&
                (state.mutation_serial & 1u) != 0 &&
                state.mutation_serial != UINT64_MAX);
    ++state.mutation_serial;
    state.writer = false;
}

bool llama_kv_cache::vbr_capture_controller_write_begin() noexcept {
    if (other != nullptr) {
        return other->vbr_capture_controller_write_begin();
    }
    std::lock_guard<std::mutex> lock(vbr_capture_unit_leases_mutex_);
    if (vbr_capture_controller_writer_) {
        return false;
    }
    for (auto & state : vbr_capture_unit_leases_) {
        if (state.writer || state.readers != 0) {
            if (state.readers != 0) {
                state.mutation_deferred = true;
            }
            return false;
        }
    }
    vbr_capture_controller_writer_ = true;
    return true;
}

void llama_kv_cache::vbr_capture_controller_write_end() noexcept {
    if (other != nullptr) {
        other->vbr_capture_controller_write_end();
        return;
    }
    std::lock_guard<std::mutex> lock(vbr_capture_unit_leases_mutex_);
    GGML_ASSERT(vbr_capture_controller_writer_);
    vbr_capture_controller_writer_ = false;
}

llama_kv_cache::vbr_unit_retier_guard::vbr_unit_retier_guard(
        llama_kv_cache * cache, uint32_t logical_unit) noexcept
    : cache_(cache), logical_unit_(logical_unit),
      active_(cache != nullptr &&
              cache->vbr_capture_unit_write_begin(logical_unit)) {
}

llama_kv_cache::vbr_unit_retier_guard::~vbr_unit_retier_guard() {
    if (active_) {
        cache_->vbr_capture_unit_write_end(logical_unit_);
    }
}

llama_kv_cache::vbr_unit_retier_guard::vbr_unit_retier_guard(
        vbr_unit_retier_guard && other) noexcept
    : cache_(other.cache_), logical_unit_(other.logical_unit_),
      active_(other.active_) {
    other.cache_ = nullptr;
    other.logical_unit_ = UINT32_MAX;
    other.active_ = false;
}

llama_kv_cache::vbr_unit_retier_guard &
llama_kv_cache::vbr_unit_retier_guard::operator=(
        vbr_unit_retier_guard && other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (active_) {
        cache_->vbr_capture_unit_write_end(logical_unit_);
    }
    cache_ = other.cache_;
    logical_unit_ = other.logical_unit_;
    active_ = other.active_;
    other.cache_ = nullptr;
    other.logical_unit_ = UINT32_MAX;
    other.active_ = false;
    return *this;
}

bool llama_kv_cache::vbr_retier_take_reconcile(const char * boundary) {
    if (vbr_capture_reconcile_pending_.exchange(
            false, std::memory_order_acq_rel)) {
        vbr_retier_reconcile_pending_ = true;
        vbr_capture_retier_attempt_boundary_ = UINT64_MAX;
        std::fill(vbr_capture_unit_attempt_boundary_.begin(),
                  vbr_capture_unit_attempt_boundary_.end(), UINT64_MAX);
    }
    if (vbr_retier_freeze_depth_ > 0 || !vbr_retier_reconcile_pending_) {
        return false;
    }
    // Do not replay a queued promote/degrade choice: only force the ordinary controller off
    // its stable fast path, where it recomputes watermark, occupancy and budgets from scratch.
    vbr_retier_reconcile_pending_ = false;
    vbr_retier_reconciles_++;
    LLAMA_LOG_INFO("VBR_RETIER_FREEZE event=reconcile controller=%s boundary=%s reconciles_total=%llu\n",
            vbr_params_.trace_label != nullptr ? vbr_params_.trace_label : "single",
            boundary, (unsigned long long) vbr_retier_reconciles_);
    return true;
}

// The cache is EMPTY: nothing is stored, so undoing every degrade is free and LOSSLESS — unlike
// container promotion this genuinely restores quality, because all future content is new. Flip
// every tensor back to its entry tier, rewind the price cursor, drop the (now stale) sink
// stashes and release every physical page; the next session refills from a clean entry-tier
// start (prefill-direct). Fires lazily from prepare() at the first decode after the cache
// empties, whatever emptied it (clear, seq_rm, server slot recycle).
void llama_kv_cache::vbr_full_reset() {
    // #9: the scope opens BEFORE the first observable mutation (type flips/unmaps below).
    vbr_mutation_op mutation_op(this, vbr_operation_kind::sequence_edit,
            vbr_operation_class::controller, -1, 0, std::numeric_limits<llama_pos>::max());
    const vbr_mutation_op::success_on_return mutation_ok(mutation_op);
    if (vbr_retier_defer("full_reset")) {
        return;
    }
    if (!vbr_capture_controller_write_begin()) {
        return;
    }
    struct capture_controller_guard {
        llama_kv_cache * cache;
        ~capture_controller_guard() {
            cache->vbr_capture_controller_write_end();
        }
    } capture_guard { this };
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
                const size_t slot = vbr_slot_span(e.t, pool.gran);
                pool.be->vmm_pool_unmap(pool.vmm, e.byte_off, slot);
            }
        }
        vbr_capture_watermark_publish(pool, 0);
        mapped += pool.be->vmm_pool_mapped(pool.vmm);
    }
    vbr_degrade_cursor_    = 0;
    vbr_hard_seal_deferred_.clear();
    vbr_capture_retier_deferred_.clear();
    vbr_budget_warned_     = false;
    vbr_stash_dirty_       = false;
    vbr_quiet_boundaries_  = 0;
    // Deliberately bumps even though the cursor is now 0: this is the low-LCP/empty-cache ABA
    // that a recurrent-only checkpoint must detect.
    vbr_attention_content_changed();
    vbr_generation_global(vbr_mutation_registrant::full_reset, vbr_operation_class::controller);
    if (vbr_ownership_) {
        vbr_ownership_->clear_all();
    }
    if (auto * tracker = vbr_generation_tracker_mut()) {
        for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
            for (uint32_t side = 0; side < 2; ++side) {
                const ggml_tensor * tensor = side != 0 ? layers[ikv].v : layers[ikv].k;
                const uint32_t unit = static_cast<uint32_t>(ikv * 2 + side);
                const auto before = tracker->unit_generation(unit);
                const int32_t target = tensor != nullptr ? static_cast<int32_t>(tensor->type) : -1;
                GGML_ASSERT(tracker->publish_unit(
                        unit,
                        before.current_type,
                        target,
                        vbr_repr_domain::full,
                        0,
                        vbr_repr_transition::full_reset,
                        vbr_mutation_registrant::full_reset,
                        vbr_cited_op()));
            }
        }
    }
    LLAMA_LOG_INFO("%s: VBR full reset: cache empty — %zu tensors back at their entry tier, pools released "
            "(%.2f MiB mapped)\n", __func__, undone, mapped/1024.0/1024.0);
}

// Occupancy DROPPED (seq_rm trimmed a session): pull the mapped watermark back down and release
// the tail pages. They are phantom cost against the budget (mapped but unreadable — attention
// pads only to used+256), and both promotion's hysteresis and its re-encode row count follow
// wm_cells, so leaving it at the old high-water lets a promotion map tiers back up at a stale
// size. The rows shrunk away keep
// valid current-tier bytes, so no scrub is needed; regrowth remaps zero-filled pages on demand.
// 25% hysteresis so growth/trim jitter cannot thrash map/unmap.
void llama_kv_cache::vbr_shrink_watermark() {
    uint32_t wm_now = vbr_watermark_cells(0);
    const bool shrink_needed = std::any_of(
        vbr_pools_.begin(), vbr_pools_.end(), [&](const auto & pool) {
            return pool.vmm != nullptr &&
                wm_now + wm_now / 4 < pool.wm_cells;
        });
    if (!shrink_needed || !vbr_capture_controller_write_begin()) {
        return;
    }
    struct capture_controller_guard {
        llama_kv_cache * cache;
        ~capture_controller_guard() {
            cache->vbr_capture_controller_write_end();
        }
    } capture_guard { this };
    wm_now = vbr_watermark_cells(0);
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
                const size_t slot = vbr_slot_span(t, pool.gran);
                pool.be->vmm_pool_unmap(pool.vmm, e.byte_off + keep, slot - keep);
            }
        }
        LLAMA_LOG_INFO("%s: VBR watermark shrink (device %d): %u -> %u cells (%.2f MiB mapped)\n",
                __func__, pool.device, pool.wm_cells, wm_now,
                pool.be->vmm_pool_mapped(pool.vmm)/1024.0/1024.0);
        vbr_capture_watermark_publish(pool, wm_now);
    }
}

// Container promotion: occupancy DROPPED (seq_rm trimmed a long session) and a higher tier now
// fits with headroom. This restores NO information — the live rows are re-encoded from their
// current recon (same quality, bigger container) — but every FUTURE row encodes at the higher
// tier, which the bathtub makes worth real quality. Walks the price cursor BACKWARDS one step
// per call; the in-place transcode runs descending tiles (see vbr-transcode.cu) after the grown
// extent is mapped up front.
bool llama_kv_cache::vbr_promote_next(uint32_t wm_next) {
    vbr_mutation_op mutation_op(this, vbr_operation_kind::controller_retier,
            vbr_operation_class::controller, -1, -1, -1);
    const vbr_mutation_op::success_on_return mutation_ok(mutation_op);
    if (vbr_retier_defer("promote")) {
        return false;
    }
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
            // tiers (t4 <-> t3 <-> t2 <-> t1); the configured entry returns losslessly at full reset.
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

        vbr_unit_retier_guard capture_guard(
            this, static_cast<uint32_t>(ikv * 2 + (st.is_v != 0)));
        if (!capture_guard) {
            return false;
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
                if (vbr_stash_rows_ > 0 && e.stash_valid > 0 && pp->stash_vmm != nullptr) {
                    stash_ptr  = (char *) pp->be->vmm_pool_base(pp->stash_vmm) + e.stash_off;
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
        const ggml_type type_A = t->type;
        vbr_set_tensor_type(t, st.is_v ? layers[ikv].v_stream : layers[ikv].k_stream, type_B);
        vbr_tier_epoch_++; // fence graph reuse off the old views (type/strides changed in place)
        vbr_representation_changed();
        if (auto * tracker = vbr_generation_tracker_mut()) {
            const uint8_t promote_hops = units.empty() ? 0 : units.front().second->promote_hops;
            GGML_ASSERT(tracker->publish_unit(
                    static_cast<uint32_t>(ikv * 2 + (st.is_v != 0)),
                    static_cast<int32_t>(type_A),
                    static_cast<int32_t>(type_B),
                    type_B == GGML_TYPE_F16 || type_B == GGML_TYPE_TURBO8_0
                            ? vbr_repr_domain::full
                            : vbr_repr_domain::tapped,
                    promote_hops,
                    vbr_repr_transition::promote,
                    vbr_mutation_registrant::promote_next,
                    vbr_cited_op()));
        }
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

llama_kv_cache::vbr_degrade_result llama_kv_cache::vbr_degrade_next(uint32_t wm_next) {
    vbr_mutation_op mutation_op(this, vbr_operation_kind::controller_retier,
            vbr_operation_class::controller, -1, -1, -1);
    const vbr_mutation_op::success_on_return mutation_ok(mutation_op);
    if (vbr_retier_defer("degrade")) {
        // The caller's freeze-pressure guard normally prevents entry. Treat a direct frozen
        // call as a no-step terminal for this pass; the reconcile edge recomputes later.
        return vbr_degrade_result::exhausted;
    }
    const size_t cursor_entry = vbr_degrade_cursor_;
    bool saw_hard_lease_block = false;
    bool saw_capture_lease_block = false;
    vbr_hard_seal_consult_session seal_session;
    GGML_ASSERT(!vbr_hard_seal_guard_ ||
        vbr_hard_seal_attempted_.size() == vbr_degrade_order_.size());
    bool attempted_ready = !vbr_hard_seal_deferred_.empty();
    if (attempted_ready) {
        std::fill(vbr_hard_seal_attempted_.begin(),
                  vbr_hard_seal_attempted_.end(), 0);
    }
    bool capture_attempted_ready =
        !vbr_capture_retier_deferred_.empty() &&
        vbr_capture_retier_attempt_boundary_ != vbr_boundary_count_;
    if (capture_attempted_ready) {
        std::fill(vbr_capture_retier_attempted_.begin(),
                  vbr_capture_retier_attempted_.end(), 0);
        vbr_capture_retier_attempt_boundary_ = vbr_boundary_count_;
    }
    size_t capture_deferred_scan = 0;
    while (true) {
        size_t order_ordinal = 0;
        bool from_deferred = false;
        bool from_capture_deferred = false;
        if (capture_attempted_ready) {
            for (; capture_deferred_scan < vbr_capture_retier_deferred_.size();
                 ++capture_deferred_scan) {
                const size_t ordinal =
                    vbr_capture_retier_deferred_[capture_deferred_scan];
                if (ordinal < vbr_capture_retier_attempted_.size() &&
                    !vbr_capture_retier_attempted_[ordinal]) {
                    vbr_capture_retier_attempted_[ordinal] = 1;
                    order_ordinal = ordinal;
                    from_capture_deferred = true;
                    ++capture_deferred_scan;
                    break;
                }
            }
        }
        if (!from_capture_deferred) {
            if (!vbr_hard_seal_next_order_step(
                    vbr_degrade_cursor_,
                    std::min(vbr_degrade_order_.size(), vbr_degrade_limit_),
                    vbr_hard_seal_deferred_, vbr_hard_seal_attempted_,
                    order_ordinal, from_deferred)) {
                break;
            }
        }
        const auto & st = vbr_degrade_order_[order_ordinal];

        const auto retire_deferred = [&]() {
            if (from_capture_deferred) {
                vbr_hard_seal_retire_step(
                    vbr_capture_retier_deferred_, order_ordinal);
            }
            if (!from_deferred) {
                return;
            }
            vbr_hard_seal_retire_step(
                vbr_hard_seal_deferred_, order_ordinal);
        };

        const auto it = map_layer_ids.find(st.il);
        if (it == map_layer_ids.end()) {
            retire_deferred();
            continue;
        }
        const int32_t ikv = it->second;
        ggml_tensor * t = st.is_v ? layers[ikv].v : layers[ikv].k; // canonical tensor: the type source of truth
        // every VMM pool holding this unit: one under -sm layer, one per device (each with its
        // shard) under -sm tensor. The tier flip is a property of the UNIT — all pools move together.
        const auto & units = vbr_units_of(ikv, st.is_v != 0);
        if (t == nullptr || units.empty()) {
            retire_deferred();
            continue;
        }
        if (!vbr_unit_movable(t->type, st.is_v != 0)) {
            retire_deferred();
            continue; // PINNED unit (explicit non-vbr side): the ladder never touches it
        }
        const ggml_type type_B = vbr_tier_type(st.tier);
        {
            // tier decision on the canonical tensor — relative row sizes are identical on every
            // instance (blocks never straddle the shard split)
            const size_t rA = ggml_row_size(t->type, t->ne[0]);
            const size_t rB = ggml_row_size(type_B,  t->ne[0]);
            if (t->type == type_B || rB >= rA) {
                retire_deferred();
                continue; // not a real degrade from the current tier (e.g. F16 band on a t8 static start)
            }
        }

        if (vbr_hard_seal_guard_) {
            if (vbr_hard_seal_step_blocked(order_ordinal, seal_session)) {
                saw_hard_lease_block = true;
                vbr_hard_seal_evidence_record(order_ordinal);
                if (!from_deferred) {
                    if (!attempted_ready) {
                        std::fill(vbr_hard_seal_attempted_.begin(),
                                  vbr_hard_seal_attempted_.end(), 0);
                        attempted_ready = true;
                    }
                    vbr_hard_seal_defer_step(
                        vbr_hard_seal_deferred_, order_ordinal,
                        &vbr_hard_seal_attempted_);
                }
                LLAMA_LOG_INFO("%s: hard lease sealed VBR order step %zu (L%d %c); trying the next unit\n",
                        __func__, order_ordinal, (int) st.il, st.is_v ? 'V' : 'K');
                continue;
            }
        }

        if (from_deferred) {
            LLAMA_LOG_INFO("%s: hard-lease release reopened VBR order step %zu (L%d %c)\n",
                    __func__, order_ordinal, (int) st.il, st.is_v ? 'V' : 'K');
        }

        const uint32_t logical_unit =
            static_cast<uint32_t>(ikv * 2 + (st.is_v != 0));
        if (logical_unit < vbr_capture_unit_attempt_boundary_.size() &&
            vbr_capture_unit_attempt_boundary_[logical_unit] ==
                vbr_boundary_count_) {
            saw_capture_lease_block = true;
            if (!from_capture_deferred) {
                vbr_hard_seal_defer_step(
                    vbr_capture_retier_deferred_, order_ordinal,
                    &vbr_capture_retier_attempted_);
            }
            continue;
        }
        vbr_unit_retier_guard capture_guard(this, logical_unit);
        if (!capture_guard) {
            saw_capture_lease_block = true;
            if (logical_unit < vbr_capture_unit_attempt_boundary_.size()) {
                vbr_capture_unit_attempt_boundary_[logical_unit] =
                    vbr_boundary_count_;
            }
            if (!from_capture_deferred) {
                if (vbr_capture_retier_attempt_boundary_ !=
                        vbr_boundary_count_) {
                    std::fill(vbr_capture_retier_attempted_.begin(),
                              vbr_capture_retier_attempted_.end(), 0);
                    vbr_capture_retier_attempt_boundary_ =
                        vbr_boundary_count_;
                }
                vbr_hard_seal_defer_step(
                    vbr_capture_retier_deferred_, order_ordinal,
                    &vbr_capture_retier_attempted_);
            }
            LLAMA_LOG_DEBUG(
                "%s: capture leased VBR order step %zu (L%d %c); "
                "trying the next unit\n",
                __func__, order_ordinal, (int) st.il,
                st.is_v ? 'V' : 'K');
            continue;
        }

        // Determine and land every stash page this unit can touch before starting any side stream,
        // capture, transcode, metadata flip, or deferred unmap. Tensor-split units reserve on all
        // devices first; a failure may retain initialized VMM pages but cannot leave mixed tiers.
        struct pending_stash {
            vbr_pool *   pool;
            vbr_extent * extent;
            uint32_t     rows;
            bool         capture;
        };
        std::vector<pending_stash> pending_stashes;
        pending_stashes.reserve(units.size());
        if (vbr_stash_rows_ > 0) {
            for (auto & [pp, ep] : units) {
                const int64_t n_cells = pp->wm_cells;
                if (n_cells <= 0) {
                    continue;
                }
                const bool capture = ep->stash_valid == 0 &&
                        ggml_is_turbo_kv_type(ep->t->type) && ep->t->type != GGML_TYPE_TURBO8_0;
                const uint32_t rows = capture
                        ? (uint32_t) std::min<int64_t>(vbr_stash_rows_, n_cells)
                        : ep->stash_valid;
                if (rows > 0) {
                    pending_stashes.push_back({ pp, ep, rows, capture });
                }
            }
            for (const auto & pending : pending_stashes) {
                const std::vector<vbr_stash_request> request = {{ pending.extent, pending.rows }};
                if (!vbr_stash_reserve(*pending.pool, request)) {
                    // Restore the exact call-entry cursor so rejection is observationally atomic:
                    // even skipped no-op order entries are reconsidered with the retried step.
                    vbr_degrade_cursor_ = cursor_entry;
                    vbr_reserve_failed_ = true;
                    LLAMA_LOG_ERROR("%s: VBR sink-stash reserve failed on device %d before tier "
                            "mutation; leaving %s at %s\n", __func__, pending.pool->device,
                            pending.extent->t->name, ggml_type_name(pending.extent->t->type));
                    return vbr_degrade_result::reserve_failed;
                }
            }
        }

        for (auto & [pp, ep] : units) {
            vbr_extent  & e   = *ep;

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
            const size_t keep_live = sp.keep_live;
            vbr_scrub_span scrub = {};

            if (n_cells > 0) {
                // Scrub math is only meaningful (and its >=keep invariant only
                // holds) for a populated extent; a 0-cell degrade is a free
                // metadata-only re-typing and must never reach the helper.
                GGML_ASSERT(vbr_scrub_span_of(
                    e.t, e.t->type, n_cells, sp, pp->gran, scrub));
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

                // f16 sink-stash: capture rows [0, stash_rows) from the first tapped-domain tier-A
                // recon, then every later hop re-encodes those rows from the stash — the sink is
                // the only region both permanently hot and permanently old
                const void * stash_ptr  = nullptr;
                int64_t      stash_rows = 0;
                if (vbr_stash_rows_ > 0) {
                    const auto pending = std::find_if(pending_stashes.begin(), pending_stashes.end(),
                            [&](const pending_stash & s) { return s.pool == pp && s.extent == &e; });
                    if (pending != pending_stashes.end()) {
                        GGML_ASSERT(pp->stash_vmm != nullptr);
                        char * sbase = (char *) pp->be->vmm_pool_base(pp->stash_vmm);
                        // Only tapped tiers provide the domain the suppressed encode tap consumes.
                        // f16/t8 hops neither allocate nor capture an unusable full-domain stash.
                        if (pending->capture) {
                            pp->be->kv_stash_capture(pp->backend, e.t, sbase + e.stash_off,
                                                    pending->rows, st.is_v != 0);
                            e.stash_valid = pending->rows;
                        }
                        stash_ptr  = sbase + e.stash_off;
                        stash_rows = e.stash_valid;
                    }
                }

                // Transcode and scrub run asynchronously on the side stream; the end-of-wave fence
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
                    /*.scrub_bytes =*/ scrub.scrub_end - scrub.keep,
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
        const ggml_type type_A = t->type;
        vbr_set_tensor_type(t, st.is_v ? layers[ikv].v_stream : layers[ikv].k_stream, type_B);
        vbr_tier_epoch_++; // fence graph reuse off the old views (type/strides changed in place)
        // Demand sheds route through this same mutation, so every shed transcode is covered here
        // without a second generation publication in the caller.
        vbr_representation_changed();
        if (auto * tracker = vbr_generation_tracker_mut()) {
            const uint8_t promote_hops = units.empty() ? 0 : units.front().second->promote_hops;
            const auto transition = type_A == GGML_TYPE_F16 && type_B == GGML_TYPE_TURBO8_0
                    ? vbr_repr_transition::degrade_f16_to_t8_admitted
                    : vbr_repr_transition::degrade_other;
            GGML_ASSERT(tracker->publish_unit(
                    static_cast<uint32_t>(ikv * 2 + (st.is_v != 0)),
                    static_cast<int32_t>(type_A),
                    static_cast<int32_t>(type_B),
                    type_B == GGML_TYPE_F16 || type_B == GGML_TYPE_TURBO8_0
                            ? vbr_repr_domain::full
                            : vbr_repr_domain::tapped,
                    promote_hops,
                    transition,
                    vbr_mutation_registrant::degrade_next,
                    vbr_cited_op()));
        }
        retire_deferred();
        return vbr_degrade_result::applied;
    }
    if (saw_hard_lease_block) {
        vbr_hard_seal_blocked_ = true;
        return vbr_degrade_result::hard_lease_blocked;
    }
    if (saw_capture_lease_block) {
        return vbr_degrade_result::capture_lease_blocked;
    }
    return vbr_degrade_result::exhausted;
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
    struct anchor_device {
        int device = -1;
        const ggml_vbr_backend_iface * be = nullptr;
        const ggml_vbr_cross_domain_iface_v1 * cross_be = nullptr;
    };
    std::vector<anchor_device> devices;
    for (const auto & p : vbr_pools_) {
        if (p.be == nullptr) {
            continue; // bookkeeping-only pool (static allocation), no backend vtable
        }
        const int dev = p.device >= 0 ? p.device : 0;
        if (std::find_if(devices.begin(), devices.end(),
                [dev](const auto & d) { return d.device == dev; }) == devices.end()) {
            devices.push_back({ dev, p.be, p.cross_be });
        }
    }
    for (const auto & device : devices) {
    const int dev = device.device;
    const auto * be = device.be;
    const auto * cross_be = device.cross_be;
    fprintf(stderr, "VBR anchor: device %d\n", dev);
    ggml_backend_t bk = be->backend_init(dev);
    if (!bk) {
        fprintf(stderr, "VBR anchor: backend_init failed\n");
        continue;
    }
    {
        const ggml_type t8 = GGML_TYPE_TURBO8_0;
        int64_t ne0 = 1024;
        llama_turbo_meansub_ref oracle_mean_ref;
        for (size_t i = 0; i < layers.size(); ++i) if (layers[i].k && layers[i].k->type == t8) {
            ne0 = layers[i].k->ne[0];
            oracle_mean_ref = layers[i].turbo_meansub_ref;
            break;
        }
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
            char nm_storage[64];
            snprintf(
                nm_storage, sizeof(nm_storage), "cache_%c_l%d_ms%d",
                ivar ? 'v' : 'k', oracle_mean_ref.layer,
                oracle_mean_ref.model_id);
            const char * nm = nm_storage;
            const ggml_type ladder[4] = { GGML_TYPE_TURBO1_TCQ, GGML_TYPE_TURBO2_TCQ,
                                          GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO4_0 };
            const size_t r_max = ggml_row_size(ladder[3], ne0);
            const size_t r_full_max = ggml_row_size(GGML_TYPE_F16, ne0);
            int mean_max_l = 0;
            int mean_max_c = 0;
            int mean_live = 0;
            const bool cross_ready =
                cross_be != nullptr &&
                cross_be->kv_cross_domain_reconstruct != nullptr &&
                cross_be->kv_transcode_workspace_reserve_v2 != nullptr &&
                oracle_mean_ref.model_id > 0 && oracle_mean_ref.layer >= 0 &&
                ggml_turbo_meansub_table(
                    oracle_mean_ref.model_id, ivar, &mean_max_l,
                    &mean_max_c, &mean_live) != nullptr &&
                oracle_mean_ref.layer < mean_max_l && mean_live > 0 &&
                ne0 <= mean_max_c;

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
                    ggml_backend_get_default_buffer_type(bk),
                    std::max(r_max, r_full_max) * (size_t) N);
            void * slab = ggml_backend_buffer_get_base(slabbuf);
            // header contexts: transcode reads type/ne[0]/nb[1]/data/name of src only
            ggml_init_params iph = { 128*ggml_tensor_overhead(), nullptr, true };
            ggml_context * hc = ggml_init(iph);
            auto alias = [&](ggml_type tt, void * base, ggml_backend_buffer_t buffer) {
                ggml_tensor * x = ggml_new_tensor_2d(hc, tt, ne0, N);
                x->data = base; x->buffer = buffer;
                ggml_set_name(x, nm);
                return x;
            };
            // raw device->host byte copy via a throwaway I8 header (tensor_get needs one)
            auto download = [&](void * base, ggml_backend_buffer_t b, void * dst, size_t nbytes) {
                ggml_tensor * d = ggml_new_tensor_1d(hc, GGML_TYPE_I8, (int64_t) nbytes);
                d->data = base; d->buffer = b;
                ggml_backend_tensor_get(d, dst, 0, nbytes);
            };
            // Exercise every direct tapped promotion edge, not only the adjacent live-promotion
            // ladder. Import reconstruction deliberately uses one direct edge, so T1->T4 and
            // T2->T4 must independently prove the same reverse-tile in-place invariant.
            for (int from = 0; from + 1 < 4; ++from) {
                const ggml_type tfrom = ladder[from];
                const size_t bytesFrom = ggml_row_size(tfrom, ne0) * (size_t) N;
                ggml_backend_buffer_t sourcebuf = ggml_backend_buft_alloc_buffer(
                        ggml_backend_get_default_buffer_type(bk), bytesFrom);
                void * source = ggml_backend_buffer_get_base(sourcebuf);
                const ggml_vbr_transcode_params tp_dn = {
                    s_t8, tfrom, source, sourcebuf, N, /*is_v=*/ivar != 0,
                    nullptr, 0, /*scrub_bytes=*/0,
                };
                be->kv_transcode(bk, &tp_dn);
                ggml_backend_synchronize(bk);
                std::vector<uint8_t> source_host(bytesFrom);
                download(source, sourcebuf, source_host.data(), bytesFrom);

                for (int to = from + 1; to < 4; ++to) {
                    const ggml_type tto     = ladder[to];
                    const size_t    bytesTo = ggml_row_size(tto, ne0) * (size_t) N;

                    ggml_backend_buffer_t hsep =
                        ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(bk), bytesTo);
                    const ggml_vbr_transcode_params tp_hs = {
                        alias(tfrom, source, sourcebuf),
                        tto,
                        ggml_backend_buffer_get_base(hsep),
                        hsep,
                        N,
                        /*is_v=*/ivar != 0,
                        nullptr,
                        0,
                        /*scrub_bytes=*/0,
                    };
                    be->kv_transcode(bk, &tp_hs);
                    ggml_backend_synchronize(bk);

                    ggml_backend_tensor_set(alias(tfrom, slab, slabbuf), source_host.data(), 0, bytesFrom);
                    const ggml_vbr_transcode_params tp_hi = {
                        alias(tfrom, slab, slabbuf), tto,     slab, slabbuf,           N,
                        /*is_v=*/ivar != 0,          nullptr, 0,    /*scrub_bytes=*/0,
                    };
                    be->kv_transcode(bk, &tp_hi);
                    ggml_backend_synchronize(bk);

                    std::vector<uint8_t> hb(bytesTo), ib(bytesTo);
                    download(ggml_backend_buffer_get_base(hsep), hsep, hb.data(), bytesTo);
                    download(slab, slabbuf, ib.data(), bytesTo);
                    size_t same = 0, first_bad = bytesTo;
                    for (size_t i = 0; i < bytesTo; ++i) {
                        if (hb[i] == ib[i]) {
                            same++;
                        } else if (first_bad == bytesTo) {
                            first_bad = i;
                        }
                    }
                    fprintf(stderr,
                            "VBR SELFTEST PROMOTE %s %s->%s in-place==separate: %.3f%% (%zu/%zu)%s first-diff byte "
                            "%lld (row %lld)\n",
                            nm, ggml_type_name(tfrom), ggml_type_name(tto), 100.0 * (double) same / (double) bytesTo,
                            same, bytesTo, same == bytesTo ? "" : " byte-MISMATCH",
                            same == bytesTo ? -1LL : (long long) first_bad,
                            same == bytesTo ? -1LL : (long long) (first_bad / ggml_row_size(tto, ne0)));
                    if (same != bytesTo) {
                        // TCQ trellis blocks carry trailing don't-care bits the decode never reads, so a
                        // byte diff is not yet corruption — adjudicate on DEQUANTIZED values instead
                        const size_t          fb = (size_t) N * ne0 * sizeof(uint16_t);
                        ggml_backend_buffer_t f1 =
                            ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(bk), fb);
                        ggml_backend_buffer_t f2 =
                            ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(bk), fb);
                        ggml_tensor * asep = ggml_new_tensor_2d(hc, tto, ne0, N);
                        asep->data         = ggml_backend_buffer_get_base(hsep);
                        asep->buffer       = hsep;
                        ggml_set_name(asep, nm);
                        be->kv_stash_capture(bk, asep, ggml_backend_buffer_get_base(f1), N, ivar != 0);
                        be->kv_stash_capture(bk, alias(tto, slab, slabbuf), ggml_backend_buffer_get_base(f2), N,
                                             ivar != 0);
                        ggml_backend_synchronize(bk);
                        std::vector<uint16_t> v1(fb / 2), v2(fb / 2);
                        download(ggml_backend_buffer_get_base(f1), f1, v1.data(), fb);
                        download(ggml_backend_buffer_get_base(f2), f2, v2.data(), fb);
                        size_t  vbad      = 0;
                        int64_t first_row = -1;
                        for (size_t i = 0; i < v1.size(); ++i) {
                            if (v1[i] != v2[i]) {
                                vbad++;
                                if (first_row < 0) {
                                    first_row = (int64_t) (i / (size_t) ne0);
                                }
                            }
                        }
                        fprintf(stderr,
                                "VBR SELFTEST PROMOTE %s %s->%s DEQUANT compare: %s (%zu/%zu f16 values differ, first "
                                "row %lld)\n",
                                nm, ggml_type_name(tfrom), ggml_type_name(tto),
                                vbad == 0 ? "IDENTICAL (slack bits only)" : "VALUE MISMATCH", vbad, v1.size(),
                                (long long) first_row);
                        ggml_backend_buffer_free(f1);
                        ggml_backend_buffer_free(f2);
                    }
                    ggml_backend_buffer_free(hsep);
                }
                if (cross_ready) {
                    const ggml_vbr_transcode_workspace_params_v2 workspace_request = {
                        N, ne0, 0, true,
                    };
                    GGML_ASSERT(cross_be->kv_transcode_workspace_reserve_v2(
                        bk, &workspace_request));
                    for (const ggml_type full_target : {
                            GGML_TYPE_TURBO8_0, GGML_TYPE_F16 }) {
                        const size_t bytes_to =
                            ggml_row_size(full_target, ne0)*(size_t) N;
                        ggml_backend_buffer_t hsep =
                            ggml_backend_buft_alloc_buffer(
                                ggml_backend_get_default_buffer_type(bk),
                                bytes_to);
                        const ggml_vbr_cross_domain_reconstruct_params separate = {
                            { alias(tfrom, source, sourcebuf), full_target,
                              ggml_backend_buffer_get_base(hsep), hsep, N,
                              ivar != 0, nullptr, 0, 0 },
                            oracle_mean_ref.model_id,
                            oracle_mean_ref.layer,
                            0,
                        };
                        GGML_ASSERT(cross_be->kv_cross_domain_reconstruct(
                            bk, &separate));
                        ggml_backend_tensor_set(
                            alias(tfrom, slab, slabbuf),
                            source_host.data(), 0, bytesFrom);
                        const ggml_vbr_cross_domain_reconstruct_params inplace = {
                            { alias(tfrom, slab, slabbuf), full_target,
                              slab, slabbuf, N, ivar != 0,
                              nullptr, 0, 0 },
                            oracle_mean_ref.model_id,
                            oracle_mean_ref.layer,
                            0,
                        };
                        GGML_ASSERT(cross_be->kv_cross_domain_reconstruct(
                            bk, &inplace));
                        ggml_backend_synchronize(bk);
                        std::vector<uint8_t> separate_host(bytes_to);
                        std::vector<uint8_t> inplace_host(bytes_to);
                        download(
                            ggml_backend_buffer_get_base(hsep), hsep,
                            separate_host.data(), bytes_to);
                        download(
                            slab, slabbuf, inplace_host.data(), bytes_to);
                        const bool identical = separate_host == inplace_host;
                        fprintf(stderr,
                            "VBR SELFTEST CROSS %s %s->%s in-place==separate: %s (%zu bytes)\n",
                            nm, ggml_type_name(tfrom),
                            ggml_type_name(full_target),
                            identical ? "IDENTICAL" : "MISMATCH", bytes_to);
                        GGML_ASSERT(identical);
                        ggml_backend_buffer_free(hsep);
                    }
                }
                ggml_backend_buffer_free(sourcebuf);
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

llama_memory_vbr_state_data_v2 llama_kv_cache::memory_vbr_state_v2(
        llama_seq_id seq_id, uint32_t n_tokens_extra) const {
    llama_memory_vbr_state_data_v2 result = {};
    auto & st = result.state;
    st.representation_epoch = vbr_representation_epoch();
    st.checkpoint_epoch     = vbr_checkpoint_epoch(seq_id);
    st.retier_freeze_depth       = other ? other->vbr_retier_freeze_depth_       : vbr_retier_freeze_depth_;
    st.retier_env_freeze         = other ? other->vbr_freeze_                    : vbr_freeze_;
    st.retier_freeze_enters      = other ? other->vbr_retier_freeze_enters_      : vbr_retier_freeze_enters_;
    st.retier_freeze_exits       = other ? other->vbr_retier_freeze_exits_       : vbr_retier_freeze_exits_;
    st.retier_deferred_decisions = other ? other->vbr_retier_deferred_decisions_ : vbr_retier_deferred_decisions_;
    st.retier_reconciles         = other ? other->vbr_retier_reconciles_         : vbr_retier_reconciles_;

    // full-reset feasibility: used cells the asking seq does not exclusively own. Cells above
    // used_max_p1 are empty by definition, so the scan is bounded by live occupancy.
    uint32_t used_cells = 0;
    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];
        used_cells += cells.get_used();
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
    result.used_cells_exclusive = used_cells - st.used_cells_other;

    if (!vbr_vmm_active() || vbr_budget_bytes_ == 0) {
        return result; // no controller: zeros besides the occupancy counts
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
        return result; // no VMM pools — controller effectively inert
    }
    st.deficit_raw     = deficit_raw;
    st.deficit_clamped = deficit_clamped;

    // bpv_if_degraded: walk the ladder from the CURRENT cursor with the same skip rules as
    // vbr_degrade_next until every pool's RAW projection fits (or the floor clamp stops it) —
    // the aggregate the controller would land at if the deficit were paid by tiers alone.
    // Mirrors the vbr_floor_clamp_order simulation; aggregate basis = VMM-pooled units.
    std::vector<ggml_type> sim;
    double  sum_bits = 0.0;
    int64_t sum_vals = 0;
    vbr_sim_seed(sim, /*pooled_only=*/true, GGML_TYPE_COUNT, GGML_TYPE_COUNT,
                 &sum_bits, &sum_vals, nullptr);
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
        size_t slot; const ggml_tensor * t; ggml_type type_B;
        if (!vbr_sim_step(sim, i, slot, t, type_B)) {
            continue;
        }
        const auto & stp = vbr_degrade_order_[i];
        const size_t ikv = slot / 2;
        const size_t rA = ggml_row_size(sim[slot], t->ne[0]);
        const size_t rB = ggml_row_size(type_B,    t->ne[0]);
        // projection deltas land in EVERY pool holding the unit, priced at each pool's shard width
        for (size_t pi = 0; pi < vbr_pools_.size(); ++pi) {
            const auto & p = vbr_pools_[pi];
            if (p.vmm == nullptr) {
                continue;
            }
            const vbr_extent & e = stp.is_v ? p.v[ikv] : p.k[ikv];
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

    return result;
}

bool llama_kv_cache::vbr_capture_readiness_cells(
        uint64_t logical_growth,
        uint64_t & committed,
        uint64_t & projected,
        uint64_t & capacity) const {
    committed = 0;
    projected = 0;
    capacity = 0;
    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];
        committed += cells.get_used();
        capacity += cells.size();
    }
    projected = committed > capacity ? capacity :
        std::min<uint64_t>(capacity, committed+std::min<uint64_t>(
            logical_growth, capacity-committed));
    return capacity != 0;
}

bool llama_kv_cache::vbr_accumulate_exclusive_cells(
        uint32_t * counts, size_t size) const {
    if (other) {
        return other->vbr_accumulate_exclusive_cells(counts, size);
    }
    if (!counts || size > LLAMA_MAX_SEQ) {
        return false;
    }
    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];
        const uint32_t top = cells.used_max_p1();
        for (uint32_t i = 0; i < top; ++i) {
            if (cells.is_empty(i) || cells.seq_count(i) != 1) {
                continue;
            }
            bool valid = true;
            cells.seq_for_each(i, [&](llama_seq_id seq_id) {
                if (seq_id < 0 || size_t(seq_id) >= size) {
                    return;
                }
                if (counts[size_t(seq_id)] == UINT32_MAX) {
                    valid = false;
                    return;
                }
                counts[size_t(seq_id)]++;
            });
            if (!valid) {
                return false;
            }
        }
    }
    return true;
}

bool llama_kv_cache::vbr_operation_armed() const {
    if (other) {
        return other->vbr_operation_armed();
    }
    return vbr_vmm_active() && vbr_budget_bytes_ > 0;
}

bool llama_kv_cache::vbr_recovery_service_pending() const {
    if (other) {
        return other->vbr_recovery_service_pending();
    }
    const auto * tracker = vbr_generation_tracker_get();
    if (tracker == nullptr) {
        return false;
    }
    return tracker->shadow_unavailable() ||
           vbr_recovery_pending_for(tracker->runtime_instance());
}

bool llama_kv_cache::vbr_retier_freeze_begin(
        const char * owner, vbr_operation_id operation_id) {
    if (other) {
        return other->vbr_retier_freeze_begin(owner, operation_id);
    }
    if (!vbr_operation_armed()) {
        return false;
    }
    if (!vbr_operation_registry_is_live(operation_id)) {
        LLAMA_LOG_ERROR("VBR_OPERATION event=reject reason=unregistered_id owner=%s operation_id=%llu\n",
                owner != nullptr ? owner : "-",
                (unsigned long long) operation_id.value);
        return false;
    }
    if (vbr_retier_freeze_depth_ >= VBR_RETIER_FREEZE_MAX_DEPTH) {
        LLAMA_LOG_ERROR("VBR_OPERATION event=reject reason=freeze_depth_limit owner=%s "
                "operation_id=%llu depth=%u\n",
                owner != nullptr ? owner : "-",
                (unsigned long long) operation_id.value,
                vbr_retier_freeze_depth_);
        return false;
    }
    if (vbr_retier_freeze_depth_ == 0 &&
        !vbr_capture_controller_write_begin()) {
        LLAMA_LOG_DEBUG(
            "VBR_OPERATION event=reject reason=capture_unit_leased owner=%s "
            "operation_id=%llu\n",
            owner != nullptr ? owner : "-",
            (unsigned long long) operation_id.value);
        return false;
    }
    const uint64_t now = llama_vram_ledger_now_ns();
    if (vbr_retier_freeze_depth_ == 0) {
        vbr_retier_outer_deferred_base_ = vbr_retier_deferred_decisions_;
    }
    vbr_retier_freeze_stack_[vbr_retier_freeze_depth_] = {
        operation_id,
        now != 0 ? now : 1,
    };
    vbr_retier_freeze_depth_++;
    vbr_retier_freeze_enters_++;
    LLAMA_LOG_INFO("VBR_RETIER_FREEZE event=enter controller=%s owner=%s operation_id=%llu "
            "depth=%u env_freeze=%u enters_total=%llu\n",
            vbr_params_.trace_label != nullptr ? vbr_params_.trace_label : "single",
            owner != nullptr ? owner : "-",
            (unsigned long long) operation_id.value,
            vbr_retier_freeze_depth_, vbr_freeze_ ? 1u : 0u,
            (unsigned long long) vbr_retier_freeze_enters_);
    return true;
}

void llama_kv_cache::vbr_retier_freeze_end(
        const char * owner, vbr_operation_id operation_id) {
    if (other) {
        other->vbr_retier_freeze_end(owner, operation_id);
        return;
    }
    GGML_ASSERT(vbr_retier_freeze_depth_ > 0);
    const vbr_retier_freeze_frame frame =
        vbr_retier_freeze_stack_[vbr_retier_freeze_depth_ - 1];
    GGML_ASSERT(frame.operation_id == operation_id);
    GGML_ASSERT(vbr_operation_registry_is_live(operation_id));
    const uint64_t now = llama_vram_ledger_now_ns();
    vbr_retier_freeze_depth_--;
    vbr_retier_freeze_stack_[vbr_retier_freeze_depth_] = {};
    vbr_retier_freeze_exits_++;
    const uint64_t duration_us =
        now > frame.started_ns ? (now - frame.started_ns) / 1000 : 0;
    const uint64_t deferred_scope =
        vbr_retier_deferred_decisions_ - vbr_retier_outer_deferred_base_;
    if (vbr_retier_freeze_depth_ == 0 && !vbr_freeze_) {
        // Even a no-op trim can change the controller's future occupancy inputs. Force a fresh
        // decision pass at the next safe boundary; never execute a stale queued choice here.
        vbr_retier_reconcile_pending_ = true;
    }
    if (vbr_retier_freeze_depth_ == 0) {
        vbr_capture_controller_write_end();
    }
    LLAMA_LOG_INFO("VBR_RETIER_FREEZE event=exit controller=%s owner=%s operation_id=%llu "
            "depth=%u duration_us=%llu "
            "deferred_scope=%llu deferred_total=%llu exits_total=%llu action=%s\n",
            vbr_params_.trace_label != nullptr ? vbr_params_.trace_label : "single",
            owner != nullptr ? owner : "-",
            (unsigned long long) operation_id.value,
            vbr_retier_freeze_depth_,
            (unsigned long long) duration_us,
            (unsigned long long) deferred_scope,
            (unsigned long long) vbr_retier_deferred_decisions_,
            (unsigned long long) vbr_retier_freeze_exits_,
            vbr_retier_freeze_depth_ > 0
                ? "remain_frozen"
                : vbr_retier_reconcile_pending_
                    ? "reevaluate_next_boundary"
                    : "env_freeze_noop");
}

bool llama_kv_cache::vbr_import_destination_pricing_begin(
        const std::vector<ggml_type> & types,
        uint32_t projected_wm_cells,
        vbr_import_destination_pricing & output) const noexcept {
    output = {};
    try {
        if (other) {
            return other->vbr_import_destination_pricing_begin(
                types, projected_wm_cells, output);
        }
        if (!vbr_vmm_active() || vbr_budget_bytes_ == 0 ||
            projected_wm_cells == 0 || types.size() != layers.size()*2) {
            return false;
        }
        output.watermark_cells = projected_wm_cells;
        output.types = types;
        output.pools.reserve(vbr_pools_.size());
        output.devices.reserve(vbr_pools_.size());
        const auto add = [&](uint64_t lhs, uint64_t rhs) {
            if (rhs > UINT64_MAX - lhs) {
                output.overflow = true;
                return UINT64_MAX;
            }
            return lhs + rhs;
        };
        const auto multiply = [&](uint64_t lhs, uint64_t rhs) {
            if (lhs != 0 && rhs > UINT64_MAX/lhs) {
                output.overflow = true;
                return UINT64_MAX;
            }
            return lhs*rhs;
        };
        for (const auto & p : vbr_pools_) {
            if (!p.vmm) {
                continue;
            }
            vbr_import_destination_pricing::pool_row row;
            row.be = p.be;
            row.device = p.device;
            row.mapped = p.be->vmm_pool_mapped(p.vmm);
            row.available = vbr_budget_eff_uncached(p);
            row.needed = p.mapped_base;
            for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
                for (int side = 0; side < 2; ++side) {
                    const auto & extent = side ? p.v[ikv] : p.k[ikv];
                    const auto * tensor = extent.t;
                    if (!tensor || types[ikv*2 + side] == GGML_TYPE_COUNT) {
                        if (tensor) {
                            return false;
                        }
                        continue;
                    }
                    uint64_t endpoint = 0;
                    if (!llama_vbr_policy::logical_endpoint_bytes(
                            ggml_row_size(types[ikv*2 + side], tensor->ne[0]),
                            projected_wm_cells, vbr_slot_span(tensor, p.gran),
                            p.gran, endpoint)) {
                        output.overflow = true;
                        endpoint = UINT64_MAX;
                    }
                    row.needed = add(row.needed, endpoint);
                }
            }
            output.pools.push_back(row);
            auto device = std::find_if(
                output.devices.begin(), output.devices.end(),
                [&](const auto & value) {
                    return value.be == p.be && value.device == p.device;
                });
            if (device == output.devices.end()) {
                vbr_import_destination_pricing::device_row created;
                created.be = p.be;
                created.device = p.device;
                size_t free_bytes = 0;
                size_t total = 0;
                p.be->get_device_memory(p.device, &free_bytes, &total);
                created.available = free_bytes;
                output.devices.push_back(created);
                device = output.devices.end() - 1;
            }
            device->scratch_k_current = std::max<uint64_t>(
                device->scratch_k_current, p.scratch_k_reserved);
            device->scratch_v_current = std::max<uint64_t>(
                device->scratch_v_current, p.scratch_v_reserved);
            for (size_t ikv = 0; ikv < layers.size(); ++ikv) {
                const auto * tk = p.k[ikv].t;
                const auto * tv = p.v[ikv].t;
                bool need_k = false;
                bool need_v = false;
                ggml_vbr_kv_dequant_sides(
                    tk ? types[ikv*2] : GGML_TYPE_F16,
                    tv ? types[ikv*2 + 1] : GGML_TYPE_F16,
                    &need_k, &need_v);
                if (need_k && tk) {
                    device->scratch_k_needed = std::max(
                        device->scratch_k_needed, multiply(
                            ggml_row_size(GGML_TYPE_F16, tk->ne[0]),
                            projected_wm_cells));
                }
                if (need_v && tv) {
                    device->scratch_v_needed = std::max(
                        device->scratch_v_needed, multiply(
                            ggml_row_size(GGML_TYPE_F16, tv->ne[0]),
                            projected_wm_cells));
                }
            }
        }
        std::sort(
            output.devices.begin(), output.devices.end(),
            [](const auto & lhs, const auto & rhs) {
                if (lhs.be != rhs.be) {
                    return std::less<const void *>{}(lhs.be, rhs.be);
                }
                return lhs.device < rhs.device;
            });
        output.active = !output.pools.empty() && !output.devices.empty();
        return output.active;
    } catch (...) {
        output = {};
        return false;
    }
}

bool llama_kv_cache::vbr_import_destination_pricing_apply(
        const llama_vbr_policy::step & step,
        vbr_import_destination_pricing & pricing) const noexcept {
    try {
        if (!pricing.active || step.slot >= pricing.types.size() ||
            pricing.types[step.slot] != ggml_type(step.type_a) ||
            step.type_b < 0 || step.type_b >= GGML_TYPE_COUNT) {
            return false;
        }
        const size_t ikv = step.slot/2;
        const bool is_v = (step.slot & 1u) != 0;
        if (ikv >= layers.size()) {
            return false;
        }
        const auto before_type = ggml_type(step.type_a);
        const auto after_type = ggml_type(step.type_b);
        size_t pool_row = 0;
        for (const auto & p : vbr_pools_) {
            if (!p.vmm) {
                continue;
            }
            if (pool_row >= pricing.pools.size()) {
                return false;
            }
            const auto * tensor = (is_v ? p.v[ikv] : p.k[ikv]).t;
            if (tensor) {
                uint64_t before = 0;
                uint64_t after = 0;
                if (!llama_vbr_policy::logical_endpoint_bytes(
                        ggml_row_size(before_type, tensor->ne[0]),
                        pricing.watermark_cells,
                        vbr_slot_span(tensor, p.gran), p.gran, before) ||
                    !llama_vbr_policy::logical_endpoint_bytes(
                        ggml_row_size(after_type, tensor->ne[0]),
                        pricing.watermark_cells,
                        vbr_slot_span(tensor, p.gran), p.gran, after) ||
                    after > before || pricing.pools[pool_row].needed < before) {
                    return false;
                }
                pricing.pools[pool_row].needed -= before - after;
            }
            ++pool_row;
        }
        pricing.types[step.slot] = after_type;
        for (auto & device : pricing.devices) {
            size_t k_row = 0;
            size_t v_row = 0;
            for (const auto & p : vbr_pools_) {
                if (!p.vmm || p.be != device.be || p.device != device.device) {
                    continue;
                }
                const auto * tk = p.k[ikv].t;
                const auto * tv = p.v[ikv].t;
                bool need_k = false;
                bool need_v = false;
                ggml_vbr_kv_dequant_sides(
                    tk ? pricing.types[ikv*2] : GGML_TYPE_F16,
                    tv ? pricing.types[ikv*2 + 1] : GGML_TYPE_F16,
                    &need_k, &need_v);
                if (need_k && tk) {
                    k_row = std::max(
                        k_row, ggml_row_size(GGML_TYPE_F16, tk->ne[0]));
                }
                if (need_v && tv) {
                    v_row = std::max(
                        v_row, ggml_row_size(GGML_TYPE_F16, tv->ne[0]));
                }
            }
            if (k_row != 0) {
                if (k_row > UINT64_MAX/pricing.watermark_cells) {
                    pricing.overflow = true;
                } else {
                    device.scratch_k_needed = std::max<uint64_t>(
                        device.scratch_k_needed,
                        k_row*pricing.watermark_cells);
                }
            }
            if (v_row != 0) {
                if (v_row > UINT64_MAX/pricing.watermark_cells) {
                    pricing.overflow = true;
                } else {
                    device.scratch_v_needed = std::max<uint64_t>(
                        device.scratch_v_needed,
                        v_row*pricing.watermark_cells);
                }
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

llama_memory_vbr_preflight_data llama_kv_cache::vbr_import_destination_preflight(
        const vbr_import_destination_pricing & pricing,
        std::vector<llama_memory_vbr_physical_growth> * physical) const noexcept {
    if (physical) {
        physical->clear();
    }
    try {
    llama_memory_vbr_preflight_data result = {};
    result.active = pricing.active;
    result.fits = pricing.active && !pricing.overflow;
    result.watermark_cells = pricing.watermark_cells;
    const auto add = [](uint64_t lhs, uint64_t rhs) {
        return rhs > UINT64_MAX - lhs ? UINT64_MAX : lhs + rhs;
    };
    for (const auto & pool : pricing.pools) {
        result.pools = result.pools == UINT32_MAX
            ? UINT32_MAX : result.pools + 1;
        result.bytes_needed = add(result.bytes_needed, pool.needed);
        result.bytes_available = add(
            result.bytes_available, pool.available);
        const int64_t deficit = pricing.overflow
            ? INT64_MAX
            : pool.needed > pool.available
                ? pool.needed - pool.available > uint64_t(INT64_MAX)
                    ? INT64_MAX : int64_t(pool.needed - pool.available)
                : 0;
        result.max_deficit = std::max(result.max_deficit, deficit);
        result.fits = result.fits && deficit == 0;
    }
    if (physical) {
        physical->reserve(pricing.devices.size());
    }
    for (const auto & device : pricing.devices) {
        uint64_t kv = 0;
        bool overflow = pricing.overflow;
        for (const auto & pool : pricing.pools) {
            if (pool.be != device.be || pool.device != device.device) {
                continue;
            }
            const uint64_t growth = pool.needed > pool.mapped
                ? pool.needed - pool.mapped : 0;
            if (growth > UINT64_MAX-kv) {
                overflow = true;
                kv = UINT64_MAX;
            } else {
                kv += growth;
            }
        }
        const uint64_t scratch_k =
            device.scratch_k_needed > device.scratch_k_current
                ? device.scratch_k_needed - device.scratch_k_current : 0;
        const uint64_t scratch_v =
            device.scratch_v_needed > device.scratch_v_current
                ? device.scratch_v_needed - device.scratch_v_current : 0;
        uint64_t needed = kv;
        for (const uint64_t value : { scratch_k, scratch_v }) {
            if (value > UINT64_MAX-needed) {
                overflow = true;
                needed = UINT64_MAX;
            } else {
                needed += value;
            }
        }
        const int64_t deficit = overflow
            ? INT64_MAX
            : needed > device.available
                ? needed - device.available > uint64_t(INT64_MAX)
                    ? INT64_MAX : int64_t(needed - device.available)
                : 0;
        result.physical_growth_needed = add(
            result.physical_growth_needed, needed);
        result.physical_growth_available = add(
            result.physical_growth_available, device.available);
        result.max_deficit = std::max(result.max_deficit, deficit);
        result.fits = result.fits && deficit == 0;
        if (physical) {
            physical->push_back({
                device.be, device.device, kv,
                device.scratch_k_needed, device.scratch_v_needed,
                device.scratch_k_current, device.scratch_v_current,
                device.available });
        }
    }
    return result;
    } catch (...) {
        if (physical) {
            physical->clear();
        }
        return {};
    }
}

llama_memory_vbr_preflight_data llama_kv_cache::vbr_import_destination_preflight(
        const std::vector<ggml_type> & types,
        uint32_t projected_wm_cells,
        std::vector<llama_memory_vbr_physical_growth> * physical) const noexcept {
    vbr_import_destination_pricing pricing;
    if (!vbr_import_destination_pricing_begin(
            types, projected_wm_cells, pricing)) {
        if (physical) {
            physical->clear();
        }
        return {};
    }
    return vbr_import_destination_preflight(pricing, physical);
}

llama_memory_vbr_preflight_data llama_kv_cache::vbr_retier_preflight(
        uint32_t n_tokens_extra,
        std::vector<llama_memory_vbr_physical_growth> * physical) const {
    if (other) {
        return other->vbr_retier_preflight(n_tokens_extra, physical);
    }
    std::vector<ggml_type> types;
    vbr_sim_seed(types, /* pooled_only = */ true,
                 GGML_TYPE_COUNT, GGML_TYPE_COUNT,
                 nullptr, nullptr, nullptr);
    return vbr_import_destination_preflight(
        types, vbr_watermark_cells(n_tokens_extra), physical);
}

// ---- co-tenancy donor side ----

const std::string & llama_kv_cache::vbr_pool_busid(vbr_pool & p) const {
    // resolved eagerly at pool arming; "-" = resolved-and-absent (never publish). The
    // empty case only exists for pools armed before that code ran (defensive).
    if (p.busid.empty()) {
        p.busid = "-";
    }
    return p.busid;
}

llama_kv_cache::vbr_pool * llama_kv_cache::vbr_find_pool(const std::string & busid) {
    for (auto & p : vbr_pools_) {
        if (p.vmm != nullptr && vbr_pool_busid(p) == busid) {
            return &p;
        }
    }
    return nullptr;
}

llama_kv_cache::vbr_tree_pool_ref llama_kv_cache::vbr_tree_find_pool(const std::string & busid) {
    llama_kv_cache * root = vbr_tree_root();
    if (auto * p = root->vbr_find_pool(busid)) {
        return { root, p };
    }
    if (root->vbr_ledger_sibling_ != nullptr) {
        if (auto * p = root->vbr_ledger_sibling_->vbr_find_pool(busid)) {
            return { root->vbr_ledger_sibling_, p };
        }
    }
    return {};
}

size_t llama_kv_cache::vbr_child_offer(
        const llama_kv_cache * child, const std::string & busid) const {
    const llama_kv_cache * root = vbr_tree_root();
    if (child == nullptr || child != root) {
        return 0;
    }
    return root->vbr_tree_offer(busid);
}

size_t llama_kv_cache::vbr_tree_offer(const std::string & busid) const {
    const llama_kv_cache * root_const = vbr_tree_root();
    const auto suppressed = root_const->vbr_offer_suppressed_until_.find(busid);
    if (suppressed != root_const->vbr_offer_suppressed_until_.end() &&
        llama_vram_ledger_now_ns() < suppressed->second) {
        return 0;
    }
    llama_kv_cache * root = const_cast<llama_kv_cache *>(root_const);
    const auto demanded = root->vbr_tree_find_pool(busid);
    if (demanded.pool == nullptr) {
        return 0;
    }

    vbr_shed_tx tx;
    tx.demanded_device = demanded.pool->device;
    auto append_child = [&](llama_kv_cache * child) {
        if (child == nullptr || !child->vbr_vmm_active()) {
            return;
        }
        vbr_tx_child state;
        state.cache = child;
        state.incoming_wm = child->vbr_watermark_cells(0);
        state.start_cursor = child->vbr_degrade_cursor_;
        state.final_cursor = state.start_cursor;
        state.start_epoch = child->vbr_tier_epoch_;
        child->vbr_sim_seed(state.types_before, /* pooled_only = */ true,
                GGML_TYPE_COUNT, GGML_TYPE_COUNT, nullptr, nullptr, nullptr);
        state.types_after = state.types_before;
        tx.children.push_back(std::move(state));
    };
    append_child(root);
    append_child(root->vbr_ledger_sibling_);

    std::vector<llama_vbr_policy::child> policy_children;
    for (const auto & state : tx.children) {
        policy_children.push_back(state.cache->vbr_policy_child_stream(
                tx.demanded_device, state.incoming_wm));
    }
    llama_vbr_policy::shortest_prefix_stream stream(std::move(policy_children));
    uint64_t best = 0;
    if (!root->vbr_tx_reprice(tx, /* actual = */ false)) {
        return 0;
    }
    const auto note_feasible = [&]() {
        const auto demanded_cost = tx.devices.find(tx.demanded_device);
        if (demanded_cost == tx.devices.end() || demanded_cost->second.capacity_signed <= 0) {
            return;
        }
        for (const auto & [device, cost] : tx.devices) {
            if (device != tx.demanded_device && cost.capacity_signed < 0) {
                return;
            }
        }
        best = std::max(best, (uint64_t) demanded_cost->second.capacity_signed);
    };
    note_feasible();
    llama_vbr_policy::selection selected;
    for (;;) {
        const auto status = stream.next(selected);
        if (status == llama_vbr_policy::result::exhausted) {
            break;
        }
        if (status != llama_vbr_policy::result::selected) {
            return 0;
        }
        tx.policy_prefix.push_back(selected);
        if (!root->vbr_tx_reprice(tx, /* actual = */ false)) {
            return 0;
        }
        note_feasible();
    }
    return best > SIZE_MAX ? SIZE_MAX : (size_t) best;
}

uint32_t llama_kv_cache::vbr_pool_n_live(const vbr_pool & p) const {
    const auto * root = vbr_tree_root();
    const auto it = root->vbr_presence_.find(p.busid);
    return it != root->vbr_presence_.end() && it->second.cur > 0 ? it->second.cur : 1;
}

bool llama_kv_cache::vbr_presence_quiet() const {
    // The presence census is normally fed by the ledger scan, which freeze mode
    // disables. vbr_scan_events_ therefore stays zero and would otherwise disable
    // promotion under freeze (the frozen schedule would be degrade-only, structurally unlike
    // production). Single-tenant frozen has no presence changes by definition -> always quiet,
    // leaving promotion gated only by the deterministic vbr_quiet_boundaries_ cooldown.
    if (vbr_freeze_) {
        return true;
    }
    const auto * root = vbr_tree_root();
    return !root->vbr_tree_forced() &&
           root->vbr_scan_events_ - root->vbr_nlive_change_scan_ >=
                   (uint32_t) LLAMA_VRAM_LEDGER_DEBOUNCE;
}

// Runtime-growth demand, demander side. Publishes a phase=runtime claim when this
// resident spent its own consent window and is still over budget (the try_map-failed case
// arrives here at the NEXT boundary — the map failure fails that batch recoverably first);
// the est carries projected - budget_eff with est_partial=0 so donors apply the shed-sizing
// formula unchanged (an explicit-cap demander whose shortage free VRAM covers nets
// shortfall <= 0 at every donor and draws no shed). CLEAR unlinks at the first boundary
// where the recomputed shortage is gone — the donors' lift signal. Skipped entirely while
// a load-phase claim is still live (satisfied pre-claim-complete: one claim per process).
void llama_kv_cache::vbr_runtime_demand_update(uint32_t wm_next, bool was_over) {
    vbr_runtime_wm_ = wm_next;
    vbr_runtime_was_over_ = was_over;

    llama_kv_cache * root = vbr_tree_root();
    if (this != root || !llama_vram_ledger_armed() || llama_vram_demand_pending_complete()) {
        return;
    }

    bool any_window_spent = false;
    auto note_window = [&](const llama_kv_cache * child) {
        if (child == nullptr) {
            return;
        }
        const size_t limit = child->vbr_demand_limit();
        any_window_spent = any_window_spent ||
                (child->vbr_vmm_active() && limit > 0 && child->vbr_degrade_cursor_ >= limit);
    };
    note_window(root);
    note_window(root->vbr_ledger_sibling_);
    if (!any_window_spent && root->vbr_runtime_live_.empty()) {
        return;
    }

    std::map<std::string, uint64_t> wanted_by_busid;
    auto sample_child = [&](llama_kv_cache * child) {
        if (child == nullptr || !child->vbr_vmm_active()) {
            return;
        }
        const size_t limit = child->vbr_demand_limit();
        const bool window_spent = limit > 0 && child->vbr_degrade_cursor_ >= limit;
        for (size_t pi = 0; pi < child->vbr_pools_.size(); ++pi) {
            auto & p = child->vbr_pools_[pi];
            if (p.vmm == nullptr || child->vbr_pool_busid(p) == "-") {
                continue;
            }
            const size_t projected = child->vbr_vmm_projected_bytes(p, child->vbr_runtime_wm_);
            const size_t beff = child->vbr_budget_eff(p);
            const bool shorted = window_spent &&
                    (child->vbr_runtime_was_over_ || projected > beff);
            if (!shorted) {
                wanted_by_busid[p.busid] += 0;
                continue;
            }
            const uint64_t add = projected > beff ? projected - beff
                    : (pi < child->vbr_pre_deficit_.size() ? child->vbr_pre_deficit_[pi] : 0);
            auto & wanted = wanted_by_busid[p.busid];
            wanted = UINT64_MAX - wanted < add ? UINT64_MAX : wanted + add;
        }
    };
    // Parent execution order is peer first and root last; root publishes one tree claim.
    sample_child(root->vbr_ledger_sibling_);
    sample_child(root);
    for (const auto & busid : root->vbr_runtime_live_) {
        wanted_by_busid[busid] += 0;
    }

    for (const auto & [busid, wanted] : wanted_by_busid) {
        const bool live = root->vbr_runtime_live_.count(busid) != 0;
        if (wanted > 0 && !live) {
            llama_vram_claim_fields f = {};
            f.phase                     = LLAMA_VRAM_CLAIM_RUNTIME;
            f.bytes_total_remaining_est = wanted;
            f.est_partial               = 0;
            f.ver                       = ++root->vbr_runtime_ver_;
            f.created_ts_ns             = llama_vram_ledger_now_ns();
            if (llama_vram_claim_publish(busid, f)) {
                root->vbr_runtime_live_.insert(busid);
                root->vbr_ledger_mtime_ = llama_vram_ledger_dir_mtime_ns();
                LLAMA_LOG_INFO("%s: runtime demand on %s: %.1f MiB (tree consent spent)\n",
                        __func__, busid.c_str(), wanted/1048576.0);
            }
        } else if (wanted == 0 && live) {
            llama_vram_claim_withdraw(busid);
            root->vbr_runtime_live_.erase(busid);
            LLAMA_LOG_INFO("%s: runtime demand on %s cleared\n", __func__, busid.c_str());
        }
    }
}

// promote gate shared by the boundary path and the tick (one env read, one gate — the
// co-tenancy freeze terms live here exactly once)
void llama_kv_cache::vbr_maybe_promote(uint32_t wm_next) {
    static const bool vbr_promote_on = [] {
        const char * e = getenv("VBR_PROMOTE"); // kill switch for experiments; default ON
        return e == nullptr || atoi(e) != 0;
    }();
    if (vbr_promote_on && vbr_degrade_cursor_ > 0 && vbr_quiet_boundaries_ >= 4 &&
        vbr_tree_total_grant_decrement() == 0 && vbr_presence_quiet()) {
        vbr_promote_next(wm_next);
    }
}

void llama_kv_cache::vbr_arm_wave_fences() {
    auto arm_child = [](llama_kv_cache * child) {
        if (child == nullptr) {
            return;
        }
        for (auto & p : child->vbr_pools_) {
            if (p.wave_pending) {
                p.be->fence_arm(p.backend);
                p.wave_pending = false;
            }
        }
    };
    arm_child(this);
    if (vbr_ledger_owner_) {
        arm_child(vbr_ledger_sibling_);
    }
}

size_t llama_kv_cache::vbr_total_grant_decrement() const {
    size_t total = 0;
    for (const auto & p : vbr_pools_) {
        total += p.grant_decrement;
    }
    return total;
}

// recompute each pool's decrement sum from the grant rows and bust the budget memos —
// called only on grant mutation / amortization change (scan events), never per boundary
void llama_kv_cache::vbr_apply_grant_decrements() {
    for (auto & p : vbr_pools_) {
        p.grant_decrement = 0;
    }
    for (const auto & g : vbr_grants_) {
        if (g.pool_idx < vbr_pools_.size()) {
            vbr_pools_[g.pool_idx].grant_decrement += g.bytes; // amortization already folded in
        }
    }
    for (auto & p : vbr_pools_) {
        p.budget_eff_stamp = ~0ull;
    }
}

// dir-mtime pre-check, every boundary OUTSIDE the stable gate (~1µs stat): a rename in the
// ledger (new claim, phase flip, peer offer) forces the full controller path this boundary
void llama_kv_cache::vbr_ledger_precheck() {
    llama_kv_cache * root = vbr_tree_root();
    if (!root->vbr_vmm_active() || !llama_vram_ledger_armed()) {
        return;
    }
    const uint64_t mtime = llama_vram_ledger_dir_mtime_ns();
    if (mtime != root->vbr_ledger_mtime_) {
        // adopt immediately: our own upcoming renames re-stat and re-adopt after writing
        root->vbr_ledger_mtime_ = mtime;
        root->vbr_tree_force();
    }
}

// full ledger pass: grant upkeep (lift / amortize), demand service (rank-0 shed sizing →
// decrement + capped waves), marker publish/beat. Runs inside the !vbr_stable branch after
// the pool's own degrade loop and before the existing fence-arm (a demand wave queued here
// is fenced by that same loop).
void llama_kv_cache::vbr_presence_census(const std::vector<llama_vram_peer_marker> & peers) {
    // ---- Presence census: N_live per device = self + live peer markers ----
    vbr_scan_events_++;
    std::map<std::string, uint32_t> raw;
    auto seed_child = [&](llama_kv_cache * child) {
        if (child == nullptr) {
            return;
        }
        for (auto & p : child->vbr_pools_) {
            if (p.vmm != nullptr && child->vbr_pool_busid(p) != "-") {
                raw[p.busid] = 1; // one process/tree even when both children share a device
            }
        }
    };
    seed_child(this);
    seed_child(vbr_ledger_sibling_);
    for (const auto & m : peers) {
        auto it = raw.find(m.busid);
        if (it != raw.end()) {
            it->second++;
        }
    }
    // A first self-only sighting adopts silently. Peer arrivals are immediate (growing
    // headroom is the safe direction); departures wait for DEBOUNCE consecutive scans.
    for (auto & [busid, n] : raw) {
        auto & st = vbr_presence_[busid];
        st.stable = (st.cur != 0 && n < st.cur && st.raw == n) ? st.stable + 1 : 0;
        st.raw = n;
        bool changed = false;
        if (st.cur == 0) {
            st.cur = n;
            changed = n > 1; // first scan can discover an already-present peer arrival
        } else if (n > st.cur || st.stable >= (uint32_t) LLAMA_VRAM_LEDGER_DEBOUNCE) {
            st.cur = n;
            st.stable = 0;
            changed = true;
        }
        if (changed) {
            vbr_nlive_change_scan_ = vbr_scan_events_;
            auto invalidate_child = [](llama_kv_cache * child) {
                if (child == nullptr) {
                    return;
                }
                for (auto & p : child->vbr_pools_) {
                    p.budget_eff_stamp = ~0ull;
                }
            };
            invalidate_child(this);
            invalidate_child(vbr_ledger_sibling_);
            // Both children must reject their next stable path and adopt the shared census.
            vbr_tree_force();
        }
    }

}

// Census, grant upkeep, demand service, and hygiene: the four phases of the full
// ledger pass, split for readability — vbr_ledger_scan_service composes them.
bool llama_kv_cache::vbr_grants_upkeep(const std::vector<llama_vram_peer_claim> & claims, uint64_t now) {
    // lift on claim-disappearance-with-live-pid / pid-death /
    // heartbeat-stall; amortize surviving demanded-device rows by the claim's bytes_now ----
    bool grants_changed = false;
    for (auto it = vbr_grants_.begin(); it != vbr_grants_.end(); ) {
        auto & g = *it;
        const llama_vram_peer_claim * claim = nullptr;
        bool any_claim_of_owner = false;
        for (const auto & c : claims) {
            if (c.pid == g.pid && c.starttime == g.starttime && c.fields.ver == g.ver) {
                any_claim_of_owner = true;
                if (c.busid == g.busid) {
                    claim = &c;
                }
            }
        }
        bool lift = false;
        if (!llama_vram_ledger_pid_alive(g.pid, g.starttime)) {
            lift = true; // trivial lift on death
        } else if (!any_claim_of_owner) {
            lift = true; // claim-complete or runtime CLEAR: disappearance with live pid
        } else if (claim != nullptr) {
            // heartbeat-stall (flat 3·BEAT rule): the ≤BEAT writer thread makes cadence
            // decode-independent, so a stalled beat means a wedged demander
            if (llama_vram_hb_observe(vbr_claim_obs_, g.busid + "-" + std::to_string(g.pid),
                                       claim->hb_counter, now)
                    > (uint64_t) LLAMA_VRAM_LEDGER_HB_STALL_MS * 1000000ull) {
                lift = true;
            }
            if (!lift && !g.collateral) {
                // clamped amortization: the demander's landed bytes release the decrement
                const uint64_t delta = claim->bytes_now > g.bytes_now_at_grant
                                       ? claim->bytes_now - g.bytes_now_at_grant : 0;
                const uint64_t decr  = delta < g.bytes ? g.bytes - delta : 0;
                if (decr != g.bytes) {
                    g.bytes = decr; // bytes now carries the LIVE decrement
                    g.bytes_now_at_grant = claim->bytes_now;
                    grants_changed = true;
                    if (g.bytes == 0) {
                        lift = true;
                    }
                }
            }
        }
        // collateral rows: delta_i = 0 — decrement holds in full until the lift event
        if (lift) {
            it = vbr_grants_.erase(it);
            grants_changed = true;
        } else {
            ++it;
        }
    }

    return grants_changed;
}

bool llama_kv_cache::vbr_tree_has_grant(const llama_vram_peer_claim & c) const {
    const llama_kv_cache * root = vbr_tree_root();
    auto child_has = [&](const llama_kv_cache * child) {
        if (child == nullptr) {
            return false;
        }
        for (const auto & g : child->vbr_grants_) {
            if (g.pid == c.pid && g.starttime == c.starttime && g.ver == c.fields.ver &&
                g.busid == c.busid && !g.collateral) {
                return true;
            }
        }
        return false;
    };
    return child_has(root) || child_has(root->vbr_ledger_sibling_);
}

bool llama_kv_cache::vbr_tx_settle_tree() {
    llama_kv_cache * root = vbr_tree_root();
    auto settle_child = [](llama_kv_cache * child) {
        if (child == nullptr) {
            return true;
        }
        child->vbr_flush_deferred_unmaps();
        for (auto & p : child->vbr_pools_) {
            if (p.wave_pending) {
                GGML_ASSERT(p.backend != nullptr);
                ggml_backend_synchronize(p.backend);
                p.wave_pending = false;
            }
            if (!p.unmap_deferred.empty()) {
                return false;
            }
        }
        return true;
    };
    const bool root_settled = settle_child(root);
    const bool sibling_settled = settle_child(root->vbr_ledger_sibling_);
    root->vbr_grant_pending_clear();
    if (root->vbr_ledger_sibling_ != nullptr) {
        root->vbr_ledger_sibling_->vbr_grant_pending_clear();
    }
    return root_settled && sibling_settled;
}

bool llama_kv_cache::vbr_tx_reprice(vbr_shed_tx & tx, bool actual) const {
    tx.steps.clear();
    tx.endpoints.clear();
    tx.workspaces.clear();
    tx.stashes.clear();
    tx.scratches.clear();
    tx.devices.clear();

    for (auto & child : tx.children) {
        child.types_after = child.types_before;
        child.final_cursor = child.start_cursor;
    }
    // Selection-coherence rule mirrors the downward projection's apply in
    // vbr_downward_project_policy_prefix — keep the two in sync.
    for (const auto & selected : tx.policy_prefix) {
        if (selected.child_index >= tx.children.size()) {
            return false;
        }
        auto & child = tx.children[selected.child_index];
        const auto & policy = selected.value;
        if (policy.slot >= child.types_after.size() ||
            child.types_after[policy.slot] != (ggml_type) policy.type_a) {
            return false;
        }
        vbr_tx_step step;
        step.child_idx = selected.child_index;
        step.order_idx = policy.order_index;
        step.slot      = policy.slot;
        step.ikv       = policy.slot/2;
        step.is_v      = (policy.slot & 1) != 0;
        step.type_a    = (ggml_type) policy.type_a;
        step.type_b    = (ggml_type) policy.type_b;
        tx.steps.push_back(step);
        child.types_after[step.slot] = step.type_b;
        child.final_cursor = step.order_idx + 1;
    }

    std::map<vbr_tx_extent_key, uint32_t> virtual_stash_valid;

    for (size_t ci = 0; ci < tx.children.size(); ++ci) {
        auto & state = tx.children[ci];
        llama_kv_cache * child = state.cache;
        for (size_t pi = 0; pi < child->vbr_pools_.size(); ++pi) {
            auto & p = child->vbr_pools_[pi];
            if (p.vmm == nullptr) {
                continue;
            }
            const uint32_t terminal_wm = std::max(p.wm_cells, state.incoming_wm);
            for (size_t ikv = 0; ikv < child->layers.size(); ++ikv) {
                for (int side = 0; side < 2; ++side) {
                    auto & e = side ? p.v[ikv] : p.k[ikv];
                    if (e.t == nullptr) {
                        continue;
                    }
                    const vbr_tx_extent_key key { ci, pi, ikv, side != 0 };
                    const size_t slot = ikv*2 + side;
                    const ggml_type type = state.types_after[slot] != GGML_TYPE_COUNT
                            ? state.types_after[slot] : e.t->type;
                    const size_t slot_span = vbr_slot_span(e.t, p.gran);
                    uint64_t final_span_u64 = 0;
                    if (!llama_vbr_physical::endpoint_bytes(
                                ggml_row_size(type, e.t->ne[0]), terminal_wm,
                                slot_span, p.gran, final_span_u64) || final_span_u64 > SIZE_MAX) {
                        return false;
                    }
                    const size_t final_span = (size_t) final_span_u64;

                    auto baseline_total_it = tx.endpoint_baseline_total.find(key);
                    if (baseline_total_it == tx.endpoint_baseline_total.end()) {
                        if (!tx.snapshot_open) {
                            return false;
                        }
                        const uint64_t total = p.be->vmm_pool_mapped_in_range(
                                p.vmm, e.byte_off, slot_span);
                        baseline_total_it = tx.endpoint_baseline_total.emplace(key, total).first;
                    }
                    auto & by_span = tx.endpoint_baseline_inside[key];
                    auto baseline_inside_it = by_span.find(final_span);
                    if (baseline_inside_it == by_span.end()) {
                        if (!tx.snapshot_open) {
                            return false;
                        }
                        const uint64_t inside = p.be->vmm_pool_mapped_in_range(
                                p.vmm, e.byte_off, final_span);
                        baseline_inside_it = by_span.emplace(final_span, inside).first;
                    }
                    const uint64_t baseline_total  = baseline_total_it->second;
                    const uint64_t baseline_inside = baseline_inside_it->second;
                    if (baseline_inside > baseline_total || baseline_inside > final_span_u64) {
                        return false;
                    }

                    uint64_t gross_release = baseline_total - baseline_inside;
                    if (actual) {
                        const uint64_t current_total = p.be->vmm_pool_mapped_in_range(
                                p.vmm, e.byte_off, slot_span);
                        const uint64_t current_inside = p.be->vmm_pool_mapped_in_range(
                                p.vmm, e.byte_off, final_span);
                        if (current_inside != final_span_u64 || current_inside > current_total) {
                            return false;
                        }
                        gross_release = current_total - current_inside;
                    }

                    vbr_tx_endpoint endpoint;
                    endpoint.pool            = &p;
                    endpoint.extent          = &e;
                    endpoint.final_span      = final_span;
                    endpoint.slot_span       = slot_span;
                    endpoint.baseline_total  = baseline_total;
                    endpoint.baseline_inside = baseline_inside;
                    endpoint.gross_release   = gross_release;
                    endpoint.live            = p.wm_cells > 0;
                    tx.endpoints.emplace(key, std::move(endpoint));
                    virtual_stash_valid[key] = e.stash_valid;

                    auto & cost = tx.devices[p.device];
                    if (!llama_vbr_transaction::add_u64(
                                cost.release, baseline_total - baseline_inside) ||
                        !llama_vbr_transaction::add_u64(
                                cost.kv_growth, final_span_u64 - baseline_inside)) {
                        return false;
                    }
                }
            }
        }
    }

    for (const auto & step : tx.steps) {
        llama_kv_cache * child = tx.children[step.child_idx].cache;
        const auto & units = child->vbr_units_of(step.ikv, step.is_v);
        for (auto & [pp, ep] : units) {
            const size_t pi = (size_t) (pp - child->vbr_pools_.data());
            const vbr_tx_extent_key ekey { step.child_idx, pi, step.ikv, step.is_v };
            auto endpoint = tx.endpoints.find(ekey);
            if (endpoint == tx.endpoints.end()) {
                return false;
            }
            endpoint->second.touched = true;
            if (pp->wm_cells == 0) {
                continue;
            }

            const bool capture = child->vbr_stash_rows_ > 0 &&
                    virtual_stash_valid[ekey] == 0 &&
                    ggml_is_turbo_kv_type(step.type_a) &&
                    step.type_a != GGML_TYPE_TURBO8_0;
            const uint32_t capture_rows = capture
                    ? std::min(child->vbr_stash_rows_, pp->wm_cells) : 0;
            const vbr_tx_pool_key pkey { step.child_idx, pi };
            auto & workspace = tx.workspaces[pkey];
            workspace.be      = pp->be;
            workspace.backend = pp->backend;
            workspace.device  = pp->device;
            workspace.requests.push_back({
                    (int64_t) pp->wm_cells, ep->t->ne[0], (int64_t) capture_rows });
            if (capture_rows > 0) {
                auto & stash = tx.stashes[pkey];
                stash.device = pp->device;
                stash.requests.push_back({ ep, capture_rows });
                virtual_stash_valid[ekey] = capture_rows;
            }
        }
    }

    // Once preflight has landed a persistent component, trimming away the step that requested it
    // does not return those bytes. Keep every snapshotted group in the actual price and query its
    // current occupancy even when the candidate prefix now has no request for it.
    for (const auto & [key, baseline] : tx.workspace_baseline) {
        GGML_UNUSED(baseline);
        if (tx.workspaces.count(key) == 0) {
            auto & pool = tx.children[key.child_idx].cache->vbr_pools_[key.pool_idx];
            auto & group = tx.workspaces[key];
            group.be = pool.be;
            group.backend = pool.backend;
            group.device = pool.device;
        }
    }
    for (const auto & [key, baseline] : tx.stash_baseline) {
        GGML_UNUSED(baseline);
        if (tx.stashes.count(key) == 0) {
            auto & pool = tx.children[key.child_idx].cache->vbr_pools_[key.pool_idx];
            auto & group = tx.stashes[key];
            group.device = pool.device;
        }
    }

    for (auto & [key, group] : tx.workspaces) {
        uint64_t physical_now = 0;
        uint64_t physical_if_reserved = 0;
        const auto project = [&](const llama_vbr_transaction::workspace_request & request,
                                 uint64_t & now, uint64_t & projected) {
                    size_t now_st = 0;
                    size_t projected_st = 0;
                    if (!group.be->kv_transcode_workspace_memory(
                                group.backend, group.device,
                                request.n_cells, request.ne0, request.stash_rows,
                                &now_st, &projected_st)) {
                        return false;
                    }
                    now = now_st;
                    projected = projected_st;
                    return true;
                };
        bool ok = true;
        if (group.requests.empty()) {
            // Zero work is the backend's read-only occupancy query.
            ok = project({ 0, 1, 0 }, physical_now, physical_if_reserved);
        } else {
            ok = llama_vbr_transaction::workspace_endpoint(
                    group.requests, project, physical_now, physical_if_reserved);
        }
        if (!ok) {
            return false;
        }
        auto baseline = tx.workspace_baseline.find(key);
        if (baseline == tx.workspace_baseline.end()) {
            if (!tx.snapshot_open) {
                return false;
            }
            baseline = tx.workspace_baseline.emplace(key, physical_now).first;
        }
        const uint64_t endpoint = actual ? physical_now : physical_if_reserved;
        if (endpoint < baseline->second) {
            return false;
        }
        if (!llama_vbr_transaction::add_u64(
                    tx.devices[group.device].workspace_growth, endpoint - baseline->second)) {
            return false;
        }
    }

    for (auto & [key, group] : tx.stashes) {
        auto & child = tx.children[key.child_idx];
        auto & pool = child.cache->vbr_pools_[key.pool_idx];
        size_t physical_now = 0;
        size_t physical_if_reserved = 0;
        if (!child.cache->vbr_stash_memory(
                    pool, group.requests, physical_now, physical_if_reserved)) {
            return false;
        }
        auto baseline = tx.stash_baseline.find(key);
        if (baseline == tx.stash_baseline.end()) {
            if (!tx.snapshot_open) {
                return false;
            }
            baseline = tx.stash_baseline.emplace(key, physical_now).first;
        }
        const uint64_t endpoint = actual ? physical_now : physical_if_reserved;
        if (endpoint < baseline->second) {
            return false;
        }
        if (!llama_vbr_transaction::add_u64(
                    tx.devices[group.device].stash_growth, endpoint - baseline->second)) {
            return false;
        }
    }

    auto query_scratch = [&](vbr_tx_scratch_group & group) {
        size_t physical_now = 0;
        size_t physical_if_reserved = 0;
        group.be->kv_dequant_scratch_memory(
                group.backend, group.k_need, group.v_need,
                &physical_now, &physical_if_reserved);
        auto baseline = tx.scratch_baseline.find(group.backend);
        if (baseline == tx.scratch_baseline.end()) {
            if (!tx.snapshot_open) {
                return false;
            }
            baseline = tx.scratch_baseline.emplace(group.backend, physical_now).first;
        }
        const uint64_t endpoint = actual ? physical_now : physical_if_reserved;
        if (endpoint < baseline->second) {
            return false;
        }
        group.physical_projected = endpoint;
        return true;
    };

    for (size_t ci = 0; ci < tx.children.size(); ++ci) {
        auto & state = tx.children[ci];
        llama_kv_cache * child = state.cache;
        for (auto & p : child->vbr_pools_) {
            if (p.be == nullptr || p.compute_backend == nullptr || p.device < 0) {
                continue;
            }
            size_t k_row = 0;
            size_t v_row = 0;
            for (size_t ikv = 0; ikv < child->layers.size(); ++ikv) {
                const ggml_tensor * tk = p.k[ikv].t;
                const ggml_tensor * tv = p.v[ikv].t;
                const ggml_type type_k = state.types_after[ikv*2] != GGML_TYPE_COUNT
                        ? state.types_after[ikv*2] : (tk ? tk->type : GGML_TYPE_F16);
                const ggml_type type_v = state.types_after[ikv*2 + 1] != GGML_TYPE_COUNT
                        ? state.types_after[ikv*2 + 1] : (tv ? tv->type : GGML_TYPE_F16);
                bool need_k = false;
                bool need_v = false;
                ggml_vbr_kv_dequant_sides(type_k, type_v, &need_k, &need_v);
                if (need_k && tk != nullptr) {
                    k_row = std::max(k_row, ggml_row_size(GGML_TYPE_F16, tk->ne[0]));
                }
                if (need_v && tv != nullptr) {
                    v_row = std::max(v_row, ggml_row_size(GGML_TYPE_F16, tv->ne[0]));
                }
            }
            const uint32_t terminal_wm = std::max(p.wm_cells, state.incoming_wm);
            if ((terminal_wm != 0 && k_row > SIZE_MAX/(size_t) terminal_wm) ||
                (terminal_wm != 0 && v_row > SIZE_MAX/(size_t) terminal_wm)) {
                return false;
            }
            auto & group = tx.scratches[p.compute_backend];
            if (group.backend == nullptr) {
                group.be = p.be;
                group.backend = p.compute_backend;
                group.device = p.device;
            } else if (group.be != p.be || group.device != p.device) {
                return false;
            }
            group.k_need = std::max(group.k_need, k_row*(size_t) terminal_wm);
            group.v_need = std::max(group.v_need, v_row*(size_t) terminal_wm);
            group.owner_backend = true;
        }
    }

    for (auto & [backend, group] : tx.scratches) {
        GGML_UNUSED(backend);
        if (!query_scratch(group)) {
            return false;
        }
    }

    for (size_t ci = 0; ci < tx.children.size(); ++ci) {
        auto & state = tx.children[ci];
        llama_kv_cache * child = state.cache;
        vbr_device_watermarks watermarks;
        for (const auto & p : child->vbr_pools_) {
            if (p.device >= 0) {
                auto & wm = watermarks[p.device];
                wm = std::max(wm, std::max(p.wm_cells, state.incoming_wm));
            }
        }
        bool reverse_ok = true;
        child->vbr_shared_scratch_visit(
                state.types_after, watermarks,
                [&](const vbr_shared_scratch_plan & plan) {
                    auto & group = tx.scratches[plan.compute_backend];
                    if (group.backend == nullptr) {
                        group.be = plan.be;
                        group.backend = plan.compute_backend;
                        group.device = plan.device;
                    } else if (group.be != plan.be || group.device != plan.device) {
                        reverse_ok = false;
                        return;
                    }
                    group.k_need = std::max(group.k_need, plan.k_bytes);
                    group.v_need = std::max(group.v_need, plan.v_bytes);
                    if (!query_scratch(group)) {
                        reverse_ok = false;
                    }
                });
        if (!reverse_ok) {
            return false;
        }
    }

    for (const auto & [backend, group] : tx.scratches) {
        GGML_UNUSED(backend);
        const auto baseline = tx.scratch_baseline.find(group.backend);
        if (baseline == tx.scratch_baseline.end() || group.physical_projected < baseline->second ||
            !llama_vbr_transaction::add_u64(
                    tx.devices[group.device].scratch_growth,
                    group.physical_projected - baseline->second)) {
            return false;
        }
    }

    for (auto & [device, cost] : tx.devices) {
        GGML_UNUSED(device);
        if (!llama_vbr_transaction::finalize(cost)) {
            return false;
        }
    }
    return true;
}

bool llama_kv_cache::vbr_tx_map_endpoints(vbr_shed_tx & tx) {
    for (auto & [key, endpoint] : tx.endpoints) {
        GGML_UNUSED(key);
        if (endpoint.final_span > 0 &&
            !endpoint.pool->be->vmm_pool_map(
                    endpoint.pool->vmm, endpoint.extent->byte_off, endpoint.final_span)) {
            return false;
        }
    }
    return true;
}

bool llama_kv_cache::vbr_tx_preflight(vbr_shed_tx & tx) {
    tx.snapshot_open = false;
    for (auto & [key, group] : tx.workspaces) {
        auto & pool = tx.children[key.child_idx].cache->vbr_pools_[key.pool_idx];
        if (pool.backend == nullptr) {
            pool.backend = pool.be->backend_init(pool.device);
            if (pool.backend == nullptr) {
                return false;
            }
        }
        group.backend = pool.backend;
    }
    for (auto & [key, group] : tx.workspaces) {
        GGML_UNUSED(key);
        for (const auto & request : group.requests) {
            if (!group.be->kv_transcode_workspace_reserve(
                        group.backend, request.n_cells, request.ne0, request.stash_rows)) {
                return false;
            }
        }
    }
    for (auto & [key, group] : tx.stashes) {
        auto & child = tx.children[key.child_idx];
        auto & pool = child.cache->vbr_pools_[key.pool_idx];
        if (!child.cache->vbr_stash_reserve(pool, group.requests)) {
            return false;
        }
    }

    for (auto & [backend, group] : tx.scratches) {
        GGML_UNUSED(backend);
        if (group.owner_backend &&
            !group.be->kv_dequant_scratch_reserve(
                    group.backend, group.k_need, group.v_need)) {
            return false;
        }
    }
    for (auto & state : tx.children) {
        vbr_device_watermarks watermarks;
        for (const auto & p : state.cache->vbr_pools_) {
            if (p.device >= 0) {
                auto & wm = watermarks[p.device];
                wm = std::max(wm, std::max(p.wm_cells, state.incoming_wm));
            }
        }
        bool reserved = true;
        state.cache->vbr_shared_scratch_visit(
                state.types_after, watermarks,
                [&](const vbr_shared_scratch_plan & plan) {
                    if (reserved && !plan.be->kv_dequant_scratch_reserve(
                                plan.compute_backend, plan.k_bytes, plan.v_bytes)) {
                        reserved = false;
                    }
                });
        if (!reserved) {
            return false;
        }
    }
    if (!vbr_tx_map_endpoints(tx)) {
        return false;
    }
    return true;
}

bool llama_kv_cache::vbr_tx_hard_seal_allowed(vbr_shed_tx & tx) {
    std::map<size_t, vbr_hard_seal_consult_session> sessions;
    for (const auto & step : tx.steps) {
        llama_kv_cache * child = tx.children[step.child_idx].cache;
        if (!child->vbr_hard_seal_guard_) {
            continue;
        }
        if (child->vbr_hard_seal_step_blocked(
                step.order_idx, sessions[step.child_idx])) {
            child->vbr_hard_seal_blocked_ = true;
            child->vbr_hard_seal_evidence_record(step.order_idx);
            return false;
        }
    }
    return true;
}

bool llama_kv_cache::vbr_tx_capture_leases_allowed(vbr_shed_tx & tx) {
    tx.unit_guards.clear();
    try {
        std::vector<std::pair<llama_kv_cache *, uint32_t>> units;
        units.reserve(tx.steps.size());
        for (const auto & step : tx.steps) {
            if (step.child_idx >= tx.children.size()) {
                return false;
            }
            auto * child = tx.children[step.child_idx].cache;
            if (child == nullptr || step.ikv > (UINT32_MAX - 1)/2) {
                return false;
            }
            units.push_back({
                child,
                static_cast<uint32_t>(step.ikv*2 + (step.is_v ? 1 : 0)),
            });
        }
        std::sort(units.begin(), units.end(), [](const auto & lhs, const auto & rhs) {
            if (lhs.first != rhs.first) {
                return std::less<llama_kv_cache *>{}(lhs.first, rhs.first);
            }
            return lhs.second < rhs.second;
        });
        units.erase(std::unique(units.begin(), units.end()), units.end());
        tx.unit_guards.reserve(units.size());
        for (const auto & [cache, logical_unit] : units) {
            tx.unit_guards.emplace_back(cache, logical_unit);
            if (!tx.unit_guards.back()) {
                tx.unit_guards.clear();
                return false;
            }
        }
        return true;
    } catch (...) {
        tx.unit_guards.clear();
        return false;
    }
}

bool llama_kv_cache::vbr_tx_prepare_commit(
        vbr_shed_tx & tx, const llama_vram_peer_claim & c) {
    tx.gross_by_pool.clear();
    tx.deferred_by_pool.clear();
    tx.planned_grants.clear();
    tx.planned_grants.reserve(tx.endpoints.size());
    std::map<vbr_tx_pool_key, size_t> tail_counts;

    for (const auto & [key, endpoint] : tx.endpoints) {
        const vbr_tx_pool_key pkey { key.child_idx, key.pool_idx };
        const uint64_t baseline_release = endpoint.baseline_total - endpoint.baseline_inside;
        if (!llama_vbr_transaction::add_u64(tx.gross_by_pool[pkey], baseline_release)) {
            return false;
        }
        if (endpoint.gross_release > 0 && endpoint.final_span < endpoint.slot_span &&
            endpoint.touched && endpoint.live) {
            tail_counts[pkey]++;
            if (!llama_vbr_transaction::add_u64(
                        tx.deferred_by_pool[pkey], endpoint.gross_release)) {
                return false;
            }
        }
    }
    for (const auto & [pkey, count] : tail_counts) {
        auto & pool = tx.children[pkey.child_idx].cache->vbr_pools_[pkey.pool_idx];
        if (count > SIZE_MAX - pool.unmap_deferred.size()) {
            return false;
        }
        pool.unmap_deferred.reserve(pool.unmap_deferred.size() + count);
    }

    uint64_t prior_credit = 0;
    for (const auto & child : tx.children) {
        for (const auto & row : child.cache->vbr_grants_) {
            if (!row.collateral && row.busid == c.busid && row.pid == c.pid &&
                row.starttime == c.starttime && row.ver == c.fields.ver &&
                !llama_vbr_transaction::add_u64(prior_credit, row.bytes)) {
                return false;
            }
        }
    }

    std::vector<size_t> grant_counts(tx.children.size(), 0);
    for (const auto & [device, cost] : tx.devices) {
        if (cost.capacity_signed < 0) {
            return false;
        }
        uint64_t credit = (uint64_t) cost.capacity_signed;
        if (device == tx.demanded_device) {
            credit = std::min(credit, tx.target);
        }
        if (credit == 0) {
            continue;
        }

        uint64_t gross_device = 0;
        std::vector<std::pair<vbr_tx_pool_key, uint64_t>> releasing;
        releasing.reserve(tx.gross_by_pool.size());
        for (const auto & [pkey, gross] : tx.gross_by_pool) {
            auto & pool = tx.children[pkey.child_idx].cache->vbr_pools_[pkey.pool_idx];
            if (pool.device == device && gross > 0) {
                releasing.push_back({ pkey, gross });
                if (!llama_vbr_transaction::add_u64(gross_device, gross)) {
                    return false;
                }
            }
        }
        if (gross_device == 0) {
            return false;
        }

        uint64_t assigned = 0;
        for (size_t i = 0; i < releasing.size(); ++i) {
            const auto [pkey, gross] = releasing[i];
            uint64_t share = 0;
            if (i + 1 == releasing.size()) {
                share = credit - assigned;
            } else {
#ifdef _MSC_VER
                share = (uint64_t) ((double) credit * gross / gross_device);
#else
                share = (uint64_t) ((__uint128_t) credit * gross / gross_device);
#endif
            }
            assigned += share;
            if (share == 0) {
                continue;
            }
            llama_kv_cache * child = tx.children[pkey.child_idx].cache;
            auto & pool = child->vbr_pools_[pkey.pool_idx];
            vbr_tx_grant_plan plan;
            plan.child_idx = pkey.child_idx;
            plan.row.busid              = c.busid;
            plan.row.pid                = c.pid;
            plan.row.starttime          = c.starttime;
            plan.row.ver                = c.fields.ver;
            plan.row.pool_idx           = pkey.pool_idx;
            plan.row.bytes              = share;
            plan.row.collateral         = child->vbr_pool_busid(pool) != c.busid;
            plan.row.bytes_now_at_grant = c.bytes_now;
            if (!plan.row.collateral) {
                if (!llama_vbr_transaction::grant_threshold(
                            c.bytes_now, prior_credit, plan.row.bytes_now_at_grant) ||
                    !llama_vbr_transaction::add_u64(prior_credit, share)) {
                    return false;
                }
            }
            tx.planned_grants.push_back(std::move(plan));
            grant_counts[pkey.child_idx]++;
        }
        if (assigned != credit) {
            return false;
        }
    }
    for (size_t ci = 0; ci < tx.children.size(); ++ci) {
        auto & grants = tx.children[ci].cache->vbr_grants_;
        if (grant_counts[ci] > SIZE_MAX - grants.size()) {
            return false;
        }
        grants.reserve(grants.size() + grant_counts[ci]);
    }
    for (const auto & [pkey, pending] : tx.deferred_by_pool) {
        llama_kv_cache * child = tx.children[pkey.child_idx].cache;
        auto & pool = child->vbr_pools_[pkey.pool_idx];
        uint64_t & current = child->vbr_grant_pending_[child->vbr_pool_busid(pool)];
        if (pending > UINT64_MAX - current) {
            return false;
        }
    }
    return true;
}

void llama_kv_cache::vbr_tx_suppress(const std::string & busid) {
    llama_kv_cache * root = vbr_tree_root();
    constexpr uint64_t retry_ns = 1000000000ull;
    root->vbr_offer_suppressed_until_[busid] = llama_vram_ledger_now_ns() + retry_ns;
    auto invalidate = [](llama_kv_cache * child) {
        if (child == nullptr) {
            return;
        }
        for (auto & p : child->vbr_pools_) {
            p.budget_eff_stamp = ~0ull;
        }
    };
    invalidate(root);
    invalidate(root->vbr_ledger_sibling_);
    root->vbr_tree_force();
}

bool llama_kv_cache::vbr_tx_publish_zero_intent(const vbr_shed_tx & tx) {
    llama_kv_cache * root = vbr_tree_root();
    std::set<std::string> busids;
    for (const auto & [device, cost] : tx.devices) {
        GGML_UNUSED(cost);
        for (auto & child : tx.children) {
            for (auto & p : child.cache->vbr_pools_) {
                if (p.vmm != nullptr && p.device == device &&
                    child.cache->vbr_pool_busid(p) != "-") {
                    busids.insert(p.busid);
                }
            }
        }
    }
    for (const auto & busid : busids) {
        uint64_t pending = root->vbr_grant_pending_[busid];
        if (root->vbr_ledger_sibling_ != nullptr) {
            if (UINT64_MAX - pending < root->vbr_ledger_sibling_->vbr_grant_pending_[busid]) {
                return false;
            }
            pending += root->vbr_ledger_sibling_->vbr_grant_pending_[busid];
        }
        llama_vram_marker_fields f = {};
        f.vbr = 1;
        f.serviced = llama_vram_marker_serviced_flag() ? 1u : 0u;
        f.shed_available = 0;
        // Withdraw only the new offer.  A prior committed wave may still own a bridge until its
        // deferred tails flush; replacing that with zero would let another donor under-shed.
        f.grant_pending = pending;
        uint64_t created_ts = 0;
        if (!llama_vram_marker_publish(busid, f, &created_ts)) {
            // The substrate may have completed the rename and then failed to reopen the new path.
            // Forget every affected cache entry so the next pass republishes instead of beating a
            // stale/unlinked descriptor; earlier successful zero-intent writes remain safe.
            for (const auto & affected : busids) {
                root->vbr_marker_pub_.erase(affected);
                root->vbr_tx_suppress(affected);
            }
            return false;
        }
        root->vbr_marker_pub_[busid] = { 0, pending };
        root->vbr_marker_created_ts_[busid] = created_ts;
        root->vbr_ledger_mtime_ = llama_vram_ledger_dir_mtime_ns();
    }
    return true;
}

void llama_kv_cache::vbr_tx_apply(vbr_shed_tx & tx, vbr_operation_id operation_id) {
    for (const auto & state : tx.children) {
        GGML_ASSERT(state.cache->vbr_degrade_cursor_ == state.start_cursor);
        GGML_ASSERT(state.cache->vbr_tier_epoch_ == state.start_epoch);
        for (size_t ikv = 0; ikv < state.cache->layers.size(); ++ikv) {
            for (int side = 0; side < 2; ++side) {
                const ggml_type expected = state.types_before[ikv*2 + side];
                const ggml_tensor * tensor = side
                        ? state.cache->layers[ikv].v : state.cache->layers[ikv].k;
                if (expected != GGML_TYPE_COUNT) {
                    GGML_ASSERT(tensor != nullptr && tensor->type == expected);
                }
            }
        }
    }
    for (const auto & step : tx.steps) {
        auto & child_state = tx.children[step.child_idx];
        llama_kv_cache * child = child_state.cache;
        GGML_ASSERT(step.ikv < child->layers.size());
        ggml_tensor * canonical = step.is_v
                ? child->layers[step.ikv].v : child->layers[step.ikv].k;
        GGML_ASSERT(canonical != nullptr && canonical->type == step.type_a);

        const auto & units = child->vbr_units_of(step.ikv, step.is_v);
        for (auto & [pp, ep] : units) {
            const size_t pi = (size_t) (pp - child->vbr_pools_.data());
            const vbr_tx_extent_key key { step.child_idx, pi, step.ikv, step.is_v };
            const auto endpoint = tx.endpoints.find(key);
            GGML_ASSERT(endpoint != tx.endpoints.end());
            GGML_ASSERT(pp->be->vmm_pool_mapped_in_range(
                    pp->vmm, ep->byte_off, endpoint->second.final_span) ==
                    endpoint->second.final_span);

            const int64_t n_cells = pp->wm_cells;
            if (n_cells <= 0) {
                continue;
            }
            GGML_ASSERT(pp->backend != nullptr);
            if (!pp->wave_pending) {
                pp->be->sync_device(pp->device);
            }

            const bool capture = child->vbr_stash_rows_ > 0 && ep->stash_valid == 0 &&
                    ggml_is_turbo_kv_type(step.type_a) &&
                    step.type_a != GGML_TYPE_TURBO8_0;
            const uint32_t capture_rows = capture
                    ? std::min(child->vbr_stash_rows_, pp->wm_cells) : 0;
            const void * stash_ptr = nullptr;
            int64_t stash_rows = 0;
            if (capture_rows > 0) {
                GGML_ASSERT(pp->stash_vmm != nullptr);
                char * base = (char *) pp->be->vmm_pool_base(pp->stash_vmm);
                pp->be->kv_stash_capture(
                        pp->backend, ep->t, base + ep->stash_off,
                        capture_rows, step.is_v);
                ep->stash_valid = capture_rows;
            }
            if (ep->stash_valid > 0 && pp->stash_vmm != nullptr) {
                stash_ptr = (char *) pp->be->vmm_pool_base(pp->stash_vmm) + ep->stash_off;
                stash_rows = ep->stash_valid;
            }

            const vbr_span span = vbr_span_of(
                    ep->t, step.type_b, n_cells,
                    std::max(pp->wm_cells, child_state.incoming_wm), pp->gran);
            vbr_scrub_span scrub;
            GGML_ASSERT(vbr_scrub_span_of(
                ep->t, step.type_a, n_cells, span, pp->gran, scrub));
            const ggml_vbr_transcode_params params = {
                /* .src         =*/ ep->t,
                /* .type_B      =*/ step.type_b,
                /* .dst         =*/ ep->t->data,
                /* .pool_buf    =*/ pp->buf,
                /* .n_cells     =*/ n_cells,
                /* .is_v        =*/ step.is_v,
                /* .stash_f16   =*/ stash_ptr,
                /* .stash_rows  =*/ stash_rows,
                /* .scrub_bytes =*/ scrub.scrub_end - scrub.keep,
            };
            pp->be->kv_transcode(pp->backend, &params);
            pp->wave_pending = true;
        }

        vbr_set_tensor_type_noalloc(
                canonical,
                step.is_v ? child->layers[step.ikv].v_stream
                          : child->layers[step.ikv].k_stream,
                step.type_b);
        child->vbr_degrade_cursor_ = step.order_idx + 1;
        child->vbr_tier_epoch_++;
        // Tree sheds bypass vbr_degrade_next(), so publish the same representation transition
        // here at the atomic tree commit point. The root operation manifest covers every armed
        // child instance and this exact id authenticates each child's unit publication.
        child->vbr_representation_changed();
        if (auto * tracker = child->vbr_generation_tracker_mut()) {
            const uint8_t promote_hops = units.empty() ? 0 : units.front().second->promote_hops;
            const auto transition = step.type_a == GGML_TYPE_F16 &&
                                            step.type_b == GGML_TYPE_TURBO8_0
                    ? vbr_repr_transition::degrade_f16_to_t8_admitted
                    : vbr_repr_transition::degrade_other;
            GGML_ASSERT(tracker->publish_unit(
                    static_cast<uint32_t>(step.ikv * 2 + (step.is_v ? 1 : 0)),
                    static_cast<int32_t>(step.type_a),
                    static_cast<int32_t>(step.type_b),
                    step.type_b == GGML_TYPE_F16 || step.type_b == GGML_TYPE_TURBO8_0
                            ? vbr_repr_domain::full
                            : vbr_repr_domain::tapped,
                    promote_hops,
                    transition,
                    vbr_mutation_registrant::degrade_next,
                    operation_id));
        }
        child->vbr_quiet_boundaries_ = 0;
    }

    for (auto & [key, endpoint] : tx.endpoints) {
        GGML_UNUSED(key);
        if (endpoint.gross_release == 0 || endpoint.final_span >= endpoint.slot_span) {
            continue;
        }
        const size_t off = endpoint.extent->byte_off + endpoint.final_span;
        const size_t len = endpoint.slot_span - endpoint.final_span;
        if (endpoint.touched && endpoint.live) {
            endpoint.pool->unmap_deferred.push_back({ off, len });
        } else {
            endpoint.pool->be->vmm_pool_unmap(endpoint.pool->vmm, off, len);
        }
    }

    for (auto & state : tx.children) {
        llama_kv_cache * child = state.cache;
        vbr_hard_seal_defer_jumped_steps(
            child->vbr_hard_seal_deferred_,
            state.sealed_deferred,
            state.final_cursor);
        vbr_hard_seal_defer_jumped_steps(
            child->vbr_capture_retier_deferred_,
            state.capture_deferred,
            state.final_cursor);
        if (!state.capture_deferred.empty()) {
            child->vbr_capture_retier_attempt_boundary_ =
                child->vbr_boundary_count_;
            for (const size_t ordinal : state.capture_deferred) {
                if (ordinal < child->vbr_capture_retier_attempted_.size()) {
                    child->vbr_capture_retier_attempted_[ordinal] = 1;
                }
            }
        }
        child->vbr_degrade_cursor_ = state.final_cursor;
        for (auto & p : child->vbr_pools_) {
            if (p.vmm != nullptr) {
                child->vbr_capture_watermark_publish(
                    p, std::max(p.wm_cells, state.incoming_wm));
                p.budget_eff_stamp = ~0ull;
                p.scratch_rows_epoch = ~0ull;
            }
        }
    }

    for (auto & plan : tx.planned_grants) {
        tx.children[plan.child_idx].cache->vbr_grants_.push_back(std::move(plan.row));
    }

    for (const auto & [pkey, pending] : tx.deferred_by_pool) {
        if (pending == 0) {
            continue;
        }
        llama_kv_cache * child = tx.children[pkey.child_idx].cache;
        auto & pool = child->vbr_pools_[pkey.pool_idx];
        uint64_t & current = child->vbr_grant_pending_[child->vbr_pool_busid(pool)];
        GGML_ASSERT(pending <= UINT64_MAX - current);
        current += pending;
    }
    for (auto & state : tx.children) {
        state.cache->vbr_apply_grant_decrements();
    }
}

llama_kv_cache::vbr_tx_result llama_kv_cache::vbr_execute_tree_shed(
        const llama_vram_peer_claim & c, uint64_t target, uint32_t n_tokens) {
    vbr_tx_result result;
    if (target == 0) {
        return result;
    }
    llama_kv_cache * root = vbr_tree_root();
    const bool root_frozen = root->vbr_retier_defer("peer_tree_shed");
    const bool sibling_frozen = root->vbr_ledger_sibling_ != nullptr &&
            root->vbr_ledger_sibling_->vbr_retier_defer("peer_tree_shed");
    if (root_frozen || sibling_frozen) {
        result.status = vbr_tx_status::retryable_no_tier_mutation;
        return result;
    }
    const auto demanded = root->vbr_tree_find_pool(c.busid);
    if (demanded.pool == nullptr) {
        return result;
    }
    if (!root->vbr_tx_settle_tree()) {
        root->vbr_tx_suppress(c.busid);
        result.status = vbr_tx_status::retryable_no_tier_mutation;
        return result;
    }

    vbr_shed_tx tx;
    tx.demanded_device = demanded.pool->device;
    tx.target = target;
    auto append_child = [&](llama_kv_cache * child) {
        if (child == nullptr || !child->vbr_vmm_active()) {
            return;
        }
        vbr_tx_child state;
        state.cache = child;
        state.incoming_wm = child->vbr_watermark_cells(n_tokens);
        state.start_cursor = child->vbr_degrade_cursor_;
        state.final_cursor = state.start_cursor;
        state.start_epoch = child->vbr_tier_epoch_;
        child->vbr_sim_seed(state.types_before, /* pooled_only = */ true,
                GGML_TYPE_COUNT, GGML_TYPE_COUNT, nullptr, nullptr, nullptr);
        state.types_after = state.types_before;
        tx.children.push_back(std::move(state));
    };
    append_child(root);
    append_child(root->vbr_ledger_sibling_);
    if (tx.children.empty()) {
        return result;
    }

    std::vector<llama_vbr_policy::child> policy_children;
    policy_children.reserve(tx.children.size());
    for (size_t i = 0; i < tx.children.size(); ++i) {
        auto policy = tx.children[i].cache->vbr_policy_child_stream(
            tx.demanded_device, tx.children[i].incoming_wm);
        tx.children[i].sealed_deferred = policy.blocked_order_indices;
        tx.children[i].capture_deferred =
            policy.capture_blocked_order_indices;
        policy_children.push_back(std::move(policy));
    }

    if (!root->vbr_tx_reprice(tx, /* actual = */ false)) {
        root->vbr_tx_suppress(c.busid);
        result.status = vbr_tx_status::retryable_no_tier_mutation;
        return result;
    }
    bool planned = llama_vbr_transaction::prefix_feasible(
            tx.devices, tx.demanded_device, tx.target);
    if (!planned) {
        llama_vbr_policy::shortest_prefix_stream stream(std::move(policy_children));
        std::vector<llama_vbr_policy::selection> prefix;
        const auto selected = stream.shortest_prefix(
                [&](const std::vector<llama_vbr_policy::selection> & candidate) {
                    tx.policy_prefix = candidate;
                    if (!root->vbr_tx_reprice(tx, /* actual = */ false)) {
                        return false;
                    }
                    return llama_vbr_transaction::prefix_feasible(
                            tx.devices, tx.demanded_device, tx.target);
                }, prefix);
        if (selected != llama_vbr_policy::result::selected) {
            return result;
        }
        tx.policy_prefix = std::move(prefix);
        planned = true;
    }
    GGML_ASSERT(planned);

    if (!root->vbr_tx_hard_seal_allowed(tx) ||
        !root->vbr_tx_capture_leases_allowed(tx)) {
        result.status = vbr_tx_status::retryable_no_tier_mutation;
        return result;
    }
    if (!root->vbr_tx_preflight(tx) ||
        !root->vbr_tx_reprice(tx, /* actual = */ true) ||
        !llama_vbr_transaction::prefix_feasible(
                tx.devices, tx.demanded_device, tx.target)) {
        root->vbr_tx_suppress(c.busid);
        result.status = vbr_tx_status::retryable_no_tier_mutation;
        return result;
    }

    while (!tx.policy_prefix.empty()) {
        const auto removed = tx.policy_prefix.back();
        tx.policy_prefix.pop_back();
        if (!root->vbr_tx_reprice(tx, /* actual = */ false)) {
            tx.policy_prefix.push_back(removed);
            GGML_ASSERT(root->vbr_tx_reprice(tx, /* actual = */ true));
            break;
        }
        if (!root->vbr_tx_map_endpoints(tx)) {
            tx.policy_prefix.push_back(removed);
            if (!root->vbr_tx_reprice(tx, /* actual = */ true) ||
                !llama_vbr_transaction::prefix_feasible(
                        tx.devices, tx.demanded_device, tx.target)) {
                root->vbr_tx_suppress(c.busid);
                result.status = vbr_tx_status::retryable_no_tier_mutation;
                return result;
            }
            break;
        }
        if (!root->vbr_tx_reprice(tx, /* actual = */ true) ||
            !llama_vbr_transaction::prefix_feasible(
                    tx.devices, tx.demanded_device, tx.target)) {
            tx.policy_prefix.push_back(removed);
            if (!root->vbr_tx_reprice(tx, /* actual = */ true) ||
                !llama_vbr_transaction::prefix_feasible(
                        tx.devices, tx.demanded_device, tx.target)) {
                root->vbr_tx_suppress(c.busid);
                result.status = vbr_tx_status::retryable_no_tier_mutation;
                return result;
            }
            break;
        }
    }

    // One operation spans the atomic tree transaction. Its immutable manifest names every
    // armed child instance, so the no-fail apply below can cite one id for all unit publications.
    vbr_operation_binding shed_binding;
    shed_binding.kind        = vbr_operation_kind::controller_retier;
    shed_binding.child_phase = vbr_operation_phase::mutate;
    for (const auto & state : tx.children) {
        const vbr_controller_instance_id instance = state.cache->vbr_instance_id();
        if (vbr_controller_instance_id_is_set(instance)) {
            GGML_ASSERT(vbr_binding_add_instance_target(
                    shed_binding,
                    vbr_operation_kind::controller_retier,
                    vbr_operation_class::controller,
                    instance, VBR_STREAM_ANY, -1, -1, -1));
        }
    }
    vbr_mutation_op shed_op(root, shed_binding, /* provenance_bearing = */ false);
    const vbr_mutation_op::success_on_return shed_ok(shed_op);
    if (shed_binding.n_targets > 0 && !shed_op.active()) {
        // The root's refused scope already latched its own tracker unavailable. Propagate the
        // same one-id refusal to every other armed child before the legacy tree mutation proceeds;
        // otherwise a sibling could retain apparently valid evidence for an uncited tier flip.
        for (const auto & state : tx.children) {
            if (state.cache == root || state.cache->vbr_generation_tracker_get() == nullptr) {
                continue;
            }
            state.cache->vbr_adopt_refused();
            {
                vbr_mutation_op refused_child(
                        state.cache,
                        vbr_operation_kind::controller_retier,
                        vbr_operation_class::controller,
                        -1, -1, -1);
            }
            state.cache->vbr_release_adopted();
        }
    }

    if (!root->vbr_tx_prepare_commit(tx, c) ||
        !root->vbr_tx_publish_zero_intent(tx)) {
        root->vbr_tx_suppress(c.busid);
        result.status = vbr_tx_status::retryable_no_tier_mutation;
        return result;
    }
    const auto demanded_cost = tx.devices.find(tx.demanded_device);
    GGML_ASSERT(demanded_cost != tx.devices.end() && demanded_cost->second.capacity_signed >= 0);
    result.credited = std::min<uint64_t>(
            tx.target, (uint64_t) demanded_cost->second.capacity_signed);
    root->vbr_tx_apply(tx, vbr_cited_op());
    root->vbr_offer_suppressed_until_.erase(c.busid);
    result.status = vbr_tx_status::committed;
    return result;
}

bool llama_kv_cache::vbr_service_demands(const std::vector<llama_vram_peer_claim> & claims,
                                         const std::vector<llama_vram_peer_marker> & peers,
                                         const std::set<std::string> & announced,
                                         uint64_t now, uint32_t n_tokens) {
    bool grants_changed = false;
    // ---- demand service: rank-0 shed sizing ----
    // one band per donor per session-generation is enforced by the band cursor itself
    // (monotone: once spent, shed_available stays 0 until vbr_full_reset)
    // idleness for runtime-demand donation: decode-based, evaluated here (a boundary
    // caller has just stamped last_prepare, so it is never idle — active-vs-active
    // residents self-serve via their own ladders; only the tick path can qualify)
    const bool donor_idle = vbr_last_prepare_ns_ != 0 &&
        now - vbr_last_prepare_ns_ >= (uint64_t) LLAMA_VRAM_LEDGER_IDLE_MS * 1000000ull;
    for (const auto & c : claims) {
        LLAMA_LOG_DEBUG("%s: claim pid %d phase %u est %.1f MiB (donor_idle %d)\n",
                __func__, c.pid, (unsigned) c.fields.phase,
                c.fields.bytes_total_remaining_est/1048576.0, (int) donor_idle);
        if (c.fields.phase != LLAMA_VRAM_CLAIM_DEMAND &&
            !(c.fields.phase == LLAMA_VRAM_CLAIM_RUNTIME && donor_idle)) {
            continue;
        }
        // already granted to this (pid, starttime, ver) on this device? one decision, one shed
        if (vbr_tree_has_grant(c) || vbr_tree_budget_explicit()) {
            continue;
        }
        // does the demand name one of our devices, and do we have an offer there?
        const auto demanded = vbr_tree_find_pool(c.busid);
        if (demanded.pool == nullptr) {
            continue;
        }
        vbr_pool * demanded_pool = demanded.pool;
        llama_kv_cache * root = vbr_tree_root();
        const size_t our_offer = vbr_child_offer(root, c.busid);
        const size_t sib_offer = vbr_child_offer(root->vbr_ledger_sibling_, c.busid);
        if (our_offer + sib_offer == 0) {
            continue;
        }
        const auto own_ts = vbr_marker_created_ts_.find(c.busid);
        if (announced.count(c.busid) != 0 || own_ts == vbr_marker_created_ts_.end() || own_ts->second == 0) {
            // A newly announced donor waits one scan so every contender ranks against a
            // pre-existing offer set. Failed/unknown publication can never imply rank zero.
            continue;
        }
        const uint64_t self_created_ts = own_ts->second;
        // rank-0 among FRESH offering markers on the demanded device (created_ts, pid) —
        // both sides use the marker registry's preserved first-publish timestamp
        bool rank0 = true;
        for (const auto & m : peers) {
            if (m.busid != c.busid || m.fields.shed_available == 0) {
                continue;
            }
            // freshness for donor selection instantiates with LONG/2
            if (llama_vram_hb_observe(vbr_claim_obs_, "m-" + m.busid + "-" + std::to_string(m.pid),
                                       m.hb_counter, now)
                    >= (uint64_t) LLAMA_VRAM_LEDGER_LONG_MS/2 * 1000000ull) {
                continue; // stale offer — not a competitor
            }
            if (m.created_ts_ns < self_created_ts ||
                (m.created_ts_ns == self_created_ts && m.pid < llama_vram_ledger_self_pid())) {
                rank0 = false;
                break;
            }
        }
        if (!rank0) {
            continue;
        }
        // shortfall = est − (free − headroom) − Σ peers' grant_pending (bridges shed→flush)
        size_t free_b = 0, total_b = 0;
        demanded_pool->be->get_device_memory(demanded_pool->device, &free_b, &total_b);
        // headroom_eff = base x N_live (spec normative — flat base would undershed by
        // (N_live-1) x base exactly when the census matters)
        const size_t headroom_eff = llama_vram_headroom_bytes() *
                demanded.child->vbr_pool_n_live(*demanded_pool);
        uint64_t peers_pending = 0;
        for (const auto & m : peers) {
            if (m.busid == c.busid) {
                peers_pending += m.fields.grant_pending;
            }
        }
        const uint64_t covered = (free_b > headroom_eff ? free_b - headroom_eff : 0) + peers_pending;
        LLAMA_LOG_DEBUG("%s: sizing pid %d: est %.1f free %.1f hr %.1f off %.1f\n",
                __func__, c.pid, c.fields.bytes_total_remaining_est/1048576.0,
                free_b/1048576.0, headroom_eff/1048576.0, (our_offer + sib_offer)/1048576.0);
        if (c.fields.bytes_total_remaining_est <= covered) {
            continue; // shortfall ≤ 0: free (or peers' in-flight sheds) already cover it
        }
        const uint64_t shortfall = c.fields.bytes_total_remaining_est - covered;
        const uint64_t target    = std::min<uint64_t>(our_offer + sib_offer, shortfall);

        // One root-owned transaction interleaves the children by the existing pure quality
        // policy, then prices every physical device before changing either child's tier state.
        const vbr_tx_result shed = root->vbr_execute_tree_shed(c, target, n_tokens);
        if (shed.status == vbr_tx_status::committed) {
            grants_changed = true;
            LLAMA_LOG_INFO("%s: atomic VBR tree shed for pid %d on %s: %.1f MiB net credit\n",
                    __func__, c.pid, c.busid.c_str(), shed.credited/1048576.0);
            // One physical wave per scan.  Its pending bridge must be published and its tails
            // flushed before another claim is sized from free memory and exact residency.
            break;
        } else if (shed.status == vbr_tx_status::retryable_no_tier_mutation) {
            // Donation is optional. A retryable planning, reservation, mapping, preparation, or
            // publication failure withdrew this donor's offer without changing tiers; it must not
            // fail the resident's otherwise-valid batch.
            break;
        }
    }
    return grants_changed;
}

void llama_kv_cache::vbr_grant_pending_clear() {
    // ---- grant_pending clear: first scan event after the wave's deferred unmaps flushed ----
    for (auto & [busid, pending] : vbr_grant_pending_) {
        if (pending == 0) {
            continue;
        }
        bool flushed = true;
        for (auto & p : vbr_pools_) {
            if (p.vmm != nullptr && vbr_pool_busid(p) == busid && !p.unmap_deferred.empty()) {
                flushed = false;
                break;
            }
        }
        if (flushed) {
            pending = 0;
        }
    }

}

void llama_kv_cache::vbr_markers_publish(std::set<std::string> * changed) {
    // ---- marker publish / beat (write discipline: rename only on field change) ----
    std::set<std::string> busids;
    auto collect_child = [&](llama_kv_cache * child) {
        if (child == nullptr) {
            return;
        }
        for (auto & p : child->vbr_pools_) {
            if (p.vmm != nullptr && child->vbr_pool_busid(p) != "-") {
                busids.insert(p.busid);
            }
        }
    };
    collect_child(this);
    collect_child(vbr_ledger_sibling_);
    for (const auto & busid : busids) {
        uint64_t pending = vbr_grant_pending_[busid];
        if (vbr_ledger_sibling_ != nullptr) {
            pending += vbr_ledger_sibling_->vbr_grant_pending_[busid];
        }
        const auto child_busy = [&](const llama_kv_cache * child) {
            if (child == nullptr) {
                return false;
            }
            for (const auto & p : child->vbr_pools_) {
                if (p.vmm != nullptr && p.busid == busid &&
                    (p.wave_pending || !p.unmap_deferred.empty())) {
                    return true;
                }
            }
            return false;
        };
        const bool tree_busy = child_busy(this) || child_busy(vbr_ledger_sibling_);
        uint64_t offer = 0;
        if (!vbr_tree_budget_explicit() && pending == 0 && !tree_busy) {
            offer = (uint64_t) vbr_child_offer(this, busid) +
                    (uint64_t) vbr_child_offer(vbr_ledger_sibling_, busid);
        }
        auto pub = vbr_marker_pub_.find(busid);
        if (pub == vbr_marker_pub_.end() || pub->second.first != offer || pub->second.second != pending) {
            llama_vram_marker_fields f = {};
            f.vbr            = 1;
            f.serviced       = llama_vram_marker_serviced_flag() ? 1u : 0u;
            f.shed_available = offer;
            f.grant_pending  = pending;
            uint64_t created_ts = 0;
            if (llama_vram_marker_publish(busid, f, &created_ts)) {
                vbr_marker_pub_[busid] = { offer, pending };
                vbr_marker_created_ts_[busid] = created_ts;
                if (changed != nullptr) {
                    changed->insert(busid);
                }
                // our own rename: re-adopt the dir mtime so we don't trip our own pre-check
                vbr_ledger_mtime_ = llama_vram_ledger_dir_mtime_ns();
            }
        } else {
            llama_vram_marker_beat(busid);
        }
    }
}

void llama_kv_cache::vbr_ledger_scan_service(uint32_t n_tokens) {
    // explicit budgets still run the pass — they publish markers (shed_available = 0,
    // demand service skipped) so the demander's presence census stays complete
    if (!vbr_ledger_owner_ || !vbr_vmm_active() || !llama_vram_ledger_armed()) {
        return;
    }
    vbr_ledger_force_ = false;
    vbr_last_scan_ns_ = llama_vram_ledger_now_ns();

    const uint64_t now = vbr_last_scan_ns_;

    // The root-owned offer is page-exact only against a settled tree.  A current boundary may
    // already have queued an ordinary budget wave on either child, so retire both children before
    // querying ranges or publishing capacity.  This rare scan path deliberately pays the sync.
    vbr_tx_settle_tree();
    // Retire the pending bridge, then announce the current offer before taking the rank snapshot.
    // A changed advert waits one scan; the post-service publish exposes the handoff immediately.
    vbr_grant_pending_clear();
    std::set<std::string> announced;
    vbr_markers_publish(&announced);

    std::vector<llama_vram_peer_claim> claims;
    llama_vram_ledger_scan(claims);
    std::vector<llama_vram_peer_marker> peers;
    llama_vram_ledger_scan_markers(peers);

    vbr_presence_census(peers);
    bool grants_changed = vbr_grants_upkeep(claims, now);
    if (vbr_ledger_sibling_ != nullptr) {
        // the owner scans once for the whole tree: the sibling's grants amortize and lift
        // against the same claim set, and its pending clears on its own flush state
        if (vbr_ledger_sibling_->vbr_grants_upkeep(claims, now)) {
            vbr_ledger_sibling_->vbr_apply_grant_decrements();
        }
        vbr_ledger_sibling_->vbr_grant_pending_clear();
    }
    grants_changed = vbr_service_demands(claims, peers, announced, now, n_tokens) || grants_changed;
    if (grants_changed) {
        vbr_apply_grant_decrements();
    }
    vbr_markers_publish();
}

void llama_kv_cache::vbr_cotenancy_accum(uint64_t & decrement, uint32_t & grants,
                                         uint64_t & offer, uint64_t & pending) const {
    if (!vbr_vmm_active()) {
        return;
    }
    decrement += vbr_total_grant_decrement();
    grants    += (uint32_t) vbr_grants_.size();
    // telemetry reports the PUBLISHED truth: an explicit budget never offers, so /slots
    // must not show peers an offer the marker does not carry
    if (vbr_ledger_owner_) {
        for (const auto & [busid, pub] : vbr_marker_pub_) {
            GGML_UNUSED(busid);
            offer += pub.first;
            pending += pub.second;
        }
    }
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
        return supports_qwen4_text_mrope_shift();
    }
    return true;
}

bool llama_kv_cache::supports_qwen4_text_mrope_shift() const {
    // This is deliberately a model/layout contract, not a generic M-RoPE
    // allowance.  Qwen4 text tokens use [t,t,t,0]; with no dimensions in the
    // fourth section a single temporal delta is exactly a NEOX K-shift over
    // the full rotary span.  Multimodal/2-D layouts do not satisfy this type.
    static constexpr std::array<int, 4> qwen4_text_sections = { 11, 11, 10, 0 };

    if (model.arch != LLM_ARCH_QWEN4EXP ||
        hparams.rope_type != LLAMA_ROPE_TYPE_IMROPE ||
        hparams.rope_sections != qwen4_text_sections) {
        return false;
    }

    for (const auto & layer : layers) {
        if (hparams.has_rope(layer.il) && hparams.n_rot(layer.il) != 64) {
            return false;
        }
    }

    return true;
}

bool llama_kv_cache::can_shift_qwen4_text_range(
        llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    if (!supports_qwen4_text_mrope_shift() ||
        seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return false;
    }

    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    const auto & cells = v_cells[seq_to_stream[seq_id]];
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1) || !cells.seq_has(i, seq_id)) {
            continue;
        }

        const llama_pos pos = cells.pos_get(i);
        const auto & ext = cells.ext_get(i);
        if (ext.x != pos || ext.y != pos) {
            return false;
        }
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

uint32_t llama_kv_cache::get_stream_for_seq(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && size_t(seq_id) < seq_to_stream.size());
    return seq_to_stream[size_t(seq_id)];
}

bool llama_kv_cache::state_empty() const {
    return std::all_of(v_cells.begin(), v_cells.end(),
        [](const llama_kv_cells & cells) { return cells.get_used() == 0; });
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

llama_turbo_meansub_ref llama_kv_cache::get_turbo_meansub_ref(int32_t il) const {
    const auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return {};
    }
    return layers.at(it->second).turbo_meansub_ref;
}

const llama_kv_cells & llama_kv_cache::get_cells(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    return v_cells[seq_to_stream[seq_id]];
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

bool llama_kv_cache::has_cell_ext() const {
    // M-RoPE needs the 2D position, the PLE n-gram hash needs the token id
    return hparams.n_pos_per_embd() > 1 || hparams.ple_n_heads > 0;
}

void llama_kv_cache::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    res.clear();
    res.resize(n_tokens*n, LLAMA_TOKEN_NULL);

    if (n == 0) {
        return;
    }

    // apply_ubatch() has already indexed the current ubatch. The canonical
    // (position, cell) index resolves the nearest predecessor directly, including
    // position gaps, instead of rebuilding a hash table by scanning used cells.

    // an embd (multimodal) ubatch can repeat one position for a whole image, so positions
    // do not encode the token order; resolve its predecessors by ubatch order instead
    std::vector<uint32_t> ord; // index among the ubatch tokens of the same seq
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idx;

    if (!ubatch.token) {
        ord.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            auto & v = seq_idx[ubatch.seq_id[i][0]];
            ord[i] = v.size();
            v.push_back(i);
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        // TODO: a token that belongs to more than one sequence has an ambiguous history.
        //       the n-gram architectures have to reject such batches
        const llama_seq_id seq_id = ubatch.seq_id[i][0];

        for (uint32_t j = 0; j < n; ++j) {
            const llama_pos d = (llama_pos) (n - j);

            llama_pos p;
            if (!ubatch.token) {
                const auto & v = seq_idx[seq_id];
                const int64_t k = (int64_t) ord[i] - d;
                // k >= 0: an earlier token of this very ubatch; k < 0: before the chunk
                p = k >= 0 ? ubatch.pos[v[k]] : ubatch.pos[v[0]] + (llama_pos) k;
            } else {
                p = ubatch.pos[i] - d;
            }

            if (p < 0) {
                continue;
            }

            GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
            res[i*n + j] = v_cells[seq_to_stream[seq_id]].seq_pos_tok_le(seq_id, p);
        }
    }
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
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

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

        if (!hparams.has_rope(il)) {
            continue;
        }

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
        // Settle in-flight degrade waves: the wave fence orders graph_compute, not the
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
            if (state_write_includes_cell(cells, i, seq_id)) {
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
    state_read_sinfo(io, seq_id, flags, nullptr, nullptr);
}

void llama_kv_cache::state_read_sinfo(
        llama_io_read_i & io,
           llama_seq_id   seq_id,
  llama_state_seq_flags   flags,
      slot_info_vec_t *   sinfos_out,
const slot_info_vec_t *   sinfos_in) {
    // Imports are provenance-bearing: recovery is reserved before the first read so a
    // partial-import unwind autorecords and the boundary drain quarantines + invalidates.
    vbr_mutation_op mutation_op(this, vbr_operation_kind::state_import,
            vbr_operation_class::state_api, seq_id, 0, std::numeric_limits<llama_pos>::max(),
            /*provenance_bearing=*/true);
    const vbr_mutation_op::success_on_return mutation_ok(mutation_op);
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (sinfos_out) {
        sinfos_out->assign(n_stream, slot_info{});
    }

    if (sinfos_in && sinfos_in->size() != n_stream) {
        throw std::runtime_error("failed to restore kv cache: mirrored slot layout has the wrong stream count");
    }

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    // a whole-context restore replaces every stream, so the cache is emptied once here
    // clear() resets all streams at once, so doing it per stream below would keep only the last one
    if (seq_id == -1) {
        clear(true);
    }

    bool imported = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            // a mirrored cache must be empty here as well, or the two no longer agree cell for cell
            if (sinfos_in && !(*sinfos_in)[s].empty()) {
                throw std::runtime_error("failed to restore kv cache: mirrored cache holds cells this one does not");
            }
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(
                io, strm, cell_count, sinfo, seq_id,
                sinfos_in ? &(*sinfos_in)[s] : nullptr);
        if (res && seq_id == -1 && vbr_vmm_active()) {
            // the whole-cache branch positions cells directly (no apply_ubatch), so nothing has
            // grown the VMM physical backing yet — state_read_data would write into unmapped VA
            vbr_vmm_ensure_mapped();
        }

        try {
            res = res && state_read_data(io, strm, cell_count, sinfo);
        } catch (...) {
            res = false;
        }

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }

        if (sinfos_out) {
            (*sinfos_out)[s] = sinfo;
        }

        imported = true;
    }

    // state_read_data adopts bytes in their native current tier (including mixed-tier caches).
    // Count the completed adoption once, independently of how many streams it populated.
    if (imported) {
        vbr_attention_content_changed();
        vbr_generation_global(
                seq_id == -1
                        ? vbr_mutation_registrant::whole_import
                        : vbr_mutation_registrant::state_read_install,
                vbr_operation_class::state_api);
        vbr_ownership_rebuild();
    }
}

// The one spelling of "update the index for every sequence that owns a cell".
// exclude_seq = -1 means none excluded; add=false removes.
void llama_kv_cache::vbr_ownership_update_all_seqs(uint32_t stream, uint32_t cell, llama_pos pos,
                                                   bool add, llama_seq_id exclude_seq) {
    if (!vbr_ownership_) {
        return;
    }
    // O(occupants) via the cell's own membership bitset (efficiency review: the all-seq probe
    // was 12.8M bit tests per whole-cache edit at 200k cells x 64 seqs).
    v_cells[stream].seq_for_each(cell, [&](llama_seq_id seq) {
        if (seq == exclude_seq) {
            return;
        }
        if (add) {
            vbr_ownership_->add_cell(stream, seq, cell, pos);
        } else {
            vbr_ownership_->remove_cell(stream, seq, cell, pos);
        }
    });
}

// One-time index rebuild at import/install boundaries (an infrequent, sanctioned scan).
void llama_kv_cache::vbr_ownership_rebuild() {
    if (!vbr_ownership_) {
        return;
    }
    vbr_ownership_->clear_all();
    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.is_empty(i)) {
                vbr_ownership_update_all_seqs(s, i, cells.pos_get(i), /*add=*/true);
            }
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

            if (has_cell_ext()) {
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

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id, const slot_info * sinfo_in) {
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

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                if (hparams.n_pos_per_embd() > 1) {
                    ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                    ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
                }

                // apply_ubatch() below restores ext.tok from the ubatch tokens
                ubatch.token[i] = ext.tok;
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

        if (sinfo_in) {
            // this cache mirrors another one, so it takes that cache's layout instead of searching for its own cells
            if (sinfo_in->empty() || sinfo_in->n_stream() != 1 || sinfo_in->idxs[0].size() != cell_count) {
                LLAMA_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                        sinfo_in->empty() ? 0 : (int) sinfo_in->idxs[0].size(), cell_count);
                return false;
            }

            sinfo = *sinfo_in;

            // the layout is cell indices, so it means the same in both caches only while their streams line up
            sinfo.s0 = strm;
            sinfo.s1 = strm;
            sinfo.strm[0] = strm;

            // seq_rm above freed exactly the cells this sequence held
            // anything else in the way is a cache that had already drifted, which this restore must not hide
            for (uint32_t i = 0; i < cell_count; ++i) {
                const uint32_t idx = sinfo.idxs[0][i];

                if (idx >= cells.size() || !cells.is_empty(idx)) {
                    LLAMA_LOG_ERROR("%s: cell %u of the mirrored slot layout is not free\n", __func__, idx);
                    return false;
                }
            }
        } else {
            sinfo = find_slot(ubatch, false);
            if (sinfo.empty()) {
                LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
                return false;
            }
        }

        // note: apply_ubatch() rebuilds llama_kv_cell_ext from the ubatch
        //       only ext.tok and the M-RoPE 2D position round-trip through it
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        // Native import publishes one controller-global generation only after state_read_data
        // completes. Do not misclassify its preparatory cell placement as an ordinary append.
        apply_ubatch(sinfo, ubatch, false);

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

        // the cells go in from 0, so a mirrored cache lands on the same ones as long as it restores the same count. the layout itself carries no more information here
        if (sinfo_in && (sinfo_in->empty() || sinfo_in->n_stream() != 1 || sinfo_in->idxs[0].size() != cell_count)) {
            LLAMA_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                    sinfo_in->empty() ? 0 : (int) sinfo_in->idxs[0].size(), cell_count);
            return false;
        }

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (has_cell_ext()) {
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

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) :
    status(status), max_graph_seqs(status == LLAMA_MEMORY_STATUS_SUCCESS ?
            std::numeric_limits<uint32_t>::max() : 0) {
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : llama_kv_cache_context(kv, std::numeric_limits<uint32_t>::max()) {
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        uint32_t         max_graph_seqs_limit) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();

    const uint32_t n_stream_physical = kv->get_n_stream();
    if (max_graph_seqs_limit == 0 || n_stream_physical == 0) {
        status = LLAMA_MEMORY_STATUS_FAILED_PREPARE;
        max_graph_seqs = 0;
        return;
    }
    // Unified KV has one physical stream but may represent every configured logical
    // sequence. Keep physical graph layout separate from the composite's logical cap.
    const uint32_t n_stream = n_stream_physical == 1
        ? 1
        : std::min(n_stream_physical, max_graph_seqs_limit);
    max_graph_seqs = max_graph_seqs_limit;

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
    // Unresolved recovery work is an update: update() owns the quarantine drain and the
    // monotone re-arm, and a NO_UPDATE short-circuit here would starve a latched tracker on
    // quiet decode streams until an unrelated shift happened to run.
    if (!do_shift && this->sc_info.empty() && !kv->vbr_recovery_service_pending()) {
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

uint32_t llama_kv_cache_context::get_max_graph_seqs() const {
    return max_graph_seqs;
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

llama_turbo_meansub_ref llama_kv_cache_context::get_turbo_meansub_ref(int32_t il) const {
    return kv->get_turbo_meansub_ref(il);
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

void llama_kv_cache_context::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    kv->get_prev_tokens(ubatch, n, res);
}
