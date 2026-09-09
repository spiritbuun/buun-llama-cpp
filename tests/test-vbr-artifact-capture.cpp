#include "llama-vbr-artifact-capture.h"
#include "llama-vbr-artifact-stage.h"
#include "llama-vbr-artifact-validate.h"
#include "llama-vbr-explicit-capture.h"
#include "llama-vbr-identity-digest.h"
#include "llama-vbr-operation.h"
#include "server-prompt-cache-payload.h"
#include "server-vbr-artifact-store.h"

#include "ggml.h"
#include "ggml-turbo-meansub.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", \
                    __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

struct synthetic_source {
    std::vector<uint8_t> bytes;
    uint64_t fail_at = UINT64_MAX;
};

static void test_exact_capture_pretransfer_quote() {
    vbr_explicit_capture_pretransfer_quote quote;
    quote.payload_bytes = 40;
    quote.stash_bytes = 8;
    quote.companion_bytes = 12;
    quote.metadata_bytes = 4;
    quote.planned_packed_bytes = 60;
    quote.conservative_host_resident_bytes = 64;
    quote.controllers = 2;
    quote.units = 16;
    quote.companions = 3;

    CHECK(vbr_explicit_capture_pretransfer_quote_admissible(quote, 0));
    CHECK(vbr_explicit_capture_pretransfer_quote_admissible(quote, 60));
    CHECK(!vbr_explicit_capture_pretransfer_quote_admissible(quote, 59));

    auto mutant = quote;
    mutant.planned_packed_bytes++;
    CHECK(!vbr_explicit_capture_pretransfer_quote_admissible(mutant, 0));
    mutant = quote;
    mutant.conservative_host_resident_bytes++;
    CHECK(!vbr_explicit_capture_pretransfer_quote_admissible(mutant, 0));
    mutant = quote;
    mutant.controllers = 0;
    CHECK(!vbr_explicit_capture_pretransfer_quote_admissible(mutant, 0));
    mutant = quote;
    mutant.units = 0;
    CHECK(!vbr_explicit_capture_pretransfer_quote_admissible(mutant, 0));
    mutant = quote;
    mutant.payload_bytes = UINT64_MAX;
    mutant.stash_bytes = 1;
    mutant.planned_packed_bytes = UINT64_MAX;
    mutant.conservative_host_resident_bytes = UINT64_MAX;
    CHECK(!vbr_explicit_capture_pretransfer_quote_admissible(mutant, 0));
}

static bool read_synthetic(
        const void * opaque, uint64_t offset,
        uint8_t * destination, size_t size) noexcept {
    const auto & source =
        *static_cast<const synthetic_source *>(opaque);
    if (offset >= source.fail_at ||
        offset > source.bytes.size() ||
        size > source.bytes.size() - offset) {
        return false;
    }
    std::memcpy(destination, source.bytes.data() + offset, size);
    return true;
}

struct blocking_capture_source {
    synthetic_source source;
    std::atomic<bool> entered { false };
    std::atomic<bool> release { false };

    static bool read(
            const void * opaque, uint64_t offset,
            uint8_t * destination, size_t size) noexcept {
        auto & self = *const_cast<blocking_capture_source *>(
            static_cast<const blocking_capture_source *>(opaque));
        self.entered.store(true, std::memory_order_release);
        while (!self.release.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        return read_synthetic(&self.source, offset, destination, size);
    }
};

static std::vector<uint8_t> read_chain(
        const artifact_segment_chain & chain) {
    std::vector<uint8_t> out(chain.size());
    CHECK(chain.read(0, out.data(), out.size()));
    return out;
}

struct prefix_validator_serials {
    uint64_t accounting = 0;
    uint64_t policy = 0;
    uint64_t memory = 0;
    uint64_t state = 0;
    uint64_t tree = 0;
    std::array<uint8_t, 32> transform_tree = {};

    static bool recheck(
            const void * opaque,
            const vbr_target_empty_fingerprint & fingerprint) noexcept {
        const auto & self =
            *static_cast<const prefix_validator_serials *>(opaque);
        return fingerprint.memory_instance_cookie == self.memory &&
               fingerprint.target_state_serial == self.state &&
               fingerprint.accounting_serial == self.accounting &&
               fingerprint.tree_shape_digest == self.tree &&
               fingerprint.policy_epoch == self.policy &&
               fingerprint.children.size() == 1;
    }

    static uint64_t read_accounting(const void * opaque) noexcept {
        return static_cast<const prefix_validator_serials *>(opaque)->accounting;
    }

    static uint64_t read_policy(const void * opaque) noexcept {
        return static_cast<const prefix_validator_serials *>(opaque)->policy;
    }

    static bool read_transform_tree(
            const void * opaque,
            std::array<uint8_t, 32> & output) noexcept {
        output = static_cast<const prefix_validator_serials *>(opaque)
            ->transform_tree;
        return std::any_of(output.begin(), output.end(), [](uint8_t value) {
            return value != 0;
        });
    }
};

static void test_attention_stem_prefix_planner() {
    const auto cell = [](uint32_t physical, llama_pos logical) {
        return vbr_artifact_cell_placement {
            physical, logical, 0, 0,
        };
    };
    const std::vector<vbr_artifact_cell_placement> ordered {
        cell(0, 0), cell(1, 1), cell(2, 2), cell(3, 3),
    };
    const auto policy = [](uint64_t minimum_tokens,
                           uint64_t max_host_bytes = 0,
                           uint64_t max_host_tokens = 0) {
        return vbr_projected_capture_frontier_policy {
            vbr_projected_capture_frontier_mode::longest_attention_stem,
            minimum_tokens,
            max_host_bytes,
            max_host_tokens,
        };
    };
    vbr_attention_stem_prefix_plan plan;
    CHECK(vbr_plan_attention_stem_prefix(
        4, ordered.data(), ordered.size(), 10, 40, 1, policy(1), plan));
    CHECK(plan.status ==
          vbr_projected_capture_frontier_status::exact);
    CHECK(plan.selected_token_count == 4);
    CHECK(plan.selected_next_position == 4);
    CHECK(plan.planned_packed_bytes == 40);
    CHECK(plan.projected_host_resident_bytes == 424);
    CHECK(plan.surveyed_cells == 4);

    // Pin the inclusive cap comparison: one byte less must lose exactly one
    // complete row, not accept an over-cap frontier or lose two rows.
    CHECK(vbr_plan_attention_stem_prefix(
        4, ordered.data(), ordered.size(), 10, 39, 1, policy(1), plan));
    CHECK(plan.status ==
          vbr_projected_capture_frontier_status::stem_selected);
    CHECK(plan.selected_token_count == 3);
    CHECK(plan.planned_packed_bytes == 30);

    const std::vector<vbr_artifact_cell_placement> out_of_order {
        cell(0, 2), cell(1, 0), cell(2, 3), cell(3, 1),
    };
    CHECK(vbr_plan_attention_stem_prefix(
        4, out_of_order.data(), out_of_order.size(), 10, 40, 1,
        policy(4), plan));
    CHECK(plan.status ==
          vbr_projected_capture_frontier_status::exact);
    CHECK(plan.selected_token_count == 4);

    const std::vector<vbr_artifact_cell_placement> duplicate {
        cell(0, 0), cell(1, 1), cell(2, 1), cell(3, 3),
    };
    CHECK(!vbr_plan_attention_stem_prefix(
        4, duplicate.data(), duplicate.size(), 10, 40, 1,
        policy(1), plan));
    CHECK(plan.status ==
          vbr_projected_capture_frontier_status::stem_unsupported);
    const std::vector<vbr_artifact_cell_placement> out_of_range {
        cell(0, 0), cell(1, 1), cell(2, 2), cell(3, 4),
    };
    CHECK(!vbr_plan_attention_stem_prefix(
        4, out_of_range.data(), out_of_range.size(), 10, 40, 1,
        policy(1), plan));
    CHECK(plan.status ==
          vbr_projected_capture_frontier_status::stem_unsupported);

    CHECK(vbr_plan_attention_stem_prefix(
        4, ordered.data(), ordered.size(), 10, 29, 1, policy(3), plan));
    CHECK(plan.status ==
          vbr_projected_capture_frontier_status::stem_below_minimum);
    CHECK(plan.selected_token_count == 2);
    CHECK(plan.planned_packed_bytes == 20);

    // One projected unit contributes exactly 256 descriptor bytes and 128
    // reference bytes. The inclusive host cap accepts all four payload rows;
    // one byte less loses exactly one complete row.
    CHECK(vbr_plan_attention_stem_prefix(
        4, ordered.data(), ordered.size(), 10, 40, 1,
        policy(1, 424), plan));
    CHECK(plan.status == vbr_projected_capture_frontier_status::exact);
    CHECK(plan.selected_token_count == 4);
    CHECK(plan.projected_host_resident_bytes == 424);
    CHECK(vbr_plan_attention_stem_prefix(
        4, ordered.data(), ordered.size(), 10, 40, 1,
        policy(1, 423), plan));
    CHECK(plan.status ==
          vbr_projected_capture_frontier_status::stem_selected);
    CHECK(plan.selected_token_count == 3);
    CHECK(plan.projected_host_resident_bytes == 414);
    CHECK(vbr_plan_attention_stem_prefix(
        4, ordered.data(), ordered.size(), 10, 40, 2,
        policy(1, 808), plan));
    CHECK(plan.status == vbr_projected_capture_frontier_status::exact);
    CHECK(plan.projected_host_resident_bytes == 808);
    CHECK(vbr_plan_attention_stem_prefix(
        4, ordered.data(), ordered.size(), 10, 40, 2,
        policy(1, 807), plan));
    CHECK(plan.status ==
          vbr_projected_capture_frontier_status::stem_selected);
    CHECK(plan.selected_token_count == 3);
    CHECK(plan.projected_host_resident_bytes == 798);

    // Token-only cache limits are exact. With a byte limit, cache semantics
    // derive the effective token limit from bytes, so max_host_tokens is not a
    // second independent ceiling.
    CHECK(vbr_plan_attention_stem_prefix(
        4, ordered.data(), ordered.size(), 10, 40, 1,
        policy(1, 0, 4), plan));
    CHECK(plan.status == vbr_projected_capture_frontier_status::exact);
    CHECK(vbr_plan_attention_stem_prefix(
        4, ordered.data(), ordered.size(), 10, 40, 1,
        policy(1, 0, 3), plan));
    CHECK(plan.status ==
          vbr_projected_capture_frontier_status::stem_selected);
    CHECK(plan.selected_token_count == 3);
    CHECK(vbr_plan_attention_stem_prefix(
        4, ordered.data(), ordered.size(), 10, 40, 1,
        policy(1, 424, 1), plan));
    CHECK(plan.status == vbr_projected_capture_frontier_status::exact);

    std::vector<vbr_artifact_cell_placement> maximum(
        VBR_PROJECTED_CAPTURE_MAX_TOKEN_IDS);
    for (uint32_t i = 0; i < maximum.size(); ++i) {
        maximum[i] = cell(
            i, llama_pos(VBR_PROJECTED_CAPTURE_MAX_TOKEN_IDS - 1 - i));
    }
    CHECK(vbr_plan_attention_stem_prefix(
        VBR_PROJECTED_CAPTURE_MAX_TOKEN_IDS,
        maximum.data(), maximum.size(), 1,
        VBR_PROJECTED_CAPTURE_MAX_TOKEN_IDS,
        1, policy(VBR_PROJECTED_CAPTURE_MAX_TOKEN_IDS), plan));
    CHECK(plan.status ==
          vbr_projected_capture_frontier_status::exact);
    CHECK(plan.selected_token_count ==
          VBR_PROJECTED_CAPTURE_MAX_TOKEN_IDS);
    CHECK(plan.surveyed_cells ==
          VBR_PROJECTED_CAPTURE_MAX_TOKEN_IDS);
    CHECK(!vbr_plan_attention_stem_prefix(
        uint64_t(VBR_PROJECTED_CAPTURE_MAX_TOKEN_IDS) + 1,
        maximum.data(), maximum.size(), 1,
        uint64_t(VBR_PROJECTED_CAPTURE_MAX_TOKEN_IDS) + 1,
        1, policy(1), plan));
    CHECK(!vbr_plan_attention_stem_prefix(
        4, ordered.data(), ordered.size(), 10, 40,
        UINT64_MAX/256, policy(1), plan));
}

static void test_segment_chain_offsets() {
    artifact_segment_chain chain;
    const uint8_t a[] = { 0, 1, 2 };
    const uint8_t b[] = { 3, 4, 5, 6, 7 };
    CHECK(chain.append(a, sizeof(a)));
    CHECK(chain.append(b, sizeof(b)));
    CHECK(chain.size() == 8);
    CHECK(chain.segment_count() == 2);
    CHECK(chain.max_segment_size() == 5);
    uint8_t middle[5] = {};
    CHECK(chain.read(2, middle, sizeof(middle)));
    CHECK(std::vector<uint8_t>(middle, middle + 5) ==
          std::vector<uint8_t>({ 2, 3, 4, 5, 6 }));
    uint8_t one = 0;
    const auto source = chain.source();
    CHECK(source.read(source.context, 7, &one, 1));
    CHECK(one == 7);
    CHECK(!chain.read(8, &one, 1));

    artifact_segment_chain known_size(8);
    CHECK(known_size.append(a, sizeof(a)));
    const auto incomplete_digest =
        vbr_capture_stream_digest(known_size);
    CHECK(!std::any_of(
        incomplete_digest.begin(), incomplete_digest.end(),
        [](uint8_t value) { return value != 0; }));
    CHECK(known_size.append(b, sizeof(b)));
    CHECK(vbr_capture_stream_digest(known_size) ==
          vbr_capture_stream_digest(chain));
    CHECK(!known_size.append(&one, 1));

    auto incomplete = std::make_unique<artifact_segment_chain>(8);
    CHECK(incomplete->append(a, sizeof(a)));
    vbr_capture_sealed_companion refused;
    CHECK(!vbr_capture_seal_companion(
        0, std::move(incomplete), refused));
}

struct range_verify_source {
    std::vector<uint8_t> bytes;
    uint64_t calls = 0;

    static bool read(
            const void * opaque, uint64_t offset,
            uint8_t * destination, size_t size) noexcept {
        auto & self = *const_cast<range_verify_source *>(
            static_cast<const range_verify_source *>(opaque));
        if (offset > self.bytes.size() ||
            size > self.bytes.size() - offset) {
            return false;
        }
        self.calls++;
        std::memcpy(destination, self.bytes.data() + offset, size);
        return true;
    }

    vbr_artifact_byte_source source() const {
        return { bytes.size(), this, read };
    }
};

static void test_authenticated_range_tree() {
    static constexpr size_t CHUNK = VBR_CAPTURE_RANGE_CHUNK_BYTES;
    std::vector<uint8_t> bytes(4*CHUNK + 123);
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = uint8_t((i*29 + i/251 + 7) & 0xff);
    }

    artifact_segment_chain chain(VBR_CAPTURE_RANGE_CHUNK_BYTES, 5);
    const auto unsealed_digest = chain.sealed_digest();
    CHECK(!std::any_of(
        unsealed_digest.begin(), unsealed_digest.end(),
        [](uint8_t value) { return value != 0; }));
    CHECK(chain.append(bytes.data(), 17));
    CHECK(chain.append(bytes.data() + 17, CHUNK + 91));
    CHECK(chain.append(
        bytes.data() + CHUNK + 108,
        bytes.size() - CHUNK - 108));
    vbr_capture_range_tree tree;
    CHECK(vbr_capture_range_seal(chain, 1024, tree));
    CHECK(tree);
    CHECK(tree.total_bytes() == bytes.size());
    CHECK(tree.chunk_bytes() == CHUNK);
    CHECK(tree.chunk_count() == 5);
    CHECK(tree.metadata_bytes() >= 15*32);
    CHECK(std::any_of(
        tree.root().begin(), tree.root().end(),
        [](uint8_t value) { return value != 0; }));
    CHECK(chain.sealed_digest() == tree.root());
    CHECK(!chain.append(bytes.data(), 1));

    // Append boundaries do not affect the authenticated root.
    artifact_segment_chain rechunked(VBR_CAPTURE_RANGE_CHUNK_BYTES, 5);
    CHECK(rechunked.append(bytes.data(), bytes.size()));
    vbr_capture_range_tree same;
    CHECK(vbr_capture_range_seal(rechunked, 1024, same));
    CHECK(same.root() == tree.root());

    vbr_capture_range_proof proof;
    const std::vector<vbr_capture_authenticated_range> ranges {
        { CHUNK + 100, 200 },
        { 4*CHUNK + 3, 100 },
    };
    CHECK(vbr_capture_range_prove(tree, ranges, {}, proof));
    CHECK(proof);
    CHECK(proof.root() == tree.root());
    CHECK(proof.ranges().size() == 2);
    CHECK(proof.selected_chunk_count() == 2);
    CHECK(proof.proof_node_count() == 4);
    CHECK(proof.metadata_bytes() <= 1024);

    range_verify_source source { bytes };
    uint64_t bytes_read = 99;
    CHECK(vbr_capture_range_verify(proof, source.source(), &bytes_read));
    CHECK(bytes_read == CHUNK + 123);
    CHECK(source.calls == 2);

    // Selected corruption is detected; omitted chunks are never read.
    source.bytes[CHUNK + 101] ^= 1;
    CHECK(!vbr_capture_range_verify(proof, source.source(), &bytes_read));
    CHECK(bytes_read == 0);
    source.bytes = bytes;
    source.bytes[2*CHUNK + 1] ^= 1;
    source.calls = 0;
    CHECK(vbr_capture_range_verify(proof, source.source(), &bytes_read));
    CHECK(source.calls == 2);

    vbr_capture_range_proof refused = proof;
    auto tight = vbr_capture_range_proof_limits{};
    tight.max_ranges = 1;
    CHECK(!vbr_capture_range_prove(tree, ranges, tight, refused));
    CHECK(!refused);
    tight = {};
    tight.max_selected_chunks = 1;
    CHECK(!vbr_capture_range_prove(tree, ranges, tight, refused));
    tight = {};
    tight.max_proof_nodes = 2;
    CHECK(!vbr_capture_range_prove(tree, ranges, tight, refused));
    tight = {};
    tight.max_metadata_bytes = proof.metadata_bytes() - 1;
    CHECK(!vbr_capture_range_prove(tree, ranges, tight, refused));
    CHECK(!vbr_capture_range_prove(
        tree, { { CHUNK, 0 } }, {}, refused));
    CHECK(!vbr_capture_range_prove(
        tree, { { 2*CHUNK, 4 }, { CHUNK, 4 } }, {}, refused));

    // A child proof is derived entirely from immutable parent authority. The
    // requested ranges may split chunks and remain noncontiguous, but may not
    // cross a byte the parent did not authorize.
    vbr_capture_range_proof broad_parent;
    const std::vector<vbr_capture_authenticated_range> broad_ranges {
        { 16, 3*CHUNK + 100 },
        { 4*CHUNK + 5, 100 },
    };
    CHECK(vbr_capture_range_prove(
        tree, broad_ranges, {}, broad_parent));
    const auto broad_root = broad_parent.root();
    const auto broad_metadata = broad_parent.metadata_bytes();
    const std::vector<vbr_capture_authenticated_range> child_ranges {
        { CHUNK - 8, 16 },
        { 3*CHUNK + 7, 11 },
        { 4*CHUNK + 20, 5 },
    };
    vbr_capture_range_proof child;
    CHECK(vbr_capture_range_restrict(
        broad_parent, child_ranges, {}, child) ==
        vbr_capture_range_restrict_status::restricted);
    CHECK(child);
    CHECK(child.root() == broad_parent.root());
    CHECK(child.ranges().size() == child_ranges.size());
    for (size_t i = 0; i < child_ranges.size(); ++i) {
        CHECK(child.ranges()[i].offset == child_ranges[i].offset);
        CHECK(child.ranges()[i].size == child_ranges[i].size);
    }
    CHECK(child.selected_chunk_count() == 4);

    range_verify_source child_source { bytes };
    bytes_read = 99;
    CHECK(vbr_capture_range_verify(
        child, child_source.source(), &bytes_read));
    CHECK(child_source.calls == 4);
    CHECK(bytes_read == 3*CHUNK + 123);
    // Chunk two is omitted by the child even though the parent selected it.
    child_source.bytes[2*CHUNK + 1] ^= 1;
    child_source.calls = 0;
    CHECK(vbr_capture_range_verify(
        child, child_source.source(), &bytes_read));
    CHECK(child_source.calls == 4);
    child_source.bytes[CHUNK - 1] ^= 1;
    child_source.calls = 0;
    CHECK(!vbr_capture_range_verify(
        child, child_source.source(), &bytes_read));
    CHECK(bytes_read == 0);

    // Exact limits accept; one below each independently refuses and clears a
    // seeded output. This pins all four bounded arenas used by restriction.
    auto child_limits = vbr_capture_range_proof_limits{};
    child_limits.max_ranges = uint32_t(child.ranges().size());
    child_limits.max_selected_chunks = child.selected_chunk_count();
    child_limits.max_proof_nodes = child.proof_node_count();
    child_limits.max_metadata_bytes = child.metadata_bytes();
    vbr_capture_range_proof bounded;
    CHECK(vbr_capture_range_restrict(
        broad_parent, child_ranges, child_limits, bounded) ==
        vbr_capture_range_restrict_status::restricted);
    CHECK(bounded.metadata_bytes() == child.metadata_bytes());
    auto one_under = child_limits;
    one_under.max_ranges--;
    bounded = child;
    CHECK(vbr_capture_range_restrict(
        broad_parent, child_ranges, one_under, bounded) ==
        vbr_capture_range_restrict_status::limit_exceeded);
    CHECK(!bounded);
    one_under = child_limits;
    one_under.max_selected_chunks--;
    bounded = child;
    CHECK(vbr_capture_range_restrict(
        broad_parent, child_ranges, one_under, bounded) ==
        vbr_capture_range_restrict_status::limit_exceeded);
    CHECK(!bounded);
    one_under = child_limits;
    CHECK(one_under.max_proof_nodes > 0);
    one_under.max_proof_nodes--;
    bounded = child;
    CHECK(vbr_capture_range_restrict(
        broad_parent, child_ranges, one_under, bounded) ==
        vbr_capture_range_restrict_status::limit_exceeded);
    CHECK(!bounded);
    one_under = child_limits;
    one_under.max_metadata_bytes--;
    bounded = child;
    CHECK(vbr_capture_range_restrict(
        broad_parent, child_ranges, one_under, bounded) ==
        vbr_capture_range_restrict_status::limit_exceeded);
    CHECK(!bounded);

    // Authorization is byte-exact, not merely chunk-local.
    bounded = child;
    CHECK(vbr_capture_range_restrict(
        broad_parent, {}, {}, bounded) ==
        vbr_capture_range_restrict_status::invalid_argument);
    CHECK(!bounded);
    CHECK(vbr_capture_range_restrict(
        proof, { { CHUNK + 250, 100 } }, {}, bounded) ==
        vbr_capture_range_restrict_status::range_unauthorized);
    CHECK(!bounded);
    CHECK(vbr_capture_range_restrict(
        proof, { { 2*CHUNK, 1 } }, {}, bounded) ==
        vbr_capture_range_restrict_status::range_unauthorized);
    CHECK(vbr_capture_range_restrict(
        broad_parent, { { CHUNK, 0 } }, {}, bounded) ==
        vbr_capture_range_restrict_status::invalid_argument);
    CHECK(vbr_capture_range_restrict(
        broad_parent,
        { { 3*CHUNK, 4 }, { CHUNK, 4 } }, {}, bounded) ==
        vbr_capture_range_restrict_status::invalid_argument);
    CHECK(vbr_capture_range_restrict(
        broad_parent,
        { { CHUNK, 8 }, { CHUNK + 4, 8 } }, {}, bounded) ==
        vbr_capture_range_restrict_status::invalid_argument);
    CHECK(vbr_capture_range_restrict(
        {}, child_ranges, {}, bounded) ==
        vbr_capture_range_restrict_status::parent_invalid);

    // Adjacent parent ranges form one continuous authority interval.
    vbr_capture_range_proof adjacent_parent;
    CHECK(vbr_capture_range_prove(
        tree,
        { { 0, CHUNK/2 }, { CHUNK/2, CHUNK/2 } },
        {}, adjacent_parent));
    CHECK(vbr_capture_range_restrict(
        adjacent_parent, { { CHUNK/4, CHUNK/2 } }, {}, bounded) ==
        vbr_capture_range_restrict_status::restricted);
    range_verify_source adjacent_source { bytes };
    CHECK(vbr_capture_range_verify(
        bounded, adjacent_source.source(), &bytes_read));
    CHECK(adjacent_source.calls == 1);
    CHECK(bytes_read == CHUNK);

    // Restriction is non-consuming and cannot mutate the parent's root or
    // proof metadata.
    range_verify_source parent_source { bytes };
    CHECK(broad_parent.root() == broad_root);
    CHECK(broad_parent.metadata_bytes() == broad_metadata);
    CHECK(vbr_capture_range_verify(
        broad_parent, parent_source.source(), &bytes_read));

    artifact_segment_chain unavailable;
    vbr_capture_range_tree unavailable_tree = tree;
    CHECK(unavailable.append(bytes.data(), 4));
    CHECK(!vbr_capture_range_seal(unavailable, 1024, unavailable_tree));
    CHECK(!unavailable_tree);
    artifact_segment_chain over_cap(VBR_CAPTURE_RANGE_CHUNK_BYTES, 1);
    CHECK(!over_cap.append(bytes.data(), CHUNK + 1));
    CHECK(over_cap.size() == 0);
    artifact_segment_chain metadata_cap(
        VBR_CAPTURE_RANGE_CHUNK_BYTES, 5);
    CHECK(metadata_cap.append(bytes.data(), bytes.size()));
    CHECK(!vbr_capture_range_seal(
        metadata_cap, tree.metadata_bytes() - 1, unavailable_tree));
    CHECK(!unavailable_tree);

    artifact_segment_chain cancelled_chain(
        VBR_CAPTURE_RANGE_CHUNK_BYTES, 5);
    CHECK(cancelled_chain.append(bytes.data(), bytes.size()));
    bool cancelled = false;
    bool allow_seal = false;
    CHECK(!vbr_capture_range_seal(
        cancelled_chain, 1024, unavailable_tree, &allow_seal,
        [](void * opaque) noexcept {
            return *static_cast<const bool *>(opaque);
        }, &cancelled));
    CHECK(cancelled);
    CHECK(!unavailable_tree);

    // Cancellation remains authoritative after leaf hashing and before the
    // authenticated root becomes visible.
    artifact_segment_chain post_leaf_cancelled_chain(
        VBR_CAPTURE_RANGE_CHUNK_BYTES, 1);
    CHECK(post_leaf_cancelled_chain.append(bytes.data(), 4));
    uint32_t seal_polls = 0;
    cancelled = false;
    CHECK(!vbr_capture_range_seal(
        post_leaf_cancelled_chain, 1024, unavailable_tree, &seal_polls,
        [](void * opaque) noexcept {
            return ++*static_cast<uint32_t *>(opaque) < 3;
        }, &cancelled));
    CHECK(seal_polls == 3);
    CHECK(cancelled);
    CHECK(!unavailable_tree);
    const auto cancelled_digest = post_leaf_cancelled_chain.sealed_digest();
    CHECK(!std::any_of(
        cancelled_digest.begin(), cancelled_digest.end(),
        [](uint8_t value) { return value != 0; }));
}

static void test_registry_quiescence_query() {
    const vbr_controller_instance_id instance { 0x1111, 0x2222 };
    const vbr_controller_instance_id other { 0x3333, 0x4444 };
    CHECK(vbr_operation_registry_quiescent_for(&instance, 1));

    {
        auto binding = vbr_mutation_binding(
            vbr_operation_kind::sequence_edit,
            0, 0, 1,
            vbr_operation_class::explicit_destructive_trim,
            instance, 0);
        vbr_scoped_operation operation(binding);
        CHECK(bool(operation));
        CHECK(!vbr_operation_registry_quiescent_for(&instance, 1));
        CHECK(vbr_operation_registry_quiescent_for(&other, 1));
    }
    CHECK(vbr_operation_registry_quiescent_for(&instance, 1));

    {
        auto binding = vbr_mutation_binding(
            vbr_operation_kind::recovery,
            -1, -1, -1,
            vbr_operation_class::state_api);
        vbr_scoped_operation operation(binding);
        CHECK(bool(operation));
        CHECK(!vbr_operation_registry_quiescent_for(&instance, 1));
        CHECK(!vbr_operation_registry_quiescent_for(&other, 1));
    }
    CHECK(vbr_operation_registry_quiescent_for(&instance, 1));
    CHECK(!vbr_operation_registry_quiescent_for(nullptr, 0));
}

static void test_cpu_ring_boundaries() {
    vbr_capture_stream_status status;
    vbr_capture_ring_create_failure failure;
    auto unavailable = vbr_pinned_chunk_ring::create(
        { {} }, 8, 8, status, nullptr, &failure);
    CHECK(!unavailable);
    CHECK(status == vbr_capture_stream_status::ring_unavailable);
    CHECK(failure ==
          vbr_capture_ring_create_failure::invalid_geometry);

    auto ring = vbr_pinned_chunk_ring::create(
        { {}, {} }, 32, 8, status);
    CHECK(ring);
    CHECK(status == vbr_capture_stream_status::ok);
    CHECK(ring->lane_count() == 2);
    CHECK(ring->capacity_bytes() == 32);

    synthetic_source input;
    input.bytes.resize(41);
    for (size_t i = 0; i < input.bytes.size(); ++i) {
        input.bytes[i] = uint8_t((i*17 + 3) & 0xff);
    }
    vbr_capture_stream_source source;
    source.lane = 1;
    source.size = input.bytes.size();
    source.context = &input;
    source.read = read_synthetic;
    artifact_segment_chain chain;
    vbr_capture_stream_stats stats;
    CHECK(ring->stream(source, chain, stats) ==
          vbr_capture_stream_status::ok);
    CHECK(stats.bytes == input.bytes.size());
    CHECK(stats.chunks == 6);
    CHECK(stats.backpressure_waits > 0);
    CHECK(stats.max_segment_size <= 8);
    CHECK(chain.max_segment_size() <= 8);
    CHECK(chain.size() > ring->capacity_bytes());
    CHECK(read_chain(chain) == input.bytes);

    // Test-local pre-refactor oracle: the historical CPU adapter appended one
    // pageable segment per ring-sized chunk and hashed the resulting byte
    // stream. This pins the D2H facade across the shared-core extraction.
    artifact_segment_chain legacy;
    for (size_t offset = 0; offset < input.bytes.size(); offset += 8) {
        const size_t count = std::min<size_t>(8, input.bytes.size() - offset);
        CHECK(legacy.append(input.bytes.data() + offset, count));
    }
    CHECK(read_chain(legacy) == read_chain(chain));
    CHECK(vbr_capture_stream_digest(legacy) == stats.streaming_digest);
    CHECK(legacy.segment_count() == stats.chunks);

    artifact_segment_chain projected;
    vbr_capture_stream_stats projected_stats;
    const std::vector<vbr_capture_stream_range> ranges {
        { 2, 3 }, { 10, 5 }, { 20, 1 },
    };
    CHECK(ring->stream_ranges(
              source, ranges, projected, projected_stats) ==
          vbr_capture_stream_status::ok);
    std::vector<uint8_t> projected_expected;
    for (const auto & range : ranges) {
        projected_expected.insert(
            projected_expected.end(),
            input.bytes.begin() + range.source_offset,
            input.bytes.begin() + range.source_offset + range.size);
    }
    CHECK(read_chain(projected) == projected_expected);
    CHECK(projected_stats.bytes == projected_expected.size());
    CHECK(projected_stats.chunks == 2);
    CHECK(projected.segment_count() == 2);
    CHECK(projected_stats.streaming_digest ==
          vbr_capture_stream_digest(projected));

    artifact_segment_chain invalid_ranges;
    projected_stats.bytes = 99;
    CHECK(ring->stream_ranges(
              source, { { 4, 3 }, { 6, 2 } },
              invalid_ranges, projected_stats) ==
          vbr_capture_stream_status::invalid_argument);
    CHECK(projected_stats.bytes == 0);
    CHECK(invalid_ranges.size() == 0);

    auto other = vbr_pinned_chunk_ring::create(
        { {} }, 14, 7, status);
    CHECK(other);
    artifact_segment_chain rechunked;
    vbr_capture_stream_stats other_stats;
    source.lane = 0;
    CHECK(other->stream(source, rechunked, other_stats) ==
          vbr_capture_stream_status::ok);
    CHECK(read_chain(rechunked) == input.bytes);
    CHECK(other_stats.streaming_digest == stats.streaming_digest);

    input.fail_at = 16;
    artifact_segment_chain short_chain;
    CHECK(other->stream(source, short_chain, other_stats) ==
          vbr_capture_stream_status::short_read);

    input.fail_at = UINT64_MAX;
    source.fail_completion_at = 1;
    artifact_segment_chain failed_completion;
    CHECK(other->stream(
        source, failed_completion, other_stats) ==
            vbr_capture_stream_status::transfer_failed);

    struct cancellation_state {
        uint32_t probes = 0;
        uint32_t allowed = 1;
    } cancellation;
    source.fail_completion_at = UINT64_MAX;
    source.continue_context = &cancellation;
    source.continue_transfer = [](void * opaque) noexcept {
        auto & state = *static_cast<cancellation_state *>(opaque);
        return state.probes++ < state.allowed;
    };
    artifact_segment_chain cancelled;
    CHECK(other->stream(source, cancelled, other_stats) ==
          vbr_capture_stream_status::cancelled);
    CHECK(cancellation.probes == 2);
    CHECK(other_stats.bytes == other->chunk_bytes());
    CHECK(other_stats.chunks == 1);
    CHECK(other_stats.submitted_bytes == other->chunk_bytes());
    CHECK(other_stats.submitted_chunks == 1);
    CHECK(cancelled.size() == other->chunk_bytes());

    // Cancellation drains the operation/chunk state; the next transfer can
    // immediately reuse the same persistent ring.
    source.continue_context = nullptr;
    source.continue_transfer = nullptr;
    artifact_segment_chain after_cancel;
    CHECK(other->stream(source, after_cancel, other_stats) ==
          vbr_capture_stream_status::ok);
    CHECK(read_chain(after_cancel) == input.bytes);
}

struct projected_snapshot_fixture {
    vbr_capture_unit_snapshot snapshot;
    uint32_t acquired = 0;
    uint32_t rechecked = 0;
    uint32_t released = 0;
    bool acquire_ok = true;
    bool recheck_ok = true;

    static bool acquire(
            void * context,
            uint64_t source_namespace,
            uint32_t child_id,
            uint32_t logical_unit_id,
            vbr_capture_unit_snapshot & output) noexcept {
        auto & self = *static_cast<projected_snapshot_fixture *>(context);
        self.acquired++;
        output = self.snapshot;
        return self.acquire_ok &&
               output.source_namespace == source_namespace &&
               output.child_id == child_id &&
               output.logical_unit_id == logical_unit_id;
    }

    static bool recheck(
            void * context,
            const vbr_capture_unit_snapshot & expected) noexcept {
        auto & self = *static_cast<projected_snapshot_fixture *>(context);
        self.rechecked++;
        return self.recheck_ok &&
               expected.generation.repr_gen ==
                   self.snapshot.generation.repr_gen &&
               expected.generation.publish_seq ==
                   self.snapshot.generation.publish_seq &&
               expected.generation.current_type ==
                   self.snapshot.generation.current_type &&
               expected.generation.last_source_type ==
                   self.snapshot.generation.last_source_type &&
               expected.generation.domain ==
                   self.snapshot.generation.domain &&
               expected.generation.promote_hops ==
                   self.snapshot.generation.promote_hops &&
               expected.generation.last_transition ==
                   self.snapshot.generation.last_transition &&
               expected.lineage_uuid == self.snapshot.lineage_uuid &&
               expected.controller_generation ==
                   self.snapshot.controller_generation &&
               expected.mutation_serial == self.snapshot.mutation_serial;
    }

    static void release(
            void * context,
            const vbr_capture_unit_snapshot &) noexcept {
        static_cast<projected_snapshot_fixture *>(context)->released++;
    }

    vbr_capture_unit_snapshot_provider provider() {
        return { this, acquire, recheck, release };
    }
};

static vbr_artifact_stream_placement projected_placement(
        uint64_t manifest,
        llama_seq_id sequence,
        std::initializer_list<uint32_t> cells) {
    GGML_UNUSED(manifest);
    vbr_artifact_stream_placement placement;
    placement.child_id = 0;
    placement.stream_index = 0;
    placement.source_sequence = sequence;
    placement.computation_frontier = 8;
    llama_pos position = 0;
    for (uint32_t cell : cells) {
        placement.cells.push_back({ cell, position++, 0, 0 });
    }
    return placement;
}

static vbr_artifact_stream_placement projected_placement(
        uint64_t manifest,
        llama_seq_id sequence,
        const std::vector<uint32_t> & cells) {
    GGML_UNUSED(manifest);
    vbr_artifact_stream_placement placement;
    placement.child_id = 0;
    placement.stream_index = 0;
    placement.source_sequence = sequence;
    placement.computation_frontier = llama_pos(cells.size());
    llama_pos position = 0;
    for (uint32_t cell : cells) {
        placement.cells.push_back({ cell, position++, 0, 0 });
    }
    return placement;
}

static std::vector<uint8_t> projected_rows(
        const std::vector<uint8_t> & source,
        uint64_t row_bytes,
        std::initializer_list<uint32_t> cells) {
    std::vector<uint8_t> output;
    for (uint32_t cell : cells) {
        const size_t offset = size_t(cell*row_bytes);
        output.insert(
            output.end(), source.begin() + offset,
            source.begin() + offset + row_bytes);
    }
    return output;
}

static void test_projected_unit_transfer() {
    vbr_capture_projection_manifest one;
    one.manifest_id = 1;
    one.placements.push_back(projected_placement(1, 1, { 1, 2, 5 }));
    vbr_capture_projection_manifest two;
    two.manifest_id = 2;
    two.placements.push_back(projected_placement(2, 2, { 2, 3 }));
    vbr_capture_projection projection;
    CHECK(vbr_artifact_project_capture_union(
        { 91, { one, two } }, {}, projection));

    synthetic_source first;
    synthetic_source second;
    first.bytes.resize(8*2);
    second.bytes.resize(8*3);
    for (size_t i = 0; i < first.bytes.size(); ++i) {
        first.bytes[i] = uint8_t(10 + i);
    }
    for (size_t i = 0; i < second.bytes.size(); ++i) {
        second.bytes[i] = uint8_t(100 + i);
    }
    vbr_capture_projected_shard_source shard_zero;
    shard_zero.shard_index = 0;
    shard_zero.row_count = 8;
    shard_zero.row_bytes = 2;
    shard_zero.source_identity = 101;
    shard_zero.source.size = first.bytes.size();
    shard_zero.source.context = &first;
    shard_zero.source.read = read_synthetic;
    vbr_capture_projected_shard_source shard_one;
    shard_one.shard_index = 1;
    shard_one.row_count = 8;
    shard_one.row_bytes = 3;
    shard_one.source_identity = 102;
    shard_one.source.size = second.bytes.size();
    shard_one.source.context = &second;
    shard_one.source.read = read_synthetic;
    std::vector<vbr_capture_projected_shard_source> sources {
        shard_one, shard_zero,
    };

    projected_snapshot_fixture snapshot;
    snapshot.snapshot.source_namespace = 91;
    snapshot.snapshot.child_id = 0;
    snapshot.snapshot.logical_unit_id = 7;
    snapshot.snapshot.lineage_uuid = { 17, 19 };
    snapshot.snapshot.controller_generation = 11;
    snapshot.snapshot.mutation_serial = 0;
    snapshot.snapshot.generation.repr_gen = 13;
    snapshot.snapshot.generation.publish_seq = 14;
    snapshot.snapshot.generation.current_type = GGML_TYPE_F16;
    snapshot.snapshot.generation.last_source_type = GGML_TYPE_F16;
    CHECK(vbr_capture_projected_shard_topology(
        sources, snapshot.snapshot.shard_count,
        snapshot.snapshot.shard_topology_digest));

    vbr_capture_stream_status status;
    auto ring = vbr_pinned_chunk_ring::create(
        { {} }, 16, 4, status);
    CHECK(ring);
    CHECK(status == vbr_capture_stream_status::ok);
    vbr_capture_projected_unit captured;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources,
              {},
              snapshot.provider(), *ring, captured) ==
          vbr_capture_stream_status::ok);
    CHECK(snapshot.acquired == 1);
    CHECK(snapshot.rechecked == 1);
    CHECK(snapshot.released == 1);
    CHECK(captured.projection() == projection);
    CHECK(captured.shards().size() == 2);
    CHECK(captured.packed_bytes() == 20);
    CHECK(captured.transfer().bytes == 20);
    CHECK(std::any_of(
        captured.transfer().streaming_digest.begin(),
        captured.transfer().streaming_digest.end(),
        [](uint8_t value) { return value != 0; }));
    // One batch-owned ring operation can serve multiple projected units
    // without reacquiring the direction-neutral transport mutex.
    auto batch_operation = ring->try_begin_operation();
    CHECK(batch_operation);
    projected_snapshot_fixture reserved_snapshot;
    reserved_snapshot.snapshot = captured.snapshot();
    vbr_capture_projected_unit reserved_capture;
    vbr_capture_stream_stats reserved_attempted;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, {},
              reserved_snapshot.provider(), *ring, reserved_capture,
              &reserved_attempted, &batch_operation) ==
          vbr_capture_stream_status::ok);
    CHECK(reserved_snapshot.acquired == 1);
    CHECK(reserved_snapshot.released == 1);
    CHECK(reserved_attempted.bytes == captured.transfer().bytes);
    reserved_snapshot = {};
    reserved_snapshot.snapshot = captured.snapshot();
    reserved_capture = {};
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, {},
              reserved_snapshot.provider(), *ring, reserved_capture,
              nullptr, &batch_operation) ==
          vbr_capture_stream_status::ok);
    CHECK(reserved_snapshot.acquired == 1);
    CHECK(reserved_snapshot.released == 1);
    batch_operation = {};
    auto rebound_sources = sources;
    rebound_sources[0].source.context = &first;
    uint32_t rebound_count = 0;
    std::array<uint8_t, 32> rebound_digest = {};
    CHECK(vbr_capture_projected_shard_topology(
        rebound_sources, rebound_count, rebound_digest));
    CHECK(rebound_digest != captured.snapshot().shard_topology_digest);
    rebound_sources = sources;
    rebound_sources[0].source.tensor_offset = 1;
    CHECK(vbr_capture_projected_shard_topology(
        rebound_sources, rebound_count, rebound_digest));
    CHECK(rebound_digest != captured.snapshot().shard_topology_digest);
    if (captured.shards().size() == 2) {
        CHECK(captured.shards()[0].shard_index == 0);
        CHECK(captured.shards()[1].shard_index == 1);
        CHECK(captured.shards()[0].authenticated_ranges);
        CHECK(captured.shards()[1].authenticated_ranges);
        CHECK(captured.shards()[0].streaming_digest ==
              captured.shards()[0].authenticated_ranges.root());
        CHECK(captured.shards()[1].streaming_digest ==
              captured.shards()[1].authenticated_ranges.root());
        CHECK(captured.shards()[0].authenticated_ranges.total_bytes() == 8);
        CHECK(captured.shards()[1].authenticated_ranges.total_bytes() == 12);
        CHECK(captured.shards()[0].authenticated_ranges.root() !=
              captured.shards()[1].authenticated_ranges.root());
        CHECK(read_chain(*captured.shards()[0].bytes) ==
              projected_rows(first.bytes, 2, { 1, 2, 3, 5 }));
        CHECK(read_chain(*captured.shards()[1].bytes) ==
              projected_rows(second.bytes, 3, { 1, 2, 3, 5 }));
        CHECK(projection->streams[0].segments.size() == 4);
        if (projection->streams[0].segments.size() == 4) {
            CHECK(projection->streams[0].segments[0].packed_first_row == 0);
            CHECK(projection->streams[0].segments[1].packed_first_row == 1);
            CHECK(projection->streams[0].segments[2].packed_first_row == 2);
            CHECK(projection->streams[0].segments[3].packed_first_row == 3);
        }
    }

    // Equal packed bytes under a different manifest dependency geometry must
    // produce a different authenticated unit digest.
    vbr_capture_projection_manifest combined;
    combined.manifest_id = 3;
    combined.placements.push_back(projected_placement(
        3, 3, { 1, 2, 3, 5 }));
    vbr_capture_projection combined_projection;
    CHECK(vbr_artifact_project_capture_union(
        { 91, { combined } }, {}, combined_projection));
    projected_snapshot_fixture combined_snapshot;
    combined_snapshot.snapshot = captured.snapshot();
    vbr_capture_projected_unit combined_capture;
    CHECK(vbr_capture_projected_unit_transfer(
              combined_projection, 0, 0, 7, sources, {},
              combined_snapshot.provider(), *ring, combined_capture) ==
          vbr_capture_stream_status::ok);
    CHECK(combined_capture.packed_bytes() == captured.packed_bytes());
    CHECK(combined_capture.transfer().streaming_digest !=
          captured.transfer().streaming_digest);

    projected_snapshot_fixture other_lineage;
    other_lineage.snapshot = captured.snapshot();
    other_lineage.snapshot.lineage_uuid.lo++;
    vbr_capture_projected_unit lineage_capture;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, {},
              other_lineage.provider(), *ring, lineage_capture) ==
          vbr_capture_stream_status::ok);
    CHECK(lineage_capture.packed_bytes() == captured.packed_bytes());
    CHECK(lineage_capture.transfer().streaming_digest !=
          captured.transfer().streaming_digest);

    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    snapshot.recheck_ok = false;
    vbr_capture_projected_unit changed = captured;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources,
              {},
              snapshot.provider(), *ring, changed) ==
          vbr_capture_stream_status::snapshot_changed);
    CHECK(snapshot.acquired == 1);
    CHECK(snapshot.rechecked == 1);
    CHECK(snapshot.released == 1);
    CHECK(!changed);
    CHECK(changed.shards().empty());

    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    sources[0].source.fail_completion_at = 0;
    vbr_capture_projected_unit failed = captured;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources,
              {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::transfer_failed);
    CHECK(snapshot.acquired == 1);
    CHECK(snapshot.rechecked == 0);
    CHECK(snapshot.released == 1);
    CHECK(!failed);
    CHECK(failed.shards().empty());

    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    snapshot.acquire_ok = false;
    sources[0].source.fail_completion_at = UINT64_MAX;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources,
              {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::snapshot_unavailable);
    CHECK(snapshot.acquired == 1);
    CHECK(snapshot.rechecked == 0);
    CHECK(snapshot.released == 0);

    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    snapshot.snapshot.controller_generation = 0;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources,
              {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::snapshot_unavailable);
    CHECK(snapshot.acquired == 1);
    CHECK(snapshot.rechecked == 0);
    CHECK(snapshot.released == 1);

    // Stable zero serials are valid; odd mutation/publish serials are not.
    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    snapshot.snapshot.mutation_serial = 1;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::snapshot_unavailable);
    CHECK(snapshot.released == 1);
    const auto expect_invalid_snapshot = [&](vbr_capture_unit_snapshot value) {
        projected_snapshot_fixture invalid;
        invalid.snapshot = value;
        CHECK(vbr_capture_projected_unit_transfer(
                  projection, 0, 0, 7, sources, {},
                  invalid.provider(), *ring, failed) ==
              vbr_capture_stream_status::snapshot_unavailable);
        CHECK(invalid.acquired == 1);
        CHECK(invalid.rechecked == 0);
        CHECK(invalid.released == 1);
    };
    auto invalid_snapshot = captured.snapshot();
    invalid_snapshot.generation.last_source_type = -1;
    expect_invalid_snapshot(invalid_snapshot);
    invalid_snapshot = captured.snapshot();
    invalid_snapshot.generation.current_type = GGML_TYPE_COUNT;
    expect_invalid_snapshot(invalid_snapshot);
    invalid_snapshot = captured.snapshot();
    invalid_snapshot.generation.domain = vbr_repr_domain(255);
    expect_invalid_snapshot(invalid_snapshot);
    invalid_snapshot = captured.snapshot();
    invalid_snapshot.generation.last_transition = vbr_repr_transition(255);
    expect_invalid_snapshot(invalid_snapshot);
    invalid_snapshot = captured.snapshot();
    invalid_snapshot.generation.flags = 1;
    expect_invalid_snapshot(invalid_snapshot);
    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    snapshot.snapshot.generation.publish_seq = 15;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::snapshot_unavailable);
    CHECK(snapshot.released == 1);

    // The snapshot authenticates the complete shard set. Reordering is
    // normalized, while omission, sparse/sentinel IDs, and substitution fail.
    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    auto omitted = sources;
    omitted.erase(omitted.begin());
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, omitted, {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::snapshot_unavailable);
    auto substituted = sources;
    substituted[0].source_identity++;
    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, substituted, {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::snapshot_unavailable);
    auto sparse = sources;
    sparse[0].shard_index = 3;
    uint32_t invalid_count = 9;
    std::array<uint8_t, 32> invalid_digest = { 1 };
    CHECK(!vbr_capture_projected_shard_topology(
        sparse, invalid_count, invalid_digest));
    CHECK(invalid_count == 0);
    auto sentinel = sources;
    sentinel[0].shard_index = UINT32_MAX;
    CHECK(!vbr_capture_projected_shard_topology(
        sentinel, invalid_count, invalid_digest));

    vbr_capture_projected_transfer_limits tight;
    tight.max_shards = 1;
    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, tight,
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::projection_invalid);
    CHECK(snapshot.acquired == 0);
    tight = {};
    tight.max_shard_segment_references = 7;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, tight,
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::projection_invalid);
    tight = {};
    tight.max_source_operations = 6;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, tight,
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::projection_invalid);
    CHECK(snapshot.acquired == 0);
    tight.max_source_operations = 7;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, tight,
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::ok);
    CHECK(snapshot.acquired == 1);
    CHECK(snapshot.rechecked == 1);
    CHECK(snapshot.released == 1);
    tight = {};
    tight.max_total_packed_bytes = 19;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, tight,
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::projection_invalid);
    tight = {};
    tight.max_authenticated_chunks = 1;
    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, tight,
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::projection_invalid);
    CHECK(snapshot.acquired == 0);
    tight = {};
    tight.max_authenticated_metadata_bytes = 63;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources, tight,
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::projection_invalid);
    CHECK(snapshot.acquired == 0);

    snapshot = {};
    snapshot.snapshot = captured.snapshot();
    sources[0].row_count = 3;
    CHECK(vbr_capture_projected_unit_transfer(
              projection, 0, 0, 7, sources,
              {},
              snapshot.provider(), *ring, failed) ==
          vbr_capture_stream_status::projection_invalid);
    CHECK(snapshot.acquired == 0);
    CHECK(snapshot.released == 0);
}

struct projected_controller_fixture {
    uint64_t rejected_manifest = 0;
    std::vector<uint64_t> rechecked;

    static bool recheck(
            void * context,
            uint64_t manifest_id,
            const vbr_capture_controller_target * targets,
            size_t target_count) noexcept {
        auto & self = *static_cast<projected_controller_fixture *>(context);
        self.rechecked.push_back(manifest_id);
        if (manifest_id == self.rejected_manifest || !targets ||
            target_count == 0) {
            return false;
        }
        for (size_t i = 0; i < target_count; ++i) {
            if (targets[i].manifest_id != manifest_id ||
                (i != 0 &&
                 targets[i - 1].child_id >= targets[i].child_id)) {
                return false;
            }
        }
        return true;
    }

    vbr_capture_controller_target_provider provider() {
        return { this, recheck };
    }
};

static vbr_unit_generation projected_generation(
        uint64_t repr_gen, ggml_type type = GGML_TYPE_F16) {
    vbr_unit_generation generation;
    generation.repr_gen = repr_gen;
    generation.publish_seq = repr_gen*2;
    generation.current_type = type;
    generation.last_source_type = type;
    return generation;
}

static vbr_capture_controller_target projected_target(
        uint64_t manifest_id,
        uint32_t child_id,
        uint64_t controller_generation,
        const vbr_unit_generation & generation) {
    vbr_capture_controller_target target;
    target.manifest_id = manifest_id;
    target.source_namespace = 707;
    target.child_id = child_id;
    target.lineage_uuid = {
        uint64_t(child_id + 1), controller_generation + 1000,
    };
    target.controller_generation = controller_generation;
    target.units = { generation };
    target.policy.child_id = child_id;
    target.policy.dependency_mode =
        checkpoint_child_dependency_mode::live_guarded;
    target.policy.degrade_order_digest.fill(
        uint8_t(1 + controller_generation + child_id));
    target.policy.policy_digest.fill(
        uint8_t(17 + controller_generation + child_id));
    target.policy.cursor = controller_generation;
    target.policy.floor_type = GGML_TYPE_Q4_0;
    target.policy.pressure_independent_settings = 9;
    target.policy.n_stream = 1;
    target.policy.unified = true;
    target.policy.wm_cells = 8;
    target.policy.current_type_vector_digest =
        vbr_type_vector_digest(std::vector<ggml_type> {
            ggml_type(generation.current_type),
        });
    target.policy.completed_wave = true;
    vbr_artifact_unit_descriptor descriptor;
    descriptor.child_id = child_id;
    descriptor.logical_unit_id = 0;
    descriptor.lineage_uuid = target.lineage_uuid;
    descriptor.repr_gen = generation.repr_gen;
    descriptor.current_type = generation.current_type;
    descriptor.last_source_type = generation.last_source_type;
    descriptor.promote_hops = generation.promote_hops;
    descriptor.last_transition = generation.last_transition;
    descriptor.representation.kind =
        generation.current_type == GGML_TYPE_F16
            ? vbr_artifact_representation_kind::raw
            : vbr_artifact_representation_kind::approximate;
    descriptor.representation.codec_id = 1;
    descriptor.representation.codec_version = 1;
    if (descriptor.representation.kind ==
            vbr_artifact_representation_kind::approximate) {
        descriptor.representation.reference_digest.fill(0x31);
    }
    descriptor.side = vbr_artifact_side::key;
    descriptor.layout = vbr_artifact_layout::row_major;
    descriptor.n_stream = target.policy.n_stream;
    descriptor.unified = target.policy.unified;
    descriptor.wm_cells = target.policy.wm_cells;
    descriptor.rank = 2;
    descriptor.dimensions = { target.policy.wm_cells, 1, 0, 0 };
    descriptor.row_alignment = 1;
    descriptor.row_codec_version = 1;
    descriptor.meansub_model_id = 1;
    descriptor.meansub_layer = int32_t(child_id);
    descriptor.meansub_baked = true;
    vbr_artifact_shard_descriptor shard;
    shard.shard_index = 0;
    shard.topology_index = 0;
    shard.device_ordinal = 0;
    shard.logical_offset = 0;
    shard.row_count = target.policy.wm_cells;
    shard.column_count = 1;
    shard.row_bytes = 1;
    shard.payload_bytes = target.policy.wm_cells;
    descriptor.shards = { shard };
    target.unit_descriptors = { descriptor };
    return target;
}

static void projected_target_streams(
        vbr_capture_controller_target & target,
        uint32_t n_stream) {
    target.policy.n_stream = n_stream;
    target.policy.unified = n_stream == 1;
    for (auto & descriptor : target.unit_descriptors) {
        descriptor.n_stream = n_stream;
        descriptor.unified = n_stream == 1;
    }
}

static void projected_target_row_bytes(
        vbr_capture_controller_target & target,
        uint64_t row_bytes) {
    for (auto & descriptor : target.unit_descriptors) {
        for (auto & shard : descriptor.shards) {
            shard.row_bytes = row_bytes;
            shard.payload_bytes = descriptor.wm_cells*row_bytes;
        }
    }
}

static void bind_projected_manifest_metadata(
        vbr_capture_projection_manifest & manifest,
        const vbr_capture_controller_target & target) {
    manifest.identity_policy_order_digest.fill(
        uint8_t(0x80 + manifest.manifest_id));
    manifest.identity.execution_identity = "projected-test";
    manifest.identity.adapter_config_identity = "adapter";
    manifest.identity.media_content_identity = "media";
    manifest.identity.sequence_epoch = manifest.manifest_id;
    manifest.identity.token_count = 8;
    manifest.identity.next_position = 8;
    manifest.token_block.tokens = {
        1, 2, 3, 4, 5, 6, 7, int32_t(manifest.manifest_id),
    };
    auto & generation = manifest.generation;
    generation.version = 1;
    generation.status = vbr_checkpoint_generation_status::complete;
    generation.identity_policy_order_digest =
        manifest.identity_policy_order_digest;
    vbr_checkpoint_generation_controller controller;
    controller.child_id = target.child_id;
    controller.dependency_mode = target.policy.dependency_mode;
    controller.lineage_uuid = target.lineage_uuid;
    controller.global_generation = target.controller_generation;
    for (const auto & unit : target.units) {
        controller.units.push_back({
            unit.repr_gen, unit.current_type, unit.last_source_type,
            unit.domain, unit.promote_hops, unit.last_transition,
        });
    }
    for (const auto & placement : manifest.placements) {
        vbr_checkpoint_generation_stream stream;
        stream.stream_index = placement.stream_index;
        stream.dependency_seq_id = placement.source_sequence;
        stream.computation_frontier = placement.computation_frontier;
        stream.captured_dependency_count = placement.cells.size();
        for (const auto & cell : placement.cells) {
            const uint32_t page =
                cell.physical_cell/VBR_GENERATION_PAGE_CELLS;
            if (stream.pages.empty() ||
                stream.pages.back().page_index != page) {
                vbr_generation_page_ref ref;
                ref.page_index = page;
                ref.captured_page_gen = 1;
                stream.pages.push_back(ref);
            }
            const uint32_t offset =
                cell.physical_cell%VBR_GENERATION_PAGE_CELLS;
            stream.pages.back().covered_mask[offset/64] |=
                uint64_t(1) << (offset%64);
        }
        controller.streams.push_back(std::move(stream));
    }
    generation.controllers.push_back(std::move(controller));
}

static vbr_capture_projected_unit capture_projected_unit_for_target(
        const vbr_capture_projection & projection,
        const vbr_capture_controller_target & target,
        uint32_t stream_index = 0,
        uint64_t row_bytes = 1,
        uint32_t logical_unit_id = 0,
        uint32_t max_source_operations = 4096,
        uint8_t payload_salt = 0) {
    uint64_t source_rows = target.policy.wm_cells;
    for (const auto & stream : projection->streams) {
        if (stream.child_id != target.child_id ||
            stream.stream_index != stream_index) {
            continue;
        }
        for (const auto & segment : stream.segments) {
            source_rows = std::max<uint64_t>(
                source_rows,
                uint64_t(segment.first_physical_cell) + segment.cell_count);
        }
    }
    CHECK(source_rows <= UINT32_MAX);
    synthetic_source source;
    source.bytes.resize(size_t(source_rows*row_bytes));
    for (size_t i = 0; i < source.bytes.size(); ++i) {
        source.bytes[i] = uint8_t(
            i + target.child_id + target.controller_generation + payload_salt);
    }
    vbr_capture_projected_shard_source shard;
    shard.shard_index = 0;
    shard.row_count = uint32_t(source_rows);
    shard.row_bytes = row_bytes;
    shard.source_identity =
        target.controller_generation*10 + target.child_id + 1;
    shard.source.size = source.bytes.size();
    shard.source.context = &source;
    shard.source.read = read_synthetic;
    std::vector<vbr_capture_projected_shard_source> sources { shard };

    projected_snapshot_fixture snapshot;
    snapshot.snapshot.source_namespace = target.source_namespace;
    snapshot.snapshot.child_id = target.child_id;
    snapshot.snapshot.logical_unit_id = logical_unit_id;
    snapshot.snapshot.lineage_uuid = target.lineage_uuid;
    snapshot.snapshot.controller_generation = target.controller_generation;
    snapshot.snapshot.mutation_serial = 0;
    snapshot.snapshot.generation = target.units[logical_unit_id];
    CHECK(vbr_capture_projected_shard_topology(
        sources, snapshot.snapshot.shard_count,
        snapshot.snapshot.shard_topology_digest));
    vbr_capture_stream_status status;
    const size_t chunk_bytes = row_bytes == 1 ? 4 :
        VBR_CAPTURE_RANGE_CHUNK_BYTES;
    auto ring = vbr_pinned_chunk_ring::create(
        { {} }, 2*chunk_bytes, chunk_bytes, status);
    CHECK(ring);
    vbr_capture_projected_unit unit;
    if (ring) {
        vbr_capture_projected_transfer_limits limits;
        limits.max_source_operations = max_source_operations;
        CHECK(vbr_capture_projected_unit_transfer(
                  projection, target.child_id, stream_index, logical_unit_id,
                  sources, limits, snapshot.provider(), *ring, unit) ==
              vbr_capture_stream_status::ok);
    }
    return unit;
}

static const vbr_capture_manifest_result * projected_manifest(
        const vbr_capture_manifest_assembly & assembly,
        uint64_t manifest_id) {
    const auto & manifests = assembly.manifests();
    const auto found = std::find_if(
        manifests.begin(), manifests.end(),
        [&](const auto & value) { return value.manifest_id == manifest_id; });
    return found == manifests.end() ? nullptr : &*found;
}

static bool assemble_projected_test_batch(
        const vbr_capture_projection & projection,
        const std::vector<vbr_capture_controller_target> & targets,
        const std::vector<vbr_capture_projected_unit> & units,
        const vbr_capture_controller_target_provider & provider,
        const vbr_capture_manifest_assembly_limits & limits,
        vbr_capture_manifest_assembly & output) {
    auto owned_targets = targets;
    auto owned_units = units;
    return vbr_capture_assemble_manifests(
        projection, std::move(owned_targets), std::move(owned_units),
        provider, limits, output);
}

static void test_preflight_unavailable_projection_rows() {
    auto target = projected_target(
        1, 0, 101, projected_generation(5));
    vbr_capture_projection_manifest available;
    available.manifest_id = 1;
    available.placements.push_back(
        projected_placement(1, 1, { 1, 2 }));
    bind_projected_manifest_metadata(available, target);
    auto unavailable = available;
    unavailable.manifest_id = 2;
    unavailable.dependencies_available = false;
    unavailable.placements.clear();

    vbr_capture_projection mixed;
    CHECK(vbr_artifact_project_capture_union(
        { 707, { unavailable, available } }, {}, mixed));
    CHECK(mixed);
    CHECK(mixed->manifest_count == 2);
    CHECK(mixed->union_cell_count == 2);
    CHECK(mixed->streams.size() == 1);
    auto unit = capture_projected_unit_for_target(mixed, target);
    projected_controller_fixture provider;
    vbr_capture_manifest_assembly assembled;
    CHECK(assemble_projected_test_batch(
        mixed, { target }, { unit }, provider.provider(), {}, assembled));
    CHECK(provider.rechecked == std::vector<uint64_t>({ 1 }));
    CHECK(projected_manifest(assembled, 1)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 2)->state ==
          vbr_capture_manifest_state::dependency_unavailable);

    auto unavailable_three = unavailable;
    unavailable_three.manifest_id = 3;
    vbr_capture_projection all_unavailable;
    CHECK(vbr_artifact_project_capture_union(
        { 707, { unavailable, unavailable_three } }, {}, all_unavailable));
    CHECK(all_unavailable);
    CHECK(all_unavailable->union_cell_count == 0);
    CHECK(all_unavailable->streams.empty());
    provider = {};
    CHECK(assemble_projected_test_batch(
        all_unavailable, {}, {}, provider.provider(), {}, assembled));
    CHECK(provider.rechecked.empty());
    CHECK(assembled.manifests().size() == 2);
    CHECK(std::all_of(
        assembled.manifests().begin(), assembled.manifests().end(),
        [](const auto & manifest) {
            return manifest.state ==
                vbr_capture_manifest_state::dependency_unavailable;
        }));

    auto invalid = unavailable;
    invalid.placements.push_back(projected_placement(2, 2, { 3 }));
    vbr_capture_projection refused = mixed;
    CHECK(!vbr_artifact_project_capture_union(
        { 707, { invalid } }, {}, refused));
    CHECK(!refused);
    invalid = unavailable;
    invalid.dependencies_available = true;
    CHECK(!vbr_artifact_project_capture_union(
        { 707, { invalid } }, {}, refused));
    CHECK(!refused);
}

static vbr_artifact_portable_topology capture_test_topology();

static vbr_projected_manifest_publication projected_publication(
        uint64_t manifest_id,
        const vbr_capture_manifest_assembly & assembly,
        const vbr_artifact_portable_topology & topology) {
    vbr_projected_manifest_publication out;
    out.manifest_id = manifest_id;
    out.topologies = { topology };
    const auto * row = projected_manifest(assembly, manifest_id);
    CHECK(row != nullptr);
    if (!row || row->state != vbr_capture_manifest_state::ready) {
        return out;
    }
    if (row->unit_count == 0 ||
        row->first_unit > assembly.unit_references().size() ||
        row->unit_count >
            assembly.unit_references().size() - row->first_unit) {
        return out;
    }
    uint64_t packed_bytes = 0;
    for (uint32_t i = 0; i < row->unit_count; ++i) {
        const uint32_t captured_index = assembly.unit_references()[
            row->first_unit + i];
        if (captured_index >= assembly.projected_units().size() ||
            assembly.projected_units()[captured_index].packed_bytes() >
                UINT64_MAX - packed_bytes) {
            return out;
        }
        packed_bytes +=
            assembly.projected_units()[captured_index].packed_bytes();
    }
    vbr_artifact_portable_accounting_row payload;
    payload.role = vbr_artifact_accounting_role::unit_payload;
    payload.domain = vbr_artifact_payload_storage_domain();
    payload.logical_bytes = packed_bytes;
    payload.resident_bytes = packed_bytes;
    out.accounting.push_back(payload);
    const vbr_artifact_portable_domain host {
        llama_cache_acct_residency::pageable_host,
        llama_cache_acct_domain_kind::not_applicable,
        UINT32_MAX, UINT16_MAX,
    };
    out.accounting.push_back({
        vbr_artifact_accounting_role::descriptor_metadata,
        host, 512, 512, llama_cache_acct_attr_kind::artifact,
    });
    out.accounting.push_back({
        vbr_artifact_accounting_role::reference_metadata,
        host, 256, 256, llama_cache_acct_attr_kind::artifact,
    });
    return out;
}

struct occupied_guard_fixture {
    vbr_target_validation_snapshot target;
    std::vector<vbr_occupied_replacement_cell> cells;
    std::vector<vbr_occupied_replacement_unit_currency> units;
    vbr_occupied_replacement_observation observation;

    void bind() noexcept {
        observation.cells = cells.data();
        observation.cell_count = cells.size();
        observation.units = units.data();
        observation.unit_count = units.size();
    }
};

static bool make_occupied_guard_fixture(
        const vbr_artifact_package_view & recovery,
        llama_seq_id destination,
        uint32_t capacity,
        occupied_guard_fixture & out) {
    out = {};
    if (!recovery || recovery.units().empty() ||
        recovery.manifest().generation.controllers.size() != 1 ||
        recovery.manifest().controller_policy.size() != 1 ||
        recovery.manifest().stream_placements.size() != 1) {
        return false;
    }
    static const uint8_t memory_cookie = 0;
    static const uint8_t pool_cookie = 0;
    const auto & controller =
        recovery.manifest().generation.controllers.front();
    const auto & placement =
        recovery.manifest().stream_placements.front();
    out.target.memory_instance_cookie = 0x4101;
    out.target.target_state_serial = 0x4102;
    out.target.accounting_serial = 0x4103;
    out.target.tree_shape_digest = 0x4104;
    out.target.policy_epoch = 0x4105;
    out.target.scheduler_idle = true;
    out.target.destination_sequence_absent = false;
    vbr_target_child_snapshot child;
    child.child_id = 0;
    child.dependency_mode = checkpoint_child_dependency_mode::live_guarded;
    child.memory_cookie = &memory_cookie;
    child.empty = false;
    child.dedicated = true;
    child.armed = true;
    child.lineage_uuid = controller.lineage_uuid;
    child.instance_id = { 0x4111, 0x4112 };
    child.state_serial = out.target.target_state_serial;
    child.policy_epoch = out.target.policy_epoch;
    child.controller_policy =
        recovery.manifest().controller_policy.front();
    for (size_t i = 0; i < recovery.units().size(); ++i) {
        const auto & descriptor = recovery.units()[i].descriptor;
        if (descriptor.logical_unit_id != i ||
            descriptor.child_id != 0 || i >= controller.units.size()) {
            return false;
        }
        const auto & generation = controller.units[i];
        vbr_target_unit_snapshot unit;
        unit.child_id = descriptor.child_id;
        unit.logical_unit_id = descriptor.logical_unit_id;
        unit.current_type = descriptor.current_type;
        unit.last_source_type = descriptor.last_source_type;
        unit.promote_hops = descriptor.promote_hops;
        unit.last_transition = descriptor.last_transition;
        unit.representation_kind = descriptor.representation.kind;
        unit.codec_id = descriptor.representation.codec_id;
        unit.codec_version = descriptor.representation.codec_version;
        unit.representation_reference_digest =
            descriptor.representation.reference_digest;
        unit.source_loss_history =
            descriptor.representation.source_loss_history;
        unit.checkpoint_codec_hops =
            descriptor.representation.checkpoint_codec_hops;
        unit.recoverability = descriptor.recoverability;
        unit.side = descriptor.side;
        unit.layout = descriptor.layout;
        unit.row_codec_version = descriptor.row_codec_version;
        unit.current_domain = generation.domain;
        unit.codebook_digest = descriptor.codebook_digest;
        unit.rotation_digest = descriptor.rotation_digest;
        unit.meansub_digest = descriptor.meansub_digest;
        unit.meansub_model_id = descriptor.meansub_model_id;
        unit.meansub_layer = descriptor.meansub_layer;
        unit.meansub_baked = descriptor.meansub_baked;
        unit.n_stream = descriptor.n_stream;
        unit.unified = descriptor.unified;
        unit.wm_cells = descriptor.wm_cells;
        unit.rank = descriptor.rank;
        unit.dimensions = descriptor.dimensions;
        unit.row_alignment = descriptor.row_alignment;
        for (const auto & shard : descriptor.shards) {
            if (shard.topology_index >= recovery.topologies().size()) {
                return false;
            }
            unit.shards.push_back({
                shard.shard_index, &pool_cookie, {},
                shard.topology_index, shard.device_ordinal,
                recovery.topologies()[shard.topology_index].digest,
                shard.logical_offset, shard.row_count, shard.row_bytes,
                shard.payload_bytes,
            });
        }
        child.units.push_back(std::move(unit));
        vbr_unit_generation live;
        live.repr_gen = generation.repr_gen;
        live.publish_seq = 0x4200 + i;
        live.current_type = generation.current_type;
        live.last_source_type = generation.last_source_type;
        live.domain = generation.domain;
        live.promote_hops = generation.promote_hops;
        live.last_transition = generation.last_transition;
        out.units.push_back({ 0, uint32_t(i), live });
    }
    out.target.children.push_back(std::move(child));
    out.cells.reserve(placement.cells.size());
    for (const auto & cell : placement.cells) {
        out.cells.push_back({
            0, cell.physical_cell, cell.logical_position,
            cell.ext_x, cell.ext_y, destination, 1, true,
        });
    }
    std::sort(out.cells.begin(), out.cells.end(),
        [](const auto & a, const auto & b) {
            return a.physical_cell < b.physical_cell;
        });
    out.observation.destination = destination;
    out.observation.sequence_epoch =
        recovery.manifest().identity.sequence_epoch;
    out.observation.controller_generation = controller.global_generation;
    out.observation.representation_epoch =
        out.target.children.front().state_serial;
    out.observation.cell_capacity = capacity;
    out.bind();
    return true;
}

static bool publish_occupied_guard_package(
        llama_vbr_artifact_catalog & catalog,
        const vbr_artifact_portable_topology & topology,
        const llama_cache_budget_config & budget,
        uint64_t manifest_id,
        uint32_t token_count,
        bool alternating_physical_cells,
        llama_cache_acct_artifact_id & reference,
        vbr_artifact_package_view & view,
        uint64_t schedule_seed = 0,
        uint64_t source_capacity_override = 0,
        uint32_t proof_count = 1,
        uint8_t payload_salt = 0,
        ggml_type representation_type = GGML_TYPE_F16) {
    reference = {};
    view.reset();
    if (token_count == 0 || proof_count == 0 || proof_count > 4096) {
        return false;
    }
    std::vector<uint32_t> physical(token_count);
    for (uint32_t i = 0; i < token_count; ++i) {
        physical[i] = alternating_physical_cells ? 2*i + 1 : i;
    }
    const uint64_t seed = schedule_seed == 0 ? manifest_id : schedule_seed;
    auto target = projected_target(
        manifest_id, 0, 5000 + seed,
        projected_generation(6000 + seed, representation_type));
    const uint64_t source_capacity = source_capacity_override != 0
        ? source_capacity_override
        : alternating_physical_cells ? uint64_t(token_count)*2 : token_count;
    target.policy.wm_cells = source_capacity;
    target.policy.current_type_vector_digest =
        vbr_type_vector_digest(
            std::vector<ggml_type> { representation_type });
    auto & descriptor = target.unit_descriptors.front();
    descriptor.wm_cells = source_capacity;
    descriptor.dimensions[0] = source_capacity;
    descriptor.shards.front().row_count = source_capacity;
    descriptor.shards.front().payload_bytes = source_capacity;
    const auto generation = target.units.front();
    const auto descriptor_template = descriptor;
    target.units.resize(proof_count, generation);
    target.unit_descriptors.resize(proof_count, descriptor_template);
    std::vector<ggml_type> types(proof_count, representation_type);
    target.policy.current_type_vector_digest = vbr_type_vector_digest(types);
    for (uint32_t i = 0; i < proof_count; ++i) {
        target.unit_descriptors[i].logical_unit_id = i;
    }

    vbr_capture_projection_manifest manifest;
    manifest.manifest_id = manifest_id;
    manifest.placements.push_back(projected_placement(
        manifest_id, llama_seq_id(manifest_id), physical));
    bind_projected_manifest_metadata(manifest, target);
    manifest.identity.token_count = token_count;
    manifest.identity.next_position = token_count;
    manifest.token_block.tokens.resize(token_count);
    for (uint32_t i = 0; i < token_count; ++i) {
        manifest.token_block.tokens[i] = llama_token(i + 1);
    }
    vbr_capture_projection projection;
    if (!vbr_artifact_project_capture_union(
            { 707, { manifest } }, {}, projection)) {
        fprintf(stderr, "occupied package projection failed: %" PRIu64 "\n",
                manifest_id);
        return false;
    }
    std::vector<vbr_capture_projected_unit> units;
    units.reserve(proof_count);
    for (uint32_t i = 0; i < proof_count; ++i) {
        units.push_back(capture_projected_unit_for_target(
            projection, target, 0, 1, i,
            std::max<uint32_t>(4096, token_count), payload_salt));
    }
    vbr_capture_manifest_assembly assembly;
    projected_controller_fixture controller;
    if (!assemble_projected_test_batch(
            projection, { target }, units, controller.provider(), {},
            assembly)) {
        fprintf(stderr, "occupied package assembly failed: %" PRIu64 "\n",
                manifest_id);
        return false;
    }
    if (assembly.manifests().empty() ||
        assembly.manifests().front().state !=
            vbr_capture_manifest_state::ready) {
        fprintf(stderr,
                "occupied package not ready: %" PRIu64 " state=%u\n",
                manifest_id, assembly.manifests().empty() ? UINT32_MAX :
                    uint32_t(assembly.manifests().front().state));
    }
    std::vector<vbr_projected_manifest_publication> publications;
    publications.push_back(projected_publication(
        manifest_id, assembly, topology));
    std::vector<vbr_projected_manifest_publish_result> results;
    if (!catalog.publish_projected_batch(
            assembly, std::move(publications), budget, results) ||
        results.size() != 1 ||
        results.front().status !=
            vbr_projected_manifest_publish_status::published) {
        fprintf(stderr,
                "occupied package publication failed: %" PRIu64
                " size=%zu status=%u\n",
                manifest_id, results.size(),
                results.empty() ? UINT32_MAX :
                    uint32_t(results.front().status));
        return false;
    }
    reference = results.front().publication.reference_artifact;
    return catalog.resolve_reference(reference, view) ==
        vbr_artifact_resolve_status::ok;
}

static void test_projected_publication_claim_preparation() {
    llama_cache_acct_ledger ledger;
    llama_vbr_artifact_catalog catalog(ledger);
    const auto topology = capture_test_topology();
    std::vector<llama_vbr_artifact_domain_binding> bindings;
    CHECK(catalog.bind_topologies({ topology }, bindings));
    CHECK(bindings.size() == 1);
    const auto device = bindings.front().domain;
    const auto host = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    const llama_cache_acct_completeness_requirement requirements[] {
        { device, llama_cache_acct_producer::live_memory },
        { host, llama_cache_acct_producer::retention_sidecar },
    };
    CHECK(ledger.configure_required_producers(requirements, 2));
    for (const auto category : {
            llama_cache_acct_category::live_attention_state,
            llama_cache_acct_category::live_recurrent_state,
            llama_cache_acct_category::recurrent_rollback_planes,
            llama_cache_acct_category::rolling_window_tape }) {
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated,
                llama_cache_acct_measure::reserved }) {
            ledger.gauge_set(category, device, measure, 0);
        }
    }
    for (const auto category : {
            llama_cache_acct_category::full_snapshot_payload,
            llama_cache_acct_category::checkpoint_state_payload,
            llama_cache_acct_category::typed_accelerator_payload }) {
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated,
                llama_cache_acct_measure::reserved }) {
            ledger.gauge_set(category, host, measure, 0);
        }
    }
    CHECK(ledger.certify_complete(
        device, llama_cache_acct_producer::live_memory));
    CHECK(ledger.certify_complete(
        host, llama_cache_acct_producer::retention_sidecar));

    const vbr_artifact_portable_domain portable_host {
        llama_cache_acct_residency::pageable_host,
        llama_cache_acct_domain_kind::not_applicable,
        UINT32_MAX, UINT16_MAX,
    };
    std::vector<vbr_artifact_portable_accounting_row> rows {
        { vbr_artifact_accounting_role::unit_payload,
          portable_host, 64, 64,
          llama_cache_acct_attr_kind::artifact },
        { vbr_artifact_accounting_role::descriptor_metadata,
          portable_host, 16, 16,
          llama_cache_acct_attr_kind::artifact },
        { vbr_artifact_accounting_role::reference_metadata,
          portable_host, 8, 8,
          llama_cache_acct_attr_kind::artifact },
    };
    llama_cache_budget_config budget;
    llama_cache_budget_device_input input;
    input.backend_device = reinterpret_cast<const void *>(uintptr_t(1));
    input.domain = device;
    input.physical_total = 1ull << 30;
    input.physical_free = 1ull << 29;
    input.phys_state = llama_cache_budget_capacity_state::known;
    input.current_compute_allocated = 0;
    input.configured_compute_reserve = 0;
    input.compute_state = llama_cache_budget_capacity_state::known;
    input.cache_cap_state = llama_cache_budget_capacity_state::unbounded;
    budget.devices.push_back(input);
    const auto baseline = ledger.snapshot().live_ops;
    auto claim = catalog.prepare_projected_publication_claim(1, rows, budget);
    CHECK(claim.ready());
    CHECK(claim.preparation().status ==
          llama_cache_prepare_status::prepared);
    CHECK(ledger.snapshot().live_ops == baseline + rows.size());

    llama_vbr_projected_publication_claim moved(std::move(claim));
    CHECK(!claim.ready());
    CHECK(moved.ready());
    moved = {};
    CHECK(ledger.snapshot().live_ops == baseline);

    std::vector<llama_vbr_projected_publication_request> batch_requests;
    for (uint64_t manifest_id = 1; manifest_id <= 4; ++manifest_id) {
        batch_requests.push_back({ manifest_id, rows, {}, false });
    }
    auto batch = catalog.prepare_projected_publication_claims(
        batch_requests, budget);
    CHECK(batch.ready());
    CHECK(batch.manifests() == 4);
    CHECK(ledger.snapshot().live_ops == baseline + rows.size()*4);
    const auto reserved_total = [&] (
            llama_cache_acct_category category,
            const llama_cache_acct_resource_domain & domain) {
        const auto snapshot = ledger.snapshot();
        const auto found = std::find_if(
            snapshot.cells.begin(), snapshot.cells.end(),
            [&](const auto & cell) {
                return cell.category == category && cell.domain == domain;
            });
        return found == snapshot.cells.end() ? UINT64_MAX :
            found->cell.measures[size_t(
                llama_cache_acct_measure::reserved)].value;
    };
    CHECK(reserved_total(
              llama_cache_acct_category::unit_version_payload,
              host) == 64);
    CHECK(reserved_total(
              llama_cache_acct_category::artifact_descriptor_metadata,
              host) == 16*4);

    std::vector<llama_vbr_projected_publication_request> survivors {
        batch_requests[1], batch_requests[3],
    };
    CHECK(catalog.shrink_projected_publication_claims(batch, survivors));
    CHECK(batch.ready());
    CHECK(batch.manifests() == 2);
    CHECK(ledger.snapshot().live_ops == baseline + rows.size()*2);
    CHECK(reserved_total(
              llama_cache_acct_category::unit_version_payload,
              host) == 64);
    CHECK(reserved_total(
              llama_cache_acct_category::artifact_descriptor_metadata,
              host) == 16*2);
    std::vector<llama_vbr_projected_publication_claim> split;
    CHECK(catalog.partition_projected_publication_claims(
        std::move(batch), split));
    CHECK(!batch.ready());
    CHECK(split.size() == 2);
    CHECK(split[0].manifest_id() == 2);
    CHECK(split[1].manifest_id() == 4);
    CHECK(split[0].ready());
    CHECK(split[1].ready());
    split.clear();
    CHECK(ledger.snapshot().live_ops == baseline);

    auto under_reserved = batch_requests;
    for (auto & request : under_reserved) {
        request.reserve_accounting_explicit = true;
    }
    auto under_reserved_claim =
        catalog.prepare_projected_publication_claims(
            under_reserved, budget);
    CHECK(!under_reserved_claim.ready());
    CHECK(ledger.snapshot().live_ops == baseline);

    auto over_reserved = batch_requests;
    over_reserved.front().reserve_accounting = { rows.front() };
    over_reserved.front().reserve_accounting.front().logical_bytes++;
    over_reserved.front().reserve_accounting.front().resident_bytes++;
    over_reserved.front().reserve_accounting_explicit = true;
    auto over_reserved_claim =
        catalog.prepare_projected_publication_claims(
            over_reserved, budget);
    CHECK(!over_reserved_claim.ready());
    CHECK(ledger.snapshot().live_ops == baseline);

    auto duplicate_rows = rows;
    duplicate_rows.push_back(rows.front());
    auto duplicate = catalog.prepare_projected_publication_claim(
        1, duplicate_rows, budget);
    CHECK(!duplicate.ready());
    CHECK(duplicate.preparation().status ==
          llama_cache_prepare_status::invalid_argument);
    CHECK(ledger.snapshot().live_ops == baseline);

    std::vector<vbr_artifact_portable_accounting_row> duplicate_scale(
        16384, rows.front());
    auto duplicate_scale_claim =
        catalog.prepare_projected_publication_claim(
            1, duplicate_scale, budget);
    CHECK(!duplicate_scale_claim.ready());
    CHECK(duplicate_scale_claim.preparation().status ==
          llama_cache_prepare_status::invalid_argument);
    CHECK(ledger.snapshot().live_ops == baseline);

    auto mismatched_rows = rows;
    mismatched_rows.front().resident_bytes--;
    auto mismatched = catalog.prepare_projected_publication_claim(
        1, mismatched_rows, budget);
    CHECK(!mismatched.ready());
    CHECK(mismatched.preparation().status ==
          llama_cache_prepare_status::invalid_argument);
    CHECK(ledger.snapshot().live_ops == baseline);

    auto wrong_kind_rows = rows;
    wrong_kind_rows.front().domain.kind =
        llama_cache_acct_domain_kind::device_topology;
    auto wrong_kind = catalog.prepare_projected_publication_claim(
        1, wrong_kind_rows, budget);
    CHECK(!wrong_kind.ready());
    CHECK(wrong_kind.preparation().status ==
          llama_cache_prepare_status::invalid_argument);
    CHECK(ledger.snapshot().live_ops == baseline);

    auto wrong_topology_rows = rows;
    wrong_topology_rows.front().domain.topology_index = 1;
    auto wrong_topology = catalog.prepare_projected_publication_claim(
        1, wrong_topology_rows, budget);
    CHECK(!wrong_topology.ready());
    CHECK(wrong_topology.preparation().status ==
          llama_cache_prepare_status::invalid_argument);
    CHECK(ledger.snapshot().live_ops == baseline);

    auto malformed_host_rows = rows;
    malformed_host_rows[1].domain.topology_index = 0;
    auto malformed_host = catalog.prepare_projected_publication_claim(
        1, malformed_host_rows, budget);
    CHECK(!malformed_host.ready());
    CHECK(malformed_host.preparation().status ==
          llama_cache_prepare_status::invalid_argument);
    CHECK(ledger.snapshot().live_ops == baseline);

    auto overflow_rows = rows;
    overflow_rows[1].logical_bytes = UINT64_MAX;
    overflow_rows[1].resident_bytes = UINT64_MAX;
    auto overflow = catalog.prepare_projected_publication_claim(
        1, overflow_rows, budget);
    CHECK(!overflow.ready());
    CHECK(overflow.preparation().status ==
          llama_cache_prepare_status::invalid_argument);
    CHECK(ledger.snapshot().live_ops == baseline);
}

static void test_dependency_scoped_projected_catalog_publication() {
    static_assert(!std::is_copy_constructible_v<
        vbr_capture_sealed_companion>);
    static_assert(std::is_nothrow_move_constructible_v<
        vbr_capture_sealed_companion>);
    std::vector<vbr_capture_projection_manifest> manifests;
    std::vector<vbr_capture_controller_target> targets;
    for (uint64_t i = 1; i <= 8; ++i) {
        vbr_capture_projection_manifest manifest;
        manifest.manifest_id = i;
        manifest.placements.push_back(projected_placement(
            i, llama_seq_id(i), { uint32_t(i - 1) }));
        auto target = projected_target(
            i, 0, 500 + i, projected_generation(20 + i));
        bind_projected_manifest_metadata(manifest, target);
        if (i == 6 || i == 8) {
            vbr_artifact_companion_payload companion;
            companion.kind = vbr_artifact_companion_kind::recurrent;
            companion.format_version = 1;
            companion.build_identity_digest.fill(0x6a);
            companion.domain = {
                llama_cache_acct_residency::pageable_host,
                llama_cache_acct_domain_kind::not_applicable,
                UINT32_MAX, UINT16_MAX,
            };
            companion.payload_bytes = 4;
            manifest.companions.push_back(companion);
        }
        manifests.push_back(std::move(manifest));
        targets.push_back(std::move(target));
    }
    vbr_capture_projection projection;
    CHECK(vbr_artifact_project_capture_union(
        { 707, std::move(manifests) }, {}, projection));
    std::vector<vbr_capture_projected_unit> units;
    for (const auto & target : targets) {
        units.push_back(
            capture_projected_unit_for_target(projection, target));
    }
    projected_controller_fixture controller;
    controller.rejected_manifest = 7;
    vbr_capture_manifest_assembly assembly;
    CHECK(assemble_projected_test_batch(
        projection, targets, units, controller.provider(), {}, assembly));
    CHECK(assembly.manifests().size() == 8);
    CHECK(projected_manifest(assembly, 7)->state ==
          vbr_capture_manifest_state::dependency_unavailable);

    const auto topology = capture_test_topology();
    const uint8_t recurrent_state[] { 9, 8, 7, 6 };
    const auto make_publications = [&]() {
        std::vector<vbr_projected_manifest_publication> values;
        for (uint64_t i = 1; i <= 8; ++i) {
            values.push_back(projected_publication(i, assembly, topology));
        }
        // Manifest 8 declares one required companion but supplies no sealed
        // evidence. It must not poison the six independent publications.
        // Manifest 6 carries real companion evidence. The projected metadata
        // authority hashes and accounts it; malformed bytes cannot hide
        // behind a caller-provided nonzero digest.
        auto & companion_publication = values[5];
        vbr_artifact_companion_payload companion;
        companion.kind = vbr_artifact_companion_kind::recurrent;
        companion.format_version = 1;
        companion.build_identity_digest.fill(0x6a);
        companion.domain = {
            llama_cache_acct_residency::pageable_host,
            llama_cache_acct_domain_kind::not_applicable,
            UINT32_MAX, UINT16_MAX,
        };
        companion.payload_bytes = sizeof(recurrent_state);
        companion_publication.accounting.push_back({
            vbr_artifact_accounting_role::recurrent_payload,
            companion.domain,
            companion.payload_bytes, companion.payload_bytes,
            llama_cache_acct_attr_kind::artifact,
        });
        auto companion_bytes = std::make_unique<artifact_segment_chain>();
        CHECK(companion_bytes->append(
            recurrent_state, sizeof(recurrent_state)));
        vbr_capture_sealed_companion sealed;
        CHECK(vbr_capture_seal_companion(
            0, std::move(companion_bytes), sealed));
        companion_publication.companions.push_back(std::move(sealed));
        return values;
    };

    llama_cache_acct_ledger ledger;
    llama_vbr_artifact_catalog catalog(ledger);
    std::vector<llama_vbr_artifact_domain_binding> bindings;
    CHECK(catalog.bind_topologies({ topology }, bindings));
    CHECK(bindings.size() == 1);

    // Whole-inventory structural refusal is preflight-only. A mismatched late
    // ID must not publish the earlier matching rows or charge the ledger.
    auto malformed_inventory = make_publications();
    malformed_inventory.back().manifest_id = 9;
    std::vector<vbr_projected_manifest_publish_result> malformed_results;
    CHECK(!catalog.publish_projected_batch(
        assembly, std::move(malformed_inventory), {}, malformed_results));
    CHECK(malformed_results.empty());
    CHECK(catalog.snapshot().references == 0);
    CHECK(ledger.snapshot().live_ops == 0);
    auto publications = make_publications();
    llama_cache_acct_resource_domain device;
    CHECK(ledger.make_device_domain(
        topology, llama_cache_acct_device_ordinal { 0 }, device));
    const auto host = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    const auto pinned = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pinned_host);
    const llama_cache_acct_completeness_requirement requirements[] {
        { device, llama_cache_acct_producer::live_memory },
        { host, llama_cache_acct_producer::retention_sidecar },
        { pinned, llama_cache_acct_producer::retention_sidecar },
    };
    CHECK(ledger.configure_required_producers(requirements, 3));
    vbr_artifact_package accounting_package;
    accounting_package.topologies = publications.front().topologies;
    accounting_package.manifest.accounting =
        publications.front().accounting;
    CHECK(catalog.configure_accounting(accounting_package));
    for (const auto category : {
            llama_cache_acct_category::live_attention_state,
            llama_cache_acct_category::live_recurrent_state,
            llama_cache_acct_category::recurrent_rollback_planes,
            llama_cache_acct_category::rolling_window_tape,
            llama_cache_acct_category::transfer_staging }) {
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated,
                llama_cache_acct_measure::reserved }) {
            ledger.gauge_set(category, device, measure, 0);
        }
    }
    for (const auto category : {
            llama_cache_acct_category::full_snapshot_payload,
            llama_cache_acct_category::checkpoint_state_payload,
            llama_cache_acct_category::typed_accelerator_payload }) {
        ledger.gauge_set(
            category, host,
            llama_cache_acct_measure::resident_allocated, 0);
        ledger.gauge_set(
            category, host,
            llama_cache_acct_measure::reserved, 0);
    }
    for (uint8_t raw = 0;
         raw < uint8_t(llama_cache_acct_category::_count); ++raw) {
        const auto category = llama_cache_acct_category(raw);
        const auto classification = llama_cache_budget_classify(category);
        if (classification.participation !=
                llama_cache_budget_capacity_participation::participating ||
            (classification.scope !=
                 llama_cache_budget_residency_scope::host &&
             classification.scope !=
                 llama_cache_budget_residency_scope::by_domain)) {
            continue;
        }
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated,
                llama_cache_acct_measure::reserved }) {
            ledger.gauge_set(category, pinned, measure, 0);
        }
    }
    CHECK(ledger.certify_complete(
        device, llama_cache_acct_producer::live_memory));
    CHECK(ledger.certify_complete(
        host, llama_cache_acct_producer::retention_sidecar));
    CHECK(ledger.certify_complete(
        pinned, llama_cache_acct_producer::retention_sidecar));
    llama_cache_budget_config budget;
    llama_cache_budget_device_input input;
    input.backend_device = reinterpret_cast<const void *>(uintptr_t(1));
    input.domain = device;
    input.physical_total = 1ull << 30;
    input.physical_free = 1ull << 29;
    input.phys_state = llama_cache_budget_capacity_state::known;
    input.current_compute_allocated = 0;
    input.configured_compute_reserve = 0;
    input.compute_state = llama_cache_budget_capacity_state::known;
    input.cache_cap_state = llama_cache_budget_capacity_state::unbounded;
    budget.devices.push_back(input);
    budget.host.pageable_state =
        llama_cache_budget_capacity_state::unbounded;
    budget.host.pinned_state =
        llama_cache_budget_capacity_state::unbounded;
    budget.host.total_state =
        llama_cache_budget_capacity_state::unbounded;
    budget.global_cap_state =
        llama_cache_budget_capacity_state::unbounded;
    std::vector<llama_vbr_projected_publication_claim> publication_claims;
    for (size_t i = 0; i < assembly.manifests().size(); ++i) {
        if (assembly.manifests()[i].state !=
                vbr_capture_manifest_state::ready) {
            continue;
        }
        auto claim = catalog.prepare_projected_publication_claim(
            assembly.manifests()[i].manifest_id,
            publications[i].accounting, budget);
        CHECK(claim.ready());
        publication_claims.push_back(std::move(claim));
    }
    CHECK(publication_claims.size() == 7);
    std::vector<vbr_projected_manifest_publish_result> results;
    vbr_projected_batch_publish_diagnostics diagnostics;
    CHECK(catalog.publish_projected_batch_claimed(
        assembly, std::move(publications), std::move(publication_claims),
        results, &diagnostics));
    CHECK(results.size() == 8);
    CHECK(diagnostics.ready_manifests == 7);
    CHECK(diagnostics.published_manifests == 6);
    CHECK(diagnostics.dependency_unavailable == 1);
    CHECK(diagnostics.main_payload_bytes_rehashed == 0);
    CHECK(diagnostics.companion_payload_hash_bytes ==
          3*sizeof(recurrent_state));
    CHECK(results[6].status ==
          vbr_projected_manifest_publish_status::dependency_unavailable);
    CHECK(results[7].status ==
          vbr_projected_manifest_publish_status::companion_unavailable);
    const auto snapshot = catalog.snapshot();
    CHECK(snapshot.references == 6);
    CHECK(snapshot.blobs == 6);

    std::vector<llama_cache_acct_artifact_id> references;
    for (const auto & result : results) {
        if (result.status != vbr_projected_manifest_publish_status::published &&
            result.status != vbr_projected_manifest_publish_status::adopted) {
            continue;
        }
        references.push_back(result.publication.reference_artifact);
        vbr_artifact_package_view view;
        CHECK(catalog.resolve_reference(
            result.publication.reference_artifact, view) ==
              vbr_artifact_resolve_status::ok);
        CHECK(view.validate() == vbr_artifact_status::ok);
        CHECK(view.projected_ranges().size() == 1);
        CHECK(view.manifest().generation.status ==
              vbr_checkpoint_generation_status::complete);
        CHECK(view.manifest().stream_placements.size() == 1);
        if (result.manifest_id == 6) {
            CHECK(view.manifest().companions.size() == 1);
        }
        view.reset();
    }
    for (const auto reference : references) {
        CHECK(catalog.retire(reference) == vbr_artifact_retire_status::retired);
    }
    CHECK(catalog.snapshot().references == 0);
    CHECK(catalog.snapshot().blobs == 0);
    CHECK(ledger.snapshot().live_ops == 0);

    // Occupied replacement is a two-artifact capability: the recovery image
    // authenticates the incumbent bytes while the incoming image supplies the
    // provisional source rows.  This model-free publication reaches the same
    // immutable catalog view used by production rather than constructing a
    // mutable package facade in the test.
    llama_cache_acct_artifact_id occupied_reference;
    vbr_artifact_package_view occupied_view;
    CHECK(publish_occupied_guard_package(
        catalog, topology, budget, 92, 8, false,
        occupied_reference, occupied_view));
    occupied_guard_fixture occupied;
    CHECK(make_occupied_guard_fixture(
        occupied_view, 92, 16, occupied));
    vbr_occupied_replacement_guard occupied_guard;
    CHECK(vbr_prepare_occupied_replacement_guard(
              occupied.target, occupied_view, occupied_view,
              occupied.observation, occupied_guard) ==
          vbr_occupied_replacement_guard_status::ready);
    CHECK(occupied_guard.ready());
    CHECK(occupied_guard.destination() == 92);
    CHECK(occupied_guard.accounting_serial() ==
          occupied.target.accounting_serial);
    CHECK(occupied_guard.incoming_artifact() == occupied_reference);
    CHECK(occupied_guard.recovery_artifact() == occupied_reference);
    CHECK(occupied_guard.cell_mapping().size() == 8);
    CHECK(occupied_guard.cell_mapping().front()
              .source_physical_cell == 0);
    CHECK(occupied_guard.cell_mapping().front().source_packed_row == 0);
    CHECK(occupied_guard.cell_mapping().front()
              .destination_physical_cell == 8);
    CHECK(occupied_guard.cell_mapping().back()
              .destination_physical_cell == 15);
    CHECK(occupied_guard.relocation_runs().size() == 1);
    CHECK(occupied_guard.relocation_runs().front()
              .first_source_packed_row == 0);
    CHECK(occupied_guard.relocation_runs().front()
              .first_destination_physical_cell == 8);
    CHECK(occupied_guard.relocation_runs().front().cell_count == 8);
    CHECK(occupied_guard.strategy() ==
          vbr_occupied_replacement_strategy::provisional_free_cells);
    CHECK(occupied_guard.recovery_runs().empty());
    CHECK(vbr_recheck_occupied_replacement_guard(
              occupied_guard, occupied.target, occupied.observation) ==
          vbr_occupied_replacement_guard_status::ready);

    occupied_guard_fixture full_pool;
    CHECK(make_occupied_guard_fixture(
        occupied_view, 92, 8, full_pool));
    vbr_occupied_replacement_guard recycle_guard;
    CHECK(vbr_prepare_occupied_replacement_guard(
              full_pool.target, occupied_view, occupied_view,
              full_pool.observation, recycle_guard) ==
          vbr_occupied_replacement_guard_status::ready);
    CHECK(recycle_guard.strategy() ==
          vbr_occupied_replacement_strategy::recycle_incumbent_cells);
    CHECK(recycle_guard.cell_mapping().size() == 8);
    CHECK(recycle_guard.relocation_runs().size() == 1);
    CHECK(recycle_guard.recovery_runs().size() == 1);
    CHECK(recycle_guard.recovery_package().reference_artifact() ==
          occupied_reference);
    for (size_t logical = 0;
         logical < recycle_guard.cell_mapping().size(); ++logical) {
        const auto & mapping = recycle_guard.cell_mapping()[logical];
        CHECK(mapping.logical_position == llama_pos(logical));
        CHECK(mapping.destination_physical_cell == logical);
    }
    CHECK(recycle_guard.recovery_runs().front().first_source_packed_row == 0);
    CHECK(recycle_guard.recovery_runs().front()
              .first_destination_physical_cell == 0);
    CHECK(recycle_guard.recovery_runs().front().cell_count == 8);
    CHECK(vbr_recheck_occupied_replacement_guard(
              recycle_guard, full_pool.target, full_pool.observation) ==
          vbr_occupied_replacement_guard_status::ready);

    // A unified cache can hold several independent server slots. The guard
    // maps the destination through its exact recovery image while retaining
    // every single-owner foreign cell in the same authenticated snapshot.
    occupied_guard_fixture shared_pool = full_pool;
    shared_pool.observation.cell_capacity = 16;
    for (uint32_t physical = 8; physical < 16; ++physical) {
        shared_pool.cells.push_back({
            0, physical, llama_pos(physical-8), 0, 0, 93, 1, false,
            llama_token(1000+physical),
        });
    }
    shared_pool.bind();
    vbr_occupied_replacement_guard shared_guard;
    CHECK(vbr_prepare_occupied_replacement_guard(
              shared_pool.target, occupied_view, occupied_view,
              shared_pool.observation, shared_guard) ==
          vbr_occupied_replacement_guard_status::ready);
    CHECK(shared_guard.strategy() ==
          vbr_occupied_replacement_strategy::recycle_incumbent_cells);
    CHECK(shared_guard.preserved_cells().size() == 8);
    CHECK(shared_guard.preserved_cells().front().owner_sequence == 93);
    CHECK(shared_guard.preserved_cells().front().token == 1008);
    CHECK(vbr_recheck_occupied_replacement_guard(
              shared_guard, shared_pool.target, shared_pool.observation) ==
          vbr_occupied_replacement_guard_status::ready);

    auto shared_mutant = shared_pool;
    ++shared_mutant.cells.back().token;
    shared_mutant.bind();
    CHECK(vbr_recheck_occupied_replacement_guard(
              shared_guard, shared_mutant.target,
              shared_mutant.observation) ==
          vbr_occupied_replacement_guard_status::currency_changed);
    CHECK(!shared_guard.ready());

    CHECK(vbr_prepare_occupied_replacement_guard(
              shared_pool.target, occupied_view, occupied_view,
              shared_pool.observation, shared_guard) ==
          vbr_occupied_replacement_guard_status::ready);

    auto shared_alias = shared_pool;
    shared_alias.cells.front().owner_sequence = -1;
    shared_alias.cells.front().reference_count = 2;
    shared_alias.bind();
    vbr_occupied_replacement_guard shared_alias_guard;
    CHECK(vbr_prepare_occupied_replacement_guard(
              shared_alias.target, occupied_view, occupied_view,
              shared_alias.observation, shared_alias_guard) ==
          vbr_occupied_replacement_guard_status::ownership_mismatch);
    shared_guard.reset();
    auto full_pool_mutant = full_pool;
    full_pool_mutant.cells.back().physical_cell = 6;
    full_pool_mutant.bind();
    CHECK(vbr_recheck_occupied_replacement_guard(
              recycle_guard, full_pool_mutant.target,
              full_pool_mutant.observation) !=
          vbr_occupied_replacement_guard_status::ready);
    CHECK(!recycle_guard.ready());

    vbr_occupied_replacement_guard moved_guard(std::move(occupied_guard));
    CHECK(!occupied_guard.ready());
    CHECK(moved_guard.ready());
    vbr_occupied_replacement_guard overwritten_guard;
    CHECK(vbr_prepare_occupied_replacement_guard(
              occupied.target, occupied_view, occupied_view,
              occupied.observation, overwritten_guard) ==
          vbr_occupied_replacement_guard_status::ready);
    overwritten_guard = std::move(moved_guard);
    CHECK(!moved_guard.ready());
    CHECK(overwritten_guard.ready());
    overwritten_guard.reset();
    CHECK(!overwritten_guard.ready());
    CHECK(overwritten_guard.cell_mapping().empty());

    const auto expect_recheck_refusal = [&] (
            occupied_guard_fixture mutant) {
        mutant.bind();
        vbr_occupied_replacement_guard guard;
        CHECK(vbr_prepare_occupied_replacement_guard(
                  occupied.target, occupied_view, occupied_view,
                  occupied.observation, guard) ==
              vbr_occupied_replacement_guard_status::ready);
        CHECK(vbr_recheck_occupied_replacement_guard(
                  guard, mutant.target, mutant.observation) !=
              vbr_occupied_replacement_guard_status::ready);
        CHECK(!guard.ready());
    };
    auto mutant = occupied;
    mutant.target.accounting_serial++;
    expect_recheck_refusal(std::move(mutant));
    mutant = occupied;
    mutant.target.tree_shape_digest++;
    expect_recheck_refusal(std::move(mutant));
    mutant = occupied;
    mutant.target.policy_epoch++;
    mutant.target.children[0].policy_epoch++;
    expect_recheck_refusal(std::move(mutant));
    mutant = occupied;
    mutant.target.children[0].lineage_uuid.lo++;
    expect_recheck_refusal(std::move(mutant));
    mutant = occupied;
    mutant.target.children[0].state_serial++;
    mutant.observation.representation_epoch++;
    expect_recheck_refusal(std::move(mutant));
    mutant = occupied;
    mutant.target.children[0].units[0].codec_version++;
    expect_recheck_refusal(std::move(mutant));
    mutant = occupied;
    mutant.units[0].generation.repr_gen++;
    expect_recheck_refusal(std::move(mutant));
    mutant = occupied;
    mutant.observation.controller_generation++;
    expect_recheck_refusal(std::move(mutant));
    mutant = occupied;
    mutant.observation.sequence_epoch++;
    expect_recheck_refusal(std::move(mutant));
    mutant = occupied;
    mutant.cells[0].owner_sequence = 93;
    expect_recheck_refusal(std::move(mutant));
    mutant = occupied;
    mutant.cells[0].reference_count = 2;
    expect_recheck_refusal(std::move(mutant));
    mutant = occupied;
    mutant.cells[0].logical_position++;
    expect_recheck_refusal(std::move(mutant));
    mutant = occupied;
    mutant.target.companions.push_back({});
    expect_recheck_refusal(std::move(mutant));

    occupied_guard_fixture cell_cap;
    CHECK(make_occupied_guard_fixture(
        occupied_view, 92, VBR_OCCUPIED_REPLACEMENT_MAX_CELLS, cell_cap));
    CHECK(vbr_prepare_occupied_replacement_guard(
              cell_cap.target, occupied_view, occupied_view,
              cell_cap.observation, occupied_guard) ==
          vbr_occupied_replacement_guard_status::ready);
    occupied_guard.reset();
    cell_cap.observation.cell_capacity =
        VBR_OCCUPIED_REPLACEMENT_MAX_CELLS + 1;
    CHECK(vbr_prepare_occupied_replacement_guard(
              cell_cap.target, occupied_view, occupied_view,
              cell_cap.observation, occupied_guard) ==
          vbr_occupied_replacement_guard_status::cell_limit_exceeded);
    CHECK(!occupied_guard.ready());
    occupied_view.reset();
    CHECK(catalog.retire(occupied_reference) ==
          vbr_artifact_retire_status::retired);

    // Incoming and recovery are separate authorities.  Three projected unit
    // proofs select the same sparse rows, but packed-row expansion is paid once
    // while the recovery placement independently authenticates the incumbent.
    llama_cache_acct_artifact_id distinct_recovery_reference;
    llama_cache_acct_artifact_id distinct_incoming_reference;
    llama_cache_acct_artifact_id wrong_role_reference;
    vbr_artifact_package_view distinct_recovery;
    vbr_artifact_package_view distinct_incoming;
    vbr_artifact_package_view wrong_role;
    CHECK(publish_occupied_guard_package(
        catalog, topology, budget, 192, 8, false,
        distinct_recovery_reference, distinct_recovery, 777, 8, 3));
    CHECK(publish_occupied_guard_package(
        catalog, topology, budget, 193, 8, false,
        distinct_incoming_reference, distinct_incoming, 777, 8, 3, 1));
    CHECK(distinct_recovery_reference.v != 0);
    CHECK(distinct_incoming_reference.v != 0);
    CHECK(distinct_recovery_reference != distinct_incoming_reference);
    occupied_guard_fixture distinct_fixture;
    CHECK(make_occupied_guard_fixture(
        distinct_recovery, 192, 8, distinct_fixture));
    CHECK(vbr_prepare_occupied_replacement_guard(
              distinct_fixture.target, distinct_incoming, distinct_recovery,
              distinct_fixture.observation, occupied_guard) ==
          vbr_occupied_replacement_guard_status::ready);
    CHECK(occupied_guard.incoming_artifact() == distinct_incoming_reference);
    CHECK(occupied_guard.recovery_artifact() == distinct_recovery_reference);
    CHECK(occupied_guard.strategy() ==
          vbr_occupied_replacement_strategy::recycle_incumbent_cells);
    CHECK(!occupied_guard.recovery_runs().empty());
    CHECK(occupied_guard.packed_rows_expanded() == 8);
    CHECK(occupied_guard.cell_mapping().front().source_physical_cell == 0);
    CHECK(occupied_guard.cell_mapping().front().source_packed_row == 0);
    CHECK(distinct_fixture.cells.front().physical_cell == 0);
    occupied_guard.reset();

    CHECK(publish_occupied_guard_package(
        catalog, topology, budget, 194, 8, false,
        wrong_role_reference, wrong_role, 778, 8, 3, 2));
    CHECK(vbr_prepare_occupied_replacement_guard(
              distinct_fixture.target, wrong_role, distinct_recovery,
              distinct_fixture.observation, occupied_guard) !=
          vbr_occupied_replacement_guard_status::ready);
    CHECK(!occupied_guard.ready());
    CHECK(vbr_prepare_occupied_replacement_guard(
              distinct_fixture.target, distinct_incoming, wrong_role,
              distinct_fixture.observation, occupied_guard) !=
          vbr_occupied_replacement_guard_status::ready);
    CHECK(!occupied_guard.ready());
    wrong_role.reset();
    distinct_incoming.reset();
    distinct_recovery.reset();
    CHECK(catalog.retire(wrong_role_reference) ==
          vbr_artifact_retire_status::retired);
    CHECK(catalog.retire(distinct_incoming_reference) ==
          vbr_artifact_retire_status::retired);
    CHECK(catalog.retire(distinct_recovery_reference) ==
          vbr_artifact_retire_status::retired);

    // A transformed occupied capability is legal only on the bounded dense
    // recycle strategy. The incoming F16 bytes are authenticated against the
    // live TURBO8 destination schedule, while the independent TURBO8 recovery
    // package remains the byte-exact rollback authority.
    llama_cache_acct_artifact_id transform_recovery_reference;
    llama_cache_acct_artifact_id transform_incoming_reference;
    vbr_artifact_package_view transform_recovery;
    vbr_artifact_package_view transform_incoming;
    CHECK(publish_occupied_guard_package(
        catalog, topology, budget, 195, 8, false,
        transform_recovery_reference, transform_recovery,
        1000, 8, 1, 3, GGML_TYPE_TURBO8_0));
    CHECK(publish_occupied_guard_package(
        catalog, topology, budget, 196, 8, false,
        transform_incoming_reference, transform_incoming,
        1000, 8, 1, 4, GGML_TYPE_F16));
    occupied_guard_fixture transform_fixture;
    CHECK(make_occupied_guard_fixture(
        transform_recovery, 195, 8, transform_fixture));
    vbr_import_schedule_quote transform_incoming_quote;
    vbr_import_schedule_quote transform_recovery_quote;
    CHECK(vbr_quote_import_schedule(
        transform_fixture.target, transform_incoming,
        transform_incoming_quote));
    CHECK(transform_incoming_quote.status() ==
          vbr_import_schedule_status::downward);
    CHECK(vbr_quote_import_schedule(
        transform_fixture.target, transform_recovery,
        transform_recovery_quote));
    CHECK(transform_recovery_quote.status() ==
          vbr_import_schedule_status::exact);
    CHECK(vbr_prepare_occupied_replacement_guard(
              transform_fixture.target, transform_incoming,
              transform_recovery, transform_fixture.observation,
              occupied_guard, &transform_incoming_quote,
              &transform_recovery_quote) ==
          vbr_occupied_replacement_guard_status::ready);
    CHECK(occupied_guard.strategy() ==
          vbr_occupied_replacement_strategy::recycle_incumbent_cells);
    CHECK(occupied_guard.relocation_runs().size() == 1);
    CHECK(occupied_guard.recovery_runs().size() == 1);
    CHECK(vbr_recheck_occupied_replacement_guard(
              occupied_guard, transform_fixture.target,
              transform_fixture.observation) ==
          vbr_occupied_replacement_guard_status::ready);
    occupied_guard.reset();

    auto transformed_validation_target = transform_fixture.target;
    const auto transformed_accounting = ledger.snapshot();
    transformed_validation_target.accounting_serial =
        transformed_accounting.serial;
    auto & transformed_live_child =
        transformed_validation_target.children.front();
    auto & transformed_live_unit = transformed_live_child.units.front();
    transformed_live_unit.shards.front().domain = device;
    transformed_live_unit.shards.front().mapped_bytes = 8;
    transformed_live_unit.downward_supported = true;
    transformed_live_unit.downward_movable = true;
    transformed_live_unit.controller_floor_type = GGML_TYPE_TURBO1_TCQ;
    transformed_live_unit.downward_type = GGML_TYPE_TURBO8_0;
    transformed_live_unit.downward_domain = vbr_repr_domain::full;
    transformed_live_unit.downward_recipe_id = VBR_DOWNWARD_RECIPE_ID;
    transformed_live_unit.downward_recipe_version =
        VBR_DOWNWARD_RECIPE_VERSION;
    CHECK(vbr_downward_resolve_recipe(
              GGML_TYPE_F16, GGML_TYPE_TURBO8_0,
              GGML_TYPE_TURBO1_TCQ, true,
              transformed_live_unit.downward_recipe) ==
          vbr_downward_recipe_status::resolved);
    transformed_live_unit.downward_row_bytes = 1;
    transformed_live_unit.downward_mapped_bytes = 8;
    transformed_live_unit.downward_transfer_bytes = 8;
    transformed_live_unit.downward_codec_workspace_bytes = 64;
    transformed_live_unit.downward_meansub_model_id = 7;
    vbr_downward_policy_child occupied_downward_child;
    occupied_downward_child.initial_types = { GGML_TYPE_F16 };
    occupied_downward_child.target_types = { GGML_TYPE_TURBO8_0 };
    occupied_downward_child.initial_cursor =
        transform_incoming.manifest().controller_policy.front().cursor;
    const auto & occupied_edge =
        transformed_live_unit.downward_recipe.edges.front();
    occupied_downward_child.policy.steps.push_back({
        0, 0,
        int32_t(occupied_edge.source_type),
        int32_t(occupied_edge.target_type), 1,
    });
    occupied_downward_child.policy.terminal_progress = 1;
    const auto occupied_downward_projection =
        vbr_downward_project_policy_prefix({ occupied_downward_child });
    CHECK(occupied_downward_projection.status ==
          vbr_downward_policy_status::coherent);
    transformed_live_child.controller_policy.cursor =
        occupied_downward_projection.final_cursors.front();
    transformed_live_child.controller_policy.current_type_vector_digest =
        occupied_downward_projection.child_type_digests.front();
    transformed_live_unit.downward_build_identity_digest =
        vbr_downward_build_identity(
            transformed_live_unit.downward_recipe,
            transformed_live_unit.downward_meansub_model_id,
            transformed_live_unit.meansub_digest,
            occupied_downward_projection.child_type_digests.front(),
            occupied_downward_projection.tree_digest);
    vbr_import_destination_projection occupied_destination;
    occupied_destination.status =
        vbr_import_destination_status::feasible_degraded;
    occupied_destination.prefix = occupied_downward_projection.prefix;
    occupied_destination.initial_types = { { GGML_TYPE_F16 } };
    occupied_destination.initial_cursors = {
        occupied_downward_child.initial_cursor,
    };
    occupied_destination.final_types =
        occupied_downward_projection.final_types;
    occupied_destination.final_cursors =
        occupied_downward_projection.final_cursors;
    occupied_destination.child_type_digests =
        occupied_downward_projection.child_type_digests;
    occupied_destination.tree_digest =
        occupied_downward_projection.tree_digest;
    vbr_import_schedule_quote occupied_transform_quote;
    CHECK(vbr_quote_import_schedule(
        transformed_validation_target, transform_incoming,
        occupied_transform_quote));
    CHECK(vbr_rebind_import_schedule_quote(
        transformed_validation_target, transform_incoming,
        occupied_destination, occupied_transform_quote));
    vbr_import_schedule_quote occupied_recovery_quote;
    CHECK(vbr_quote_import_schedule(
        transformed_validation_target, transform_recovery,
        occupied_recovery_quote));
    vbr_occupied_replacement_guard validated_transform_guard;
    CHECK(vbr_prepare_occupied_replacement_guard(
              transformed_validation_target, transform_incoming,
              transform_recovery, transform_fixture.observation,
              validated_transform_guard, &occupied_transform_quote,
              &occupied_recovery_quote) ==
          vbr_occupied_replacement_guard_status::ready);
    prefix_validator_serials transformed_serials;
    transformed_serials.accounting = transformed_accounting.serial;
    transformed_serials.policy = transformed_validation_target.policy_epoch;
    transformed_serials.memory =
        transformed_validation_target.memory_instance_cookie;
    transformed_serials.state =
        transformed_validation_target.target_state_serial;
    transformed_serials.tree =
        transformed_validation_target.tree_shape_digest;
    transformed_serials.transform_tree =
        occupied_downward_projection.tree_digest;
    llama_cache_budget_plan occupied_transform_plan;
    occupied_transform_plan.accounting_serial =
        transformed_accounting.serial;
    vbr_adopt_policy occupied_transform_policy;
    occupied_transform_policy.authorized = true;
    occupied_transform_policy.identity.execution_identity =
        transform_incoming.manifest().identity.execution_identity;
    occupied_transform_policy.identity.adapter_config_identity =
        transform_incoming.manifest().identity.adapter_config_identity;
    occupied_transform_policy.identity.media_content_identity =
        transform_incoming.manifest().identity.media_content_identity;
    occupied_transform_policy.identity.sequence_epoch =
        transform_incoming.manifest().identity.sequence_epoch;
    occupied_transform_policy.identity.requested_frontier =
        transform_incoming.manifest().identity.next_position;
    occupied_transform_policy.identity.tokens =
        &transform_incoming.manifest().token_block.tokens;
    occupied_transform_policy.destination_sequence = 195;
    occupied_transform_policy.adoption_nonce = 0x4195;
    occupied_transform_policy.domain_bindings = bindings;
    occupied_transform_policy.domain_bindings.push_back({
        UINT32_MAX, UINT16_MAX, host,
    });
    occupied_transform_policy.accounting_snapshot =
        &transformed_accounting;
    occupied_transform_policy.budget_config = &budget;
    occupied_transform_policy.transform_budget_plan =
        &occupied_transform_plan;
    occupied_transform_policy.downward_projection =
        &occupied_downward_projection;
    occupied_transform_policy.schedule_quote = &occupied_transform_quote;
    occupied_transform_policy.context = &transformed_serials;
    occupied_transform_policy.read_accounting_serial =
        prefix_validator_serials::read_accounting;
    occupied_transform_policy.read_policy_epoch =
        prefix_validator_serials::read_policy;
    occupied_transform_policy.read_transform_tree_digest =
        prefix_validator_serials::read_transform_tree;
    occupied_transform_policy.occupied_replacement =
        &validated_transform_guard;
    occupied_transform_policy.occupied_representation_identity = [](
            const void *, int32_t, bool, int32_t,
            vbr_explicit_representation_identity &) noexcept {
        return false;
    };
    auto occupied_transformed = vbr_validate_unit_manifest_snapshot(
        transformed_validation_target, transform_incoming,
        occupied_transform_policy);
    CHECK(occupied_transformed.status ==
          vbr_manifest_validation_status::validated);
    CHECK(occupied_transformed.decision ==
          vbr_import_decision::downward_rebase);
    CHECK(occupied_transformed.proof);
    CHECK(!validated_transform_guard.ready());
    if (occupied_transformed.proof) {
        CHECK(occupied_transformed.proof->is_occupied_replacement());
        CHECK(occupied_transformed.proof->children().size() == 1);
        CHECK(occupied_transformed.proof->children().front().transform_kind ==
              vbr_import_transform_kind::downward);
    }
    occupied_transformed.proof.reset();
    occupied_guard_fixture transform_free_fixture;
    CHECK(make_occupied_guard_fixture(
        transform_recovery, 195, 16, transform_free_fixture));
    CHECK(vbr_prepare_occupied_replacement_guard(
              transform_free_fixture.target, transform_incoming,
              transform_recovery, transform_free_fixture.observation,
              occupied_guard, &transform_incoming_quote,
              &transform_recovery_quote) ==
          vbr_occupied_replacement_guard_status::unsupported_layout);
    CHECK(!occupied_guard.ready());
    transform_incoming.reset();
    transform_recovery.reset();
    CHECK(catalog.retire(transform_incoming_reference) ==
          vbr_artifact_retire_status::retired);
    CHECK(catalog.retire(transform_recovery_reference) ==
          vbr_artifact_retire_status::retired);

    const auto validate_recycle_fanout = [&](uint32_t proof_count,
                                             uint64_t manifest_base,
                                             bool accepted) {
        llama_cache_acct_artifact_id recovery_reference;
        llama_cache_acct_artifact_id incoming_reference;
        vbr_artifact_package_view recovery_view;
        vbr_artifact_package_view incoming_view;
        CHECK(publish_occupied_guard_package(
            catalog, topology, budget, manifest_base, 8, false,
            recovery_reference, recovery_view, 900, 8, proof_count));
        CHECK(publish_occupied_guard_package(
            catalog, topology, budget, manifest_base + 1, 8, false,
            incoming_reference, incoming_view, 900, 8, proof_count, 1));
        CHECK(recovery_reference.v != 0 && incoming_reference.v != 0);
        CHECK(recovery_reference != incoming_reference);

        occupied_guard_fixture fixture;
        CHECK(make_occupied_guard_fixture(
            recovery_view, llama_seq_id(manifest_base), 8, fixture));
        const auto accounting = ledger.snapshot();
        fixture.target.accounting_serial = accounting.serial;
        for (auto & unit : fixture.target.children.front().units) {
            for (auto & shard : unit.shards) {
                shard.domain = device;
                shard.mapped_bytes = unit.wm_cells*shard.row_bytes;
            }
        }
        vbr_occupied_replacement_guard guard;
        CHECK(vbr_prepare_occupied_replacement_guard(
                  fixture.target, incoming_view, recovery_view,
                  fixture.observation, guard) ==
              vbr_occupied_replacement_guard_status::ready);
        CHECK(guard.strategy() ==
              vbr_occupied_replacement_strategy::recycle_incumbent_cells);
        CHECK(guard.relocation_runs().size() == 1);
        CHECK(guard.recovery_runs().size() == 1);

        prefix_validator_serials serials;
        serials.accounting = accounting.serial;
        serials.policy = fixture.target.policy_epoch;
        serials.memory = fixture.target.memory_instance_cookie;
        serials.state = fixture.target.target_state_serial;
        serials.tree = fixture.target.tree_shape_digest;
        vbr_adopt_policy policy;
        policy.authorized = true;
        policy.identity.execution_identity =
            incoming_view.manifest().identity.execution_identity;
        policy.identity.adapter_config_identity =
            incoming_view.manifest().identity.adapter_config_identity;
        policy.identity.media_content_identity =
            incoming_view.manifest().identity.media_content_identity;
        policy.identity.sequence_epoch =
            incoming_view.manifest().identity.sequence_epoch;
        policy.identity.requested_frontier =
            incoming_view.manifest().identity.next_position;
        policy.identity.tokens = &incoming_view.manifest().token_block.tokens;
        policy.destination_sequence = llama_seq_id(manifest_base);
        policy.adoption_nonce = manifest_base + 0x4106;
        policy.domain_bindings = bindings;
        policy.domain_bindings.push_back({ UINT32_MAX, UINT16_MAX, host });
        policy.accounting_snapshot = &accounting;
        policy.budget_config = &budget;
        policy.context = &serials;
        policy.read_accounting_serial = prefix_validator_serials::read_accounting;
        policy.read_policy_epoch = prefix_validator_serials::read_policy;
        policy.occupied_replacement = &guard;
        policy.occupied_representation_identity = [](
                const void *, int32_t, bool, int32_t,
                vbr_explicit_representation_identity &) noexcept {
            return false;
        };
        auto validated = vbr_validate_unit_manifest_snapshot(
            fixture.target, incoming_view, policy);
        CHECK(validated.status == (accepted
            ? vbr_manifest_validation_status::validated
            : vbr_manifest_validation_status::geometry_mismatch));
        CHECK(bool(validated.proof) == accepted);
        CHECK(accepted ? !guard.ready() : guard.ready());
        validated.proof.reset();
        guard.reset();
        incoming_view.reset();
        recovery_view.reset();
        CHECK(catalog.retire(incoming_reference) ==
              vbr_artifact_retire_status::retired);
        CHECK(catalog.retire(recovery_reference) ==
              vbr_artifact_retire_status::retired);
    };
    // One incoming and one recovery descriptor per shard: 2048*2 is the
    // exact validator limit; one additional shard fails before staging.
    validate_recycle_fanout(2048, 300, true);
    validate_recycle_fanout(2049, 400, false);

    llama_cache_acct_artifact_id run_reference;
    vbr_artifact_package_view run_view;
    CHECK(publish_occupied_guard_package(
        catalog, topology, budget, 93,
        VBR_OCCUPIED_REPLACEMENT_MAX_RUNS, true,
        run_reference, run_view));
    occupied_guard_fixture run_fixture;
    CHECK(make_occupied_guard_fixture(
        run_view, 93, 2*VBR_OCCUPIED_REPLACEMENT_MAX_RUNS,
        run_fixture));
    const auto occupied_accounting = ledger.snapshot();
    run_fixture.target.accounting_serial = occupied_accounting.serial;
    for (auto & unit : run_fixture.target.children.front().units) {
        for (auto & shard : unit.shards) {
            shard.domain = device;
            shard.mapped_bytes = unit.wm_cells*shard.row_bytes;
        }
    }
    CHECK(vbr_prepare_occupied_replacement_guard(
              run_fixture.target, run_view, run_view,
              run_fixture.observation, occupied_guard) ==
          vbr_occupied_replacement_guard_status::ready);
    CHECK(occupied_guard.relocation_runs().size() ==
          VBR_OCCUPIED_REPLACEMENT_MAX_RUNS);
    CHECK(occupied_guard.cell_mapping().front().source_physical_cell == 1);
    CHECK(occupied_guard.cell_mapping().front().source_packed_row == 0);
    CHECK(occupied_guard.cell_mapping().back().source_physical_cell ==
          2*VBR_OCCUPIED_REPLACEMENT_MAX_RUNS-1);
    CHECK(occupied_guard.cell_mapping().back().source_packed_row ==
          VBR_OCCUPIED_REPLACEMENT_MAX_RUNS-1);

    prefix_validator_serials occupied_serials;
    occupied_serials.accounting = occupied_accounting.serial;
    occupied_serials.policy = run_fixture.target.policy_epoch;
    occupied_serials.memory = run_fixture.target.memory_instance_cookie;
    occupied_serials.state = run_fixture.target.target_state_serial;
    occupied_serials.tree = run_fixture.target.tree_shape_digest;
    vbr_adopt_policy occupied_policy;
    occupied_policy.authorized = true;
    occupied_policy.identity.execution_identity =
        run_view.manifest().identity.execution_identity;
    occupied_policy.identity.adapter_config_identity =
        run_view.manifest().identity.adapter_config_identity;
    occupied_policy.identity.media_content_identity =
        run_view.manifest().identity.media_content_identity;
    occupied_policy.identity.sequence_epoch =
        run_view.manifest().identity.sequence_epoch;
    occupied_policy.identity.requested_frontier =
        run_view.manifest().identity.next_position;
    occupied_policy.identity.tokens = &run_view.manifest().token_block.tokens;
    occupied_policy.destination_sequence = 93;
    occupied_policy.adoption_nonce = 0x4106;
    occupied_policy.domain_bindings = bindings;
    occupied_policy.domain_bindings.push_back({ UINT32_MAX, UINT16_MAX, host });
    occupied_policy.accounting_snapshot = &occupied_accounting;
    occupied_policy.budget_config = &budget;
    occupied_policy.context = &occupied_serials;
    occupied_policy.read_accounting_serial =
        prefix_validator_serials::read_accounting;
    occupied_policy.read_policy_epoch = prefix_validator_serials::read_policy;
    occupied_policy.occupied_replacement = &occupied_guard;
    occupied_policy.occupied_representation_identity = [](
            const void *, int32_t, bool, int32_t,
            vbr_explicit_representation_identity &) noexcept {
        return false;
    };
    auto occupied_validated = vbr_validate_unit_manifest_snapshot(
        run_fixture.target, run_view, occupied_policy);
    if (occupied_validated.status !=
            vbr_manifest_validation_status::validated) {
        fprintf(stderr, "sparse occupied validation status: %s\n",
            vbr_manifest_validation_status_name(occupied_validated.status));
    }
    CHECK(occupied_validated.status ==
          vbr_manifest_validation_status::validated);
    CHECK(occupied_validated.proof);
    CHECK(!occupied_guard.ready());
    CHECK(occupied_validated.proof &&
          occupied_validated.proof->relocation_runs().size() ==
          VBR_OCCUPIED_REPLACEMENT_MAX_RUNS);
    CHECK(occupied_validated.proof &&
          occupied_validated.proof->relocation_runs().front()
              .first_source_packed_row == 0);
    CHECK(occupied_validated.proof &&
          occupied_validated.proof->relocation_runs().back()
              .first_source_packed_row ==
          VBR_OCCUPIED_REPLACEMENT_MAX_RUNS-1);

    vbr_adopt_stage_policy occupied_stage_policy;
    occupied_stage_policy.ledger = &ledger;
    occupied_stage_policy.budget = &budget;
    occupied_stage_policy.pinned_domain = pinned;
    occupied_stage_policy.pinned_ring_bytes = 8192;
    occupied_stage_policy.chunk_bytes = 4096;
    occupied_stage_policy.lanes.push_back({ device, nullptr, nullptr, false });
    auto occupied_staged = occupied_validated.proof
        ? vbr_stage_validated_manifest(
            std::move(occupied_validated.proof), occupied_stage_policy)
        : vbr_adopt_stage_result {};
    if (occupied_staged.status != vbr_adopt_stage_status::staged) {
        fprintf(stderr, "sparse occupied stage status: %s\n",
            vbr_adopt_stage_status_name(occupied_staged.status));
    }
    CHECK(occupied_staged.status == vbr_adopt_stage_status::staged);
    CHECK(occupied_staged.staged);
    if (occupied_staged.staged) {
        CHECK(occupied_staged.staged->reads().size() ==
              VBR_OCCUPIED_REPLACEMENT_MAX_RUNS);
        CHECK(occupied_staged.staged->reads().front()
                  .projection_ranges.front().source_offset == 0);
        CHECK(occupied_staged.staged->reads().back()
                  .projection_ranges.front().source_offset ==
              VBR_OCCUPIED_REPLACEMENT_MAX_RUNS-1);
    }
    occupied_staged.staged.reset();
    occupied_staged.manifest.reset();
    run_view.reset();
    CHECK(catalog.retire(run_reference) ==
          vbr_artifact_retire_status::retired);

    CHECK(publish_occupied_guard_package(
        catalog, topology, budget, 94,
        VBR_OCCUPIED_REPLACEMENT_MAX_RUNS + 1, true,
        run_reference, run_view));
    CHECK(make_occupied_guard_fixture(
        run_view, 94, 2*(VBR_OCCUPIED_REPLACEMENT_MAX_RUNS + 1),
        run_fixture));
    CHECK(vbr_prepare_occupied_replacement_guard(
              run_fixture.target, run_view, run_view,
              run_fixture.observation, occupied_guard) ==
          vbr_occupied_replacement_guard_status::run_limit_exceeded);
    CHECK(!occupied_guard.ready());
    run_view.reset();
    CHECK(catalog.retire(run_reference) ==
          vbr_artifact_retire_status::retired);
    CHECK(ledger.snapshot().live_ops == 0);

    // Keep the shared batch fence intact until catalog validation chooses a
    // runnable physical owner. The original lower-ID owner is deliberately
    // malformed; its sibling must inherit the one fresh-unit reservation.
    vbr_capture_projection_manifest shared_first;
    shared_first.manifest_id = 101;
    shared_first.placements.push_back(
        projected_placement(101, 101, { 0 }));
    vbr_capture_projection_manifest shared_second;
    shared_second.manifest_id = 102;
    shared_second.placements.push_back(
        projected_placement(102, 102, { 0 }));
    auto shared_target_first = projected_target(
        101, 0, 1111, projected_generation(44));
    auto shared_target_second = shared_target_first;
    shared_target_second.manifest_id = 102;
    bind_projected_manifest_metadata(shared_first, shared_target_first);
    bind_projected_manifest_metadata(shared_second, shared_target_second);
    vbr_capture_projection shared_projection;
    CHECK(vbr_artifact_project_capture_union(
        { 707, { shared_first, shared_second } }, {}, shared_projection));
    auto shared_unit = capture_projected_unit_for_target(
        shared_projection, shared_target_first);
    vbr_capture_manifest_assembly shared_assembly;
    projected_controller_fixture shared_controller;
    CHECK(assemble_projected_test_batch(
        shared_projection,
        { shared_target_first, shared_target_second },
        { shared_unit }, shared_controller.provider(), {}, shared_assembly));
    std::vector<vbr_projected_manifest_publication> shared_publications;
    shared_publications.push_back(
        projected_publication(101, shared_assembly, topology));
    shared_publications.push_back(
        projected_publication(102, shared_assembly, topology));
    auto unexpected_companion =
        std::make_unique<artifact_segment_chain>();
    const uint8_t unexpected_byte = 0x7c;
    CHECK(unexpected_companion->append(&unexpected_byte, 1));
    vbr_capture_sealed_companion unexpected_sealed;
    CHECK(vbr_capture_seal_companion(
        0, std::move(unexpected_companion), unexpected_sealed));
    shared_publications.front().companions.push_back(
        std::move(unexpected_sealed));
    std::vector<llama_vbr_projected_publication_request> shared_requests {
        { 101, shared_publications[0].accounting, {}, false },
        { 102, shared_publications[1].accounting, {}, false },
    };
    auto shared_claim = catalog.prepare_projected_publication_claims(
        shared_requests, budget);
    CHECK(shared_claim.ready());
    std::vector<vbr_projected_manifest_publish_result> shared_results;
    CHECK(catalog.publish_projected_batch_claimed(
        shared_assembly, std::move(shared_publications),
        std::move(shared_claim), shared_results));
    CHECK(shared_results.size() == 2);
    CHECK(shared_results[0].status ==
          vbr_projected_manifest_publish_status::companion_unavailable);
    CHECK(shared_results[1].status ==
          vbr_projected_manifest_publish_status::published);
    CHECK(catalog.retire(
              shared_results[1].publication.reference_artifact) ==
          vbr_artifact_retire_status::retired);
    CHECK(catalog.snapshot().references == 0);
    CHECK(catalog.snapshot().blobs == 0);
    CHECK(ledger.snapshot().live_ops == 0);

    // A projected artifact stores only the cited packed union. Its live
    // controller watermark remains eight rows and must not be conflated with
    // the two-row storage extent.
    auto sparse_target = projected_target(
        90, 0, 909, projected_generation(21));
    vbr_capture_projection_manifest sparse_manifest;
    sparse_manifest.manifest_id = 90;
    sparse_manifest.placements.push_back(
        projected_placement(90, 90, { 2, 5 }));
    bind_projected_manifest_metadata(sparse_manifest, sparse_target);
    vbr_capture_projection sparse_projection;
    CHECK(vbr_artifact_project_capture_union(
        { 707, { sparse_manifest } }, {}, sparse_projection));
    std::vector<vbr_capture_projected_unit> sparse_units {
        capture_projected_unit_for_target(
            sparse_projection, sparse_target),
    };
    vbr_capture_manifest_assembly sparse_assembly;
    projected_controller_fixture sparse_controller;
    CHECK(assemble_projected_test_batch(
        sparse_projection, { sparse_target }, sparse_units,
        sparse_controller.provider(), {}, sparse_assembly));
    auto sparse_publication = projected_publication(
        90, sparse_assembly, topology);
    vbr_artifact_package sparse_accounting_package;
    sparse_accounting_package.topologies = sparse_publication.topologies;
    sparse_accounting_package.manifest.accounting =
        sparse_publication.accounting;
    CHECK(catalog.configure_accounting(sparse_accounting_package));
    results.clear();
    diagnostics = {};
    std::vector<vbr_projected_manifest_publication> sparse_publications;
    sparse_publications.push_back(std::move(sparse_publication));
    CHECK(catalog.publish_projected_batch(
        sparse_assembly, std::move(sparse_publications), budget,
        results, &diagnostics));
    CHECK(results.size() == 1);
    CHECK(results[0].status ==
          vbr_projected_manifest_publish_status::published);
    vbr_artifact_package_view sparse_view;
    CHECK(catalog.resolve_reference(
        results[0].publication.reference_artifact, sparse_view) ==
          vbr_artifact_resolve_status::ok);
    CHECK(sparse_view.validate() == vbr_artifact_status::ok);
    CHECK(sparse_view.units().size() == 1);
    CHECK(sparse_view.units()[0].descriptor.wm_cells == 8);
    CHECK(sparse_view.units()[0].descriptor.shards[0].row_count == 2);
    sparse_view.reset();
    const auto first_sparse_reference =
        results[0].publication.reference_artifact;
    auto adopted_publication = projected_publication(
        90, sparse_assembly, topology);
    std::vector<llama_vbr_projected_publication_claim> adopted_claims;
    auto adopted_claim = catalog.prepare_projected_publication_claim(
        90, adopted_publication.accounting, budget);
    CHECK(adopted_claim.ready());
    adopted_claims.push_back(std::move(adopted_claim));
    std::vector<vbr_projected_manifest_publication> adopted_publications;
    adopted_publications.push_back(std::move(adopted_publication));
    results.clear();
    CHECK(catalog.publish_projected_batch_claimed(
        sparse_assembly, std::move(adopted_publications),
        std::move(adopted_claims), results, &diagnostics));
    CHECK(results.size() == 1);
    CHECK(results[0].status ==
          vbr_projected_manifest_publish_status::adopted);
    CHECK(catalog.retire(results[0].publication.reference_artifact) ==
          vbr_artifact_retire_status::retired);
    CHECK(catalog.retire(first_sparse_reference) ==
          vbr_artifact_retire_status::retired);

    // A prefix projection is a borrow of one immutable parent, not a new
    // catalog reference. Physical and logical order are deliberately
    // unrelated: the source runs must name packed union rows rather than
    // treating captured physical cells as byte offsets.
    auto prefix_target = projected_target(
        91, 0, 919, projected_generation(31));
    const auto second_prefix_generation = projected_generation(32);
    prefix_target.units.push_back(second_prefix_generation);
    auto second_prefix_descriptor = prefix_target.unit_descriptors.front();
    second_prefix_descriptor.logical_unit_id = 1;
    second_prefix_descriptor.repr_gen = second_prefix_generation.repr_gen;
    second_prefix_descriptor.current_type =
        second_prefix_generation.current_type;
    second_prefix_descriptor.last_source_type =
        second_prefix_generation.last_source_type;
    prefix_target.unit_descriptors.push_back(second_prefix_descriptor);
    prefix_target.policy.current_type_vector_digest =
        vbr_type_vector_digest(std::vector<ggml_type> {
            ggml_type(prefix_target.units[0].current_type),
            ggml_type(prefix_target.units[1].current_type),
        });
    vbr_capture_projection_manifest prefix_manifest;
    prefix_manifest.manifest_id = 91;
    auto prefix_placement = projected_placement(
        91, 91, { 0, 1, 2, 3, 4, 5, 6, 7 });
    const llama_pos shuffled_logical[] { 3, 5, 1, 7, 6, 2, 4, 0 };
    for (size_t i = 0; i < prefix_placement.cells.size(); ++i) {
        prefix_placement.cells[i].logical_position = shuffled_logical[i];
    }
    prefix_manifest.placements.push_back(std::move(prefix_placement));
    bind_projected_manifest_metadata(prefix_manifest, prefix_target);
    prefix_manifest.identity.execution_identity.assign(4096, 'e');
    prefix_manifest.identity.adapter_config_identity.assign(4096, 'a');
    prefix_manifest.identity.media_content_identity.assign(4096, 'm');
    vbr_capture_projection prefix_capture;
    CHECK(vbr_artifact_project_capture_union(
        { 707, { prefix_manifest } }, {}, prefix_capture));
    auto prefix_unit = capture_projected_unit_for_target(
        prefix_capture, prefix_target);
    auto second_prefix_unit = capture_projected_unit_for_target(
        prefix_capture, prefix_target, 0, 1, 1);
    projected_controller_fixture prefix_controller;
    vbr_capture_manifest_assembly prefix_assembly;
    CHECK(assemble_projected_test_batch(
        prefix_capture, { prefix_target },
        { prefix_unit, second_prefix_unit },
        prefix_controller.provider(), {}, prefix_assembly));
    auto prefix_publication = projected_publication(
        91, prefix_assembly, topology);
    std::vector<vbr_projected_manifest_publication> prefix_publications;
    prefix_publications.push_back(std::move(prefix_publication));
    results.clear();
    CHECK(catalog.publish_projected_batch(
        prefix_assembly, std::move(prefix_publications), budget, results));
    CHECK(results.size() == 1);
    CHECK(results[0].status ==
          vbr_projected_manifest_publish_status::published);
    const auto prefix_reference =
        results[0].publication.reference_artifact;
    vbr_artifact_package_view prefix_parent;
    CHECK(catalog.resolve_reference(prefix_reference, prefix_parent) ==
          vbr_artifact_resolve_status::ok);
    CHECK(prefix_parent.validate() == vbr_artifact_status::ok);

    const llama_token divergent[] { 1, 2, 99, 100 };
    vbr_artifact_attention_prefix_request prefix_request;
    prefix_request.tokens = divergent;
    prefix_request.token_count = sizeof(divergent)/sizeof(divergent[0]);
    prefix_request.lcp_tokens = 2;
    prefix_request.text_only = true;
    vbr_artifact_attention_prefix_projection prefix_projection;
    CHECK(catalog.project_attention_prefix(
              prefix_parent, prefix_request, {}, prefix_projection) ==
          vbr_artifact_prefix_projection_status::projected);
    CHECK(prefix_projection);
    CHECK(prefix_projection.parent_artifact() == prefix_reference);
    CHECK(prefix_projection.parent_manifest_digest() ==
          prefix_parent.manifest().manifest_digest);
    CHECK(prefix_projection.identity().token_count == 2);
    CHECK(prefix_projection.identity().next_position == 2);
    CHECK(prefix_projection.prefix_tokens() ==
          std::vector<llama_token>({ 1, 2 }));
    CHECK(prefix_projection.token_digest().valid());
    CHECK(prefix_projection.digest().valid());
    CHECK(prefix_projection.cell_runs().size() == 2);
    CHECK(prefix_projection.cell_runs()[0].first_logical_position == 0);
    CHECK(prefix_projection.cell_runs()[0].first_physical_cell == 7);
    CHECK(prefix_projection.cell_runs()[0].first_packed_row == 7);
    CHECK(prefix_projection.cell_runs()[1].first_logical_position == 1);
    CHECK(prefix_projection.cell_runs()[1].first_physical_cell == 2);
    CHECK(prefix_projection.cell_runs()[1].first_packed_row == 2);
    CHECK(prefix_projection.source_runs().size() == 4);
    CHECK(prefix_projection.selected_bytes() == 4);
    CHECK(prefix_projection.proofs().size() == 2);
    CHECK(prefix_projection.proofs()[0].proof.ranges().size() == 2);
    uint64_t selected_read = 0;
    CHECK(vbr_capture_range_verify(
        prefix_projection.proofs()[0].proof,
        prefix_parent.units()[0].payload_shards[0]->source(),
        &selected_read));
    CHECK(selected_read == 8); // one declared 64 KiB leaf, eight real bytes

    const llama_token shorter[] { 1, 2, 3 };
    prefix_request.tokens = shorter;
    prefix_request.token_count = 3;
    prefix_request.lcp_tokens = 3;
    vbr_artifact_attention_prefix_projection shorter_projection;
    CHECK(catalog.project_attention_prefix(
              prefix_parent, prefix_request, {}, shorter_projection) ==
          vbr_artifact_prefix_projection_status::projected);
    CHECK(shorter_projection.selected_bytes() == 6);

    const auto expect_projection_limit = [&](const auto & cap) {
        vbr_artifact_attention_prefix_projection capped;
        CHECK(catalog.project_attention_prefix(
                  prefix_parent, prefix_request, cap, capped) ==
              vbr_artifact_prefix_projection_status::limit_exceeded);
        CHECK(!capped);
    };
    auto catalog_cap = vbr_artifact_prefix_projection_limits {};
    catalog_cap.max_tokens = 7;
    expect_projection_limit(catalog_cap);
    catalog_cap = {};
    catalog_cap.max_units = 1;
    expect_projection_limit(catalog_cap);
    catalog_cap = {};
    catalog_cap.max_placements = 7;
    expect_projection_limit(catalog_cap);
    catalog_cap = {};
    catalog_cap.max_proofs = 1;
    expect_projection_limit(catalog_cap);

    // The retained metadata cap includes the three copied semantic identity
    // strings; omitting them would incorrectly fit this boundary.
    auto identity_cap = vbr_artifact_prefix_projection_limits {};
    identity_cap.max_metadata_bytes =
        prefix_parent.manifest().identity.execution_identity.size() +
        prefix_parent.manifest().identity.adapter_config_identity.size() +
        prefix_parent.manifest().identity.media_content_identity.size();
    vbr_artifact_attention_prefix_projection identity_refused;
    CHECK(catalog.project_attention_prefix(
              prefix_parent, prefix_request, identity_cap,
              identity_refused) ==
          vbr_artifact_prefix_projection_status::limit_exceeded);
    CHECK(!identity_refused);

    auto tight_prefix_limits = vbr_artifact_prefix_projection_limits {};
    tight_prefix_limits.max_source_runs = 1;
    vbr_artifact_attention_prefix_projection refused_projection;
    CHECK(catalog.project_attention_prefix(
              prefix_parent, prefix_request, tight_prefix_limits,
              refused_projection) ==
          vbr_artifact_prefix_projection_status::limit_exceeded);
    CHECK(!refused_projection);
    tight_prefix_limits = {};
    tight_prefix_limits.proof.max_ranges = 2;
    CHECK(catalog.project_attention_prefix(
              prefix_parent, prefix_request, tight_prefix_limits,
              refused_projection) ==
          vbr_artifact_prefix_projection_status::limit_exceeded);
    CHECK(!refused_projection);
    tight_prefix_limits = {};
    tight_prefix_limits.proof.max_selected_chunks = 0;
    CHECK(catalog.project_attention_prefix(
              prefix_parent, prefix_request, tight_prefix_limits,
              refused_projection) ==
          vbr_artifact_prefix_projection_status::limit_exceeded);
    CHECK(!refused_projection);

    // The canonical validator consumes a separate projection capability. It
    // authenticates only the selected prefix, but derives a fresh dense live
    // image: packed rows 7 and 2 become destination rows 0 and 1 for every
    // unit instead of preserving either source offset space.
    prefix_request.tokens = divergent;
    prefix_request.token_count = sizeof(divergent)/sizeof(divergent[0]);
    prefix_request.lcp_tokens = 2;
    vbr_artifact_attention_prefix_projection validated_projection;
    CHECK(catalog.project_attention_prefix(
              prefix_parent, prefix_request, {}, validated_projection) ==
          vbr_artifact_prefix_projection_status::projected);
    const auto accounting = ledger.snapshot();
    prefix_validator_serials validation_serials;
    validation_serials.accounting = accounting.serial;
    validation_serials.policy = 0x901;
    validation_serials.memory = 0x902;
    validation_serials.state = 0x903;
    validation_serials.tree = 0x904;

    vbr_target_validation_snapshot validation_target;
    validation_target.memory_instance_cookie = validation_serials.memory;
    validation_target.target_state_serial = validation_serials.state;
    validation_target.accounting_serial = validation_serials.accounting;
    validation_target.tree_shape_digest = validation_serials.tree;
    validation_target.policy_epoch = validation_serials.policy;
    validation_target.scheduler_idle = true;
    validation_target.destination_sequence_absent = true;
    vbr_target_child_snapshot validation_child;
    static const uint8_t validation_memory_cookie = 0;
    static const uint8_t validation_pool_cookie = 0;
    validation_child.child_id = 0;
    validation_child.dependency_mode =
        checkpoint_child_dependency_mode::live_guarded;
    validation_child.memory_cookie = &validation_memory_cookie;
    validation_child.empty = true;
    validation_child.dedicated = true;
    validation_child.armed = true;
    validation_child.lineage_uuid = { 0x901, 0x902 };
    validation_child.instance_id = { 0x903, 0x904 };
    validation_child.state_serial = validation_target.target_state_serial;
    validation_child.policy_epoch = validation_target.policy_epoch;
    validation_child.controller_policy =
        prefix_parent.manifest().controller_policy.front();
    const auto & source_controller =
        prefix_parent.manifest().generation.controllers.front();
    for (size_t unit_index = 0;
         unit_index < prefix_parent.units().size(); ++unit_index) {
        const auto & descriptor =
            prefix_parent.units()[unit_index].descriptor;
        vbr_target_unit_snapshot target_unit;
        target_unit.child_id = descriptor.child_id;
        target_unit.logical_unit_id = descriptor.logical_unit_id;
        target_unit.current_type = descriptor.current_type;
        target_unit.last_source_type = descriptor.last_source_type;
        target_unit.promote_hops = descriptor.promote_hops;
        target_unit.last_transition = descriptor.last_transition;
        target_unit.representation_kind = descriptor.representation.kind;
        target_unit.codec_id = descriptor.representation.codec_id;
        target_unit.codec_version = descriptor.representation.codec_version;
        target_unit.representation_reference_digest =
            descriptor.representation.reference_digest;
        target_unit.source_loss_history =
            descriptor.representation.source_loss_history;
        target_unit.checkpoint_codec_hops =
            descriptor.representation.checkpoint_codec_hops;
        target_unit.recoverability = descriptor.recoverability;
        target_unit.side = descriptor.side;
        target_unit.layout = descriptor.layout;
        target_unit.row_codec_version = descriptor.row_codec_version;
        target_unit.current_domain =
            source_controller.units[unit_index].domain;
        target_unit.codebook_digest = descriptor.codebook_digest;
        target_unit.rotation_digest = descriptor.rotation_digest;
        target_unit.meansub_digest = descriptor.meansub_digest;
        target_unit.n_stream = descriptor.n_stream;
        target_unit.unified = descriptor.unified;
        target_unit.wm_cells = descriptor.wm_cells;
        target_unit.rank = descriptor.rank;
        target_unit.dimensions = descriptor.dimensions;
        target_unit.row_alignment = descriptor.row_alignment;
        for (size_t shard_index = 0;
             shard_index < descriptor.shards.size(); ++shard_index) {
            const auto & shard = descriptor.shards[shard_index];
            target_unit.shards.push_back({
                uint32_t(shard_index), &validation_pool_cookie,
                bindings[shard.device_ordinal].domain,
                shard.topology_index, shard.device_ordinal,
                prefix_parent.topologies()[shard.topology_index].digest,
                shard.logical_offset, shard.row_count, shard.row_bytes,
                shard.payload_bytes,
            });
        }
        validation_child.units.push_back(std::move(target_unit));
    }
    validation_target.children.push_back(std::move(validation_child));

    const std::vector<llama_token> validated_tokens { 1, 2 };
    vbr_adopt_policy validation_policy;
    validation_policy.authorized = true;
    validation_policy.identity.execution_identity =
        validated_projection.identity().execution_identity;
    validation_policy.identity.adapter_config_identity =
        validated_projection.identity().adapter_config_identity;
    validation_policy.identity.media_content_identity =
        validated_projection.identity().media_content_identity;
    validation_policy.identity.sequence_epoch =
        validated_projection.identity().sequence_epoch;
    validation_policy.identity.requested_frontier = 2;
    validation_policy.identity.tokens = &validated_tokens;
    validation_policy.destination_sequence = 12;
    validation_policy.adoption_nonce = 0x905;
    validation_policy.domain_bindings = bindings;
    validation_policy.domain_bindings.push_back({
        UINT32_MAX, UINT16_MAX, host,
    });
    validation_policy.accounting_snapshot = &accounting;
    validation_policy.budget_config = &budget;
    validation_policy.context = &validation_serials;
    validation_policy.recheck_target_empty =
        prefix_validator_serials::recheck;
    validation_policy.read_accounting_serial =
        prefix_validator_serials::read_accounting;
    validation_policy.read_policy_epoch =
        prefix_validator_serials::read_policy;
    const auto parent_manifest_digest =
        prefix_parent.manifest().manifest_digest;
    auto validated = vbr_validate_attention_prefix_projection(
        validation_target, std::move(validated_projection),
        validation_policy);
    CHECK(!validated_projection);
    CHECK(validated.status == vbr_manifest_validation_status::validated);
    CHECK(validated.decision == vbr_import_decision::live_rebased);
    CHECK(validated.proof);
    CHECK(validated.proof->is_prefix_projection());
    CHECK(validated.proof->projection_transfer_ready());
    CHECK(validated.proof->source_artifact() == prefix_reference);
    CHECK(validated.proof->manifest_digest() != parent_manifest_digest);
    CHECK(validated.proof->authenticated_identity().requested_frontier == 2);
    CHECK(validated.proof->token_block().tokens == validated_tokens);
    CHECK(validated.proof->companions().empty());
    CHECK(validated.proof->children().size() == 2);
    CHECK(validated.proof->tracker_install().children.size() == 1);
    CHECK(validated.proof->tracker_install().children[0].transition ==
          vbr_tracker_install_transition::whole_import);
    CHECK(validated.proof->source_controller(0));
    CHECK(validated.proof->source_controller(0)->streams.size() == 1);
    CHECK(validated.proof->source_controller(0)->streams[0]
              .computation_frontier == 2);
    CHECK(validated.proof->source_controller(0)->streams[0]
              .captured_dependency_count == 2);
    CHECK(validated.proof->source_controller(0)->streams[0].pages.empty());
    size_t dense_placement_count = 0;
    size_t dense_placement_cells = 0;
    size_t validated_projection_runs = 0;
    for (size_t child_index = 0;
         child_index < validated.proof->children().size(); ++child_index) {
        const auto & child = validated.proof->children()[child_index];
        CHECK(child.descriptor.wm_cells == 2);
        CHECK(child.placements.size() == (child_index == 0 ? 1 : 0));
        dense_placement_count += child.placements.size();
        if (child_index == 0) {
            CHECK(child.placements[0].computation_frontier == 2);
            CHECK(child.placements[0].cells.size() == 2);
            CHECK(child.placements[0].cells[0].physical_cell == 0);
            CHECK(child.placements[0].cells[1].physical_cell == 1);
            dense_placement_cells += child.placements[0].cells.size();
        }
        CHECK(child.shards.size() == 1);
        CHECK(child.shards[0].projection_proof);
        CHECK(child.shards[0].projection_runs.size() == 2);
        CHECK(child.shards[0].projection_runs[0].source_offset == 7);
        CHECK(child.shards[0].projection_runs[0].destination_offset == 0);
        CHECK(child.shards[0].projection_runs[0].size == 1);
        CHECK(child.shards[0].projection_runs[1].source_offset == 2);
        CHECK(child.shards[0].projection_runs[1].destination_offset == 1);
        CHECK(child.shards[0].projection_runs[1].size == 1);
        validated_projection_runs +=
            child.shards[0].projection_runs.size();
    }
    // The dense destination placement is canonical child-wide metadata, not
    // a prefix-sized vector copied once per logical unit. Proof/source-run
    // consumption is likewise exactly once across the grouped inventory.
    CHECK(dense_placement_count == 1);
    CHECK(dense_placement_cells == validated_tokens.size());
    CHECK(validated_projection_runs ==
          validated.proof->source_projection().source_runs().size());
    for (size_t gate = 0; gate < 32; ++gate) {
        CHECK(validated.proof->projection_transfer_ready());
    }
    auto moved_prefix_proof = std::make_unique<vbr_validated_manifest>(
        std::move(*validated.proof));
    validated.proof = std::move(moved_prefix_proof);
    CHECK(validated.proof->authenticated_identity().tokens ==
          &validated.proof->token_block().tokens);

    vbr_adopt_stage_policy prefix_stage_policy;
    prefix_stage_policy.ledger = &ledger;
    prefix_stage_policy.budget = &budget;
    prefix_stage_policy.pinned_domain = pinned;
    prefix_stage_policy.pinned_ring_bytes = 32;
    prefix_stage_policy.chunk_bytes = 8;
    for (const auto & binding : bindings) {
        prefix_stage_policy.lanes.push_back({
            binding.domain, nullptr, nullptr, false,
        });
    }
    auto staged_prefix = vbr_stage_validated_manifest(
        std::move(validated.proof), prefix_stage_policy);
    if (staged_prefix.status != vbr_adopt_stage_status::staged) {
        fprintf(stderr, "prefix stage status: %s\n",
            vbr_adopt_stage_status_name(staged_prefix.status));
    }
    CHECK(staged_prefix.status == vbr_adopt_stage_status::staged);
    CHECK(staged_prefix.manifest);
    CHECK(staged_prefix.staged);
    CHECK(staged_prefix.staged->read_count() == 2);
    if (staged_prefix.staged) {
        for (const auto & read : staged_prefix.staged->reads()) {
            CHECK(read.size == 2);
            // Production staging must execute the restricted Merkle verifier,
            // not merely carry its root beside unauthenticated H2D ranges.
            CHECK(read.proof_verified_bytes == 8);
            CHECK(read.destination_offset == 0);
            CHECK(read.projection_ranges.size() == 2);
            CHECK(read.projection_ranges[0].source_offset == 7);
            CHECK(read.projection_ranges[0].size == 1);
            CHECK(read.projection_ranges[1].source_offset == 2);
            CHECK(read.projection_ranges[1].size == 1);
        }
    }
    staged_prefix.staged.reset();
    CHECK(staged_prefix.manifest &&
          staged_prefix.manifest->projection_transfer_ready());

    // Prefix selection and representation selection compose into one
    // authenticated transaction. The sparse two-row prefix is read using the
    // compact F16 source geometry, while its fresh destination image is sized
    // and addressed using the wider TURBO8 geometry. No parent suffix row is
    // admitted to the staged reads.
    vbr_artifact_attention_prefix_projection transformed_projection;
    CHECK(catalog.project_attention_prefix(
              prefix_parent, prefix_request, {}, transformed_projection) ==
          vbr_artifact_prefix_projection_status::projected);
    auto transformed_target = validation_target;
    const auto transform_accounting = ledger.snapshot();
    transformed_target.accounting_serial = transform_accounting.serial;
    validation_serials.accounting = transform_accounting.serial;
    auto & transformed_child = transformed_target.children[0];
    vbr_downward_policy_child downward_child;
    downward_child.initial_cursor = transformed_child.controller_policy.cursor;
    for (size_t unit_index = 0;
         unit_index < transformed_child.units.size(); ++unit_index) {
        auto & unit = transformed_child.units[unit_index];
        const auto source_type = static_cast<ggml_type>(
            prefix_parent.units()[unit_index].descriptor.current_type);
        const auto target_type = GGML_TYPE_TURBO8_0;
        downward_child.initial_types.push_back(source_type);
        downward_child.target_types.push_back(target_type);
        unit.current_type = target_type;
        unit.current_domain = vbr_repr_domain::full;
        unit.downward_supported = true;
        unit.downward_movable = true;
        unit.controller_floor_type = GGML_TYPE_TURBO1_TCQ;
        unit.downward_type = target_type;
        unit.downward_domain = vbr_repr_domain::full;
        unit.downward_recipe_id = VBR_DOWNWARD_RECIPE_ID;
        unit.downward_recipe_version = VBR_DOWNWARD_RECIPE_VERSION;
        CHECK(vbr_downward_resolve_recipe(
                  source_type, target_type, GGML_TYPE_TURBO1_TCQ, true,
                  unit.downward_recipe) ==
              vbr_downward_recipe_status::resolved);
        CHECK(unit.downward_recipe.n_edges != 0);
        for (size_t edge_index = 0;
             edge_index < unit.downward_recipe.n_edges; ++edge_index) {
            const auto & edge = unit.downward_recipe.edges[edge_index];
            downward_child.policy.steps.push_back({
                downward_child.policy.steps.size(), unit_index,
                int32_t(edge.source_type),
                int32_t(edge.target_type), 1,
            });
        }
        unit.downward_row_bytes = 2;
        unit.downward_mapped_bytes = 16;
        unit.downward_transfer_bytes = 2;
        unit.downward_codec_workspace_bytes = 64;
        unit.downward_meansub_model_id = 7;
        unit.shards[0].row_bytes = 2;
        unit.shards[0].mapped_bytes = 16;
    }
    downward_child.policy.terminal_progress =
        int64_t(downward_child.policy.steps.size());
    const auto downward_projection =
        vbr_downward_project_policy_prefix({ downward_child });
    CHECK(downward_projection.status ==
          vbr_downward_policy_status::coherent);
    transformed_child.controller_policy.cursor =
        downward_projection.final_cursors[0];
    transformed_child.controller_policy.current_type_vector_digest =
        downward_projection.child_type_digests[0];
    for (auto & unit : transformed_child.units) {
        unit.downward_build_identity_digest = vbr_downward_build_identity(
            unit.downward_recipe, unit.downward_meansub_model_id,
            unit.meansub_digest,
            downward_projection.child_type_digests[0],
            downward_projection.tree_digest);
    }
    vbr_import_destination_projection destination;
    destination.status =
        vbr_import_destination_status::feasible_degraded;
    destination.prefix = downward_projection.prefix;
    destination.initial_types = { downward_child.initial_types };
    destination.initial_cursors = { downward_child.initial_cursor };
    destination.final_types = downward_projection.final_types;
    destination.final_cursors = {
        transformed_child.controller_policy.cursor,
    };
    destination.child_type_digests =
        downward_projection.child_type_digests;
    destination.tree_digest = downward_projection.tree_digest;
    vbr_import_schedule_quote transformed_quote;
    CHECK(vbr_quote_import_schedule(
        transformed_target, prefix_parent, transformed_quote));
    CHECK(transformed_quote.status() ==
          vbr_import_schedule_status::downward);
    CHECK(vbr_rebind_import_schedule_quote(
        transformed_target, prefix_parent, destination,
        transformed_quote));
    llama_cache_budget_plan transform_plan;
    transform_plan.accounting_serial = transform_accounting.serial;
    auto transformed_policy = validation_policy;
    transformed_policy.accounting_snapshot = &transform_accounting;
    transformed_policy.adoption_nonce = 0x906;
    transformed_policy.schedule_quote = &transformed_quote;
    transformed_policy.downward_projection = &downward_projection;
    transformed_policy.transform_budget_plan = &transform_plan;
    validation_serials.transform_tree = downward_projection.tree_digest;
    transformed_policy.read_transform_tree_digest =
        prefix_validator_serials::read_transform_tree;
    auto transformed = vbr_validate_attention_prefix_projection(
        transformed_target, std::move(transformed_projection),
        transformed_policy);
    CHECK(transformed.status ==
          vbr_manifest_validation_status::validated);
    CHECK(transformed.decision ==
          vbr_import_decision::downward_rebase);
    CHECK(transformed.proof);
    CHECK(transformed.proof->children().size() == 2);
    for (const auto & plan : transformed.proof->children()) {
        CHECK(plan.transform_kind ==
              vbr_import_transform_kind::downward);
        CHECK(plan.transfer_bytes == 2);
        CHECK(plan.target_mapped_bytes == 4);
        CHECK(plan.shards.size() == 1);
        CHECK(plan.shards[0].row_bytes == 1);
        CHECK(plan.shards[0].target_row_bytes == 2);
        CHECK(plan.shards[0].payload_bytes == 2);
        CHECK(plan.shards[0].target_mapped_bytes == 4);
    }
    auto transformed_stage_policy = prefix_stage_policy;
    transformed_stage_policy.transform_context = nullptr;
    transformed_stage_policy.reserve_transform = [](
            void *, const std::vector<vbr_validated_child_plan> & plans,
            llama_cache_acct_ledger &,
            const llama_cache_budget_config &,
            vbr_downward_stage_reservation & output) noexcept {
        if (plans.size() != 2 || std::any_of(
                plans.begin(), plans.end(),
                [](const vbr_validated_child_plan & plan) {
                    return plan.transform_kind !=
                        vbr_import_transform_kind::downward;
                })) {
            return false;
        }
        output.status = vbr_downward_reserve_status::reserved;
        return true;
    };
    auto staged_transform = vbr_stage_validated_manifest(
        std::move(transformed.proof), transformed_stage_policy);
    if (staged_transform.status != vbr_adopt_stage_status::staged) {
        fprintf(stderr, "transformed prefix stage status: %s\n",
            vbr_adopt_stage_status_name(staged_transform.status));
    }
    CHECK(staged_transform.status == vbr_adopt_stage_status::staged);
    CHECK(staged_transform.manifest);
    CHECK(staged_transform.manifest->decision() ==
          vbr_import_decision::downward_rebase);
    CHECK(staged_transform.staged);
    CHECK(staged_transform.staged->read_count() == 2);
    if (staged_transform.staged) {
        for (const auto & read : staged_transform.staged->reads()) {
            CHECK(read.size == 2);
            CHECK(read.destination_offset == 0);
            CHECK(read.projection_ranges.size() == 2);
            CHECK(read.projection_ranges[0].source_offset == 7);
            CHECK(read.projection_ranges[1].source_offset == 2);
        }
    }
    staged_transform.staged.reset();
    staged_transform.manifest.reset();

    // The prefix path consumes the same immutable schedule capability as the
    // full-frontier validator. A target snapshot drift must not be accepted
    // merely because the quoted direction still says "downward".
    vbr_artifact_attention_prefix_projection stale_quote_projection;
    CHECK(catalog.project_attention_prefix(
              prefix_parent, prefix_request, {}, stale_quote_projection) ==
          vbr_artifact_prefix_projection_status::projected);
    auto stale_quote_target = transformed_target;
    ++stale_quote_target.target_state_serial;
    auto stale_quote_result = vbr_validate_attention_prefix_projection(
        stale_quote_target, std::move(stale_quote_projection),
        transformed_policy);
    CHECK(stale_quote_result.status ==
          vbr_manifest_validation_status::unavailable);
    CHECK(!stale_quote_result.proof);

    // A target topology mutation consumes and releases the attempted
    // capability without affecting the already-validated sibling proof.
    vbr_artifact_attention_prefix_projection topology_mutant_projection;
    CHECK(catalog.project_attention_prefix(
              prefix_parent, prefix_request, {},
              topology_mutant_projection) ==
          vbr_artifact_prefix_projection_status::projected);
    auto topology_mutant = validation_target;
    topology_mutant.children[0].units[0].shards[0].topology_digest = {};
    auto refused_validation = vbr_validate_attention_prefix_projection(
        topology_mutant, std::move(topology_mutant_projection),
        validation_policy);
    CHECK(!topology_mutant_projection);
    CHECK(refused_validation.status ==
          vbr_manifest_validation_status::topology_mismatch);
    CHECK(!refused_validation.proof);

    // Dropping the caller's package lease cannot retire storage while either
    // independently borrowed projection remains alive.
    prefix_parent.reset();
    CHECK(catalog.retire(prefix_reference) ==
          vbr_artifact_retire_status::busy);
    prefix_projection.reset();
    CHECK(catalog.retire(prefix_reference) ==
          vbr_artifact_retire_status::busy);
    shorter_projection.reset();
    staged_prefix.manifest.reset();
    CHECK(catalog.retire(prefix_reference) ==
          vbr_artifact_retire_status::retired);
    CHECK(ledger.snapshot().live_ops == 0);
}

static void test_manifest_coherent_assembly() {
    vbr_capture_projection_manifest first;
    first.manifest_id = 10;
    first.placements.push_back(projected_placement(10, 10, { 1, 2 }));
    vbr_capture_projection_manifest second;
    second.manifest_id = 20;
    auto second_placement = projected_placement(20, 20, { 3, 4 });
    second_placement.child_id = 1;
    second.placements.push_back(std::move(second_placement));
    vbr_capture_projection_manifest shared;
    shared.manifest_id = 30;
    shared.placements.push_back(projected_placement(30, 30, { 2, 5 }));
    auto shared_second = projected_placement(30, 31, { 4, 6 });
    shared_second.child_id = 1;
    shared.placements.push_back(std::move(shared_second));
    vbr_capture_projection projection;
    CHECK(vbr_artifact_project_capture_union(
        { 707, { shared, second, first } }, {}, projection));

    const auto generation_a = projected_generation(5);
    const auto generation_b = projected_generation(7, GGML_TYPE_Q8_0);
    const auto generation_c = projected_generation(9, GGML_TYPE_Q6_K);
    auto target_first = projected_target(10, 0, 101, generation_a);
    auto target_second = projected_target(20, 1, 202, generation_b);
    auto target_shared_a = projected_target(30, 0, 101, generation_a);
    auto target_shared_b = projected_target(30, 1, 303, generation_c);

    // Representation equality deliberately ignores reference-local identity,
    // but authenticates policy and the complete pointer-free unit schema.
    CHECK(vbr_capture_controller_representation_equal(
        target_first, target_shared_a));
    auto schema_mutant = target_shared_a;
    schema_mutant.policy.cursor++;
    CHECK(!vbr_capture_controller_representation_equal(
        target_first, schema_mutant));
    schema_mutant = target_shared_a;
    schema_mutant.unit_descriptors[0].shards[0].payload_bytes++;
    CHECK(!vbr_capture_controller_representation_equal(
        target_first, schema_mutant));
    schema_mutant = target_shared_a;
    schema_mutant.unit_descriptors[0].shards[0].section_checksum[0] ^= 1;
    CHECK(!vbr_capture_controller_representation_equal(
        target_first, schema_mutant));
    schema_mutant = target_shared_a;
    schema_mutant.unit_descriptors[0].meansub_layer++;
    CHECK(!vbr_capture_controller_representation_equal(
        target_first, schema_mutant));
    schema_mutant = target_shared_a;
    schema_mutant.unit_descriptors[0].meansub_baked = false;
    CHECK(!vbr_capture_controller_representation_equal(
        target_first, schema_mutant));

    auto stash_target = target_first;
    auto stash_peer = target_shared_a;
    for (auto * value : { &stash_target, &stash_peer }) {
        auto & descriptor = value->unit_descriptors[0];
        descriptor.clean_stash_state =
            vbr_artifact_clean_stash_state::present;
        descriptor.clean_stash.valid_rows = 4;
        descriptor.clean_stash.domain = vbr_repr_domain::full;
        descriptor.clean_stash.layout = vbr_artifact_layout::row_major;
        descriptor.clean_stash.row_count = 4;
        descriptor.clean_stash.column_count = 1;
        descriptor.clean_stash.row_bytes = 2;
        std::array<uint8_t, 32> stash_id = {};
        stash_id.fill(0x5a);
        descriptor.clean_stash.payload_id =
            vbr_stash_payload_id::from_sha256(stash_id);
        auto stash_shard = descriptor.shards[0];
        stash_shard.row_count = 4;
        stash_shard.row_bytes = 2;
        stash_shard.payload_bytes = 8;
        stash_shard.section_checksum.fill(0x6b);
        descriptor.clean_stash.shards = { stash_shard };
    }
    CHECK(vbr_capture_controller_representation_equal(
        stash_target, stash_peer));
    stash_peer.unit_descriptors[0].clean_stash.row_bytes++;
    CHECK(!vbr_capture_controller_representation_equal(
        stash_target, stash_peer));
    stash_peer = stash_target;
    stash_peer.unit_descriptors[0].clean_stash.shards[0]
        .section_checksum[0] ^= 1;
    CHECK(!vbr_capture_controller_representation_equal(
        stash_target, stash_peer));

    std::vector<vbr_capture_controller_target> targets {
        target_shared_b, target_first, target_shared_a, target_second,
    };
    std::vector<vbr_capture_projected_unit> units {
        capture_projected_unit_for_target(projection, target_shared_b),
        capture_projected_unit_for_target(projection, target_second),
        capture_projected_unit_for_target(projection, target_first),
    };

    projected_controller_fixture controller;
    vbr_capture_manifest_assembly assembled;
    CHECK(assemble_projected_test_batch(
        projection, targets, units, controller.provider(), {}, assembled));
    CHECK(bool(assembled));
    CHECK(controller.rechecked == std::vector<uint64_t>({ 10, 20, 30 }));
    CHECK(assembled.controller_targets().size() == 4);
    CHECK(assembled.projected_units().size() == 3);
    CHECK(assembled.range_proofs().size() == 4);
    CHECK(assembled.manifests().size() == 3);
    CHECK(assembled.manifests()[0].manifest_id == 10);
    CHECK(assembled.manifests()[1].manifest_id == 20);
    CHECK(assembled.manifests()[2].manifest_id == 30);
    for (const auto & manifest : assembled.manifests()) {
        CHECK(manifest.state == vbr_capture_manifest_state::ready);
    }
    const auto * manifest_first = projected_manifest(assembled, 10);
    const auto * manifest_shared = projected_manifest(assembled, 30);
    CHECK(manifest_first != nullptr);
    CHECK(manifest_shared != nullptr);
    if (manifest_first && manifest_shared) {
        CHECK(manifest_first->controller_count == 1);
        CHECK(manifest_first->unit_count == 1);
        CHECK(manifest_first->range_proof_count == 1);
        CHECK(manifest_shared->controller_count == 2);
        CHECK(manifest_shared->unit_count == 2);
        CHECK(manifest_shared->range_proof_count == 2);
        const uint32_t first_unit = assembled.unit_references()[
            manifest_first->first_unit];
        const uint32_t shared_unit = assembled.unit_references()[
            manifest_shared->first_unit];
        CHECK(first_unit == shared_unit);
        const auto & first_proof = assembled.range_proofs()[
            manifest_first->first_range_proof].proof;
        const auto & shared_proof = assembled.range_proofs()[
            manifest_shared->first_range_proof].proof;
        CHECK(first_proof.ranges().size() == 1);
        CHECK(shared_proof.ranges().size() == 1);
        if (first_proof.ranges().size() == 1 &&
            shared_proof.ranges().size() == 1) {
            CHECK(first_proof.ranges()[0].offset == 0);
            CHECK(first_proof.ranges()[0].size == 2);
            CHECK(shared_proof.ranges()[0].offset == 1);
            CHECK(shared_proof.ranges()[0].size == 2);
        }
    }
    for (const auto & range : assembled.range_proofs()) {
        CHECK(range.unit_index < assembled.projected_units().size());
        if (range.unit_index >= assembled.projected_units().size()) {
            continue;
        }
        const auto & unit = assembled.projected_units()[range.unit_index];
        CHECK(range.shard_index < unit.shards().size());
        if (range.shard_index < unit.shards().size()) {
            uint64_t bytes_read = 0;
            CHECK(vbr_capture_range_verify(
                range.proof,
                unit.shards()[range.shard_index].bytes->source(),
                &bytes_read));
            CHECK(bytes_read ==
                  unit.shards()[range.shard_index].bytes->size());
        }
    }
    uint64_t all_ready_proof_metadata = 0;
    for (const auto & range : assembled.range_proofs()) {
        all_ready_proof_metadata += range.proof.metadata_bytes() +
            2*sizeof(vbr_capture_manifest_range_proof);
    }
    CHECK(all_ready_proof_metadata > 80);

    // Losing one target generation invalidates only manifests that name it;
    // independently sealed units and manifests remain available.
    auto partial_units = units;
    partial_units.erase(partial_units.begin());
    controller = {};
    CHECK(assemble_projected_test_batch(
        projection, targets, partial_units,
        controller.provider(), {}, assembled));
    CHECK(projected_manifest(assembled, 10)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 20)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 30)->state ==
          vbr_capture_manifest_state::dependency_unavailable);
    CHECK(projected_manifest(assembled, 30)->range_proof_count == 0);
    CHECK(assembled.projected_units().size() == 2);
    CHECK(assembled.range_proofs().size() == 2);

    auto stale_targets = targets;
    stale_targets.front().controller_generation = 404;
    stale_targets.front().policy.cursor = 404;
    controller = {};
    CHECK(assemble_projected_test_batch(
        projection, stale_targets, units,
        controller.provider(), {}, assembled));
    CHECK(projected_manifest(assembled, 10)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 20)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 30)->state ==
          vbr_capture_manifest_state::dependency_unavailable);
    CHECK(assembled.projected_units().size() == 2);
    CHECK(projected_manifest(assembled, 30)->controller_count == 0);
    CHECK(projected_manifest(assembled, 30)->unit_count == 0);

    stale_targets = targets;
    stale_targets.front().units[0].repr_gen++;
    stale_targets.front().unit_descriptors[0].repr_gen++;
    controller = {};
    CHECK(assemble_projected_test_batch(
        projection, stale_targets, units,
        controller.provider(), {}, assembled));
    CHECK(projected_manifest(assembled, 10)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 30)->state ==
          vbr_capture_manifest_state::dependency_unavailable);

    stale_targets = targets;
    stale_targets.front().lineage_uuid.lo++;
    stale_targets.front().unit_descriptors[0].lineage_uuid.lo++;
    controller = {};
    CHECK(assemble_projected_test_batch(
        projection, stale_targets, units,
        controller.provider(), {}, assembled));
    CHECK(projected_manifest(assembled, 10)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 20)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 30)->state ==
          vbr_capture_manifest_state::dependency_unavailable);

    auto missing_targets = targets;
    missing_targets.erase(missing_targets.begin());
    controller = {};
    CHECK(assemble_projected_test_batch(
        projection, missing_targets, units,
        controller.provider(), {}, assembled));
    CHECK(controller.rechecked == std::vector<uint64_t>({ 10, 20 }));
    CHECK(projected_manifest(assembled, 10)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 20)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 30)->state ==
          vbr_capture_manifest_state::dependency_unavailable);

    controller = {};
    CHECK(assemble_projected_test_batch(
        projection, {}, units, controller.provider(), {}, assembled));
    CHECK(controller.rechecked.empty());
    CHECK(assembled.manifests().size() == 3);
    CHECK(std::all_of(
        assembled.manifests().begin(), assembled.manifests().end(),
        [](const auto & manifest) {
            return manifest.state ==
                vbr_capture_manifest_state::dependency_unavailable;
        }));
    CHECK(assembled.projected_units().empty());

    // A manifest cannot relabel an otherwise shared captured unit. Complete
    // descriptor schema is controller-certified across every alias.
    auto mismatched_schema = targets;
    mismatched_schema[2].unit_descriptors[0].side =
        vbr_artifact_side::value;
    CHECK(!assemble_projected_test_batch(
        projection, mismatched_schema, units,
        controller.provider(), {}, assembled));
    CHECK(!assembled);

    controller = {};
    controller.rejected_manifest = 30;
    CHECK(assemble_projected_test_batch(
        projection, targets, units, controller.provider(), {}, assembled));
    CHECK(projected_manifest(assembled, 10)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 20)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 30)->state ==
          vbr_capture_manifest_state::dependency_unavailable);
    CHECK(assembled.projected_units().size() == 2);

    // Exact flat-arena bounds are accepted; one-less limits fail
    // transactionally and clear an earlier successful capability.
    vbr_capture_manifest_assembly_limits exact;
    exact.max_controller_targets = 4;
    exact.max_projected_units = 3;
    exact.max_unit_descriptor_shards = 4;
    exact.max_unit_descriptor_metadata_bytes =
        4*(sizeof(vbr_artifact_unit_descriptor) +
           sizeof(vbr_artifact_shard_descriptor));
    exact.max_manifests = 3;
    exact.max_controller_references = 4;
    exact.max_unit_references = 4;
    exact.max_range_proofs = 4;
    exact.max_range_proof_metadata_bytes = all_ready_proof_metadata;
    controller = {};
    CHECK(assemble_projected_test_batch(
        projection, targets, units, controller.provider(), exact, assembled));
    const auto expect_limit_refusal = [&](auto tighten) {
        auto limited = exact;
        tighten(limited);
        CHECK(!assemble_projected_test_batch(
            projection, targets, units,
            controller.provider(), limited, assembled));
        CHECK(!assembled);
    };
    expect_limit_refusal([](auto & value) {
        value.max_controller_targets = 3;
    });
    expect_limit_refusal([](auto & value) {
        value.max_projected_units = 2;
    });
    expect_limit_refusal([](auto & value) {
        value.max_unit_descriptor_shards = 3;
    });
    expect_limit_refusal([](auto & value) {
        value.max_unit_descriptor_metadata_bytes--;
    });
    expect_limit_refusal([](auto & value) {
        value.max_manifests = 2;
    });
    expect_limit_refusal([](auto & value) {
        value.max_controller_references = 3;
    });
    expect_limit_refusal([](auto & value) {
        value.max_unit_references = 3;
    });
    expect_limit_refusal([](auto & value) {
        value.max_range_proofs = 3;
    });
    expect_limit_refusal([&](auto & value) {
        value.max_range_proof_metadata_bytes =
            all_ready_proof_metadata - 1;
    });

    auto conflicting_targets = targets;
    conflicting_targets[2].units[0].repr_gen++;
    conflicting_targets[2].policy.current_type_vector_digest =
        target_shared_a.policy.current_type_vector_digest;
    CHECK(!assemble_projected_test_batch(
        projection, conflicting_targets, units,
        controller.provider(), {}, assembled));
    CHECK(!assembled);

    conflicting_targets = targets;
    conflicting_targets[2].lineage_uuid.lo++;
    CHECK(!assemble_projected_test_batch(
        projection, conflicting_targets, units,
        controller.provider(), {}, assembled));
    CHECK(!assembled);

    auto duplicate_targets = targets;
    duplicate_targets.push_back(target_first);
    CHECK(!assemble_projected_test_batch(
        projection, duplicate_targets, units,
        controller.provider(), {}, assembled));
    CHECK(!assembled);

    auto malformed_units = units;
    malformed_units.front() = {};
    CHECK(!assemble_projected_test_batch(
        projection, targets, malformed_units,
        controller.provider(), {}, assembled));
    CHECK(!assembled);

    // One controller tuple spans all streams of a child; every projected
    // stream must supply the complete unit vector before the manifest is ready.
    vbr_capture_projection_manifest multi_stream;
    multi_stream.manifest_id = 40;
    multi_stream.placements.push_back(
        projected_placement(40, 40, { 1, 3 }));
    auto stream_one_placement =
        projected_placement(40, 41, { 2, 4 });
    stream_one_placement.stream_index = 1;
    multi_stream.placements.push_back(std::move(stream_one_placement));
    vbr_capture_projection stream_projection;
    CHECK(vbr_artifact_project_capture_union(
        { 707, { multi_stream } }, {}, stream_projection));
    auto stream_target = projected_target(40, 0, 505, generation_a);
    projected_target_streams(stream_target, 2);
    std::vector<vbr_capture_controller_target> stream_targets {
        stream_target,
    };
    std::vector<vbr_capture_projected_unit> stream_units {
        capture_projected_unit_for_target(stream_projection, stream_target, 1),
        capture_projected_unit_for_target(stream_projection, stream_target, 0),
    };
    controller = {};
    CHECK(assemble_projected_test_batch(
        stream_projection, stream_targets, stream_units,
        controller.provider(), {}, assembled));
    CHECK(projected_manifest(assembled, 40)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 40)->controller_count == 1);
    CHECK(projected_manifest(assembled, 40)->unit_count == 2);
    stream_units.pop_back();
    CHECK(assemble_projected_test_batch(
        stream_projection, stream_targets, stream_units,
        controller.provider(), {}, assembled));
    CHECK(projected_manifest(assembled, 40)->state ==
          vbr_capture_manifest_state::dependency_unavailable);

    vbr_capture_projection_manifest selected_stream;
    selected_stream.manifest_id = 50;
    auto selected_placement = projected_placement(50, 50, { 2, 4 });
    selected_placement.stream_index = 1;
    selected_stream.placements.push_back(std::move(selected_placement));
    vbr_capture_projection selected_projection;
    CHECK(vbr_artifact_project_capture_union(
        { 707, { selected_stream } }, {}, selected_projection));
    auto selected_target = projected_target(50, 0, 606, generation_b);
    projected_target_streams(selected_target, 2);
    std::vector<vbr_capture_controller_target> selected_targets {
        selected_target,
    };
    std::vector<vbr_capture_projected_unit> selected_units {
        capture_projected_unit_for_target(
            selected_projection, selected_target, 1),
    };
    controller = {};
    CHECK(assemble_projected_test_batch(
        selected_projection, selected_targets, selected_units,
        controller.provider(), {}, assembled));
    CHECK(projected_manifest(assembled, 50)->state ==
          vbr_capture_manifest_state::ready);
    CHECK(projected_manifest(assembled, 50)->controller_count == 1);
    CHECK(projected_manifest(assembled, 50)->unit_count == 1);

    // Manifest proofs select only their own canonical chunks even when one
    // projected unit physically packs several independent prefixes.
    vbr_capture_projection_manifest chunk_zero;
    chunk_zero.manifest_id = 60;
    chunk_zero.placements.push_back(
        projected_placement(60, 60, { 0 }));
    vbr_capture_projection_manifest chunk_one;
    chunk_one.manifest_id = 70;
    chunk_one.placements.push_back(
        projected_placement(70, 70, { 3 }));
    vbr_capture_projection chunk_projection;
    CHECK(vbr_artifact_project_capture_union(
        { 707, { chunk_zero, chunk_one } }, {}, chunk_projection));
    auto chunk_target_zero = projected_target(60, 0, 707, generation_a);
    auto chunk_target_one = projected_target(70, 0, 707, generation_a);
    projected_target_row_bytes(
        chunk_target_zero, VBR_CAPTURE_RANGE_CHUNK_BYTES);
    projected_target_row_bytes(
        chunk_target_one, VBR_CAPTURE_RANGE_CHUNK_BYTES);
    std::vector<vbr_capture_controller_target> chunk_targets {
        chunk_target_zero, chunk_target_one,
    };
    std::vector<vbr_capture_projected_unit> chunk_units {
        capture_projected_unit_for_target(
            chunk_projection, chunk_target_zero, 0,
            VBR_CAPTURE_RANGE_CHUNK_BYTES),
    };
    controller = {};
    CHECK(assemble_projected_test_batch(
        chunk_projection, chunk_targets, chunk_units,
        controller.provider(), {}, assembled));
    const auto * manifest_zero = projected_manifest(assembled, 60);
    const auto * manifest_one = projected_manifest(assembled, 70);
    CHECK(manifest_zero && manifest_one);
    if (manifest_zero && manifest_one) {
        CHECK(manifest_zero->range_proof_count == 1);
        CHECK(manifest_one->range_proof_count == 1);
        const auto & proof_zero = assembled.range_proofs()[
            manifest_zero->first_range_proof].proof;
        const auto & proof_one = assembled.range_proofs()[
            manifest_one->first_range_proof].proof;
        CHECK(proof_zero.ranges().size() == 1);
        CHECK(proof_one.ranges().size() == 1);
        if (proof_zero.ranges().size() == 1 &&
            proof_one.ranges().size() == 1) {
            CHECK(proof_zero.ranges()[0].offset == 0);
            CHECK(proof_zero.ranges()[0].size ==
                  VBR_CAPTURE_RANGE_CHUNK_BYTES);
            CHECK(proof_one.ranges()[0].offset ==
                  VBR_CAPTURE_RANGE_CHUNK_BYTES);
            CHECK(proof_one.ranges()[0].size ==
                  VBR_CAPTURE_RANGE_CHUNK_BYTES);
        }
        const auto unit_index = assembled.range_proofs()[
            manifest_zero->first_range_proof].unit_index;
        const auto & bytes = assembled.projected_units()[unit_index].
            shards()[0].bytes;
        uint64_t zero_bytes = 0;
        uint64_t one_bytes = 0;
        CHECK(vbr_capture_range_verify(
            proof_zero, bytes->source(), &zero_bytes));
        CHECK(vbr_capture_range_verify(
            proof_one, bytes->source(), &one_bytes));
        CHECK(zero_bytes == VBR_CAPTURE_RANGE_CHUNK_BYTES);
        CHECK(one_bytes == VBR_CAPTURE_RANGE_CHUNK_BYTES);
    }
}

struct generated_h2d_source {
    uint64_t size = 0;

    static uint8_t byte_at(uint64_t offset) noexcept {
        return uint8_t((offset*29 + 17) & 0xff);
    }

    static bool read(
            const void * context, uint64_t offset,
            uint8_t * destination, size_t size) noexcept {
        const auto & source =
            *static_cast<const generated_h2d_source *>(context);
        if (offset > source.size || size > source.size - offset) {
            return false;
        }
        for (size_t i = 0; i < size; ++i) {
            destination[i] = byte_at(offset + i);
        }
        return true;
    }
};

struct fake_h2d_destination {
    struct pending {
        uint64_t offset = 0;
        const uint8_t * data = nullptr;
        size_t size = 0;
        uint64_t digest = 0;
    };
    std::unordered_map<uint64_t, pending> operations;
    uint64_t bytes = 0;
    bool valid = true;

    static uint64_t digest(const uint8_t * data, size_t size) noexcept {
        uint64_t value = 1469598103934665603ull;
        for (size_t i = 0; i < size; ++i) {
            value = (value ^ data[i])*1099511628211ull;
        }
        return value;
    }

    static bool issue(
            void * context, uint64_t ticket, uint64_t offset,
            const uint8_t * data, size_t size,
            bool) noexcept {
        auto & destination =
            *static_cast<fake_h2d_destination *>(context);
        if (!data || size == 0 || destination.operations.count(ticket) != 0) {
            return false;
        }
        for (size_t i = 0; i < size; ++i) {
            if (data[i] != generated_h2d_source::byte_at(offset + i)) {
                destination.valid = false;
                return false;
            }
        }
        destination.operations.emplace(ticket, pending {
            offset, data, size, digest(data, size),
        });
        return true;
    }

    static bool complete(void * context, uint64_t ticket) noexcept {
        auto & destination =
            *static_cast<fake_h2d_destination *>(context);
        const auto found = destination.operations.find(ticket);
        if (found == destination.operations.end()) {
            return false;
        }
        // If the ring reused this pinned chunk before completion, its digest
        // changed and the fake event fails.
        if (digest(found->second.data, found->second.size) !=
                found->second.digest) {
            destination.valid = false;
            return false;
        }
        destination.bytes += found->second.size;
        destination.operations.erase(found);
        return true;
    }
};

static void test_h2d_bounded_streaming() {
    const uint64_t transfer_bytes =
        VBR_PINNED_RING_MAX_BYTES + 1024*1024 + 17;
    generated_h2d_source generated { transfer_bytes };
    vbr_h2d_status status;
    auto ring = vbr_h2d_chunk_ring::create(
        { {} }, 8*1024*1024, 4*1024*1024, status);
    CHECK(ring && status == vbr_h2d_status::ok);
    CHECK(ring->capacity_bytes() < transfer_bytes);

    fake_h2d_destination event_destination;
    vbr_h2d_transfer transfer;
    transfer.source = {
        transfer_bytes, &generated, generated_h2d_source::read,
    };
    transfer.size = transfer_bytes;
    transfer.fake = {
        &event_destination,
        fake_h2d_destination::issue,
        fake_h2d_destination::complete,
        true,
    };
    vbr_h2d_stats stats;
    CHECK(ring->stream(transfer, stats) == vbr_h2d_status::ok);
    CHECK(event_destination.valid);
    CHECK(event_destination.operations.empty());
    CHECK(event_destination.bytes == transfer_bytes);
    CHECK(stats.bytes == transfer_bytes);
    CHECK(stats.backpressure_waits > 0);
    CHECK(stats.event_completions == stats.chunks);
    CHECK(stats.synchronous_fallbacks == 0);
    CHECK(stats.peak_pinned_bytes <= ring->capacity_bytes());

    fake_h2d_destination sync_destination;
    transfer.fake.context = &sync_destination;
    transfer.fake.supports_events = false;
    CHECK(ring->stream(transfer, stats) == vbr_h2d_status::ok);
    CHECK(sync_destination.valid);
    CHECK(sync_destination.bytes == transfer_bytes);
    CHECK(stats.synchronous_fallbacks == stats.chunks);
    CHECK(stats.event_completions == 0);

    fake_h2d_destination failed;
    transfer.fake.context = &failed;
    transfer.fake.supports_events = true;
    transfer.fail_completion_at = 1;
    CHECK(ring->stream(transfer, stats) ==
          vbr_h2d_status::transfer_failed);
    CHECK(failed.operations.empty());

    // Production uses one physical direction-neutral ring. The two
    // adapters share its exact capacity and contend at the whole-operation
    // boundary rather than consuming one another's outstanding chunks.
    vbr_pinned_ring_create_failure failure;
    const uint64_t live_before =
        vbr_pinned_ring_live_capacity_bytes();
    auto core = std::shared_ptr<vbr_bounded_pinned_ring_core>(
        vbr_bounded_pinned_ring_core::create(
            { {} }, 16, 8, nullptr, failure));
    CHECK(core);
    CHECK(failure == vbr_pinned_ring_create_failure::none);
    auto capture = vbr_pinned_chunk_ring::attach(core);
    auto adoption = vbr_h2d_chunk_ring::attach(core, { { } });
    CHECK(capture && adoption);
    CHECK(capture && capture->capacity_bytes() == 16);
    CHECK(adoption && adoption->capacity_bytes() == 16);
    CHECK(vbr_pinned_ring_live_capacity_bytes() == live_before + 16);

    blocking_capture_source d2h_input;
    d2h_input.source.bytes.resize(9, 0x5a);
    vbr_capture_stream_source d2h_source;
    d2h_source.size = d2h_input.source.bytes.size();
    d2h_source.context = &d2h_input;
    d2h_source.read = blocking_capture_source::read;
    artifact_segment_chain blocked_chain;
    vbr_capture_stream_stats blocked_stats;
    generated_h2d_source h2d_input { 9 };
    fake_h2d_destination blocked_destination;
    vbr_h2d_transfer blocked_transfer;
    blocked_transfer.source = {
        h2d_input.size, &h2d_input, generated_h2d_source::read,
    };
    blocked_transfer.size = h2d_input.size;
    blocked_transfer.fake = {
        &blocked_destination, fake_h2d_destination::issue,
        fake_h2d_destination::complete, true,
    };
    vbr_capture_stream_status capture_result =
        vbr_capture_stream_status::internal_error;
    std::thread capture_thread([&]() {
        capture_result = capture->stream(
            d2h_source, blocked_chain, blocked_stats);
    });
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(1);
    while (!d2h_input.entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    CHECK(d2h_input.entered.load(std::memory_order_acquire));
    CHECK(adoption->stream(blocked_transfer, stats) ==
          vbr_h2d_status::ring_unavailable);
    CHECK(blocked_chain.size() == 0);
    CHECK(blocked_destination.bytes == 0);
    d2h_input.release.store(true, std::memory_order_release);
    capture_thread.join();
    CHECK(capture_result == vbr_capture_stream_status::ok);
    CHECK(read_chain(blocked_chain) == d2h_input.source.bytes);
    CHECK(adoption->stream(blocked_transfer, stats) == vbr_h2d_status::ok);
    CHECK(blocked_destination.valid);
    CHECK(blocked_destination.bytes == h2d_input.size);

    auto retained_operation = capture->try_begin_operation();
    CHECK(retained_operation);
    capture.reset();
    adoption.reset();
    CHECK(vbr_pinned_ring_live_capacity_bytes() == live_before + 16);
    core.reset();
    CHECK(vbr_pinned_ring_live_capacity_bytes() == live_before + 16);
    retained_operation = {};
    CHECK(vbr_pinned_ring_live_capacity_bytes() == live_before);
}

static uint64_t ring_resident(
        llama_cache_acct_ledger & ledger,
        const llama_cache_acct_resource_domain & domain) {
    for (const auto & row : ledger.snapshot().cells) {
        if (row.category ==
                llama_cache_acct_category::pinned_preimage_ring &&
            row.domain == domain) {
            return row.cell.measures[size_t(
                llama_cache_acct_measure::
                    resident_allocated)].value;
        }
    }
    return 0;
}

struct ring_charge_fault_context {
    llama_cache_acct_ledger * ledger = nullptr;
};

static void inject_ring_charge_fault(void * opaque) noexcept {
    auto & context = *static_cast<ring_charge_fault_context *>(opaque);
    context.ledger->gauge_set(
        llama_cache_acct_category::pinned_preimage_ring,
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pageable_host),
        llama_cache_acct_measure::resident_allocated, 1);
}

static void test_ring_accounting_once() {
    llama_cache_acct_ledger ledger;
    const auto domain =
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pinned_host);
    const llama_cache_acct_completeness_requirement required {
        domain, llama_cache_acct_producer::retention_sidecar,
    };
    CHECK(ledger.configure_required_producers(&required, 1));
    CHECK(server_vbr_artifact_store_configure_pinned_accounting(
        ledger, domain));
    llama_cache_budget_config budget;
    budget.host.pinned_cap = 1024;
    budget.host.pinned_state =
        llama_cache_budget_capacity_state::known;
    vbr_capture_ring_accounting accounting {
        &ledger, domain, &budget,
    };
    {
        const auto snapshot = ledger.snapshot();
        llama_cache_budget_coordinator coordinator;
        CHECK(coordinator.reset(snapshot, budget));
        llama_cache_budget_plan plan;
        plan.accounting_serial = snapshot.serial;
        plan.entries.push_back({ domain, 16, 0 });
        const auto fit = coordinator.fits(plan);
        CHECK(fit.state == llama_cache_budget_fit_state::fits);
    }
    vbr_capture_stream_status status;
    vbr_capture_ring_create_failure failure;
    llama_cache_budget_config refused_budget = budget;
    refused_budget.host.pinned_cap = 8;
    vbr_capture_ring_accounting refused_accounting {
        &ledger, domain, &refused_budget,
    };
    auto refused = vbr_pinned_chunk_ring::create(
        { {} }, 16, 8, status, &refused_accounting, &failure);
    CHECK(!refused);
    CHECK(status == vbr_capture_stream_status::accounting_refused);
    CHECK(failure ==
          vbr_capture_ring_create_failure::budget_exceeded);

    // A fault raised after physical chunk allocation but during the final
    // C gauge remains accounting_unavailable, exactly as before ring factoring.
    ring_charge_fault_context fault_context { &ledger };
    auto charge_fault_accounting = accounting;
    charge_fault_accounting.charge_fault_context = &fault_context;
    charge_fault_accounting.inject_charge_fault =
        inject_ring_charge_fault;
    auto charge_failed = vbr_pinned_chunk_ring::create(
        { {} }, 16, 8, status, &charge_fault_accounting, &failure);
    CHECK(!charge_failed);
    CHECK(status == vbr_capture_stream_status::accounting_unavailable);
    CHECK(failure ==
          vbr_capture_ring_create_failure::accounting_charge_failed);
    CHECK(ring_resident(ledger, domain) == 0);

    auto ring = vbr_pinned_chunk_ring::create(
        { {} }, 16, 8, status, &accounting, &failure);
    CHECK(ring);
    CHECK(failure == vbr_capture_ring_create_failure::none);
    CHECK(ring_resident(ledger, domain) == 16);
    auto duplicate = vbr_pinned_chunk_ring::create(
        { {} }, 16, 8, status, &accounting);
    CHECK(!duplicate);
    CHECK(ring_resident(ledger, domain) == 16);
    ring.reset();
    CHECK(ring_resident(ledger, domain) == 0);
}

static bool sample_unbounded_host_budget(
        void *,
        llama_cache_budget_config & output) noexcept {
    output = {};
    output.host.pageable_state =
        llama_cache_budget_capacity_state::unbounded;
    output.host.pinned_cap = 1024;
    output.host.pinned_state =
        llama_cache_budget_capacity_state::known;
    output.host.total_state =
        llama_cache_budget_capacity_state::unbounded;
    output.global_cap_state =
        llama_cache_budget_capacity_state::unbounded;
    return true;
}

static vbr_artifact_portable_topology capture_test_topology() {
    llama_cache_acct_shard_topology topology;
    CHECK(llama_cache_acct_build_shard_topology(
        { "synthetic-capture-device" },
        LLAMA_SPLIT_MODE_NONE, 0, nullptr, topology));
    return topology;
}

struct projected_store_budget_context {
    llama_cache_acct_resource_domain device;
    bool available = true;
};

static bool sample_projected_store_budget(
        void * opaque,
        llama_cache_budget_config & output) noexcept {
    const auto * context =
        static_cast<const projected_store_budget_context *>(opaque);
    if (!context || !context->available) {
        return false;
    }
    output = {};
    llama_cache_budget_device_input input;
    input.backend_device = reinterpret_cast<const void *>(uintptr_t(1));
    input.domain = context->device;
    input.physical_total = 1ull << 30;
    input.physical_free = 1ull << 29;
    input.phys_state = llama_cache_budget_capacity_state::known;
    input.current_compute_allocated = 0;
    input.configured_compute_reserve = 0;
    input.compute_state = llama_cache_budget_capacity_state::known;
    input.cache_cap_state =
        llama_cache_budget_capacity_state::unbounded;
    output.devices.push_back(input);
    output.host.pageable_state =
        llama_cache_budget_capacity_state::unbounded;
    output.host.pinned_state =
        llama_cache_budget_capacity_state::unbounded;
    output.host.total_state =
        llama_cache_budget_capacity_state::unbounded;
    output.global_cap_state =
        llama_cache_budget_capacity_state::unbounded;
    return true;
}

static void test_projected_host_batch_store_adapter() {
    const auto topology = capture_test_topology();
    std::vector<vbr_capture_projection_manifest> manifests;
    std::vector<vbr_capture_controller_target> targets;
    for (uint64_t id = 1; id <= 2; ++id) {
        vbr_capture_projection_manifest manifest;
        manifest.manifest_id = id;
        manifest.placements.push_back(projected_placement(
            id, llama_seq_id(id), { uint32_t(id - 1) }));
        auto target = projected_target(
            id, 0, 700 + id, projected_generation(40 + id));
        bind_projected_manifest_metadata(manifest, target);
        manifests.push_back(std::move(manifest));
        targets.push_back(std::move(target));
    }
    vbr_capture_projection projection;
    CHECK(vbr_artifact_project_capture_union(
        { 707, std::move(manifests) }, {}, projection));
    std::vector<vbr_capture_projected_unit> units;
    for (const auto & target : targets) {
        units.push_back(capture_projected_unit_for_target(
            projection, target));
    }
    projected_controller_fixture controller;
    controller.rejected_manifest = 2;
    vbr_capture_manifest_assembly assembly;
    CHECK(assemble_projected_test_batch(
        projection, targets, units, controller.provider(), {}, assembly));

    const auto make_publications = [&](const auto & source_assembly) {
        std::vector<vbr_projected_manifest_publication> publications;
        publications.push_back(projected_publication(
            1, source_assembly, topology));
        publications.push_back(projected_publication(
            2, source_assembly, topology));
        return publications;
    };

    llama_cache_acct_ledger ledger;
    llama_cache_acct_resource_domain device;
    CHECK(ledger.make_device_domain(
        topology, llama_cache_acct_device_ordinal { 0 }, device));
    const auto pageable = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    const auto pinned = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pinned_host);
    const llama_cache_acct_completeness_requirement requirements[] {
        { device, llama_cache_acct_producer::live_memory },
        { pageable, llama_cache_acct_producer::retention_sidecar },
        { pinned, llama_cache_acct_producer::retention_sidecar },
    };
    CHECK(ledger.configure_required_producers(requirements, 3));
    for (const auto category : {
            llama_cache_acct_category::live_attention_state,
            llama_cache_acct_category::live_recurrent_state,
            llama_cache_acct_category::recurrent_rollback_planes,
            llama_cache_acct_category::rolling_window_tape }) {
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated,
                llama_cache_acct_measure::reserved }) {
            ledger.gauge_set(category, device, measure, 0);
        }
    }
    CHECK(server_vbr_artifact_store_observe_empty_accounting(
        ledger, device));
    CHECK(server_vbr_artifact_store_observe_empty_accounting(
        ledger, pageable));
    CHECK(ledger.certify_complete(
        device, llama_cache_acct_producer::live_memory));
    CHECK(ledger.certify_complete(
        pageable, llama_cache_acct_producer::retention_sidecar));
    CHECK(server_vbr_artifact_store_configure_pinned_accounting(
        ledger, pinned));

    projected_store_budget_context budget_context { device, true };
    server_vbr_artifact_store_config config;
    config.ledger = &ledger;
    config.pinned_domain = pinned;
    config.topologies.push_back(topology);
    config.pool_bindings.push_back({
        { 0x3333, 0x4444 }, 0, 0, 0, 0,
    });
    config.lanes.push_back({});
    config.attention_children = 1;
    config.ring_bytes = 16;
    config.chunk_bytes = 8;
    config.budget_context = &budget_context;
    config.sample_budget = sample_projected_store_budget;
    server_vbr_artifact_capture_status create_status;
    auto store = server_vbr_artifact_store::create(
        config, create_status);
    CHECK(store);
    CHECK(create_status == server_vbr_artifact_capture_status::ok);
    if (!store) {
        return;
    }

    const uint64_t baseline_ops = ledger.snapshot().live_ops;
    {
        struct checkpoint_context {
            uint32_t calls = 0;
        } checkpoint;
        server_vbr_projected_capture_admission admission;
        admission.context = &checkpoint;
        admission.continue_capture = +[](void * opaque) noexcept {
            auto * state = static_cast<checkpoint_context *>(opaque);
            if (state) {
                state->calls++;
            }
            return false;
        };
        std::vector<server_vbr_projected_host_publish_result> cancelled;
        server_vbr_projected_host_capture_diagnostics diagnostics;
        diagnostics.capture_status = vbr_explicit_capture_status::ok;
        diagnostics.capture_phase =
            vbr_explicit_capture_phase::post_transfer_stability;
        CHECK(!server_vbr_artifact_store_test_door::
            publish_after_capture_checkpoint(
                *store, assembly, make_publications(assembly), cancelled,
                admission, diagnostics));
        CHECK(checkpoint.calls == 1);
        CHECK(cancelled.empty());
        CHECK(diagnostics.capture_status ==
              vbr_explicit_capture_status::cancelled);
        CHECK(diagnostics.capture_phase ==
              vbr_explicit_capture_phase::publication);
        CHECK(diagnostics.inner_stream_status ==
              vbr_capture_stream_status::cancelled);
        CHECK(ledger.snapshot().live_ops == baseline_ops);
    }
    vbr_projected_capture_batch_request::pretransfer_quote staging_quote;
    staging_quote.planned_packed_bytes = 48;
    staging_quote.staging = {
        {
            {
                llama_cache_acct_residency::device,
                llama_cache_acct_domain_kind::device_topology,
                0, 0,
            },
            32,
        },
        {
            {
                llama_cache_acct_residency::pageable_host,
                llama_cache_acct_domain_kind::not_applicable,
                UINT32_MAX, UINT16_MAX,
            },
            16,
        },
    };
    llama_cache_budget_config staging_budget;
    CHECK(sample_projected_store_budget(
        &budget_context, staging_budget));
    staging_budget.devices.front().configured_cache_cap = 32;
    staging_budget.devices.front().cache_cap_state =
        llama_cache_budget_capacity_state::known;
    staging_budget.host.pageable_cap = 16;
    staging_budget.host.pageable_state =
        llama_cache_budget_capacity_state::known;
    const std::vector<llama_vbr_artifact_domain_binding> staging_bindings {
        { 0, 0, device },
    };
    const auto reserved = [&](
            const llama_cache_acct_snapshot & snapshot,
            const llama_cache_acct_resource_domain & domain) {
        const auto found = std::find_if(
            snapshot.cells.begin(), snapshot.cells.end(),
            [&](const auto & row) {
                return row.category ==
                           llama_cache_acct_category::transfer_staging &&
                       row.domain == domain;
            });
        return found == snapshot.cells.end() ? UINT64_MAX :
            found->cell.measures[size_t(
                llama_cache_acct_measure::reserved)].value;
    };
    auto shrink_quote = staging_quote;
    shrink_quote.planned_packed_bytes = 24;
    shrink_quote.staging[0].bytes = 16;
    shrink_quote.staging[1].bytes = 8;
    auto growth_quote = staging_quote;
    growth_quote.planned_packed_bytes = 49;
    growth_quote.staging[0].bytes = 33;
    auto resource_quote = staging_quote;
    resource_quote.union_cells = 1;
    resource_quote.manifests = 1;
    resource_quote.projected_units = 1;
    auto resource_publications = make_publications(assembly);
    std::vector<vbr_artifact_portable_accounting_row>
        resource_reserve_accounting;
    std::copy_if(
        resource_publications.front().accounting.begin(),
        resource_publications.front().accounting.end(),
        std::back_inserter(resource_reserve_accounting),
        [](const auto & row) {
            return row.role ==
                vbr_artifact_accounting_role::unit_payload;
        });
    resource_quote.durable.push_back({
        1, resource_publications.front().accounting,
        resource_reserve_accounting,
    });
    {
        server_vbr_artifact_store_test_door::
            projected_staging_lifecycle_result resources;
        CHECK(server_vbr_artifact_store_test_door::
            projected_resource_initial(
                *store, resource_quote, true, resources));
        CHECK(resources.scheduler_calls == 1);
        CHECK(resources.budget_samples == 1);
        CHECK(resources.durable_claims == 1);
        CHECK(resources.live_at_publication);
        CHECK(resources.scheduler.live_ops ==
              baseline_ops + resource_quote.staging.size() +
                  resource_quote.durable.front().accounting.size());
        CHECK(resources.after.live_ops == baseline_ops);
    }
    {
        auto shared_quote = resource_quote;
        shared_quote.manifests = 4;
        shared_quote.durable.clear();
        for (uint64_t manifest_id = 1; manifest_id <= 4; ++manifest_id) {
            shared_quote.durable.push_back({
                manifest_id,
                resource_publications.front().accounting,
                manifest_id == 1
                    ? resource_reserve_accounting
                    : std::vector<
                          vbr_artifact_portable_accounting_row> {},
            });
        }
        server_vbr_artifact_store_test_door::
            projected_staging_lifecycle_result shared;
        CHECK(server_vbr_artifact_store_test_door::
            projected_resource_initial(
                *store, shared_quote, true, shared));
        CHECK(shared.scheduler_calls == 1);
        CHECK(shared.budget_samples == 1);
        CHECK(shared.durable_claims == 4);
        CHECK(shared.scheduler.live_ops ==
              baseline_ops + shared_quote.staging.size() +
                  shared_quote.durable.size() *
                      shared_quote.durable.front().accounting.size());
        const auto reserved_for = [&] (
                vbr_artifact_accounting_role role) {
            const auto category = vbr_artifact_accounting_category(role);
            uint64_t bytes = 0;
            for (const auto & row : shared.scheduler.cells) {
                if (row.category == category) {
                    bytes += row.cell.measures[size_t(
                        llama_cache_acct_measure::reserved)].value;
                }
            }
            return bytes;
        };
        const auto expected_for = [&] (
                vbr_artifact_accounting_role role) {
            uint64_t bytes = 0;
            for (const auto & row :
                 shared_quote.durable.front().accounting) {
                if (row.role == role) {
                    bytes += row.resident_bytes;
                }
            }
            return bytes;
        };
        CHECK(reserved_for(
                  vbr_artifact_accounting_role::unit_payload) ==
              expected_for(vbr_artifact_accounting_role::unit_payload));
        CHECK(reserved_for(
                  vbr_artifact_accounting_role::descriptor_metadata) ==
              expected_for(
                  vbr_artifact_accounting_role::descriptor_metadata) * 4);
        CHECK(shared.after.live_ops == baseline_ops);
    }
    {
        server_vbr_artifact_store_test_door::
            projected_staging_lifecycle_result refused_resources;
        CHECK(!server_vbr_artifact_store_test_door::
            projected_resource_initial(
                *store, resource_quote, false, refused_resources));
        CHECK(refused_resources.scheduler_calls == 1);
        CHECK(refused_resources.budget_samples == 1);
        CHECK(refused_resources.scheduler.live_ops ==
              baseline_ops + resource_quote.staging.size() +
                  resource_quote.durable.front().accounting.size());
        CHECK(refused_resources.after.live_ops == baseline_ops);
    }
    server_vbr_artifact_store_test_door::
        projected_staging_lifecycle_result lifecycle;
    CHECK(server_vbr_artifact_store_test_door::
        projected_staging_lifecycle(
            ledger, staging_budget, staging_bindings,
            staging_quote, shrink_quote, growth_quote, lifecycle));
    CHECK(lifecycle.scheduler_calls == 1);
    CHECK(lifecycle.budget_samples == 1);
    CHECK(lifecycle.preparation.status ==
          llama_cache_prepare_status::prepared);
    CHECK(reserved(lifecycle.scheduler, device) == 32);
    CHECK(reserved(lifecycle.scheduler, pageable) == 16);
    CHECK(lifecycle.scheduler.live_ops == baseline_ops + 2);
    CHECK(reserved(lifecycle.initial, device) == 32);
    CHECK(reserved(lifecycle.initial, pageable) == 16);
    CHECK(lifecycle.initial.live_ops == baseline_ops + 2);
    CHECK(reserved(lifecycle.shrunk, device) == 16);
    CHECK(reserved(lifecycle.shrunk, pageable) == 8);
    CHECK(lifecycle.shrunk.live_ops == baseline_ops + 2);
    CHECK(reserved(lifecycle.publication, device) == 16);
    CHECK(reserved(lifecycle.publication, pageable) == 8);
    CHECK(lifecycle.after.live_ops == baseline_ops);
    CHECK(reserved(lifecycle.after, device) == 0);
    CHECK(reserved(lifecycle.after, pageable) == 0);
    {
        vbr_projected_capture_batch_request::pretransfer_quote zero;
        server_vbr_artifact_store_test_door::
            projected_staging_lifecycle_result no_work;
        CHECK(server_vbr_artifact_store_test_door::
            projected_staging_initial(
                ledger, staging_budget, staging_bindings,
                zero, true, no_work));
        CHECK(no_work.resources ==
              server_vbr_projected_host_capture_diagnostics::
                  resource_status::zero_work_admitted);
        CHECK(no_work.scheduler_calls == 0);
        CHECK(no_work.budget_samples == 0);
        CHECK(no_work.after.live_ops == baseline_ops);
    }
    {
        server_vbr_artifact_store_test_door::
            projected_staging_lifecycle_result refused;
        CHECK(!server_vbr_artifact_store_test_door::
            projected_staging_initial(
                ledger, staging_budget, staging_bindings,
                staging_quote, false, refused));
        CHECK(refused.resources ==
              server_vbr_projected_host_capture_diagnostics::
                  resource_status::scheduler_refused);
        CHECK(refused.scheduler_calls == 1);
        CHECK(refused.budget_samples == 1);
        CHECK(refused.preparation.status ==
              llama_cache_prepare_status::prepared);
        CHECK(refused.preparation.admission_status ==
              llama_cache_admission_status::admitted);
        CHECK(reserved(refused.scheduler, device) == 32);
        CHECK(reserved(refused.scheduler, pageable) == 16);
        CHECK(refused.scheduler.live_ops == baseline_ops + 2);
        CHECK(refused.initial.live_ops == baseline_ops);
        CHECK(refused.after.live_ops == baseline_ops);
    }
    {
        auto refused_budget = staging_budget;
        refused_budget.devices.front().configured_cache_cap = 31;
        server_vbr_artifact_store_test_door::
            projected_staging_lifecycle_result refused;
        CHECK(!server_vbr_artifact_store_test_door::
            projected_staging_lifecycle(
                ledger, refused_budget, staging_bindings,
                staging_quote, shrink_quote, growth_quote, refused));
        CHECK(!refused.initial_admitted);
        CHECK(refused.preparation.status ==
              llama_cache_prepare_status::admission_refused);
        CHECK(refused.preparation.admission_status ==
              llama_cache_admission_status::exceeds_budget);
        CHECK(refused.scheduler_calls == 0);
        CHECK(refused.budget_samples == 1);
        CHECK(refused.after.live_ops == baseline_ops);
    }
    CHECK(ledger.snapshot().live_ops == baseline_ops);

    std::vector<server_vbr_projected_host_publish_result> results;
    server_vbr_projected_host_publish_diagnostics diagnostics;
    CHECK(store->publish_projected_host_batch(
        assembly, make_publications(assembly), results, &diagnostics));
    CHECK(results.size() == 2);
    if (results.size() != 2) {
        return;
    }
    CHECK(results[0].manifest_id == 1);
    CHECK(results[0].status ==
              vbr_projected_manifest_publish_status::published ||
          results[0].status ==
              vbr_projected_manifest_publish_status::adopted);
    CHECK(results[0].payload);
    if (results[0].payload) {
        CHECK(results[0].payload->retirement_owned());
        CHECK(results[0].payload->accounted_by(&ledger));
        CHECK(results[0].payload->reference_artifact().v != 0);
    }
    CHECK(results[1].manifest_id == 2);
    CHECK(results[1].status ==
          vbr_projected_manifest_publish_status::dependency_unavailable);
    CHECK(!results[1].payload);
    CHECK(diagnostics.catalog.ready_manifests == 1);
    CHECK(diagnostics.catalog.published_manifests == 1);
    CHECK(diagnostics.catalog.dependency_unavailable == 1);
    CHECK(diagnostics.host_payloads_retained == 1);
    CHECK(diagnostics.postpublish_retirements == 0);
    CHECK(ledger.snapshot().live_ops > baseline_ops);
    // The typed host door accepts this store-owned capability without a
    // tenant reference and reaches the shared import kernel. The intentionally
    // empty target request then fails at kernel validation, not authorization.
    const auto authenticated_before =
        store->counters().host_imports_authenticated;
    server_vbr_artifact_import_target empty_import;
    const auto host_import = store->import_host_payload(
        std::move(empty_import), results[0].payload);
    CHECK(host_import.status ==
          server_vbr_artifact_import_status::unavailable);
    CHECK(store->counters().host_imports_authenticated ==
          authenticated_before + 1);
    results.clear();
    CHECK(ledger.snapshot().live_ops == baseline_ops);

    auto malformed = make_publications(assembly);
    malformed.back().manifest_id = 3;
    results.resize(1);
    diagnostics.catalog.ready_manifests = 99;
    CHECK(!store->publish_projected_host_batch(
        assembly, std::move(malformed), results, &diagnostics));
    CHECK(results.empty());
    CHECK(diagnostics.catalog.ready_manifests == 0);
    CHECK(ledger.snapshot().live_ops == baseline_ops);

    // A failure after both catalog rows commit must retire only the failed
    // row. Its sibling remains a typed host owner, and releasing that owner
    // returns the complete catalog/ledger state to the pre-batch baseline.
    controller.rejected_manifest = 0;
    vbr_capture_manifest_assembly ready_assembly;
    CHECK(assemble_projected_test_batch(
        projection, targets, units, controller.provider(), {},
        ready_assembly));
    server_vbr_artifact_store_test_door::
        fail_projected_host_adoption_once(*store);
    CHECK(store->publish_projected_host_batch(
        ready_assembly, make_publications(ready_assembly),
        results, &diagnostics));
    CHECK(results.size() == 2);
    if (results.size() == 2) {
        CHECK(results[0].status ==
              vbr_projected_manifest_publish_status::internal_error);
        CHECK(!results[0].payload);
        CHECK(results[1].status ==
                  vbr_projected_manifest_publish_status::published ||
              results[1].status ==
                  vbr_projected_manifest_publish_status::adopted);
        CHECK(results[1].payload);
    }
    CHECK(diagnostics.catalog.published_manifests == 2);
    CHECK(diagnostics.host_payloads_retained == 1);
    CHECK(diagnostics.postpublish_retirements == 1);
    CHECK(ledger.snapshot().live_ops > baseline_ops);
    results.clear();
    CHECK(ledger.snapshot().live_ops == baseline_ops);

    // Re-publishing the same sealed batch adopts the existing physical unit
    // allocations while creating distinct logical reference owners. Dropping
    // either generation of host owners must preserve the other.
    std::vector<server_vbr_projected_host_publish_result> first_owners;
    CHECK(store->publish_projected_host_batch(
        ready_assembly, make_publications(ready_assembly),
        first_owners, &diagnostics));
    CHECK(first_owners.size() == 2);
    std::vector<llama_cache_acct_artifact_id> first_references;
    for (const auto & row : first_owners) {
        CHECK(row.payload);
        CHECK(row.status ==
                  vbr_projected_manifest_publish_status::published ||
              row.status ==
                  vbr_projected_manifest_publish_status::adopted);
        if (row.payload) {
            first_references.push_back(
                row.payload->reference_artifact());
        }
    }
    CHECK(first_references.size() == 2);
    std::vector<server_vbr_projected_host_publish_result> adopted_owners;
    CHECK(store->publish_projected_host_batch(
        ready_assembly, make_publications(ready_assembly),
        adopted_owners, &diagnostics));
    CHECK(adopted_owners.size() == 2);
    for (size_t i = 0; i < adopted_owners.size(); ++i) {
        CHECK(adopted_owners[i].status ==
              vbr_projected_manifest_publish_status::adopted);
        CHECK(adopted_owners[i].payload);
        if (adopted_owners[i].payload && i < first_references.size()) {
            CHECK(adopted_owners[i].payload->reference_artifact() !=
                  first_references[i]);
        }
    }
    CHECK(diagnostics.catalog.published_manifests == 2);
    CHECK(diagnostics.host_payloads_retained == 2);
    if (adopted_owners.size() == 2 && adopted_owners[0].payload &&
        adopted_owners[1].payload) {
        const auto occupied_authenticated_before =
            store->counters().host_imports_authenticated;
        server_vbr_artifact_import_target empty_occupied;
        const auto occupied_import =
            store->import_host_occupied_replacement(
                std::move(empty_occupied), adopted_owners[0].payload,
                adopted_owners[1].payload);
        CHECK(occupied_import.status ==
              server_vbr_artifact_import_status::unavailable);
        CHECK(store->counters().host_imports_authenticated ==
              occupied_authenticated_before + 1);
        server_vbr_artifact_import_target duplicate_occupied;
        const auto duplicate_import =
            store->import_host_occupied_replacement(
                std::move(duplicate_occupied), adopted_owners[0].payload,
                adopted_owners[0].payload);
        CHECK(duplicate_import.status ==
              server_vbr_artifact_import_status::unavailable);
        CHECK(store->counters().host_imports_authenticated ==
              occupied_authenticated_before + 1);
    }
    const uint64_t adopted_ops = ledger.snapshot().live_ops;
    CHECK(adopted_ops > baseline_ops);
    first_owners.clear();
    CHECK(ledger.snapshot().live_ops > baseline_ops);
    CHECK(ledger.snapshot().live_ops < adopted_ops);
    adopted_owners.clear();
    CHECK(ledger.snapshot().live_ops == baseline_ops);

    budget_context.available = false;
    results.resize(1);
    diagnostics.catalog.ready_manifests = 99;
    CHECK(!store->publish_projected_host_batch(
        assembly, make_publications(assembly), results, &diagnostics));
    CHECK(results.empty());
    CHECK(diagnostics.catalog.ready_manifests == 0);
    CHECK(ledger.snapshot().live_ops == baseline_ops);

    // Two stores may share the global ledger, but a host owner remains bound
    // to the catalog that minted it. Exercise the store gate itself rather
    // than only its catalog predicate.
    budget_context.available = true;
    llama_vbr_artifact_catalog foreign_catalog(ledger);
    std::vector<llama_vbr_artifact_domain_binding> foreign_bindings;
    CHECK(foreign_catalog.bind_topologies(
        { topology }, foreign_bindings));
    llama_cache_budget_config foreign_budget;
    CHECK(sample_projected_store_budget(
        &budget_context, foreign_budget));
    std::vector<vbr_projected_manifest_publish_result> foreign_results;
    CHECK(foreign_catalog.publish_projected_batch(
        ready_assembly, make_publications(ready_assembly), foreign_budget,
        foreign_results));
    std::vector<llama_cache_acct_artifact_id> foreign_references;
    for (const auto & row : foreign_results) {
        if (row.status == vbr_projected_manifest_publish_status::published ||
            row.status == vbr_projected_manifest_publish_status::adopted) {
            foreign_references.push_back(
                row.publication.reference_artifact);
        }
    }
    std::vector<vbr_artifact_package_view> foreign_packages;
    CHECK(foreign_catalog.claim_fresh_host_batch(
        foreign_references, foreign_packages));
    CHECK(foreign_packages.size() == 2);
    if (!foreign_packages.empty()) {
        auto foreign_payload =
            server_prompt_cache_vbr_payload::adopt(
                std::move(foreign_packages.front()));
        CHECK(foreign_payload);
        if (foreign_payload) {
            const auto authenticated =
                store->counters().host_imports_authenticated;
            server_vbr_artifact_import_target foreign_import;
            const auto rejected = store->import_host_payload(
                std::move(foreign_import), foreign_payload);
            CHECK(rejected.status ==
                  server_vbr_artifact_import_status::not_found);
            CHECK(store->counters().host_imports_authenticated ==
                  authenticated);
        }
    }
    foreign_packages.clear();
    CHECK(ledger.snapshot().live_ops == baseline_ops);
}

static void test_capture_reservation_domain_preparation() {
    llama_cache_acct_ledger ledger;
    const auto topology = capture_test_topology();
    llama_cache_acct_resource_domain device;
    CHECK(ledger.make_device_domain(
        topology, llama_cache_acct_device_ordinal { 0 },
        device));
    const llama_cache_acct_completeness_requirement requirement {
        device, llama_cache_acct_producer::live_memory,
    };
    CHECK(ledger.configure_required_producers(&requirement, 1));
    for (const auto category : {
            llama_cache_acct_category::live_attention_state,
            llama_cache_acct_category::live_recurrent_state,
            llama_cache_acct_category::recurrent_rollback_planes,
            llama_cache_acct_category::rolling_window_tape }) {
        ledger.gauge_set(
            category, device,
            llama_cache_acct_measure::resident_allocated, 0);
    }
    CHECK(ledger.certify_complete(
        device, llama_cache_acct_producer::live_memory));
    // A partially activated capture row is not dormant: known resident with
    // unknown reserved evidence makes the coordinator fail closed.
    ledger.gauge_set(
        llama_cache_acct_category::unit_version_payload,
        device, llama_cache_acct_measure::logical_payload, 0);
    ledger.gauge_set(
        llama_cache_acct_category::unit_version_payload,
        device, llama_cache_acct_measure::resident_allocated, 0);

    llama_cache_budget_config budget;
    llama_cache_budget_device_input input;
    input.backend_device =
        reinterpret_cast<const void *>(uintptr_t(1));
    input.domain = device;
    input.physical_total = 1024;
    input.physical_free = 1024;
    input.phys_state =
        llama_cache_budget_capacity_state::known;
    input.current_compute_allocated = 0;
    input.configured_compute_reserve = 0;
    input.compute_state =
        llama_cache_budget_capacity_state::known;
    input.cache_cap_state =
        llama_cache_budget_capacity_state::unbounded;
    budget.devices.push_back(input);

    llama_cache_transaction_leaf leaf;
    leaf.category =
        llama_cache_acct_category::unit_version_payload;
    leaf.domain = device;
    leaf.expected_logical = 16;
    leaf.reserve_resident = 16;
    leaf.stage_resident = 16;
    llama_cache_acct_op_id committed;
    leaf.committed_op = &committed;
    std::vector<llama_cache_transaction_leaf> leaves { leaf };
    {
        auto unavailable =
            llama_cache_prepare_reservation_transaction(
                ledger, budget, leaves);
        CHECK(!unavailable.ready());
        CHECK(unavailable.preparation().status ==
              llama_cache_prepare_status::admission_refused);
    }
    CHECK(server_vbr_artifact_store_observe_empty_accounting(
        ledger, device));
    CHECK(server_vbr_artifact_store_verify_accounting(
        ledger, { device }));
    {
        auto prepared =
            llama_cache_prepare_reservation_transaction(
                ledger, budget, leaves);
        CHECK(prepared.ready());
    }
    CHECK(ledger.snapshot().live_ops == 0);
}

static void test_server_store_construction_and_lifetime() {
    llama_cache_acct_ledger ledger;
    const auto pinned =
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pinned_host);
    const llama_cache_acct_completeness_requirement requirement {
        pinned, llama_cache_acct_producer::retention_sidecar,
    };
    CHECK(ledger.configure_required_producers(&requirement, 1));
    // Reproduce the real-load failure: observing only the ring leaves the
    // host-scoped payload cells in this manifested pinned domain unknown.
    for (const auto measure : {
            llama_cache_acct_measure::logical_payload,
            llama_cache_acct_measure::resident_allocated,
            llama_cache_acct_measure::reserved }) {
        ledger.gauge_set(
            llama_cache_acct_category::pinned_preimage_ring,
            pinned, measure, 0);
    }
    CHECK(ledger.certify_complete(
        pinned, llama_cache_acct_producer::retention_sidecar));

    llama_cache_budget_config budget;
    budget.host.pinned_cap = 1024;
    budget.host.pinned_state =
        llama_cache_budget_capacity_state::known;
    {
        llama_cache_budget_coordinator coordinator;
        const auto snapshot = ledger.snapshot();
        CHECK(coordinator.reset(snapshot, budget));
        llama_cache_budget_plan plan;
        plan.accounting_serial = snapshot.serial;
        plan.entries.push_back({ pinned, 16, 0 });
        CHECK(coordinator.fits(plan).state ==
              llama_cache_budget_fit_state::unavailable);
    }
    CHECK(server_vbr_artifact_store_configure_pinned_accounting(
        ledger, pinned));
    CHECK(!server_vbr_artifact_store_configure_pinned_accounting(
        ledger,
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pageable_host)));

    server_vbr_artifact_store_config config;
    config.ledger = &ledger;
    config.pinned_domain = pinned;
    config.topologies.push_back(capture_test_topology());
    config.pool_bindings.push_back({
        { 0x1111, 0x2222 }, 0, 0, 0, 0,
    });
    config.lanes.push_back({});
    config.attention_children = 1;
    // The physical owner rounds down to complete chunks; both direction
    // adapters must use that constructed capacity rather than the request.
    config.ring_bytes = 17;
    config.chunk_bytes = 8;
    config.sample_budget = sample_unbounded_host_budget;

    const auto baseline = ledger.snapshot();
    CHECK(sample_unbounded_host_budget(nullptr, budget));
    {
        llama_cache_budget_coordinator coordinator;
        const auto snapshot = ledger.snapshot();
        CHECK(coordinator.reset(snapshot, budget));
        llama_cache_budget_plan plan;
        plan.accounting_serial = snapshot.serial;
        plan.entries.push_back({ pinned, 16, 0 });
        CHECK(coordinator.fits(plan).state ==
              llama_cache_budget_fit_state::fits);
    }
    vbr_capture_ring_accounting direct_accounting {
        &ledger, pinned, &budget,
    };
    vbr_capture_stream_status direct_status;
    {
        auto direct = vbr_pinned_chunk_ring::create(
            config.lanes, config.ring_bytes, config.chunk_bytes,
            direct_status, &direct_accounting);
        CHECK(direct);
    }
    CHECK(ring_resident(ledger, pinned) == 0);
    server_vbr_artifact_capture_status status;
    server_vbr_artifact_store_create_diagnostics diagnostics;
    const uint64_t transport_before =
        vbr_pinned_ring_live_capacity_bytes();
    {
        auto store =
            server_vbr_artifact_store::create(
                config, status, &diagnostics);
        CHECK(store);
        CHECK(status == server_vbr_artifact_capture_status::ok);
        CHECK(diagnostics.failure ==
              server_vbr_artifact_store_create_failure::none);
        CHECK(diagnostics.ring_status ==
              vbr_capture_stream_status::ok);
        CHECK(diagnostics.ring_failure ==
              vbr_capture_ring_create_failure::none);
        CHECK(diagnostics.requested_ring_bytes == 17);
        CHECK(diagnostics.constructed_ring_bytes == 16);
        if (!store) {
            return;
        }
        CHECK(store->attention_children() == 1);
        CHECK(store->counters().pinned_bytes == 16);
        CHECK(store->counters().requested == 0);
        CHECK(ring_resident(ledger, pinned) == 16);
        // Capture and import adapters share this one physical allocation.
        CHECK(vbr_pinned_ring_live_capacity_bytes() ==
              transport_before + 16);
        vbr_adopt_stage_policy import_policy;
        CHECK(server_vbr_artifact_store_test_door::
              import_transport_policy(*store, import_policy));
        CHECK(import_policy.ledger == &ledger);
        CHECK(import_policy.persistent_ring);
        CHECK(import_policy.pinned_domain == pinned);
        CHECK(import_policy.pinned_ring_bytes == 16);
        CHECK(import_policy.chunk_bytes == 8);
        CHECK(import_policy.lanes.size() == 1);
        CHECK(vbr_pinned_ring_live_capacity_bytes() ==
              transport_before + 16);
        import_policy.persistent_ring.reset();
    }
    const auto after = ledger.snapshot();
    CHECK(ring_resident(ledger, pinned) == 0);
    CHECK(vbr_pinned_ring_live_capacity_bytes() == transport_before);
    CHECK(after.live_ops == baseline.live_ops);
    CHECK(after.faults_invalid_transition ==
          baseline.faults_invalid_transition);
    CHECK(after.faults_overflow == baseline.faults_overflow);
    CHECK(after.faults_allocation == baseline.faults_allocation);

    config.attention_children = 0;
    auto rejected =
        server_vbr_artifact_store::create(
            config, status, &diagnostics);
    CHECK(!rejected);
    CHECK(status ==
          server_vbr_artifact_capture_status::unavailable);
    CHECK(diagnostics.failure ==
          server_vbr_artifact_store_create_failure::
              attention_child_missing);
}

static void test_server_capture_status_vocabulary() {
    for (size_t i = 0;
         i < size_t(server_vbr_artifact_capture_status::_count);
         ++i) {
        const auto * name =
            server_vbr_artifact_capture_status_name(
                server_vbr_artifact_capture_status(i));
        CHECK(name != nullptr);
        CHECK(std::string(name) != "_count");
    }
    CHECK(std::string(server_vbr_artifact_capture_status_name(
              server_vbr_artifact_capture_status::_count)) ==
          "_count");
    for (size_t i = 0;
         i < size_t(server_vbr_artifact_import_status::_count);
         ++i) {
        const auto * name = server_vbr_artifact_import_status_name(
            server_vbr_artifact_import_status(i));
        CHECK(name != nullptr);
        CHECK(std::string(name) != "_count");
    }
    CHECK(std::string(server_vbr_artifact_import_status_name(
              server_vbr_artifact_import_status::_count)) ==
          "_count");
    for (size_t i = 0;
         i < size_t(vbr_explicit_capture_phase::_count);
         ++i) {
        const auto * name =
            vbr_explicit_capture_phase_name(
                vbr_explicit_capture_phase(i));
        CHECK(name != nullptr);
        CHECK(std::string(name) != "_count");
    }
    for (size_t i = 0;
         i < size_t(vbr_explicit_generation_failure::_count);
         ++i) {
        const auto * name =
            vbr_explicit_generation_failure_name(
            vbr_explicit_generation_failure(i));
        CHECK(name != nullptr);
        CHECK(std::string(name) != "_count");
    }
    for (size_t i = 0;
         i < size_t(vbr_explicit_size_failure::_count);
         ++i) {
        const auto * name =
            vbr_explicit_size_failure_name(
                vbr_explicit_size_failure(i));
        CHECK(name != nullptr);
        CHECK(std::string(name) != "_count");
    }
    for (size_t i = 0;
         i < size_t(vbr_capture_reservation_group::_count);
         ++i) {
        const auto * name =
            vbr_capture_reservation_group_name(
                vbr_capture_reservation_group(i));
        CHECK(name != nullptr);
        CHECK(std::string(name) != "invalid");
    }
}

static void test_server_reference_tenant_authorization() {
    server_vbr_artifact_reference_index index;
    const llama_cache_acct_artifact_id expected { 73 };
    CHECK(index.publish("vbrref_alpha", "tenant-a", expected));
    CHECK(!index.publish("vbrref_alpha", "tenant-a", expected));
    CHECK(!index.publish("malformed", "tenant-a", expected));
    CHECK(!index.publish("vbrref_zero", "tenant-a", {}));

    llama_cache_acct_artifact_id resolved { 999 };
    CHECK(index.authorize("vbrref_alpha", "tenant-a", resolved));
    CHECK(resolved == expected);
    for (const auto & denied : std::vector<std::pair<std::string, std::string>> {
            { "vbrref_alpha", "tenant-b" },
            { "vbrref_missing", "tenant-a" },
            { "malformed", "tenant-a" },
            { "vbrref_alpha", "" },
        }) {
        resolved = { 999 };
        CHECK(!index.authorize(denied.first, denied.second, resolved));
        // Wrong-tenant, nonexistent and malformed tokens expose the same
        // closed miss shape and never return the underlying artifact id.
        CHECK(resolved.v == 0);
    }
}

static void test_server_import_route_classification() {
    using status = server_vbr_artifact_import_status;
    server_vbr_artifact_import_output untouched;
    CHECK(untouched.downward_reserve_status ==
          vbr_downward_reserve_status::not_attempted);
    CHECK(server_vbr_artifact_import_route_precheck(
              false, false, false, false, false) == status::unsupported);
    CHECK(server_vbr_artifact_import_route_precheck(
              true, false, false, true, true) == status::invalid_slot);
    CHECK(server_vbr_artifact_import_route_precheck(
              true, true, true, true, true) == status::slot_processing);
    CHECK(server_vbr_artifact_import_route_precheck(
              true, true, false, false, true) == status::unavailable);
    CHECK(server_vbr_artifact_import_route_precheck(
              true, true, false, true, false) == status::slot_not_empty);
    CHECK(server_vbr_artifact_import_route_precheck(
              true, true, false, true, true) == status::ok);

    for (const auto decision : {
            vbr_import_decision::native_import,
            vbr_import_decision::live_rebased,
            vbr_import_decision::downward_rebase,
            vbr_import_decision::upward_reconstruct }) {
        CHECK(server_vbr_artifact_import_validation_disposition(
                  vbr_manifest_validation_status::validated,
                  decision) == status::ok);
    }
    for (const auto decision : {
            vbr_import_decision::rebuild,
            vbr_import_decision::cold }) {
        CHECK(server_vbr_artifact_import_validation_disposition(
                  vbr_manifest_validation_status::validated,
                  decision) == status::report_only);
    }
    CHECK(server_vbr_artifact_import_validation_disposition(
              vbr_manifest_validation_status::validated,
              vbr_import_decision::reject) == status::validation_failed);
    CHECK(server_vbr_artifact_import_validation_disposition(
              vbr_manifest_validation_status::unavailable,
              vbr_import_decision::native_import) ==
          status::validation_failed);

    server_vbr_artifact_import_output fallback;
    for (const auto retryable : {
            status::validation_failed,
            status::report_only,
            status::stage_failed }) {
        fallback = {};
        fallback.status = retryable;
        fallback.adopt_attempted = false;
        CHECK(server_vbr_artifact_import_variant_fallback_safe(fallback));
    }
    fallback.adopt_attempted = true;
    CHECK(!server_vbr_artifact_import_variant_fallback_safe(fallback));
    fallback.adopt_attempted = false;
    fallback.h2d_bytes = 1;
    CHECK(!server_vbr_artifact_import_variant_fallback_safe(fallback));
    fallback.h2d_bytes = 0;
    fallback.h2d_chunks = 1;
    CHECK(!server_vbr_artifact_import_variant_fallback_safe(fallback));
    for (const auto terminal : {
            status::ok,
            status::unsupported,
            status::not_found,
            status::adopt_failed,
            status::unavailable,
            status::internal_error }) {
        fallback = {};
        fallback.status = terminal;
        fallback.adopt_attempted = false;
        CHECK(!server_vbr_artifact_import_variant_fallback_safe(fallback));
    }
}

static void test_server_import_schedule_actionability() {
    using schedule = vbr_import_schedule_status;
    using snapshot = vbr_import_target_snapshot_status;
    const auto unit = [](uint32_t logical_unit_id,
                         ggml_type source, ggml_type target) {
        return vbr_import_schedule_unit {
            0, logical_unit_id, source, target,
            vbr_downward_tier_domain(source),
            vbr_downward_tier_domain(target),
        };
    };

    // An exact sibling does not hide or poison one supported tapped-domain
    // upward row in the authenticated quote.
    const std::vector<vbr_import_schedule_unit> tapped_upward {
        unit(0, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0),
        unit(1, GGML_TYPE_TURBO1_TCQ, GGML_TYPE_TURBO4_0),
    };
    CHECK(vbr_classify_import_schedule_units(tapped_upward) ==
          schedule::upward_same_domain);
    CHECK(vbr_explicit_import_schedule_actionability(
              schedule::upward_same_domain, tapped_upward) ==
          snapshot::actionable);

    const std::vector<vbr_import_schedule_unit> mixed {
        unit(0, GGML_TYPE_F16, GGML_TYPE_TURBO8_0),
        unit(1, GGML_TYPE_TURBO1_TCQ, GGML_TYPE_TURBO4_0),
    };
    CHECK(vbr_classify_import_schedule_units(mixed) ==
          schedule::mixed_direction_unsupported);
    CHECK(vbr_explicit_import_schedule_actionability(
              schedule::mixed_direction_unsupported, mixed) ==
          snapshot::report_only);

    const std::vector<vbr_import_schedule_unit> cross_domain {
        unit(0, GGML_TYPE_TURBO4_0, GGML_TYPE_F16),
    };
    CHECK(vbr_classify_import_schedule_units(cross_domain) ==
          schedule::upward_cross_domain);
    CHECK(vbr_explicit_import_schedule_actionability(
              schedule::upward_cross_domain, cross_domain) ==
          snapshot::actionable);

    // A caller cannot relabel identical evidence into an actionable route.
    CHECK(vbr_explicit_import_schedule_actionability(
              schedule::upward_same_domain, cross_domain) ==
          snapshot::unavailable);
    CHECK(vbr_explicit_import_schedule_actionability(
              schedule::exact, {}) == snapshot::unavailable);
}

static void test_fresh_f16_size_generation() {
    // Production ordinary decode maps a padded nonzero watermark even before
    // any retier has created the VBR side stream. That never-degraded state is
    // a complete full-domain F16 extent with zero promote hops.
    vbr_unit_generation fresh;
    fresh.current_type = GGML_TYPE_F16;
    fresh.last_source_type = GGML_TYPE_F16;
    fresh.domain = vbr_repr_domain::full;
    fresh.promote_hops = 0;
    CHECK(vbr_explicit_capture_validate_extent_generation(
              256, GGML_TYPE_F16, 0, fresh) ==
          vbr_explicit_size_failure::none);
    CHECK(vbr_explicit_capture_validate_extent_generation(
              0, GGML_TYPE_F16, 0, fresh) ==
          vbr_explicit_size_failure::wm_cells_zero);

    auto wrong_domain = fresh;
    wrong_domain.domain = vbr_repr_domain::tapped;
    CHECK(vbr_explicit_capture_validate_extent_generation(
              256, GGML_TYPE_F16, 0, wrong_domain) ==
          vbr_explicit_size_failure::domain_mismatch);

    vbr_unit_generation tapped = fresh;
    tapped.current_type = GGML_TYPE_TURBO4_0;
    tapped.last_source_type = GGML_TYPE_F16;
    tapped.domain = vbr_repr_domain::tapped;
    tapped.promote_hops = 1;
    CHECK(vbr_explicit_capture_validate_extent_generation(
              256, GGML_TYPE_TURBO4_0, 1, tapped) ==
          vbr_explicit_size_failure::none);
    CHECK(vbr_explicit_capture_validate_extent_generation(
              256, GGML_TYPE_TURBO4_0, 0, tapped) ==
          vbr_explicit_size_failure::promote_hops_mismatch);
    CHECK(vbr_explicit_capture_validate_extent_generation(
              256, GGML_TYPE_TURBO3_TCQ, 1, tapped) ==
          vbr_explicit_size_failure::extent_type_mismatch);
}

static void test_library_representation_identity() {
    static constexpr char BUILD_A[] = "capture-build-a";
    static constexpr char BUILD_B[] = "capture-build-b";
    const vbr_explicit_representation_policy policy_a {
        BUILD_A, sizeof(BUILD_A) - 1,
    };
    const vbr_explicit_representation_policy policy_b {
        BUILD_B, sizeof(BUILD_B) - 1,
    };
    vbr_explicit_representation_identity a;
    vbr_explicit_representation_identity b;
    CHECK(vbr_explicit_capture_representation_identity(
        &policy_a, GGML_TYPE_F16, false, 0, a));
    CHECK(vbr_explicit_capture_representation_identity(
        &policy_b, GGML_TYPE_F16, false, 0, b));
    CHECK(a.codec_id == uint32_t(GGML_TYPE_F16) + 1);
    CHECK(a.codec_version == 1);
    CHECK(a.codebook_digest != b.codebook_digest);
    CHECK(a.rotation_digest == b.rotation_digest);
    CHECK(a.meansub_digest == b.meansub_digest);
    CHECK(!a.meansub_baked);

    const int baked_model = ggml_turbo_meansub_model_id(
        "qwen35", 64, 5120);
    CHECK(baked_model > 0);
    const auto before =
        vbr_explicit_representation_identity_diagnostics_snapshot();
    vbr_explicit_representation_identity baked_first;
    CHECK(vbr_explicit_capture_representation_identity(
        &policy_a, GGML_TYPE_F16, false, baked_model, baked_first));
    const auto after_first =
        vbr_explicit_representation_identity_diagnostics_snapshot();
    vbr_explicit_representation_identity baked_second;
    CHECK(vbr_explicit_capture_representation_identity(
        &policy_a, GGML_TYPE_F16, false, baked_model, baked_second));
    const auto after_second =
        vbr_explicit_representation_identity_diagnostics_snapshot();
    CHECK(baked_first.meansub_baked);
    CHECK(baked_second.meansub_baked);
    CHECK(baked_first.meansub_digest == baked_second.meansub_digest);
    CHECK(after_first.baked_table_hashes == before.baked_table_hashes + 1);
    CHECK(after_second.baked_table_hashes == after_first.baked_table_hashes);
    vbr_explicit_representation_identity missing;
    CHECK(!vbr_explicit_capture_representation_identity(
        nullptr, GGML_TYPE_F16, false, 0, missing));
}

static void test_cuda_ring() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * candidate = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(candidate) ==
                GGML_BACKEND_DEVICE_TYPE_GPU) {
            device = candidate;
            break;
        }
    }
    if (!device) {
        fprintf(stderr, "FAIL: no GPU backend for VBR streaming CUDA synthetic gate\n");
        failures++;
        return;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    CHECK(backend != nullptr);
    if (!backend) {
        return;
    }
    const size_t n = 5*1024*1024 + 3;
    std::vector<uint8_t> expected(n);
    for (size_t i = 0; i < n; ++i) {
        expected[i] = uint8_t((i*29 + 11) & 0xff);
    }
    ggml_init_params params = {
        2*ggml_tensor_overhead(), nullptr, true,
    };
    ggml_context * context = ggml_init(params);
    CHECK(context != nullptr);
    ggml_tensor * tensor =
        context ? ggml_new_tensor_1d(
            context, GGML_TYPE_I8, n) : nullptr;
    ggml_backend_buffer_t buffer =
        tensor ? ggml_backend_alloc_ctx_tensors(
            context, backend) : nullptr;
    CHECK(tensor && buffer);
    if (tensor && buffer) {
        ggml_backend_tensor_set(
            tensor, expected.data(), 0, expected.size());
        vbr_capture_stream_status status;
        auto ring = vbr_pinned_chunk_ring::create(
            { { device, backend } },
            2*1024*1024, 1024*1024, status);
        CHECK(ring);
        artifact_segment_chain chain;
        vbr_capture_stream_stats stats;
        vbr_capture_stream_source source;
        source.lane = 0;
        source.size = expected.size();
        source.backend = backend;
        source.device = device;
        source.tensor = tensor;
        CHECK(ring->stream(source, chain, stats) ==
              vbr_capture_stream_status::ok);
        CHECK(chain.max_segment_size() <= 1024*1024);
        CHECK(read_chain(chain) == expected);
        CHECK(stats.event_completions > 0);

        auto sync_ring = vbr_pinned_chunk_ring::create(
            { { device, backend, true } },
            2*1024*1024, 1024*1024, status);
        CHECK(sync_ring);
        artifact_segment_chain sync_chain;
        vbr_capture_stream_stats sync_stats;
        CHECK(sync_ring->stream(source, sync_chain, sync_stats) ==
              vbr_capture_stream_status::ok);
        CHECK(sync_stats.synchronous_fallbacks > 0);
        CHECK(sync_stats.streaming_digest ==
              stats.streaming_digest);
        CHECK(read_chain(sync_chain) == expected);
    }
    if (buffer) {
        ggml_backend_buffer_free(buffer);
    }
    if (context) {
        ggml_free(context);
    }
    ggml_backend_free(backend);
}

static void benchmark_fragmented_range_packing() {
    static constexpr uint32_t RANGE_COUNT = 1048576;
    static constexpr size_t CHUNK_BYTES = 64*1024;
    generated_h2d_source generated;
    generated.size = uint64_t(RANGE_COUNT)*2;
    std::vector<vbr_capture_stream_range> ranges;
    ranges.reserve(RANGE_COUNT);
    for (uint32_t i = 0; i < RANGE_COUNT; ++i) {
        ranges.push_back({ uint64_t(i)*2, 1 });
    }
    vbr_capture_stream_source source;
    source.size = generated.size;
    source.context = &generated;
    source.read = generated_h2d_source::read;
    vbr_capture_stream_status status;
    auto ring = vbr_pinned_chunk_ring::create(
        { {} }, 2*CHUNK_BYTES, CHUNK_BYTES, status);
    CHECK(ring);
    artifact_segment_chain chain(
        VBR_CAPTURE_RANGE_CHUNK_BYTES, 262144);
    vbr_capture_stream_stats stats;
    const auto begin = std::chrono::steady_clock::now();
    CHECK(ring && ring->stream_ranges(
              source, ranges, chain, stats) ==
          vbr_capture_stream_status::ok);
    vbr_capture_range_tree tree;
    CHECK(vbr_capture_range_seal(
        chain, uint64_t(32)*1024*1024, tree));
    vbr_capture_range_proof proof;
    vbr_capture_range_proof_limits proof_limits;
    CHECK(vbr_capture_range_prove(
        tree, { { 0, RANGE_COUNT } }, proof_limits, proof));
    uint64_t proof_bytes_read = 0;
    CHECK(vbr_capture_range_verify(
        proof, chain.source(), &proof_bytes_read));
    const auto elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin).count();
    CHECK(stats.bytes == RANGE_COUNT);
    CHECK(stats.chunks == RANGE_COUNT/CHUNK_BYTES);
    CHECK(chain.segment_count() == stats.chunks);
    CHECK(chain.max_segment_size() == CHUNK_BYTES);
    CHECK(tree.chunk_count() == stats.chunks);
    CHECK(proof.selected_chunk_count() == stats.chunks);
    CHECK(proof_bytes_read == RANGE_COUNT);
    std::vector<uint8_t> packed(RANGE_COUNT);
    CHECK(chain.read(0, packed.data(), packed.size()));
    for (uint32_t i = 0; i < RANGE_COUNT; ++i) {
        if (packed[i] != generated_h2d_source::byte_at(uint64_t(i)*2)) {
            CHECK(false);
            break;
        }
    }
    printf("VBR_CAPTURE_RANGE_PACK_BENCH ranges=%u chunks=%" PRIu64
           " bytes=%" PRIu64 " root_chunks=%u proof_nodes=%u"
           " proof_bytes=%" PRIu64 " elapsed_us=%lld\n",
           RANGE_COUNT, stats.chunks, stats.bytes, tree.chunk_count(),
           proof.proof_node_count(), proof_bytes_read,
           (long long) elapsed);
}

int main(int argc, char ** argv) {
    if (argc == 2 &&
        std::string(argv[1]) == "--range-pack-bench") {
        benchmark_fragmented_range_packing();
        return failures == 0 ? 0 : 1;
    }
    test_exact_capture_pretransfer_quote();
    test_attention_stem_prefix_planner();
    test_segment_chain_offsets();
    test_authenticated_range_tree();
    test_registry_quiescence_query();
    test_cpu_ring_boundaries();
    test_projected_unit_transfer();
    test_preflight_unavailable_projection_rows();
    test_manifest_coherent_assembly();
    test_projected_publication_claim_preparation();
    test_dependency_scoped_projected_catalog_publication();
    test_h2d_bounded_streaming();
    test_ring_accounting_once();
    test_capture_reservation_domain_preparation();
    test_server_store_construction_and_lifetime();
    test_projected_host_batch_store_adapter();
    test_server_capture_status_vocabulary();
    test_server_reference_tenant_authorization();
    test_server_import_route_classification();
    test_server_import_schedule_actionability();
    test_fresh_f16_size_generation();
    test_library_representation_identity();
    if (argc == 2 && std::string(argv[1]) == "--cuda") {
        test_cuda_ring();
    }
    if (failures != 0) {
        fprintf(stderr, "%d VBR streaming capture test(s) failed\n", failures);
        return 1;
    }
    printf("VBR artifact capture: PASS\n");
    return 0;
}
