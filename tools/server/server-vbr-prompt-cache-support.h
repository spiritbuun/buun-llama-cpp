#pragma once

#include <cstdint>

// One typed owner for the automatic VBR host-cache support boundary. These
// statuses describe semantic support, not transient capture/admission failures.
enum class server_vbr_prompt_cache_support_status : uint8_t {
    supported = 0,
    codec_unsupported,
    draft_context_unsupported,
    speculative_slot_unsupported,
    media_prompt_unsupported,
    alora_invocation_unsupported,
    artifact_topology_unavailable,
    accounting_unavailable,
    artifact_store_unavailable,
    _count,
};

const char * server_vbr_prompt_cache_support_status_name(
    server_vbr_prompt_cache_support_status status) noexcept;

// Classify global and entry-local topology facts in one deterministic order.
// Callers may omit entry-local facts during startup.
server_vbr_prompt_cache_support_status
server_vbr_prompt_cache_support_for(
    bool has_draft_context,
    bool speculative_slot,
    bool media_prompt,
    bool alora_invocation) noexcept;

enum class server_vbr_prompt_cache_fallback_action : uint8_t {
    enabled = 0,
    live_only,
    startup_error,
    _count,
};

// Automatic activation degrades unsupported configurations to live-only.
// An explicit operator request remains strict and fails startup instead.
server_vbr_prompt_cache_fallback_action
server_vbr_prompt_cache_fallback_action_for(
    bool automatic_activation,
    server_vbr_prompt_cache_support_status status) noexcept;
