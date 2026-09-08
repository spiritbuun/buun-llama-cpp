#include "server-vbr-prompt-cache-support.h"

const char * server_vbr_prompt_cache_support_status_name(
        server_vbr_prompt_cache_support_status status) noexcept {
    switch (status) {
        case server_vbr_prompt_cache_support_status::supported:
            return "supported";
        case server_vbr_prompt_cache_support_status::codec_unsupported:
            return "codec_unsupported";
        case server_vbr_prompt_cache_support_status::draft_context_unsupported:
            return "draft_context_unsupported";
        case server_vbr_prompt_cache_support_status::speculative_slot_unsupported:
            return "speculative_slot_unsupported";
        case server_vbr_prompt_cache_support_status::media_prompt_unsupported:
            return "media_prompt_unsupported";
        case server_vbr_prompt_cache_support_status::alora_invocation_unsupported:
            return "alora_invocation_unsupported";
        case server_vbr_prompt_cache_support_status::artifact_topology_unavailable:
            return "artifact_topology_unavailable";
        case server_vbr_prompt_cache_support_status::accounting_unavailable:
            return "accounting_unavailable";
        case server_vbr_prompt_cache_support_status::artifact_store_unavailable:
            return "artifact_store_unavailable";
        case server_vbr_prompt_cache_support_status::_count:
            return "invalid";
    }
    return "invalid";
}

server_vbr_prompt_cache_support_status
server_vbr_prompt_cache_support_for(
        bool has_draft_context,
        bool speculative_slot,
        bool media_prompt,
        bool alora_invocation) noexcept {
    (void) has_draft_context;
    (void) speculative_slot;
    if (media_prompt) {
        return server_vbr_prompt_cache_support_status::
            media_prompt_unsupported;
    }
    if (alora_invocation) {
        return server_vbr_prompt_cache_support_status::
            alora_invocation_unsupported;
    }
    return server_vbr_prompt_cache_support_status::supported;
}

server_vbr_prompt_cache_fallback_action
server_vbr_prompt_cache_fallback_action_for(
        bool automatic_activation,
        server_vbr_prompt_cache_support_status status) noexcept {
    if (status == server_vbr_prompt_cache_support_status::supported) {
        return server_vbr_prompt_cache_fallback_action::enabled;
    }
    return automatic_activation
        ? server_vbr_prompt_cache_fallback_action::live_only
        : server_vbr_prompt_cache_fallback_action::startup_error;
}
