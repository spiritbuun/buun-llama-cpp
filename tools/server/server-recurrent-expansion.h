#pragma once

enum class server_recurrent_expansion_state {
    expanded,
    contracted,
};

enum class server_recurrent_speculation_action {
    rs_plane,
    backup_ready,
    try_expand,
    non_speculative,
};

// Process-local lifecycle for deferred recurrent backup cells. It deliberately
// contains no allocation or slot ownership: the server performs those actions,
// then publishes their terminal here exactly once.
struct server_recurrent_expansion_lifecycle {
    server_recurrent_expansion_state state = server_recurrent_expansion_state::expanded;
    bool retry_deferred = false;

    server_recurrent_speculation_action action(bool uses_rs_plane) const {
        if (uses_rs_plane) {
            return server_recurrent_speculation_action::rs_plane;
        }
        if (retry_deferred) {
            return server_recurrent_speculation_action::non_speculative;
        }
        switch (state) {
            case server_recurrent_expansion_state::expanded:
                return server_recurrent_speculation_action::backup_ready;
            case server_recurrent_expansion_state::contracted:
                return server_recurrent_speculation_action::try_expand;
        }
        return server_recurrent_speculation_action::non_speculative;
    }

    server_recurrent_speculation_action complete_expand(bool success) {
        if (success) {
            state = server_recurrent_expansion_state::expanded;
            retry_deferred = false;
            return server_recurrent_speculation_action::backup_ready;
        }
        state = server_recurrent_expansion_state::contracted;
        retry_deferred = true;
        return server_recurrent_speculation_action::non_speculative;
    }

    void defer() {
        retry_deferred = true;
    }

    bool rearm_if_idle(bool all_slots_idle) {
        if (!retry_deferred || !all_slots_idle) {
            return false;
        }
        retry_deferred = false;
        return true;
    }
};
