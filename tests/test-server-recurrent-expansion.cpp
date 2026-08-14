#include "server-recurrent-expansion.h"

#include <cstdio>

int main() {
    server_recurrent_expansion_lifecycle lifecycle;

    if (lifecycle.action(true) != server_recurrent_speculation_action::rs_plane ||
        lifecycle.state != server_recurrent_expansion_state::expanded) {
        std::fprintf(stderr, "RS-plane action mutated expansion ownership\n");
        return 1;
    }

    lifecycle.state = server_recurrent_expansion_state::contracted;
    if (lifecycle.action(true) != server_recurrent_speculation_action::rs_plane ||
        lifecycle.action(false) != server_recurrent_speculation_action::try_expand ||
        lifecycle.complete_expand(false) != server_recurrent_speculation_action::non_speculative ||
        lifecycle.state != server_recurrent_expansion_state::contracted ||
        !lifecycle.retry_deferred) {
        std::fprintf(stderr, "failed expansion did not enter non-speculative ownership\n");
        return 1;
    }

    if (lifecycle.rearm_if_idle(false) ||
        lifecycle.state != server_recurrent_expansion_state::contracted ||
        !lifecycle.retry_deferred ||
        !lifecycle.rearm_if_idle(true) ||
        lifecycle.state != server_recurrent_expansion_state::contracted ||
        lifecycle.retry_deferred ||
        lifecycle.complete_expand(true) != server_recurrent_speculation_action::backup_ready ||
        lifecycle.state != server_recurrent_expansion_state::expanded) {
        std::fprintf(stderr, "idle retry did not recover expansion ownership\n");
        return 1;
    }

    lifecycle.defer();
    if (lifecycle.action(false) != server_recurrent_speculation_action::non_speculative ||
        lifecycle.action(true) != server_recurrent_speculation_action::rs_plane ||
        lifecycle.state != server_recurrent_expansion_state::expanded ||
        !lifecycle.rearm_if_idle(true) ||
        lifecycle.state != server_recurrent_expansion_state::expanded ||
        lifecycle.action(false) != server_recurrent_speculation_action::backup_ready) {
        std::fprintf(stderr, "deferred backup incorrectly blocked RS rollback\n");
        return 1;
    }

    return 0;
}
