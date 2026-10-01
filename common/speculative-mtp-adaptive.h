#pragma once

#include <algorithm>

// Reversible per-request draft-depth control between the two bounds. The depth
// starts at the maximum and drops to the minimum when full-depth probing stops
// paying for itself; learned depths persist across requests, periodic probes
// and a matching streak recover the full depth.
class common_speculative_mtp_adaptive {
public:
    explicit common_speculative_mtp_adaptive(int minimum, int maximum)
        : minimum_depth(std::max(1, std::min(minimum, maximum))), maximum_depth(maximum), cap(maximum) {}

    int depth() const { return cap; }

    void reset() { *this = common_speculative_mtp_adaptive(minimum_depth, maximum_depth); }

    // Keep a learned depth across requests, but treat the new prefix as an
    // opportunity to recover. Restarting every request at the full depth
    // needlessly repeats the losing probe on a stream of low-match requests.
    void begin() {
        attempts = full = prefix_full = full_streak = 0;
        // Preserve the periodic countdown too: many short requests must not
        // postpone a full-depth probe indefinitely.
    }

    void accept(int drafted, int accepted, bool other) {
        // Clipped/failed drafts and another implementation's proposals do not
        // measure our selected depth. Duplicate carry refreshes have no draft.
        if (other || drafted != cap || accepted < 0 || accepted > drafted || minimum_depth == maximum_depth) {
            return;
        }

        if (cap == minimum_depth) {
            full_streak = accepted == minimum_depth ? full_streak + 1 : 0;
            // A matching streak signals a phase change only if the leading
            // positions were not already near-perfect in the full-depth probe.
            // Periodic probes also recover when no such streak is observed.
            if (--hold == 0 || (prefix_full < 12 && full_streak >= 8)) {
                reset();
            }
            return;
        }

        full += accepted == cap;
        prefix_full += accepted >= cap - 1;
        if (++attempts == 16) {
            if (full < 8) {
                cap = minimum_depth;
                hold = 256;
            } else {
                prefix_full = 0;
            }
            attempts = full = 0;
        }
    }

private:
    int minimum_depth;
    int maximum_depth;
    int cap;
    int attempts = 0;
    int full = 0;
    int prefix_full = 0;
    int full_streak = 0;
    int hold = 0;
};
