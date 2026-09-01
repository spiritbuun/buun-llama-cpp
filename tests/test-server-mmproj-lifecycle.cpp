#include "server-context.h"

#include <cstdio>

int main() {
    const auto result = server_mmproj_lifecycle_for_test();
    const bool passed =
        result.null_binding_clears_views &&
        result.restored_binding_updates_views &&
        result.failed_recreation_stays_null &&
        result.normal_restore_once &&
        result.thrown_media_restore_once &&
        result.thrown_callback_restore_once &&
        result.throwing_restore_not_retried &&
        result.incompatible_draft_disables_shift &&
        result.incompatible_draft_not_shifted &&
        result.compatible_draft_enables_shift &&
        result.compatible_draft_shifted;
    if (!passed) {
        std::fprintf(stderr,
            "mmproj lifecycle regression failed: "
            "null=%d restored=%d failed=%d normal=%d media=%d callback=%d throwing=%d "
            "draft_off=%d draft_not_shifted=%d draft_on=%d draft_shifted=%d\n",
            result.null_binding_clears_views,
            result.restored_binding_updates_views,
            result.failed_recreation_stays_null,
            result.normal_restore_once,
            result.thrown_media_restore_once,
            result.thrown_callback_restore_once,
            result.throwing_restore_not_retried,
            result.incompatible_draft_disables_shift,
            result.incompatible_draft_not_shifted,
            result.compatible_draft_enables_shift,
            result.compatible_draft_shifted);
        return 1;
    }
    return 0;
}
