#pragma once

#include "common.h"

// Per-family speculative defaults: a model family that ships its own drafter
// (a built-in MTP head) gets that drafter on by default when the user did not
// choose --spec-type. Adding a family = adding a row to the table in
// spec-defaults.cpp.

// read general.architecture and <arch>.nextn_predict_layers from the GGUF header of
// params.model.path (no tensor data) and apply the family default, logging one INFO line
// when it fires. Call after the model path is resolved and before the model is loaded, so
// that -fit and the context sizing see the drafter exactly as with explicit flags.
// The default is skipped when --spec-type was given (any value, including none) or when a
// draft model is already attached.
void common_speculative_apply_model_default(common_params & params);
