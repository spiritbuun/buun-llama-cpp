# Marlin CUDA templates

This directory contains the dependency-free subset of the Marlin implementation
vendored from vLLM 0.28.0 for the native group-affine Q4 CUDA executor. vLLM is
licensed under Apache-2.0; its implementation is derived from
IST-DASLab/Marlin. The original notices in `marlin_template.h` are retained.

Local changes remove the PyTorch registration and scalar-type dependencies.
The GGML launch, repack, storage, and reverse-repack adapters live outside this
directory.
