/**
 * flash_namespace_config.h — stub for standalone (non-vLLM) builds.
 *
 * The upstream FlashAttention CMake generates this file. In the llama.cpp
 * SM70 plugin the FA2 base headers are used standalone, so we pin the
 * namespace directly:
 */
#pragma once

#ifndef FLASH_NAMESPACE
#define FLASH_NAMESPACE flash_sm70
#endif
