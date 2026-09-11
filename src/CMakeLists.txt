llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

# DeepSeek-V4 checkpoints do not currently ship their canonical chat template.
# Keep the native safetensors loader and GGUF converter on the same repository
# asset without requiring that asset to be present beside an installed binary.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../models/templates/deepseek-ai-DeepSeek-V4.jinja"
          LLAMA_SAFETENSORS_DEEPSEEK4_CHAT_TEMPLATE)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../models/templates/deepseek-ai-DeepSeek-V4-Flash-0731.jinja"
          LLAMA_SAFETENSORS_DEEPSEEK4_0731_CHAT_TEMPLATE)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/llama-safetensors-templates.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/llama-safetensors-templates.h"
    @ONLY)

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-kv-cache-dsa.cpp
            llama-kv-cache-dsa-iswa.cpp
            llama-kv-cache-msa.cpp
            llama-kv-cache-dsv4.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-hybrid-idx.cpp
            llama-memory-recurrent.cpp
            llama-memory-tree.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-safetensors.cpp
            llama-safetensors-loader.cpp
            llama-safetensors-metadata.cpp
            llama-safetensors-names.cpp
            llama-safetensors-quant.cpp
            llama-safetensors-tensor.cpp
            llama-safetensors-qwen3.cpp
            llama-safetensors-qwen35.cpp
    llama-safetensors-qwen4exp.cpp
    llama-safetensors-deepseek4.cpp
            llama-vocab.cpp
            llama-cache-accounting.cpp
            llama-cache-budget.cpp
            llama-cache-authority.cpp
            llama-vbr-hard-seal.cpp
            llama-vbr-extent.cpp
            llama-vbr-artifact.cpp
            llama-vbr-pinned-ring.cpp
            llama-vbr-artifact-capture.cpp
            llama-vbr-artifact-catalog.cpp
            llama-vbr-occupied-replacement.cpp
            llama-vbr-artifact-validate.cpp
            llama-vbr-artifact-stage.cpp
            llama-vbr-artifact-adopt.cpp
            llama-vbr-qsa-index.cpp
            llama-vbr-downward.cpp
            llama-vbr-upward.cpp
            llama-vbr-explicit-capture.cpp
            llama-vbr-controller-id.cpp
            llama-vbr-generation.cpp
            llama-vbr-operation.cpp
            llama-vram-demand.cpp
            llama-vram-ledger.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_compile_definitions(llama PRIVATE
    LLAMA_VERSION="${LLAMA_VERSION}"
    LLAMA_COMMIT="${LLAMA_BUILD_COMMIT}"
)

target_include_directories(llama PRIVATE . ../vendor "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if (WIN32)
        # Internal C++ implementation seams are consumed by llama-common and
        # server-context but are not part of the installed C API. Export them
        # from the Windows DLL without annotating them as stable LLAMA_API.
        set_target_properties(llama PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
