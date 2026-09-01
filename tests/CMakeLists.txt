llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE llama llama-common)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE llama-common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

llama_build_and_test(test-server-recurrent-expansion.cpp)
target_include_directories(test-server-recurrent-expansion PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)

llama_build(test-safetensors-registry.cpp)
target_include_directories(test-safetensors-registry PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_test(test-safetensors-registry)
llama_build(test-save-load-state.cpp)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-unicode.cpp)

    llama_build(test-recurrent-state-rollback.cpp)
    target_include_directories(test-recurrent-state-rollback PRIVATE ${PROJECT_SOURCE_DIR}/src)

    llama_build(test-vbr-representation-epoch.cpp)
    target_include_directories(test-vbr-representation-epoch PRIVATE ${PROJECT_SOURCE_DIR}/src)
    add_test(
        NAME test-vbr-controller-identity-cpu
        COMMAND $<TARGET_FILE:test-vbr-representation-epoch> --identity-cpu)
    add_test(
        NAME test-vbr-generation-tracker-cpu
        COMMAND $<TARGET_FILE:test-vbr-representation-epoch> --generation-cpu)
    add_test(
        NAME test-vbr-operation-recovery-cpu
        COMMAND $<TARGET_FILE:test-vbr-representation-epoch> --operation-cpu)

    add_test(
        NAME check-memory-tree-coverage
        COMMAND ${CMAKE_COMMAND} -DSOURCE_ROOT=${PROJECT_SOURCE_DIR}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/check-memory-tree-coverage.cmake)

    llama_build_and_test(test-cache-accounting.cpp)
    target_include_directories(test-cache-accounting PRIVATE ${PROJECT_SOURCE_DIR}/src)

    llama_build_and_test(test-cache-budget.cpp)
    target_include_directories(test-cache-budget PRIVATE ${PROJECT_SOURCE_DIR}/src)

    llama_build_and_test(test-cache-authority.cpp)
    target_include_directories(test-cache-authority PRIVATE ${PROJECT_SOURCE_DIR}/src)

    llama_build_and_test(
        test-vbr-artifact.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-lease.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-vbr-proof.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-prompt-cache-payload.cpp)
    target_include_directories(test-vbr-artifact PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tools/server)
    llama_build_and_test(test-vbr-artifact-adopt.cpp)
    target_include_directories(test-vbr-artifact-adopt PRIVATE ${PROJECT_SOURCE_DIR}/src)
    set_property(TEST test-vbr-artifact-adopt APPEND PROPERTY LABELS vbr-cache)

    llama_build_and_test(
        test-vbr-artifact-capture.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-vbr-artifact-store.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-prompt-cache-payload.cpp)
    target_include_directories(test-vbr-artifact-capture PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tools/server)
    llama_test_cmd(test-vbr-artifact
        NAME test-vbr-capture-projection-scale
        LABEL "vbr-cache;slow"
        ARGS --capture-projection-bench)
    llama_test_cmd(test-vbr-artifact-capture
        NAME test-vbr-range-proof-scale
        LABEL "vbr-cache;slow"
        ARGS --range-pack-bench)
    set_property(TEST test-vbr-artifact APPEND PROPERTY LABELS vbr-cache)
    set_property(TEST test-vbr-artifact-capture APPEND PROPERTY LABELS vbr-cache)

    llama_build_and_test(
        test-cache-lease.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-lease.cpp)
    target_include_directories(test-cache-lease PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tools/server)

    llama_build_and_test(
        test-cache-yield.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-yield.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-lease.cpp)
    target_include_directories(test-cache-yield PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tools/server)

    llama_build_and_test(
        test-retention-policy-corpus.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-retention-sidecar.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-yield.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-lease.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-destruction-quote.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-plan-authority.cpp)
    target_include_directories(test-retention-policy-corpus PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tools/server)

    llama_build_and_test(
        test-cache-destruction-quote.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-destruction-quote.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-retention-proof.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-yield.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-lease.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-plan-authority.cpp)
    target_include_directories(test-cache-destruction-quote PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tools/server)

    llama_build_and_test(test-cache-plan-record.cpp)
    target_include_directories(test-cache-plan-record PRIVATE ${PROJECT_SOURCE_DIR}/src)

    llama_build_and_test(test-vbr-hard-seal.cpp)
    target_include_directories(test-vbr-hard-seal PRIVATE ${PROJECT_SOURCE_DIR}/src)

    llama_build_and_test(
        test-cache-control.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-control.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-control-wire.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-retention-proof.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-vbr-proof.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-vbr-artifact-store.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-prompt-cache-payload.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-retention-sidecar.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-destruction-quote.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-plan-authority.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-yield.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-lease.cpp)
    target_include_directories(test-cache-control PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tools/server)
    target_compile_definitions(test-cache-control PRIVATE
        CACHE_CONTROL_GOLDEN_PATH="${PROJECT_SOURCE_DIR}/tests/fixtures/cache-control-v1.json")

    llama_build_and_test(test-cache-plan-estimate.cpp)
    target_include_directories(test-cache-plan-estimate PRIVATE ${PROJECT_SOURCE_DIR}/src)

    llama_build_and_test(
        test-cache-plan-authority.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-plan-authority.cpp)
    target_include_directories(test-cache-plan-authority PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tools/server)

    llama_build_and_test(
        test-cache-plan-preflight.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-plan-preflight.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-plan-authority.cpp)
    target_include_directories(test-cache-plan-preflight PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tools/server)
    target_compile_definitions(test-cache-plan-preflight PRIVATE
        CACHE_PLAN_PREFLIGHT_GOLDEN_PATH="${PROJECT_SOURCE_DIR}/tests/fixtures/cache-plan-preflight-v1.json")

    llama_build_and_test(
        test-retention-sidecar.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-retention-sidecar.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-lease.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-destruction-quote.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-yield.cpp
        ${PROJECT_SOURCE_DIR}/tools/server/server-cache-plan-authority.cpp)
    target_include_directories(test-retention-sidecar PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tools/server)
    set_property(TEST test-retention-sidecar APPEND PROPERTY LABELS vbr-cache)
    set_property(TEST test-vbr-controller-identity-cpu APPEND PROPERTY LABELS vbr-cache)
    set_property(TEST test-vbr-generation-tracker-cpu APPEND PROPERTY LABELS vbr-cache)
    set_property(TEST test-vbr-operation-recovery-cpu APPEND PROPERTY LABELS vbr-cache)

    llama_build(test-dflash-tape-vs-plane.cpp)
    target_include_directories(test-dflash-tape-vs-plane PRIVATE ${PROJECT_SOURCE_DIR}/src)

    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-batch-alloc.cpp)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        llama_build_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        target_include_directories(test-json-schema-to-grammar PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    endif()

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    llama_build_and_test(test-llama-archs.cpp)
    llama_build(test-lora-determinism.cpp)

    set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-models
        LABEL main
        ARGS -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models PROPERTIES
        FIXTURES_SETUP generate-models
    )

    llama_test(
        test-recurrent-state-rollback
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback PROPERTIES
        FIXTURES_REQUIRED generate-models
    )

    # test-vbr-representation-epoch requires an iSWA Gemma-4 fixture. test-llama-archs
    # deliberately does not save Gemma-4 yet (its synthetic model/context support is
    # incomplete), so registering this target against a generated non-iSWA model is
    # a false gate. Hardware validation invokes the built binary with the canonical
    # real Gemma-4 fixture explicitly.

    set(LORA_MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-lora-models/")
    file(MAKE_DIRECTORY "${LORA_MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-lora-model
        LABEL main
        ARGS -a qwen35 -s 0 -o "${LORA_MODEL_DIR}"
    )
    set_tests_properties(test-generate-lora-model PROPERTIES
        FIXTURES_SETUP generate-lora-model
    )

    llama_test(
        test-lora-determinism
        LABEL main
        ARGS "${LORA_MODEL_DIR}/qwen35-dense.gguf" "${LORA_MODEL_DIR}"
    )
    set_tests_properties(test-lora-determinism PROPERTIES
        FIXTURES_REQUIRED generate-lora-model
    )

    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-nemotron-h
        LABEL main
        ARGS -m "${MODEL_DIR}/nemotron_h-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-nemotron-h PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-dsv4
        LABEL main
        ARGS -m "${MODEL_DIR}/deepseek4-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-dsv4 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )

    # Test state save/load functionality across all architectures, using the generated dummy models
    llama_test(
        test-save-load-state
        LABEL main
        ARGS --models "${MODEL_DIR}" --fit off -ctk f16 -ctv f16
    )
    set_tests_properties(test-save-load-state PROPERTIES FIXTURES_REQUIRED generate-models)
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
# debug tool for chat template differential analysis (not registered as a test, run it manually)
llama_build(test-chat-analysis.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${CMAKE_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp)
llama_build_and_test(test-mtp-vocab-trim.cpp)
if (MSVC)
    target_compile_options(test-arg-parser PRIVATE /UNDEBUG)
    target_compile_options(test-mtp-vocab-trim PRIVATE /UNDEBUG)
else()
    target_compile_options(test-arg-parser PRIVATE -UNDEBUG)
    target_compile_options(test-mtp-vocab-trim PRIVATE -UNDEBUG)
endif()
llama_build_and_test(test-vbr-policy.cpp)
llama_build_and_test(test-vbr-downward.cpp)
llama_build_and_test(test-vbr-physical.cpp)
llama_build_and_test(test-vbr-transaction.cpp)
if (MSVC)
    target_compile_options(test-vbr-policy PRIVATE /UNDEBUG)
    target_compile_options(test-vbr-downward PRIVATE /UNDEBUG)
    target_compile_options(test-vbr-physical PRIVATE /UNDEBUG)
    target_compile_options(test-vbr-transaction PRIVATE /UNDEBUG)
else()
    target_compile_options(test-vbr-policy PRIVATE -UNDEBUG)
    target_compile_options(test-vbr-downward PRIVATE -UNDEBUG)
    target_compile_options(test-vbr-physical PRIVATE -UNDEBUG)
    target_compile_options(test-vbr-transaction PRIVATE -UNDEBUG)
endif()
llama_build_and_test(test-model-resolution.cpp)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()
llama_build_and_test(test-backend-ops.cpp)
if (TARGET ggml-cuda AND GGML_CUDA_Q8_MMV_TEST_INSTRUMENTATION)
    llama_test(test-backend-ops NAME test-cuda-q8-mmv-dispatch
        LABEL "main;backend"
        ARGS test -b CUDA0 -o MUL_MAT -p "type_a=q8_0.*m=257,n=1,k=5120")
    set_property(TEST test-cuda-q8-mmv-dispatch PROPERTY
        ENVIRONMENT "GGML_TEST_Q8_0_MMV_EXPECT_PATH=auto")
endif()
llama_build_and_test(test-cuda-q8-mmv-selector.cpp)
if (TARGET ggml-cuda AND NOT GGML_BACKEND_DL)
    llama_build(test-cuda-hc-combine-match.cpp)
    target_link_libraries(test-cuda-hc-combine-match PRIVATE ggml-cuda)
    llama_test(test-cuda-hc-combine-match LABEL "main;backend")
    llama_build(test-cuda-hc-stream-mean-match.cpp)
    target_link_libraries(test-cuda-hc-stream-mean-match PRIVATE ggml-cuda)
    llama_test(test-cuda-hc-stream-mean-match LABEL "main;backend")
    llama_build(test-cuda-hc-stream-mean-zero.cpp)
    llama_test(test-cuda-hc-stream-mean-zero NAME test-cuda-hc-stream-mean-zero-enabled LABEL "main;backend")
    llama_test(test-cuda-hc-stream-mean-zero NAME test-cuda-hc-stream-mean-zero-disabled LABEL "main;backend")
    set_property(TEST test-cuda-hc-stream-mean-zero-disabled PROPERTY ENVIRONMENT "GGML_CUDA_DISABLE_FUSION=1")
    set_tests_properties(test-cuda-hc-stream-mean-zero-enabled test-cuda-hc-stream-mean-zero-disabled
        PROPERTIES SKIP_RETURN_CODE 77)
    llama_build(test-cuda-hc-grouped-rms-match.cpp)
    target_link_libraries(test-cuda-hc-grouped-rms-match PRIVATE ggml-cuda)
    llama_test(test-cuda-hc-grouped-rms-match LABEL "main;backend")
    llama_build(test-cuda-hc-grouped-rms-special.cpp)
    llama_test(test-cuda-hc-grouped-rms-special LABEL "main;backend")
    set_tests_properties(test-cuda-hc-grouped-rms-special PROPERTIES SKIP_RETURN_CODE 77)
    llama_build(test-cuda-mmvq-post-silu-match.cpp)
    target_link_libraries(test-cuda-mmvq-post-silu-match PRIVATE ggml-cuda)
    llama_test(test-cuda-mmvq-post-silu-match LABEL "main;backend")
    llama_build(test-cuda-mmvq-post-silu-special.cpp)
    llama_test(test-cuda-mmvq-post-silu-special LABEL "main;backend")
    set_tests_properties(test-cuda-mmvq-post-silu-special PROPERTIES SKIP_RETURN_CODE 77)
endif()
if (TARGET ggml-cuda AND GGML_CUDA_HC_COMBINE_TEST_INSTRUMENTATION)
    llama_test(test-backend-ops NAME test-cuda-hc-combine-dispatch
        LABEL "main;backend"
        ARGS test -b CUDA0 -o HC_COMBINE_FUSION -p "n_embd=127,hc=4,nt=5,variant=7")
    set_property(TEST test-cuda-hc-combine-dispatch PROPERTY
        ENVIRONMENT "GGML_TEST_HC_COMBINE_EXPECT_DISPATCH=enabled")
    llama_test(test-backend-ops NAME test-cuda-hc-combine-fusion-disabled
        LABEL "main;backend"
        ARGS test -b CUDA0 -o HC_COMBINE_FUSION -p "n_embd=257,hc=4,nt=17,variant=0")
    set_property(TEST test-cuda-hc-combine-fusion-disabled PROPERTY
        ENVIRONMENT "GGML_TEST_HC_COMBINE_EXPECT_DISPATCH=disabled;GGML_CUDA_DISABLE_FUSION=1")
endif()
if (TARGET ggml-cuda AND GGML_CUDA_HC_STREAM_MEAN_TEST_INSTRUMENTATION)
    llama_test(test-backend-ops NAME test-cuda-hc-stream-mean-dispatch
        LABEL "main;backend"
        ARGS test -b CUDA0 -o HC_STREAM_MEAN_FUSION -p "n_embd=257,nt=17,variant=0")
    set_property(TEST test-cuda-hc-stream-mean-dispatch PROPERTY
        ENVIRONMENT "GGML_TEST_HC_STREAM_MEAN_EXPECT_DISPATCH=enabled")
    llama_test(test-backend-ops NAME test-cuda-hc-stream-mean-fusion-disabled
        LABEL "main;backend"
        ARGS test -b CUDA0 -o HC_STREAM_MEAN_FUSION -p "n_embd=257,nt=17,variant=0")
    set_property(TEST test-cuda-hc-stream-mean-fusion-disabled PROPERTY
        ENVIRONMENT "GGML_TEST_HC_STREAM_MEAN_EXPECT_DISPATCH=disabled;GGML_CUDA_DISABLE_FUSION=1")
endif()
if (TARGET ggml-cuda AND GGML_CUDA_HC_GROUPED_RMS_TEST_INSTRUMENTATION)
    llama_test(test-backend-ops NAME test-cuda-hc-grouped-rms-dispatch
        LABEL "main;backend"
        ARGS test -b CUDA0 -o HC_GROUPED_RMS_FUSION -p "n_embd=257,hc=4,nt=17,variant=0")
    set_property(TEST test-cuda-hc-grouped-rms-dispatch PROPERTY
        ENVIRONMENT "GGML_TEST_HC_GROUPED_RMS_EXPECT_DISPATCH=enabled")
    llama_test(test-backend-ops NAME test-cuda-hc-grouped-rms-disabled
        LABEL "main;backend"
        ARGS test -b CUDA0 -o HC_GROUPED_RMS_FUSION -p "n_embd=257,hc=4,nt=17,variant=0")
    set_property(TEST test-cuda-hc-grouped-rms-disabled PROPERTY
        ENVIRONMENT "GGML_TEST_HC_GROUPED_RMS_EXPECT_DISPATCH=disabled;GGML_CUDA_DISABLE_FUSION=1")
endif()
if (TARGET ggml-cuda AND GGML_CUDA_Q8_POST_SILU_TEST_INSTRUMENTATION)
    llama_test(test-backend-ops NAME test-cuda-q8-post-silu-dispatch
        LABEL "main;backend"
        ARGS test -b CUDA0 -o MMVQ_POST_SILU_FUSION -p "m=257,k=5120,variant=0")
    set_property(TEST test-cuda-q8-post-silu-dispatch PROPERTY
        ENVIRONMENT "GGML_TEST_Q8_POST_SILU_EXPECT_DISPATCH=enabled")
    llama_test(test-backend-ops NAME test-cuda-q8-post-silu-disabled
        LABEL "main;backend"
        ARGS test -b CUDA0 -o MMVQ_POST_SILU_FUSION -p "m=257,k=5120,variant=0")
    set_property(TEST test-cuda-q8-post-silu-disabled PROPERTY
        ENVIRONMENT "GGML_TEST_Q8_POST_SILU_EXPECT_DISPATCH=disabled;GGML_CUDA_DISABLE_FUSION=1")
    llama_test(test-backend-ops NAME test-cuda-q8-post-silu-force-mmq-refused
        LABEL "main;backend"
        ARGS test -b CUDA0 -o MMVQ_POST_SILU_FUSION -p "m=127,k=5120,variant=3")
    set_property(TEST test-cuda-q8-post-silu-force-mmq-refused PROPERTY
        ENVIRONMENT "GGML_TEST_Q8_POST_SILU_EXPECT_DISPATCH=disabled")
    llama_test(test-backend-ops NAME test-cuda-q8-post-silu-padded-view-refused
        LABEL "main;backend"
        ARGS test -b CUDA0 -o MMVQ_POST_SILU_FUSION -p "m=127,k=5088,variant=4")
    set_property(TEST test-cuda-q8-post-silu-padded-view-refused PROPERTY
        ENVIRONMENT "GGML_TEST_Q8_POST_SILU_EXPECT_DISPATCH=disabled")
    llama_test(test-backend-ops NAME test-cuda-q8-post-silu-small-k-refused
        LABEL "main;backend"
        ARGS test -b CUDA0 -o MMVQ_POST_SILU_FUSION -p "m=127,k=512,variant=5")
    set_property(TEST test-cuda-q8-post-silu-small-k-refused PROPERTY
        ENVIRONMENT "GGML_TEST_Q8_POST_SILU_EXPECT_DISPATCH=disabled")
endif()
if (NOT LLAMA_USE_SYSTEM_GGML)
    llama_build(test-moe-cache.cpp)
    llama_test(test-moe-cache LABEL "main;backend")
    set_tests_properties(test-moe-cache PROPERTIES TIMEOUT 120)
    llama_build_and_test(test-cuda-moe-cache-flat-hits-selector.cpp)
    if (TARGET ggml-cuda AND NOT GGML_BACKEND_DL)
        find_package(CUDAToolkit REQUIRED)
        llama_build(test-cuda-moe-cache-flat-hits.cpp)
        target_link_libraries(test-cuda-moe-cache-flat-hits PRIVATE ggml-cuda CUDA::cudart)
        llama_test(test-cuda-moe-cache-flat-hits LABEL "main;backend")
        set_property(TEST test-cuda-moe-cache-flat-hits PROPERTY ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")
        set_tests_properties(test-cuda-moe-cache-flat-hits PROPERTIES TIMEOUT 120)
        if (GGML_CUDA_MOE_CACHE_FLAT_HITS_TEST_INSTRUMENTATION)
            target_compile_definitions(test-cuda-moe-cache-flat-hits
                PRIVATE GGML_CUDA_MOE_CACHE_FLAT_HITS_TEST_INSTRUMENTATION)
        endif()
    endif()
    if (TARGET ggml-cuda AND GGML_CUDA_MOE_CACHE_FLAT_HITS_TEST_INSTRUMENTATION)
        llama_test(test-moe-cache NAME test-moe-cache-flat-hits-baseline LABEL "main;backend")
        set_property(TEST test-moe-cache-flat-hits-baseline PROPERTY
            ENVIRONMENT "GGML_TEST_MOE_CACHE_FLAT_HITS_EXPECT_BASELINE=1;CUDA_VISIBLE_DEVICES=0")
        set_tests_properties(test-moe-cache-flat-hits-baseline PROPERTIES TIMEOUT 180)
    endif()
endif()

llama_build_and_test(test-moe-cache-fit.cpp)
llama_build_and_test(test-auto-memory-policy.cpp)
llama_build_and_test(test-kv-cell-position-index.cpp)

if (TARGET ggml-cuda AND NOT GGML_BACKEND_DL)
    llama_build(test-vbr-vmm.cpp)
    target_link_libraries(test-vbr-vmm PRIVATE ggml-cuda)
    llama_test(test-vbr-vmm LABEL "backend")
endif()

llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)
    llama_build_and_test(test-col2im-1d.cpp)
endif()

# libmtmd
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
unset(LLAMA_TEST_NAME)

llama_build_and_test(test-mtmd-impl.cpp)
target_link_libraries(test-mtmd-impl PRIVATE mtmd)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data LABEL "model")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model")
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()
