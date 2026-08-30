include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-cache-authority.cpp
    server-cache-authority.h
    server-cache-control.cpp
    server-cache-control-wire.cpp
    server-cache-control.h
    server-cache-destruction-quote.cpp
    server-cache-destruction-quote.h
    server-cache-plan-authority.cpp
    server-cache-plan-authority.h
    server-cache-plan-preflight.cpp
    server-cache-plan-preflight.h
    server-cache-plan-preflight-internal.h
    server-cache-lease.cpp
    server-cache-lease.h
    server-cache-retention-proof.cpp
    server-cache-retention-proof.h
    server-cache-vbr-proof.cpp
    server-cache-vbr-proof.h
    server-cache-yield.cpp
    server-cache-yield.h
    server-prompt-cache-payload.cpp
    server-prompt-cache-payload.h
    server-vbr-artifact-store.cpp
    server-vbr-artifact-store.h
    server-vbr-capture-readiness.cpp
    server-vbr-capture-readiness.h
    server-vbr-prompt-cache-support.cpp
    server-vbr-prompt-cache-support.h
    server-retention-sidecar.cpp
    server-retention-sidecar.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-stream.cpp
    server-stream.h
    server-tools.cpp
    server-tools.h
    server-mcp.cpp
    server-mcp.h
    server-schema.cpp
    server-schema.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# C++ unit test for server_tokens::get_common_prefix (U3 empty-id media guard). Lives here rather
# than tests/ because that subdir is configured before tools/, so server-context/mtmd targets are
# not yet defined there. No model or mmproj needed — runs on CPU.
if (LLAMA_BUILD_TESTS)
    add_executable(test-server-media-prefix ${CMAKE_SOURCE_DIR}/tests/test-server-media-prefix.cpp)
    target_link_libraries(test-server-media-prefix PRIVATE server-context mtmd llama llama-common)
    target_include_directories(test-server-media-prefix PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/tools/mtmd)
    target_compile_features(test-server-media-prefix PRIVATE cxx_std_17)
    add_test(NAME test-server-media-prefix COMMAND test-server-media-prefix)
    set_property(TEST test-server-media-prefix PROPERTY LABELS main)

    add_executable(test-server-mmproj-lifecycle
        ${CMAKE_SOURCE_DIR}/tests/test-server-mmproj-lifecycle.cpp)
    target_link_libraries(test-server-mmproj-lifecycle PRIVATE
        server-context mtmd llama llama-common)
    target_include_directories(test-server-mmproj-lifecycle PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/tools/mtmd)
    target_compile_features(test-server-mmproj-lifecycle PRIVATE cxx_std_17)
    add_test(NAME test-server-mmproj-lifecycle
        COMMAND test-server-mmproj-lifecycle)
    set_property(TEST test-server-mmproj-lifecycle PROPERTY LABELS main)

    add_executable(test-server-prompt-cache ${CMAKE_SOURCE_DIR}/tests/test-server-prompt-cache.cpp)
    target_link_libraries(test-server-prompt-cache PRIVATE server-context mtmd llama llama-common)
    target_include_directories(test-server-prompt-cache PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/tools/mtmd)
    target_compile_features(test-server-prompt-cache PRIVATE cxx_std_17)
    add_test(NAME test-server-prompt-cache COMMAND test-server-prompt-cache)
    set_property(TEST test-server-prompt-cache PROPERTY LABELS main)
    set_property(TEST test-server-prompt-cache APPEND PROPERTY LABELS vbr-cache)
    add_test(NAME test-server-prompt-cache-clone-fault
        COMMAND test-server-prompt-cache --clone-fault)
    set_tests_properties(test-server-prompt-cache-clone-fault PROPERTIES
        LABELS "main;vbr-cache"
        ENVIRONMENT "LLAMA_SERVER_FAULT=load_clone_fail")
    add_test(NAME test-server-prompt-cache-accounting-fault
        COMMAND test-server-prompt-cache --accounting-fault)
    set_tests_properties(test-server-prompt-cache-accounting-fault PROPERTIES
        LABELS "main;vbr-cache"
        ENVIRONMENT "LLAMA_SERVER_FAULT=cache_lifecycle_after_commit")
    add_test(NAME test-server-prompt-cache-restore-checkpoint-fault
        COMMAND test-server-prompt-cache --restore-checkpoint-fault)
    set_tests_properties(
        test-server-prompt-cache-restore-checkpoint-fault PROPERTIES
        LABELS "main;vbr-cache"
        ENVIRONMENT "LLAMA_SERVER_FAULT=restore_checkpoint_artifact_missing")

    # The catalog test target is declared before the server targets. Add the
    # production prompt-cache seam to that translation unit instead of
    # compiling its large fixture under a second main(). Internal artifact
    # tests are intentionally absent on shared-library Windows builds.
    if (TARGET test-vbr-artifact)
        target_compile_definitions(test-vbr-artifact PRIVATE
            VBR_PROMPT_CACHE_PUBLICATION_TEST=1)
        target_link_libraries(test-vbr-artifact PRIVATE server-context mtmd)
        target_include_directories(test-vbr-artifact PRIVATE
            ${CMAKE_SOURCE_DIR}/tools/mtmd)
        add_test(NAME test-vbr-artifact-vbr-prefix-fault
            COMMAND test-vbr-artifact)
        set_tests_properties(test-vbr-artifact-vbr-prefix-fault PROPERTIES
            LABELS "main;vbr-cache"
            ENVIRONMENT "LLAMA_SERVER_FAULT=vbr_prompt_cache_prefix_fail")
        add_test(NAME test-vbr-artifact-vbr-anchor-prepare-fault
            COMMAND test-vbr-artifact)
        set_tests_properties(
            test-vbr-artifact-vbr-anchor-prepare-fault PROPERTIES
            LABELS "main;vbr-cache"
            ENVIRONMENT "LLAMA_SERVER_FAULT=vbr_anchor_prepare_fail")
        add_test(NAME test-vbr-artifact-vbr-pair-prepare-fault
            COMMAND test-vbr-artifact)
        set_tests_properties(
            test-vbr-artifact-vbr-pair-prepare-fault PROPERTIES
            LABELS "main;vbr-cache"
            ENVIRONMENT "LLAMA_SERVER_FAULT=vbr_prompt_cache_pair_prepare_fail")
    endif()

    if (TARGET test-vbr-artifact AND
        TARGET test-vbr-artifact-capture AND
        TARGET test-vbr-artifact-adopt AND
        TARGET test-retention-sidecar AND
        TARGET test-vbr-representation-epoch)
        add_custom_target(vbr-cache-gate
            COMMAND ${CMAKE_COMMAND} -E make_directory
                ${CMAKE_BINARY_DIR}/tmp
            COMMAND ${CMAKE_COMMAND} -E env
                TMPDIR=${CMAKE_BINARY_DIR}/tmp
                ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR}
                    --output-on-failure -L vbr-cache
            DEPENDS
                test-server-prompt-cache
                test-vbr-artifact
                test-vbr-artifact-capture
                test-vbr-artifact-adopt
                test-retention-sidecar
                test-vbr-representation-epoch
            USES_TERMINAL
            COMMENT "Run the CPU-deterministic VBR prompt-cache release gate")
    endif()

    # The deferred occupied-destination hardware gate composes the real store,
    # prompt-cache ticket, and model-backed adopter. Its default CPU execution
    # remains the existing model-free test and does not enter this CLI arm.
    if (TARGET test-vbr-artifact-adopt)
        target_link_libraries(test-vbr-artifact-adopt PRIVATE server-context mtmd)
        target_include_directories(test-vbr-artifact-adopt PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/tools/mtmd)
    endif()
endif()
