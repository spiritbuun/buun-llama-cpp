#include "common.h"
#include "../src/llama-model-loader.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#if defined(__linux__)
#include <sched.h>
#endif

static int failures = 0;

static void expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

int main() {
    constexpr uint64_t GiB = UINT64_C(1024) * 1024 * 1024;

    expect(!llama_mmap_prefetch_resolve(
            LLAMA_MMAP_PREFETCH_MODE_OFF, GiB, 16*GiB, true),
            "explicit off must disable bulk prefetch");
    expect(llama_mmap_prefetch_resolve(
            LLAMA_MMAP_PREFETCH_MODE_ON, 128*GiB, 16*GiB, true),
            "explicit on must preserve bulk prefetch");
    expect(llama_mmap_prefetch_resolve(
            LLAMA_MMAP_PREFETCH_MODE_AUTO, 12*GiB, 16*GiB, true),
            "auto must accept the exact 75 percent boundary");
    expect(!llama_mmap_prefetch_resolve(
            LLAMA_MMAP_PREFETCH_MODE_AUTO, 12*GiB + 1, 16*GiB, true),
            "auto must refuse a mapping above the safe boundary");
    expect(llama_mmap_prefetch_resolve(
            LLAMA_MMAP_PREFETCH_MODE_AUTO, 128*GiB, 0, false),
            "unknown memory must preserve historical behavior");
    expect(!llama_mmap_prefetch_resolve(
            LLAMA_MMAP_PREFETCH_MODE_AUTO, GiB, 0, true),
            "known exhausted memory must disable bulk prefetch");

#if defined(__linux__)
    const std::filesystem::path cgroup_root =
        std::filesystem::temp_directory_path() / "llama-cgroup-memory-policy";
    std::error_code ec;
    std::filesystem::remove_all(cgroup_root, ec);
    std::filesystem::create_directories(cgroup_root / "mount/leaf");
    {
        std::ofstream(cgroup_root / "cgroup") << "0::/parent/leaf\n";
        std::ofstream(cgroup_root / "mountinfo")
            << "35 25 0:31 /other " << (cgroup_root / "wrong-mount").string()
            << " rw - cgroup2 cgroup rw\n"
            << "36 25 0:32 /parent " << (cgroup_root / "mount").string()
            << " rw - cgroup2 cgroup rw\n";
        std::ofstream(cgroup_root / "mount/leaf/memory.max") << "max\n";
        std::ofstream(cgroup_root / "mount/leaf/memory.current") << "100\n";
        std::ofstream(cgroup_root / "mount/memory.max") << "1000\n";
        std::ofstream(cgroup_root / "mount/memory.current") << "400\n";
    }
    uint64_t cgroup_available = 0;
    expect(llama_linux_cgroup_memory_available(
            (cgroup_root / "cgroup").c_str(),
            (cgroup_root / "mountinfo").c_str(), cgroup_available) &&
            cgroup_available == 600,
            "matching finite cgroup ancestor must win over an unrelated first mount");
    std::filesystem::remove_all(cgroup_root, ec);
#endif

    uint8_t digest_a[32] = {};
    uint8_t digest_b[32] = {};
    digest_a[0] = 0x12;
    digest_b[0] = 0x34;
    const std::string first = common_moe_cache_profile_file(digest_a);
    const std::string again = common_moe_cache_profile_file(digest_a);
    const std::string other = common_moe_cache_profile_file(digest_b);
    expect(first == again, "model heatmap path must be stable");
    expect(first != other, "distinct semantic model families must not share heatmaps");
    expect(first.rfind(fs_get_cache_directory(), 0) == 0,
            "heatmap must live in the canonical llama.cpp cache directory");

    expect(common_cpu_get_num_physical_cores() > 0,
            "physical-core policy must return a usable thread count");
    expect(common_cpu_get_num_math() > 0,
            "math-core policy must return a usable thread count");
    expect(common_cpu_resolve_moe_threads(4) == 4,
            "small CPUs should use every physical core for host experts");
    expect(common_cpu_resolve_moe_threads(10) == 10,
            "Dorei-class hybrid CPUs should use every physical core");
    expect(common_cpu_resolve_moe_threads(24) == 12,
            "large CPUs should avoid memory-bound host-expert oversubscription");
    expect(common_cpu_get_num_moe_threads() > 0 &&
            common_cpu_get_num_moe_threads() <= 12,
            "detected host-expert thread policy must stay within its saturation cap");

#if defined(__linux__)
    cpu_set_t original;
    if (sched_getaffinity(0, sizeof(original), &original) == 0 && CPU_COUNT(&original) > 2) {
        cpu_set_t restricted;
        CPU_ZERO(&restricted);
        int selected = 0;
        for (int cpu = 0; cpu < CPU_SETSIZE && selected < 2; ++cpu) {
            if (CPU_ISSET(cpu, &original)) {
                CPU_SET(cpu, &restricted);
                selected++;
            }
        }
        if (selected == 2 && sched_setaffinity(0, sizeof(restricted), &restricted) == 0) {
            expect(common_cpu_get_num_physical_cores() <= selected,
                    "physical-core policy must respect inherited process affinity");
            expect(common_cpu_get_num_moe_threads() <= selected,
                    "host-expert threads must not exceed an inherited CPU affinity");
            expect(sched_setaffinity(0, sizeof(original), &original) == 0,
                    "CPU affinity test must restore the original mask");
        }
    }
#endif

    return failures == 0 ? 0 : 1;
}
