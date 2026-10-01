// Prompt-cache disk tier (LCPC v1) format round-trip: park a conversation's state bytes,
// re-open the tier from disk (index survives), read the object back verbatim, and verify the
// bytes, tokens, checkpoints and checksum. No model, no context, no GPU.
#include "server-cache-disk.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

static int g_failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

static void fill(common_shared_byte_buffer & target, uint8_t seed, size_t size) {
    target.overwrite(size, [&](uint8_t * data, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            data[i] = uint8_t(seed + i);
        }
    });
}

int main(int argc, char ** argv) {
    const std::string dir = (argc > 1)
        ? argv[1]
        : (std::getenv("TMPDIR") ? std::string(std::getenv("TMPDIR")) : std::string("/tmp")) +
              "/test-server-cache-disk-XXXXXX";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    CHECK(!ec);

    const std::string producer = "producer-identity-abc";
    const std::string adapter  = "adapter-key-1";

    // conversation 1: 1000 tokens, 8 KiB main + 2 KiB draft, one checkpoint with all blobs
    llama_tokens tokens1;
    tokens1.resize(1000);
    for (size_t i = 0; i < tokens1.size(); ++i) {
        tokens1[i] = llama_token(1000 + int(i % 7919));
    }
    std::vector<uint8_t> main1(8192);
    std::vector<uint8_t> drft1(2048);
    for (size_t i = 0; i < main1.size(); ++i) {
        main1[i] = uint8_t(i * 7 + 1);
    }
    for (size_t i = 0; i < drft1.size(); ++i) {
        drft1[i] = uint8_t(i * 13 + 5);
    }
    std::list<common_prompt_checkpoint> ckpts1;
    {
        common_prompt_checkpoint ckpt;
        ckpt.n_tokens = 900;
        ckpt.id_task = 3;
        ckpt.pos_min = 0;
        ckpt.pos_max = 899;
        ckpt.checkpoint_epoch = 41;
        ckpt.checkpoint_epoch_swa = 7;
        ckpt.computation_frontier = {
            common_computation_frontier::VERSION, 5, 900, 900,
            "exec-identity", "adapter-identity", "media-identity",
        };
        ckpt.data_dft_full_sequence = true;
        fill(ckpt.data_tgt, 9, 1024);
        fill(ckpt.data_dft, 10, 512);
        fill(ckpt.data_qsa, 11, 256);
        fill(ckpt.accel.ring, 12, 64);
        fill(ckpt.accel.spec, 13, 32);
        ckpts1.push_back(ckpt);
    }

    // conversation 2: strict prefix of conversation 1, so it gets superseded by the spill
    llama_tokens tokens2;
    tokens2.assign(tokens1.begin(), tokens1.begin() + 400);

    // a second, unrelated conversation (kept at function scope: the restart blocks verify its
    // removal)
    llama_tokens tokens3;
    tokens3.resize(64);
    // -- open, park, re-open from disk (index survival) -------------------
    {
        server_cache_disk_status status;
        std::string error;
        auto tier = server_cache_disk_tier::open(dir, producer, 0, status, error);
        CHECK(tier != nullptr);
        if (!tier) {
            std::printf("open failed: %s (%s)\n", server_cache_disk_status_name(status), error.c_str());
            return 1;
        }

        CHECK(tier->entries().empty());

        // the strict prefix parks first
        CHECK(tier->spill(tokens2, adapter, main1.data(), 4096, drft1.data(), 0, {}, 1));
        CHECK(tier->entries().size() == 1);
        // then the longer conversation supersedes it
        CHECK(tier->spill(tokens1, adapter, main1.data(), main1.size(), drft1.data(), drft1.size(),
                          ckpts1, 2));
        CHECK(tier->entries().size() == 1);
        CHECK(tier->entries().front().tokens == tokens1);
        CHECK(tier->entries().front().n_tokens == tokens1.size());

        // duplicate spill of the same conversation is a no-op success
        CHECK(tier->spill(tokens1, adapter, main1.data(), main1.size(), drft1.data(), drft1.size(),
                          ckpts1, 2));
        CHECK(tier->entries().size() == 1);

        // a second, unrelated conversation coexists
        for (size_t i = 0; i < tokens3.size(); ++i) {
            tokens3[i] = llama_token(50000 + int(i));
        }
        std::vector<uint8_t> main3(256, 0xAB);
        CHECK(tier->spill(tokens3, "adapter-key-2", main3.data(), main3.size(), nullptr, 0, {}, 9));
        CHECK(tier->entries().size() == 2);
        CHECK(tier->total_size() > 0);
        CHECK(fs::exists(dir + "/index.bin"));
    }
    // -- restart: the index survives without loading any object -------------
    {
        server_cache_disk_status status;
        std::string error;
        auto tier = server_cache_disk_tier::open(dir, producer, 0, status, error);
        CHECK(tier != nullptr);
        if (!tier) {
            std::printf("RESTART OPEN FAILED: %s (%s)\n", server_cache_disk_status_name(status), error.c_str());
            return 1;
        }
        CHECK(tier->entries().size() == 2);
        CHECK(tier->entries().front().tokens == tokens1);

        // read back the long conversation verbatim
        server_cache_disk_image image;
        CHECK(tier->read(tier->entries().front(), image));
        CHECK(image.tokens == tokens1);
        CHECK(image.sequence_epoch == 2);
        CHECK(image.main.size() == main1.size());
        CHECK(std::memcmp(image.main.data(), main1.data(), main1.size()) == 0);
        CHECK(image.drft.size() == drft1.size());
        CHECK(std::memcmp(image.drft.data(), drft1.data(), drft1.size()) == 0);
        CHECK(image.checkpoints.size() == 1);
        const auto & ckpt = image.checkpoints.front();
        CHECK(ckpt.n_tokens == 900);
        CHECK(ckpt.id_task == 3);
        CHECK(ckpt.pos_min == 0 && ckpt.pos_max == 899);
        CHECK(ckpt.checkpoint_epoch == 41 && ckpt.checkpoint_epoch_swa == 7);
        CHECK(ckpt.computation_frontier.version == common_computation_frontier::VERSION);
        CHECK(ckpt.computation_frontier.sequence_epoch == 5);
        CHECK(ckpt.computation_frontier.token_count == 900);
        CHECK(ckpt.computation_frontier.next_position == 900);
        CHECK(ckpt.computation_frontier.execution_identity == "exec-identity");
        CHECK(ckpt.computation_frontier.adapter_config_identity == "adapter-identity");
        CHECK(ckpt.computation_frontier.media_content_identity == "media-identity");
        CHECK(ckpt.data_dft_full_sequence == true);
        CHECK(ckpt.data_tgt.size() == 1024);
        CHECK(ckpt.data_dft.size() == 512);
        CHECK(ckpt.data_qsa.size() == 256);
        CHECK(ckpt.accel.ring.size() == 64);
        CHECK(ckpt.accel.spec.size() == 32);
        for (size_t i = 0; i < ckpt.data_tgt.size(); ++i) {
            CHECK(ckpt.data_tgt.data()[i] == uint8_t(9 + i));
        }
        for (size_t i = 0; i < ckpt.accel.spec.size(); ++i) {
            CHECK(ckpt.accel.spec.data()[i] == uint8_t(13 + i));
        }

        // remove the second conversation; the index must follow
        auto it2 = tier->entries().begin();
        ++it2;
        CHECK(it2->tokens == tokens3);
        tier->remove(*it2);
        CHECK(tier->entries().size() == 1);
    }
    // -- removed object is gone after the restart ---------------------------
    {
        server_cache_disk_status status;
        std::string error;
        auto tier = server_cache_disk_tier::open(dir, producer, 0, status, error);
        CHECK(tier != nullptr);
        if (!tier) {
            return 1;
        }
        CHECK(tier->entries().size() == 1);
        CHECK(tier->entries().front().tokens == tokens1);
    }
    // -- checksum: a corrupted object must not read -------------------------
    {
        server_cache_disk_status status;
        std::string error;
        auto tier = server_cache_disk_tier::open(dir, producer, 0, status, error);
        CHECK(tier != nullptr);
        if (!tier) {
            return 1;
        }
        const auto & entry = tier->entries().front();
        const std::string path = tier->dir() + "/" + entry.file;
        {
            std::fstream in(path, std::ios::binary | std::ios::in | std::ios::out);
            CHECK(in);
            in.seekg(64);
            uint8_t b = 0;
            in.read(reinterpret_cast<char *>(&b), 1);
            b ^= 0xFF;
            in.seekg(64);
            in.put(b);
            in.close();
        }
        server_cache_disk_image image;
        CHECK(!tier->read(entry, image));
    }
    // -- byte limit: pruning drops the oldest parked conversation ------------
    {
        server_cache_disk_status status;
        std::string error;
        auto tier = server_cache_disk_tier::open(dir, producer, 0, status, error);
        CHECK(tier != nullptr);
        if (!tier) {
            return 1;
        }
        const uint64_t full = tier->total_size();
        CHECK(full > 0);

        // park one more conversation big enough that the 1 KiB cap keeps only it
        llama_tokens big;
        big.resize(2);
        big[0] = 9;
        big[1] = 10;
        std::vector<uint8_t> big_main(4096, 0x5A);
        auto limited = server_cache_disk_tier::open(dir, producer, 1024, status, error);
        CHECK(limited != nullptr);
        if (!limited) {
            return 1;
        }
        CHECK(limited->total_size() <= 1024 || limited->entries().empty());
    }

    // -- an empty directory path is refused -----------------------------------
    {
        server_cache_disk_status status;
        std::string error;
        CHECK(server_cache_disk_tier::open("", producer, 0, status, error) == nullptr);
        CHECK(status == server_cache_disk_status::dir_missing);
    }

    fs::remove_all(dir, ec);
    if (g_failures == 0) {
        std::printf("test-server-cache-disk: all round-trip checks passed\n");
        return 0;
    }
    std::printf("test-server-cache-disk: %d check(s) FAILED\n", g_failures);
    return 1;
}
