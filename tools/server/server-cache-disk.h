#pragma once

#include "llama.h"
#include "common.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <vector>

// Prompt-cache disk tier (P4 host-prompt-cache gap, docs/development/server-resume-format.md):
// a parked host-prompt-cache conversation spills here as one file per conversation plus a small
// index, so a restarted server can see what is on disk without loading the state. The parked
// bytes are the exact in-memory image of a fixed host entry (main + draft sequence state, and
// the checkpoint images). Parking never re-quantizes and restore writes the bytes back verbatim,
// so a restored conversation is bit-identical to the parked one.
//
// File layout, little-endian, magic "LCPC", version 1:
//   magic[4] "LCPC" | u32 version | u32 n_tokens | i32 tokens[n_tokens]
//   u32 len | u8 producer_identity[len]
//   u32 len | u8 adapter_config_key[len]
//   u64 main_len | u8 main[main_len]
//   u64 drft_len | u8 drft[drft_len]
//   u32 n_checkpoints
//   per checkpoint:
//     i64 n_tokens | i32 id_task | i32 pos_min | i32 pos_max
//     u64 checkpoint_epoch | u64 checkpoint_epoch_swa
//     u32 frontier_version | u64 sequence_epoch | i64 token_count | i64 next_position
//     u32 len | u8 execution_identity[len]
//     u32 len | u8 adapter_config_identity[len]
//     u32 len | u8 media_content_identity[len]
//     u8  data_dft_full_sequence
//     u64 len | u8 data_tgt[len]
//     u64 len | u8 data_dft[len]
//     u64 len | u8 data_qsa[len]
//     u64 len | u8 accel_ring[len]
//     u64 len | u8 accel_spec[len]
//   trailer: u64 body_size | u8 sha256[32]
// where the sha256 covers the whole file minus the 40-byte trailer.
//
// Index, magic "LCPI", version 1, one file per directory, published after every spill/removal:
//   magic[4] "LCPI" | u32 version | u32 n_entries
//   per entry:
//     u32 len | u8 file[len]              basename of the pc-*.bin object
//     u64 file_size
//     u32 n_tokens
//     u64 saved_unix_ms
//     u32 len | u8 producer_identity[len]
//     u32 len | u8 adapter_config_key[len]
//     u8 sha256[32]                       of the whole object file
//   trailer: u64 body_size | u8 sha256[32]
enum class server_cache_disk_status : uint8_t {
    ok = 0,
    dir_missing,      // the directory could not be created
    dir_unwritable,
    index_corrupt,
    io_error,
    _count,
};

const char * server_cache_disk_status_name(server_cache_disk_status status) noexcept;

struct server_cache_disk_entry {
    std::string file; // basename under the tier directory
    uint64_t    size = 0; // bytes on disk
    uint32_t    n_tokens = 0;
    llama_tokens tokens;
    std::string producer_identity;
    std::string adapter_config_key;
    uint64_t    saved_unix_ms = 0;
    std::array<uint8_t, 32> sha256 = {};
};

// Everything read back from one object: the tokens, the prompt lineage, the checkpoint ring and
// the raw state images. A read that fails the checksum leaves out unchanged.
struct server_cache_disk_image {
    llama_tokens tokens;
    uint64_t sequence_epoch = 0;
    std::list<common_prompt_checkpoint> checkpoints;
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;
};

class server_cache_disk_tier {
public:
    // Opens dir (creating it when missing), verifies it is writable, and loads the index.
    // Without an index the tier rebuilds one by scanning pc-*.bin files (hashing each object);
    // with one it trusts the sizes and verifies the object checksums at read time. Entries of a
    // producer_identity different from the one given are removed: their bytes belong to another
    // model or cache configuration. limit_bytes == 0 means unlimited.
    static std::unique_ptr<server_cache_disk_tier> open(
            const std::string & dir,
            const std::string & producer_identity,
            uint64_t limit_bytes,
            server_cache_disk_status & status,
            std::string & error);

    ~server_cache_disk_tier();

    server_cache_disk_tier(const server_cache_disk_tier &) = delete;
    server_cache_disk_tier & operator=(const server_cache_disk_tier &) = delete;

    const std::string & dir() const { return dir_; }
    uint64_t            limit_bytes() const { return limit; }

    // Parked conversations, oldest first. Tokens stay in RAM so selection never touches disk.
    const std::list<server_cache_disk_entry> & entries() const { return entries_; }
    uint64_t total_size() const { return total_size_; }

    // Exact (tokens, adapter) presence among the parked conversations.
    bool contains(const llama_tokens & tokens, const std::string & adapter_config_key) const;

    // Parks one conversation. Refuses to write a duplicate of an already parked (tokens,
    // adapter) pair; drops parked conversations that are a strict prefix of the new one.
    // Returns false when the object could not be written (the entry list is then unchanged).
    bool spill(
            const llama_tokens & tokens,
            const std::string & adapter_config_key,
            const uint8_t * main, size_t main_size,
            const uint8_t * drft, size_t drft_size,
            const std::list<common_prompt_checkpoint> & checkpoints,
            uint64_t sequence_epoch);

    // Reads one object back, verifying the size and the whole-file checksum.
    bool read(const server_cache_disk_entry & entry, server_cache_disk_image & out) const;

    // Removes one parked conversation (object and index row). Idempotent.
    void remove(const server_cache_disk_entry & entry);

    // Enforces the byte limit, dropping the oldest parked conversations first.
    void prune();

private:
    struct entry_file;
    struct index_row;

    server_cache_disk_tier(std::string dir, std::string producer_identity, uint64_t limit);

    std::string dir_;
    std::string producer_identity;
    uint64_t limit = 0;

    std::list<server_cache_disk_entry> entries_;
    uint64_t total_size_ = 0;

    void drop(const std::list<server_cache_disk_entry>::iterator it);
    void publish_index();
    bool read_index(std::string & error);
    bool build_index_from_files(std::string & error);
    std::string object_path(const std::string & name) const;
    static std::string new_object_name(uint64_t now_us);
};
