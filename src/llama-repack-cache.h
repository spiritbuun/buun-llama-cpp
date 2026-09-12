#pragma once

#include "llama-mmap.h"

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

// Local, disposable derived data, not an interchange format. Source and cache
// files must remain immutable while loaded, just like the original mmap path.
// Linux stat identity catches ordinary edits/replacements without reading every
// weight on a cache hit; --check-tensors additionally verifies cached payload SHA.
class llama_repack_cache {
public:
    llama_repack_cache(const std::filesystem::path & source, const std::filesystem::path & directory);
    void validate_source() const;
    std::unique_ptr<llama_file> get(
        const std::string & layout, size_t expected, bool check,
        const std::function<void()> & poll,
        const std::function<void(const std::function<void(const void *, size_t)> &)> & produce) const;

private:
    std::filesystem::path source_, directory_;
    std::string identity_, index_identity_;
    std::vector<std::filesystem::path> indexed_shards_;
};
