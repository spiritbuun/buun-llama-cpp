# Prepared safetensors backing

`--repack-cache /path/to/dedicated-directory` opts into retaining prepared host
weights across launches. Without it, prepared backing is disposable and removed
after its last owner exits. Original-compatible file regions continue to map the
original safetensors; the cache does not make another copy of those tensors or of
GPU-only tensors. Selection still follows ordinary placement and mmap/lazy policy.

This initial persistent implementation is Linux-only. The underlying disposable
file-backed path has its own platform support and does not require this option.

## Storage and lifecycle

- Choose a dedicated directory **outside the model source directory**, on a disk
  with room for the selected host tensors. A large MoE can retain tens of GiB.
- Each new entry logs its expected retained bytes before preparation (`-lv 4`
  shows model-loader details). The CLI announces the selected directory. An entry
  contains `weights` and `receipt.json`; the latter records layout and checksum.
- There is no automatic eviction/quota in this first implementation. Inspect usage
  with `du -sh /path/to/dedicated-directory`. Removing old entries is a user action;
  turning the flag off does not delete anything created by earlier launches.
- For complete cleanup, stop loaders/servers using the directory, then delete that
  **specific dedicated directory**. Do not remove lock files while loaders are
  running. A crash can leave a `.preparing-*` directory; it is not reused as a valid
  entry and can be removed when no loaders are using the cache.
- Original source weights remain required. Cached files are derived data, not a
  portable substitute for a model repository.

The cache uses per-entry locks and publishes complete directories by rename. A
cancelled preparation removes its staging files. Existing open POSIX mappings
survive unlinking an entry, but never edit or truncate live weight files in place.

## Reuse and validation

Keys include the source directory, file identities (device/inode/size and nanosecond
modification/change times), canonical tensor name/type/shape, size, and a versioned
repacking contract. Ordinary edits, replacement, changed configuration, and a new
layout invalidate reuse. Moving/copying a model may cause conservative misses.
This avoids hashing the entire source model again on each launch.

Completed payloads have a SHA256 receipt. Default hits validate the payload's file
identity and size. `--check-tensors` additionally reads and hashes cached payloads;
it may therefore add substantial startup I/O and CPU time. The first preparation
also pays for checksum generation. Neither mode promises a decode speedup.

This assumes immutable local source/cache files while loaded, as ordinary mmap
does. Default stat validation does not detect silent storage corruption without a
metadata change. This is not an authenticated artifact format or a defense against
someone able to rewrite both payload and receipt. Use checksum verification when
that integrity check is wanted; retain the original source so entries can be rebuilt.

## Maintainer checklist

If a change alters canonical prepared bytes without changing name/type/shape,
bump the `llama-prepared-weights-v1` domain in `llama-repack-cache.cpp`. Changes only
to execution kernels do not require invalidating model-sized backing.
New importer metadata sidecars must also join its source-identity file list.

Run `test-repack-cache` for entry lifecycle, concurrent loaders, corruption,
cancellation, and invalidation. `test-safetensors-lazy` compares cached native
BF16/EXL3 n-gram fixtures with GGUF logits and checks no-alloc probes and cache hits
do not write backing. Real-model startup and memory-pressure gates remain necessary
when changing preparation or mapping policy, rather than for unrelated edits.
