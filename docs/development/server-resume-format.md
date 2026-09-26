# Persistent server resume — format and install contract (P1)

Status: **contract, implemented through P2** (§10 items 1 to 5, with the
measured results and the limits of v1 there) **and P3** (§11, dynamic VBR). Where the build differs from the
first design the text says "as built".
Phases: P0 measurement of the existing code, P1 this contract, P2 fixed-type
KV, P3 dynamic VBR, P4 open (host prompt cache, adapter transitions). Use the
symbols, not line numbers, when lines move.

Scope of v1: fixed-type KV (f16, q8_0, turbo, TCQ), dense, hybrid-recurrent and
SWA/iSWA models, one entry per conversation, direct install into a slot. Images
and audio are stored on models where a media chunk of n cells takes n consecutive
positions (§4a). Dynamic VBR takes its own route through the same entries (§11,
P3). The v2 extension adds shared-position (M-RoPE/IMROPE/VISION) media using
whole-sequence objects and position-aware checkpoint metadata (§4a). Out of scope: drafter state
on the fixed route (never saved), adapter changes between producer and consumer
(P4).

## 1. Review of the existing code

What P0 and the P1 code reading established, and what the design takes from it.

| Code | Fact | Consequence |
|---|---|---|
| Library sequence-state file (`llama-context.cpp`, file v3) | 24-byte header with the declared total size and one FNV-1a-64 over the whole payload; `llama_state_seq_file_snapshot_prepare` reads the whole file into host memory and hashes it before anything is installed; the writer buffers the whole payload, publishes by rename, no `fsync` | Integrity exists but is all-or-nothing: no appending, no partial read, no bounded staging, no durability. A resume entry cannot live in this container |
| Per-sequence state blob (`llama_kv_cache::state_write`) | `n_stream`; per stream `cell_count`, a meta block (per cell `pos`, seq ids, optional ext), then per layer the K rows and the V rows of all written cells, then a TCQ footer when a TCQ type is present. Rows are contiguous per layer. Cells are written in cell-index order, which is not guaranteed to be position order | A token range is expressible as the same layout restricted to the cells of that range. No new blob grammar is needed, only a range filter and an ordering rule |
| Single-sequence reader (`state_read_meta`) | Clears the destination sequence first, places all `cell_count` cells through `find_slot`, all-or-nothing; validates layer count, `v_trans`, per-layer type and row size, TCQ fingerprint | Installing a second chunk needs an append mode that does not clear the sequence. The structural checks stay the single authority for "does this blob fit this cache" |
| Composite memories (hybrid, iSWA, hybrid-iSWA and relatives) | Write the base part (full-attention KV) unless `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`, then always the partial part (recurrent state, SWA cache) | The base part is position-addressable and chunkable. The partial part is a point-in-time image: it is valid only at the position where it was taken. A server context checkpoint is exactly such an image |
| Slot semantic envelope `BUUNSLOT` v2 (`server-context.cpp`) | 224-byte header: counts, next position, five SHA-256 digests (runtime identity, adapter identity, token digest, serialized tokens, logits), then the serialized `server_tokens`, then optional logits. The parser refuses any other version, header size or unknown flag bit with `format_mismatch` | Reused as the entry's token ledger at version 3. Older builds refuse a v3 envelope at the header |
| Slot runtime identity (`buun.server.slot-file-runtime-identity/v2`) | Hashes the build label, context sizes, slot layout and index next to the family digest and the KV settings; any difference reports `model_family_mismatch` | Resume needs its own key (§5) and its own reason codes (§8). The legacy identity is not changed |
| `SERVER_TASK_TYPE_SLOT_RESTORE` | Defers while the slot is processing; parses the envelope; installs state; on failure runs `mandatory_recovery_reset(restore_failure)`; on success invalidates the frontier record, runs the slot's recovery-reset bookkeeping (drops the destination's checkpoints and draft state), sets the ledger, publishes the slot to the retention observer. It installs no checkpoints | This is the establishment path the resume installer reuses for the slot. Checkpoints need a second, existing door (next row) |
| Host-cache restore (`server_prompt_cache::commit_restore_delivery`, `server_prompt_cache_mirror_restore_retention`) | Delivers a whole `server_prompt` (tokens and checkpoints) to the slot, then admits the checkpoints as a batch through the publish authority (`admit_live_checkpoints`) and attaches their release operations on the retention observer. Fail-closed: checkpoints that cannot be admitted are retired and dropped, the restore still stands | The existing door for imported checkpoints. Resume does not insert into `slot.prompt.checkpoints` by hand |
| Checkpoint validity (`checkpoint_frontier_is_current`) | A checkpoint is usable only if its computation frontier carries this process's execution identity (random per model load), the slot's current sequence epoch, the current adapter identity, and token/position counts equal to the checkpoint's | A checkpoint read from disk can never pass as saved. Import is a named provenance transition that re-stamps it (§7) |
| `SERVER_TASK_TYPE_SLOT_SAVE` | Mutates: a one-token target decode to align frontier logits, clears the draft sequence, resets the DFlash ring | Resume capture is a separate read-only path. It saves no logits |
| Shutdown (`server.cpp`) | The inference loop returns, then `clean_up()` destroys the context. HTTP threads are still alive in between. A second signal exits from the handler. The router force-kills a child after 10 s | Save hook sits between the two. Per-entry commit, most recently used entry first, so a forced kill loses the tail of the list, not the store |
| Sleep (`handle_sleeping_state`, `load_model`) | State is destroyed on sleep and the model reloaded on wake inside one process | Same save and restore calls; the first P2 integration test, no restart needed |
| Streaming handlers (`server-http.cpp`) | A handler parked in the chunked provider is released only by the SSE ping timer (30 s) or a client disconnect | P2 prerequisite: a shutdown flag in the three `should_stop` closures. Without it a supervisor's grace period is spent waiting, not saving |
| Durable I/O already in the tree | `llama-repack-cache.cpp`: `flock`, staged files, streamed hashing with `sync_write()` every 64 MiB, directory `fsync`, rename. `llama-vram-ledger.cpp`: 0700 directory with ownership check, stale-owner detection by pid and start time. `fs_get_cache_file` forbids subdirectories and `fs_create_directory_with_parents` creates 0755 | The store reuses these patterns behind its own root helper (0700 directories, 0600 files) on `fs_get_cache_directory()` |

Slot files do have a payload checksum (the
library file header above). What is missing is a checksum on the raw state API and
any integrity unit smaller than the whole file.

## 2. Container: a directory of objects, manifest published last

```text
<fs_get_cache_directory()>/resume/<semantic-family-digest>/
    writer.lock
    execution-identity       dynamic VBR only (§11)
    entries/<entry-id>/
        commit               manifest; atomic rename makes the entry visible
        c-<p0>-<p1>-<gen>    base-state chunk for token positions [p0, p1)
        t-<pos>-<gen>        tail state (recurrent / SWA part) taken at <pos>
        v-<gen>              dynamic VBR artifact of the whole sequence (§11)
        s-<gen>              complete fixed-KV sequence for shared-position media (§4a)
        tmp-*                staging; removed by the next writer
```

An entry is one conversation: one slot's sequence at one captured boundary.
`<entry-id>` is 32 lowercase hex digits drawn at random when the conversation is
first saved. `<gen>` is the manifest generation that wrote the object.

Why not one file with appended chunks, which is how the chunk idea was first
sketched: a single file needs its own extent allocator, a double commit record
with torn-write reasoning, and truncation to give space back. With one object per
file the filesystem is the allocator, `rename` is the commit, `unlink` returns
space at once, and the existing repack-cache code is the template. Chunk overhead
is unchanged (one 64-byte header and one block of rounding per ≈54 MB chunk on a
27B TCQ cache, under 0.01 %). A 200k-token conversation is about 50 chunk files.
In the store an entry is always a directory; the one single-file form is the
slot file, an exported copy of one entry (§8.1).

Why not the library sequence file with a new envelope version inside it: see the
first row of §1. What declares a resume manifest is therefore the
manifest's own magic and version. Builds that predate it cannot
open an entry at all: given a manifest by name they fail the library magic check.

### 2.1 Object file (`c-*`, `t-*`)

64-byte little-endian header, then the payload, nothing after it.

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `BUUNRSMO` |
| 8 | 4 | object format version = 1 |
| 12 | 4 | header size = 64 |
| 16 | 4 | kind: 1 = base chunk, 2 = tail state, 3 = VBR artifact, 4 = placement, 5 = whole fixed-KV sequence |
| 20 | 4 | flags = 0 (unknown bits refuse the object) |
| 24 | 8 | payload bytes |
| 32 | 8 | XXH3-64 of the payload |
| 40 | 4 | `p0` (chunk) or position (tail state) |
| 44 | 4 | `p1` (chunk) or 0 |
| 48 | 8 | manifest generation that wrote it |
| 56 | 8 | XXH3-64 of bytes 0..55 |

The file size must equal 64 + payload bytes. XXH3-64 (vendored under
`vendor/hash`) detects corruption at memory-copy speed; SHA-256 stays the hash
for identities. Neither authenticates: anyone who can write the store can forge
it.

- **Base chunk payload** — the per-sequence state blob of §1 for the base part
  only, restricted to cells with `p0 <= pos < p1`, **cells in ascending position
  order**. Every chunk carries its own TCQ footer. Chunk boundaries are multiples
  of `chunk_tokens` (4096 in v1, recorded per entry; a reader accepts any gapless
  tiling). The last chunk of an entry may be short.
- **Tail state payload** — the `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` blob of the
  sequence taken when the sequence ended at `pos`. A server context checkpoint's
  `data_tgt` is this blob already; the state at the captured boundary is the same
  kind of image taken at save time. Dense models have no tail states.

So for every model class an entry is *base chunks covering [0, N)* plus *tail
states at some positions ≤ N*, and "the saved state" and "a checkpoint" stop being
different things on disk:

| Model class | Restorable positions |
|---|---|
| Dense | any `p <= N` |
| Hybrid, SWA, hybrid-SWA | each `p` that has a tail state; `N` always has one |

### 2.2 Manifest (`commit`)

| Part | Content |
|---|---|
| Header, 64 bytes LE | magic `BUUNRSMM`, manifest format version = 1 (ordinary positions) or 2 (shared-position media), header size, flags = 0, JSON bytes, ledger bytes, generation, XXH3-64 of JSON + ledger, XXH3-64 of the header |
| JSON document | records below; UTF-8, at most 1 MiB, parsed with a depth limit |
| Ledger | `BUUNSLOT` envelope **version 3**: the v2 layout, flag `RESUME_KEY_BOUND` (bit 2) set, `RUNTIME_FAMILY_BOUND` and `HAS_LOGITS` clear, the identity field holding the resume compatibility key (§5). The v3 parser accepts v2 only on the legacy route and v3 only on the resume route |

JSON records:

- `resume_key`, `family_digest`, `adapter_identity` (hex), duplicated from the
  ledger for listing without parsing it.
- `n_tokens`, `chunk_tokens`, `saved_unix_ms`, `last_used_unix_ms`, `slot_hint`.
- `chunks[]`: `p0`, `p1`, `gen`, `bytes`, `xxh3`, `prefix_digest`, `producer`.
  `prefix_digest` is SHA-256 (domain `buun.server.resume-prefix/v1`) over the
  ledger's token ids `[0, p1)`: a chunk is bound to its whole prefix, because KV
  depends on it. The first cell of a media chunk adds the chunk's cell count and
  content id (§4a); a text-only ledger hashes the ids alone.
- `tail_states[]`: `pos`, `pos_min`, `pos_max`, `n_tokens`, `gen`, `bytes`,
  `xxh3`, `prefix_digest`, `producer`, `role` (`frontier`, `turn`, `early`).
- `producers[]`: provenance only, never a gate — model name and file basename as
  loaded, projector file basename (`mmproj_file`, optional), weight quantization, build label, KV types, adapter labels, host
  time. `producer` fields index this table; a conversation continued by a second
  model has chunks from both. A save keeps the records of the objects it carries
  over and drops the rest, so the table lists who wrote what the entry holds
  now, not everyone who ever wrote to it.
- Object names are derived from the records, never stored, so a manifest cannot
  name a path.

Bounded decode limits, checked before any allocation: manifest file ≤ 64 MiB;
≤ 4096 chunks; ≤ 64 tail states; ≤ 16 producers; `n_tokens` ≤ 2^24; chunks tile
`[0, n_tokens)` exactly; every tail state satisfies `pos <= n_tokens` and
`pos_max + 1 == pos == n_tokens(record)`; every declared size equals the file's
size; the ledger's token count equals `n_tokens` and its next position equals
`n_tokens` (a slot whose positions are not the identity, for instance after a
context shift, is not saved in v1).

## 3. Library additions (implemented in P2)

The range writer and the append reader are the only format work in the library.
Both reuse `state_write_meta` / `state_write_data` and their readers.

```c
#define LLAMA_STATE_SEQ_RANGE_VERSION 1

// size / write the base part of seq_id for positions [p0, p1), cells in
// ascending position order. 0 on failure
size_t llama_state_seq_get_size_range(ctx, seq_id, p0, p1);
size_t llama_state_seq_get_data_range(ctx, dst, size, seq_id, p0, p1);

// append a range blob: does not clear the sequence; p0 must be the sequence's
// pos_max + 1 in the base cache (0 on an empty sequence); cells at
// pos >= p_limit are read and dropped. All-or-nothing per call: on failure the
// cells of this call are removed and the earlier ones stay. 0 on failure
size_t llama_state_seq_append_data(ctx, src, size, seq_id, p0, p1, p_limit);
```

Range blob: 16 bytes `{magic "gqsr", LLAMA_STATE_SEQ_RANGE_VERSION, p0, p1}`,
then `cell_count`, the meta block and the data block of §1 for one stream.

- **No `n_stream` word.** A range belongs to one sequence, so the blob is the
  same under unified and split KV and installs under either. Tested both ways on
  a dense, a hybrid and an SWA model.
- **No `BASE_ONLY` flag and no `flags` argument**, as first sketched: the range
  calls are base-only by definition. Composite memories forward them to their
  attention / base cache. A purely recurrent memory writes an empty payload
  (blob size 16 means "no base part"); memory classes without an
  implementation throw, which the API reports as 0.
- **Ranges are strict.** The writer refuses a sequence that does not hold
  exactly the positions `p0 .. p1-1`; the reader requires `cell_count == p1 - p0`,
  `pos[i] == p0 + i` and the blob's `p0`/`p1` to equal the caller's. That bounds
  `cell_count` before the batch reservation it feeds.
- Ascending position order is what lets `p_limit` cut a chunk by row count: the
  reader places the first `p_limit - p0` cells and skips the remaining rows of
  every layer block.
- The reader does not check for trailing bytes; the store's exact file size and
  checksum cover that.
- A blob layout change bumps `LLAMA_STATE_SEQ_RANGE_VERSION`, which is part of
  the resume key. With the build label out of the key this constant and the
  reader's structural checks are what stop a stale layout.
- Dynamic VBR keeps refusing (the writer shares `state_write`'s settle/refuse
  step); P3.
- `llama_memory_seq_pos_max` on a composite memory is the minimum over both
  parts, so it stays -1 until the partial part is installed. Code that tracks an
  install in progress counts chunks; it does not ask the memory.

Two findings that shape the server side:

- `PARTIAL_ONLY` is ignored by a plain KV cache: on a dense model it writes the
  full state. The server decides "has a partial part" itself:
  `llama_model_is_hybrid || llama_model_is_recurrent || llama_model_n_swa > 0`.
- The SWA partial blob is an ordinary KV-cache blob and carries `n_stream`. It
  does **not** install across stream layouts. On a model with SWA the stream
  count is therefore part of the resume key (§5). Recurrent partial blobs have
  no such word, so dense and hybrid entries move between slot counts.

One flag was added for the `frontier` tail state of an SWA model:

```c
// write every cell the sequence still holds in an SWA cache, not only the cells
// inside the attention window
#define LLAMA_STATE_SEQ_FLAGS_SWA_HELD_CELLS 4
```

The default SWA image holds exactly the window. The server's reuse check is more
conservative than the window (`pos_min` must lie below
`pos_next - n_swa`, minus one when the request brings no new token), so a restored
exact-window image fails it on the first request and falls back to a checkpoint,
reprocessing the turn since. The live cache passes the check because it still
holds older cells. With the flag the frontier image carries those cells and the
restored slot behaves as the live one did: measured on a 9k conversation, the
first request after a restart reuses all 9047 tokens. The flag is additive and
off by default; context checkpoints and the legacy slot files keep the
exact-window image.

## 4. Capture (read-only)

Runs on the inference thread after the loop has returned (shutdown) or before
`destroy()` (sleep). Skipped when the server is already sleeping. Never from a
signal handler or a destructor, and never after a failed startup.

Per nonempty slot, most recently used first:

1. `llama_synchronize`. Read the ledger (`slot.prompt.tokens`) as it is. A slot
   that was generating is saved "one behind": the sampled-but-undecoded token is
   simply not part of the entry. No decode, no draft-sequence change, no ring
   reset, no logits.
2. Skip with a reason: media without a content id (§4a), a Qwen4 QSA
   index (the index image is outside the partial state; unverified), positions
   that are not the identity, or a memory whose `pos_max + 1` differs from the
   ledger length on a model with a partial part (`frontier_inconsistent`). On a
   dense model extra KV positions are clipped by the range writer.
3. Decide reuse: the slot remembers the entry id it was restored from or last
   saved as. A chunk of that entry is kept when its `prefix_digest`, the resume
   key and the adapter identity still match **and the live cells of its range
   serialize to the object's size and checksum**. Everything after the first
   mismatch is rewritten. Equal tokens do not prove an equal state: a request
   with `cache_prompt: false`, a legacy slot restore or another conversation in
   the slot recompute the cells, and after a restore from another model of the
   family the recomputed cells are another model's. Reading the ranges back
   costs 18 ms on a 9k-token hybrid 4B save (two 71 MB chunks and the short
   last one; 135 ms against 117 ms). A tail state that the slot still holds
   (the frontier, a checkpoint) is compared the same way. An `early` tail is
   selected only from validated live checkpoints. Once it leaves the live ring,
   its disk-only copy is not retained: matching base chunks do not authenticate
   recurrent/SWA state, and pure recurrent models have no base payload at all.
   Measured: base model
   saves, a fine-tune restores and refills cold, its save holds the fine-tune's
   chunks and tails (the bytes of a fresh fine-tune save); the same restore
   followed by an ordinary turn, and a restart after it, keep the base model's
   chunks. An inherited early checkpoint is reusable while still in the live ring.

   Which entry is a different question from which bytes. The entry of a
   conversation is the one whose chunks, all but the last, lead the ledger: the
   slot's own, else one no slot holds. That is how a conversation that came
   back from the host cache, or was sent again after a start that had no slot
   for it, goes on in its own entry. The last chunk is excused because a client
   re-renders the last turn; the price is that a new conversation sharing all
   but the last chunk of an entry takes it over, which costs the other
   conversation at most one chunk and its tail. An entry of a single chunk
   therefore goes with its slot. A shared first chunk is not enough: a second
   agent with the same long system prompt gets its own entry and the first
   keeps its history (measured: 5000 shared tokens, 12k and 6.5k conversations
   through one slot of two; both restored, the first with its whole history).
   One exception saves writes without costing anything: when the slot's
   previous entry would not outlive this save under the retention rule below
   (one slot, no host cache), the new conversation takes that entry and keeps
   whatever leading chunks are still the live state.
4. Stream new objects: range blob into a staging buffer of one chunk (≈54 MB for
   a 27B TCQ cache, ≈210 MB at f16), XXH3 while writing, `sync_write` every
   64 MiB, `fsync`, rename from `tmp-*`.
5. Tail states per §6, copied from the slot's checkpoints (`data_tgt` only) and
   one `PARTIAL_ONLY` read of the live sequence for the frontier.
6. Write the manifest to `tmp-*`, `fsync`, rename over `commit`, `fsync` the
   directory. The entry is now the new generation.
7. Unlink objects the new manifest does not reference.

Disk bound. What a save makes obsolete is decided before its bytes are written
and goes first, free space or not. A save that replaces more than the last
chunk of its entry (an edited history, a state that is no longer the stored
bytes) removes `commit` first (`fsync`), unlinks the objects it does not keep,
then writes. A save that needs a new entry applies the retention rule below
first, counting the new entry. An interruption then loses the affected entry,
which is the agreed trade; entries that are neither replaced nor due under
retention are never touched. Only the append keeps the old commit until the new
one is published, and it duplicates at most one chunk and the changed tail
states; the preflight takes that away too when it would not fit. Measured as
the peak of allocated bytes, sampled every 2 ms through the save, 13k-token
dense entry of 794 MB: history edited one chunk in, peak 794 MB (was 1308);
another conversation in the only slot, peak 794 MB (was 1527, both snapshots).

Retention, with no knob: the namespace keeps at most `n_parallel` entries of
this server's resume key, the ones slots hold first, then newest `last_used`,
and at most `max(8, 4 × n_parallel)` entries overall; the rest are unlinked,
before a new entry is written and after the live slots are committed. Entries of another key (a different KV type, another stream count on
an SWA model) cannot be read by this server and say nothing about its
conversations, so only the overall bound ends them: a run under other settings
does not delete what the usual settings saved. The same holds for entries of
this key whose conversation is known to be without a slot: the ones that found
no slot at install (§7), and, when there is a host cache, the one a slot held
before another conversation took the slot. They count against the overall bound
only, so a start with fewer slots does not end the conversations it had no slot
for. An entry is not deleted merely because its
conversation is not in a slot at save time. With unified KV and several slots
this is what keeps the conversations that were evicted to the host cache during
the run: their entries from the previous start survive, stale by the turns made
since, until `--resume-host-cache` (P4) saves them properly.

One writer per family namespace: `flock` on `writer.lock`, held for the life of
the process. As built there is no stale-owner detection by pid and start time:
the kernel drops the lock with its owner, so a killed server leaves nothing to
detect. The pid written into the file is for a person looking at the directory.
A second server runs without persistence and says so. Retention also removes an
entry whose manifest is damaged; one of an unsupported version is kept, it may
belong to a newer build. On Windows the store opens as `store_unwritable`
(encoding and decoding are portable, the durable I/O is POSIX).

### 4a. Media

The ledger already carries media: `server_tokens::serialize()` writes each image
or audio chunk as a placeholder (cell count, position type, content id, no
pixels or embeddings), a few hundred bytes per chunk. The content id is the
SHA-256 of the file the client sent. The KV cells of a chunk are ordinary cells
and go into the token ranges like any other.

- **Positions.** A token range names cells by position and requires
  `pos[i] == p0 + i`. That holds where a chunk of n cells takes n consecutive
  positions (Gemma 3/4, SmolVLM, LLaVA-style). Under M-RoPE the cells of a chunk
  share temporal positions and carry spatial coordinates. Such conversations use
  manifest v2 (`shared_positions: true`); text-only conversations still use v1.
  VBR preserves the coordinates already present in its artifact placements and
  requires uniqueness of the complete `(position, x, y)` tuple per sequence.
  Fixed KV uses a kind-5 `sequence` object: the existing full sequence serializer,
  including attention coordinates and recurrent state. This object is installed
  whole or refused when the destination lacks cell capacity; it cannot be
  truncated to a smaller context. Saves rewrite the whole object rather than
  retaining incremental chunks and currently stage one full object in host RAM
  (bounded by the existing 8 GiB object limit).
  Publication retains the previous committed whole object until the new
  generation is durable, so replacement temporarily needs space for both.
  Older builds refuse v2 at the manifest version check, without misinterpreting
  media positions as token counts or deleting an unsupported entry.
- **Identity.** Media cells are `LLAMA_TOKEN_NULL` in the ledger, so ids alone
  would make two images equal. The prefix digest adds, at the first cell of a
  chunk, the chunk's cell count and its content id. A chunk without an id is
  `unsupported_media`.
- **Dynamic VBR.** The exact artifact route supports both consecutive- and shared-position
  media. Restore checks the artifact's media identity, cell count and next position
  against the integrity-checked ledger, then publishes that ledger with its media
  placeholders intact. It does not reconstruct images from null token ids or run
  the projector again for the reused prefix. Automatic host capture and restore
  also support complete media prefixes and sealed checkpoint frontiers. Lookup
  checks each complete-media identity scope, so adding another image does not
  hide a saved earlier prefix. Arbitrary truncation of a larger host artifact is
  still text-only: the media path never cuts through an image or invents a
  recurrent checkpoint. Occupied-slot replacement retains its lease, ownership
  and rollback checks, with recovery identity including media content ids.
  This is in-process host caching. The shutdown pass that persists conversations
  held **only** in host memory still skips media (§11); media in live slots is
  persisted. A displaced image conversation must be live again before shutdown
  to be included in the current resume store.
- **Checkpoints.** A shared-position checkpoint records its logical cell count
  separately from its temporal `pos_min`/`pos_max`. Restore verifies the latter
  against the media ledger before importing it through the normal checkpoint
  authority. This is necessary for Qwen chat-template rewinds on the first turn
  after restart; saving only the terminal image would otherwise cold-prefill.
- **Boundaries.** A chunk boundary may fall inside a media chunk: objects are
  data, and the digest of a boundary inside an image already covers the image. A
  restore never stops inside one. A checkpoint whose frontier lies inside a
  media chunk is not stored as a tail state, a tail state there is no restore
  candidate, and on a dense model the cap of a smaller context, or the last
  complete chunk after a failure, moves down to the first cell of the chunk.
- **Projector.** The key binds whether a projector is loaded, not which one. A
  follow-up request that sends the same file reuses the stored cells whichever
  projector encoded them, the same kind of handoff as between two models of one
  family (§5). `mmproj_file` in the producer record says which it was.
  Measured for one model with its projector at two precisions (SmolVLM2-500M,
  Q8_0 saved, f16 resumed, q8_0 KV): the two later turns, one of them a
  text-only question about the stored image, are token-identical to the f16
  projector reading the same history cold. Two differently trained projectors
  for one model were not available to measure; that case is approximate reuse
  by the same decision as a fine-tune, not a measured one.

Measured on the 3090: a three-turn conversation with two
images, one of them across the first chunk boundary, over two restarts against
the same conversation in one process. SmolVLM2-500M (dense, q8_0 KV) and Gemma 4
E2B (SWA, turbo3_tcq KV, below and above the window): all turns token-identical,
`cache_n` equal to the one-process run after each restart, so no image was
encoded again. A 4096-token context that ends inside the first image restores
the 4033 tokens before it. Two media conversations saved under `-np 2` and
restarted under `-np 1` with a host cache, on SmolVLM2: one `installed_host`,
both continue identically. On Gemma 4 the stream count is part of the resume
key (§5), so that restart installs nothing (`resume_key_mismatch`) and both
conversations prefill cold: identical to the one-process run under f16 KV, one
of the two second turns in other words under TCQ 3-bit. Before the v2 extension,
Qwen3.5-4B with a projector had media slots skipped with
`unsupported_positions`, a text-only conversation identical across restarts.
Before this change `--resume` with a projector loaded hit an assert at the first
save, with or without media in the slot.

The v2 Qwen image regression on the 3090 covers Qwen3.5-4B and Qwen3.8-27B
with fixed Turbo3 and dynamic VBR, including two successive restarts and
recurrent checkpoint rewinds. All three replies match uninterrupted controls,
and restored turns reuse the prefixes rather than silently replaying them.
The 27B also passes with native MTP and through the slot-file route. A ~4k
text-plus-image run forces F16-to-T8 transitions and reuses 4344/4514 cells
after the two restarts. Video and Qwen audio are not covered by this test.
At 140 MiB the same run reaches 62 degradation steps, including mixed lower
TCQ tiers, and still restores both prefixes with identical replies. A fixed
Turbo3 + MTP run also passes; restarting it with a 512-cell context explicitly
refuses `context_too_small` without installing a partial media image.

## 5. Resume compatibility key

SHA-256, domain `buun.server.resume-compat/v1`, over:

| In the key | Why |
|---|---|
| Semantic family digest (v4) | structure and tokenizer; MTP head and pad token already unbound |
| Manifest, object and `LLAMA_STATE_SEQ_RANGE_VERSION` numbers, byte order | portability versions instead of the build label |
| `cache_type_k`, `cache_type_v` | the bytes are in that codec |
| RoPE / YaRN parameters, `n_ctx_orig_yarn`, group-attention `n`/`w` | K is stored rotated |
| Control vectors | state-affecting, like adapters |
| `swa_full` | changes which SWA cells exist |
| mmproj presence | a ledger with media needs a projector to be read; which projector is provenance, not a gate (§4a) |
| Stream count, on a model with SWA only (1 under unified KV, else `n_parallel`; 0 without SWA) | the SWA partial blob records it (§3). Base chunks and recurrent partial blobs do not, so a dense or hybrid conversation saved under `-np 2` continues under `-np 1`, measured bit-exact |

| Left out | Becomes |
|---|---|
| Build label (`llama_commit()`) | producer provenance |
| `n_ctx`, `n_ctx_seq`, slot count, slot index | install-time fit (§8) |
| Flash-attention type, `v_trans`, layer count, per-layer type and row size, TCQ codebook fingerprint | the state reader's structural checks |
| `no_fused_gdn`, `logits_all` | provenance / irrelevant |
| Weight values, quantization, file name | provenance; fine-tunes stay eligible by decision |

The adapter identity stays a separate digest and must be equal in v1. The
producer-to-consumer adapter transition is left to a later phase.

Documented property of family reuse (measured before implementation): the restored
history is a mixed-model history. The typical next token follows the consumer
model (median KLD to the consumer's own prefill 3–6× below the distance between
the two models under f16 KV, about 2× under TCQ, top-1 agreement 0.93–0.98);
strongly history-determined predictions, mostly in the first ≈50 tokens, can
follow the producer. This is kept, not gated. The manifest records the producer
so that logs never call such a restore an exact-model hit.

## 6. Checkpoint set and budget

Tail states saved per entry on a model with a partial part:

1. `frontier` at `N` — mandatory; it *is* the state. Without it only a dense
   model could be restored.
2. `turn` — mandatory when the slot holds one: the slot's newest context
   checkpoint. P0: the first request after a restart re-renders the last reply
   and rewinds behind `N` even with thinking off; without this companion a
   hybrid reprocesses the whole history and an SWA model has no valid rewind.
3. `early` — at most one: the slot's oldest validated live checkpoint, when
   distinct from `turn`. It can support a partial-prefix restore into a smaller
   context. A subsequent save drops it once it leaves the live ring; preserving
   disk-only early history would require a separate proof of state lineage.

Budget: tail states 2 and 3 only. Each costs the model's fixed partial size
(52.7 MB on a 4B hybrid, 156.9 MB on a 27B; 6 MiB on the measured SWA model),
install 4–12 ms. More than three is opt-in and not part of v1. Pinning a
checkpoint at the preamble boundary in the live ring would make `early`
reliable; that belongs to the checkpoint code and is only noted here.

Partial SWA images are valid only on the base chunks of the same entry; they are
never shared between entries.

## 7. Install and the checkpoint import transition

Runs at the tail of `load_model`, after the slots exist and before readiness
(startup and wake). Failure of any entry degrades that conversation to a cold
start; it never fails the load and never leaves half a slot.

1. Take the namespace lock; enumerate `entries/*/commit` (bounded); parse
   header, JSON and ledger; check the resume key and adapter identity. Sort by
   `last_used`.
2. Map entries to slots: `slot_hint` if free, otherwise any free slot. More
   entries of this key than slots: with a host prompt cache (`--cache-ram`,
   fixed-type KV) the oldest go there first, one at a time through a slot that
   is still empty: installed as below, saved by the call that saves an idle
   slot, cleared (`installed_host`, oldest first so that the cache evicts it
   first). The next request of such a conversation finds it by the ordinary
   prefix match. The cache applies its own budget; a state it does not take is
   `host_cache_rejected`. Without a host cache they are `no_free_slot`. In all
   three cases the entry stays in the store (§4).
3. Choose the install position `p`: the largest restorable position (§2.1) with
   `p < n_ctx_seq` of the destination. `p == N` is a full install, `0 < p < N` a
   prefix install, none is `context_too_small`.
4. For each chunk with `p0 < p`: verify header, size and XXH3 while reading into
   the one-chunk staging buffer, then `llama_state_seq_append_data(...,
   p_limit = p)`. Then the tail state at `p`, if the model has a partial part,
   through `llama_state_seq_set_data_ext(PARTIAL_ONLY)`.
   If the cache runs out of cells part-way (unified KV shares them between
   slots), or an object fails its check: a dense model keeps the complete chunks
   it has; a model with a partial part clears the sequence and retries at the
   largest tail-state position inside what did fit, and so on down the tail
   states. The outcome names the first failure as its `why`. No rewind of
   installed state is involved either way.
5. Establish the slot exactly as `SLOT_RESTORE` does after a successful state
   install (frontier-record invalidation, recovery-reset bookkeeping, ledger
   truncated to `p`, fresh sequence epoch, retention publication), with no
   frontier logits. The next request decodes at least one token and gets its own.
   Speculative state: event `target_restored_without_draft`; drafters rebind as
   in the `--mmproj-gpu-swap` path.
6. Select the newest tail states with `pos < p` within the configured checkpoint
   budget, then import them chronologically as context checkpoints — transition
   `resume_import`. This retains the recent turn before optional early history:
   - new `common_prompt_checkpoint` with `n_tokens`, `pos_min`, `pos_max` and
     `data_tgt` from the record; `data_dft`, `data_qsa`, `accel` empty; VBR
     epochs 0;
   - computation frontier filled exactly as the checkpoint creation site fills it
     (inline there today; P2 factors that into one helper used by both), so it
     carries this process's execution identity, the slot's new sequence epoch,
     the current adapter identity, `token_count = n_tokens` and
     `next_position = pos_max + 1`;
   - cache-family binding of the slot with the producer recorded, never relabeled
     as computed here;
   - admitted through the creation/host-restore door (retention publish,
     `admit_live_checkpoints`, release operations). Fail-closed: a checkpoint
     that is not admitted is dropped and reported; the install stands.
7. Any failure after step 4 began: `mandatory_recovery_reset(restore_failure)`
   on the slot, reason logged, next entry.

Rewinds of restored state happen only through the server's `pos_min` guard or a
restored checkpoint. Resume code never calls `llama_memory_seq_rm` on SWA or
recurrent state (P0: it succeeds and the output is silently wrong). It does not
need to: `p_limit` cuts the last chunk while it is read.

Open P2 check: a checkpoint restore with empty `data_dft` while a drafter is
active must take the target-only speculative transition.

## 8. Outcomes and reason codes

One log line and one `/slots` field per entry. The string
`model_family_mismatch` is never produced by resume.

| Outcome | Meaning |
|---|---|
| `installed_full` | `p == N` |
| `installed_prefix` | `0 < p < N`; reports `p`, `N` and why (`context_smaller`, `cells_exhausted`) |
| `installed_host` | no slot for the entry: restored through an empty slot into the host prompt cache; `restored` says full or prefix |
| `skipped` | destination untouched |
| `failed` | destination reset to empty |

| Reason | When |
|---|---|
| `resume_key_mismatch` | key differs (KV type, RoPE, versions …); the differing field is not recoverable from a hash, so the log prints the producer's provenance next to the current settings |
| `adapter_mismatch` | adapter identity differs |
| `format_unsupported` | unknown manifest/object/envelope version or flag |
| `manifest_corrupt`, `ledger_invalid` | checksum, bounds or token validation |
| `object_missing`, `object_size_mismatch`, `object_checksum_mismatch` | per object; a failed chunk ends the usable prefix at its `p0` if a restorable position remains below it, else `failed` |
| `companion_missing` | a model with a partial part and no tail state at or below the fit position |
| `context_too_small` | no restorable position fits |
| `no_free_slot` | more entries than slots and no host prompt cache; the entry is kept |
| `host_cache_rejected` | more entries than slots and the host prompt cache did not take the state (its size limit); the entry is kept |
| `host_restore_refused` | save side, dynamic VBR: the VBR host cache's restore did not bring a hosted conversation into the staging slot (§11); an entry it had from an earlier pass is kept |
| `entry_in_use` | the restore action named an entry another slot was restored from or saved as; two slots never write one entry |
| `state_rejected` | the library refused a blob (type, shape, TCQ fingerprint) |
| `state_too_large` | a whole fixed-KV media sequence exceeds the bounded object size; capture is skipped |
| `unsupported_media` | a media chunk without a content id (§4a); capture-side skip, logged at save |
| `unsupported_positions` | invalid position geometry (shared-position media uses the v2 route, §4a) |
| `unsupported_qsa`, `frontier_inconsistent` | capture-side skips, logged at save |
| `unsupported_artifact`, `capture_refused`, `slot_busy`, `cache_shared`, `tier_mismatch`, `slot_not_empty`, `precision_refused`, `execution_identity_unavailable` | dynamic VBR route (§11). On a slot-file restore beside other slots `cache_shared` and `tier_mismatch` are a 400 (§8.1) |
| `store_locked`, `store_unwritable`, `no_space`, `io_error` | store level; the server runs without persistence |
| `checkpoints_dropped=<n>` | warning attached to an `installed_*` outcome |
| `provenance_limit` | capture skipped before disk mutation because the objects this save carries over already come from 16 producers and the current producer is new; the previous entry is retained rather than misattributing new bytes |

**No silent downgrade.** An entry that fails a check is skipped or failed with
its reason. It is never installed by the legacy slot-file route, and the legacy
route never reads a manifest. Empty slots are skipped without a line.

`/slots/<id>?action=restore` routes by what it is given: a `resume_entry` id is
resolved inside the resume namespace and takes steps 3–7 above; a `filename`
under `--slot-save-path` is sniffed, and a slot file in the resume format takes
the same installer (§8.1), while a legacy library file takes the unchanged
legacy route on fixed-type caches and is refused with 501 under dynamic VBR.
The `resume_entry` form is the restart-free test entry for the installer.

As built: `POST /slots/<id>?action=restore` with `{"resume_entry": "<32 hex>"}`
is accepted whenever the server runs with `--resume`, with or without
`--slot-save-path`. A body carrying both `resume_entry` and `filename`, or an id
that is not 32 hex digits, is a 400. The manifest is read before the slot is
cleared. The response is the usual restore result plus a `resume` object
holding the outcome of this section; a missing entry is `skipped` /
`object_missing` with the destination unchanged. Paths of the
host appear in the server log only, never in the response.

Explicit slot `erase` ends the conversation on disk as well. The entry removed
is the one of the conversation in the slot by the rule of §4 step 3, found at
the time of the erase, not the id the slot saved into last: after another
conversation took the slot that id still names the previous one. Measured:
save, restart, erase, restart installs nothing; A saved and restored, B takes
the slot and is erased unsaved, A's entry survives and installs at the next
start; with both saved, erasing B's slot leaves A's entry alone. Within the
last-chunk tradeoff an entry of a single chunk goes with its slot here too.

### 8.1 Slot files (`--slot-save-path`)

`POST /slots/<id>?action=save {"filename"}` writes one file in the resume
format: exactly one exported entry (`server_resume_store::export_entry`). The
slot is captured as §4 describes into a staging store rooted at
`<slot_save_path>.staging`, which is not durable (its objects are not
`fsync`ed); the entry is taken out of the staging store (`take_entry`, a
rename into `<store>/taken/`) and exported to the file off the main loop
(`export_taken`), so the other slots go on decoding meanwhile. A dynamic VBR
cache keeps only its capture on the main loop: the export job writes the
artifact and the ring's checkpoints as an entry of its own store
(`<slot_save_path>.staging/exports`), commits and exports it, and removes it.
Exports run one at a time in the order of their requests, and the save answers
when its file is written. The file is written to a staging file, `fsync`ed,
renamed and the directory synced, so it is replaced whole or not at all. No host
cache is included; that is what `--resume` is for.

File layout, little-endian:

| Offset | Size | Part |
|---|---|---|
| 0 | 64 | entry header: magic `BUUNRSME`, u32 version = 1, u32 header size = 64, u32 object count, u32 flags = 0, u64 manifest bytes, u64 file bytes, 16 zero bytes, u64 XXH3-64 of bytes 0..55 |
| 64 | manifest bytes | the manifest exactly as `commit` holds it (§2.2) |
| … | 64 + payload, per object | each object the manifest references, header and payload, exactly as its file in the entry holds it (§2.1), in manifest order |

The file size must equal the header's file bytes and the sum the manifest
implies. `restore {"filename"}` sniffs the magic (`is_entry_file`): a resume
entry file is imported into the staging store (`import_entry`: header seal,
version and flags, sizes against the header and the manifest, each object's
header against its record, the payload checked as it is read) under a fresh
entry id, installed by §7 with checksums, required companions and context
checkpoints, and dropped again. The import copies nothing: the entry reads the
objects at their offsets in the file, through the descriptor opened at import,
so a file replaced meanwhile does not change what is installed. The export
copies each object with `copy_file_range` where the kernel has it. A file
restores into any slot index and after a restart; the resume key and adapter identity decide as for any entry.
A file without the magic is a legacy library file (§8).

Under dynamic VBR the entry is one artifact (§11). Into an otherwise empty
cache it is imported whole. Beside other slots it is inserted: the other
slots' cells stay where they are, the file's rows take free cells, and a
recurrent state takes a free state cell. The insertion is refused with a 400,
leaving the other slots untouched, when the other slots hold the cache at
tiers other than the file's (`tier_mismatch`; restore it while they are
empty), when the cache has sliding-window or indexed attention
(`cache_shared`), and when the free cells do not hold the file
(`context_too_small`). A placed entry is
not whole without its pool and is refused by export and import; a slot-file
save therefore captures a whole, compact (packed rows) artifact.

Measured on an RTX 3090, turbo3_tcq KV, save / restore in ms, resume file
against legacy file: dense 220/103 against 246/225; hybrid 216/119 against
135/124 (the resume file carries two checkpoints, 218 MB against 114 MB, and
the restore is warm); SWA 38/9 against 21/19. Dynamic VBR, dense: 2.47 s save,
3.79 s restore for 1.06 GB.

## 9. Install route: recommendation and alternative

Recommended for v1, and what §7 describes: **direct slot install**. It reuses an
establishment path that P0 exercised across restart and sleep/wake, streams
chunks straight into the cache with one chunk of staging, and supports prefix
installs.

Alternative kept open: install entries as **host prompt-cache entries** and let
ordinary prefix matching pull them into whichever slot a request lands on. It
gets slot mapping, fewer-slots handling and checkpoint delivery from existing
code, and pinned requests now see host entries. It needs `--cache-ram`, holds
every entry in host memory, and loses chunk-level reads unless host entries
learn chunk lists. It is the natural route for `--resume-host-cache` in P4, where
the saved roots are host entries anyway.

As built, the two meet for the entries that get no slot (§7 step 2): a host
entry is a whole sequence image, a resume entry is range blobs and tail states,
so the entry is installed into an empty slot and saved from there by the
existing call. No second reader, and the host cache keeps its own format, budget
and eviction. Measured on the 3090: 5 ms (dense 0.6B, 647 tokens) and 54 ms
(hybrid 4B) per entry at start; the conversation's next turn reuses all of its
tokens from the host cache and is token-identical to the one-process run.

Open question for the maintainer: with `--kv-unified` and several slots, each
launch moves the other idle slots to the host cache, so at shutdown usually one
conversation is live. The retention rule of §4 keeps the others' previous
entries, but they are stale by the turns made since the last start. Exact
coverage for that configuration needs P4, or an earlier decision to save host
entries that were live slots.

## 10. What P2 builds from this

1. **Done.** Shutdown flag in the `should_stop` closures. SIGTERM mid-prefill on
   the 27B at 32k exits in 0.7–2.8 s (was 30 s), mid-decode in 0.3 s. Residual: a
   stream attached to a conversation pipe (`X-Conversation-Id`) polls only the
   pipe's cancel flag and still waits for its ping.
2. **Done.** Library: range writer, append reader, version constant; test 11 of
   `test-save-load-state` (three blobs plus the partial state equal the whole
   state byte for byte, cross-layout install, `p_limit`, refusals).
3. **Done.** Store (`tools/server/server-resume-store`): 0700/0600 root helper,
   lock, object and manifest I/O, bounded parsing (depth, counts, integer
   ranges), fault seams (write stopped at half a payload, `ENOSPC` before an
   object is published, kill between objects and manifest).
   `test-server-resume-store`: after each stopped write a reopened store lists
   the previous generation and every object of it verifies.
4. **Done.** Capture and install in the server, `--resume`, `--resume-path`
   (default: the llama.cpp cache directory), the `resume_entry` restore action,
   reason codes, the held-cells flag of §3. Self-tests of the v3 envelope in
   `test-server-prompt-cache`: ledger round trip, logits refused, the legacy and
   the resume route refuse each other's envelopes, a changed key is
   `resume_key_mismatch`.
5. **Done**, single RTX 3090, TCQ 3-bit KV, 9k-token conversation, greedy
   continuations compared token for token with the same conversation in one
   process:

   | | dense 0.6B | hybrid 4B | SWA (Gemma E2B) | hybrid 27B + MTP |
   |---|---|---|---|---|
   | entry size | 211 MB | 218 MB, 3 tail states | 16.5 MB | 582 MB |
   | save at shutdown | 0.19 s | 0.20 s | 0.09 s | 0.43 s |
   | install at startup | 54 ms | 70 ms | 4.6 ms | 184 ms (prefill: 9.6 s) |
   | restart, sleep/wake, restore action | identical | identical | see below | identical |
   | rewind | identical | identical | identical | |
   | smaller context | prefix at 4607, continuation identical | `context_too_small` (no tail state fits) | | |
   | `-np 2` → `-np 1` | identical | identical | `resume_key_mismatch` by design | |
   | second save after one more turn | 28 MB | 113 MB | | 319 MB |

   A save killed part-way (SIGKILL 0.10–0.15 s after SIGTERM) leaves no entry on
   a first save and the previous generation on a later one; the next start
   installs that generation, continues identically and sweeps the leftovers.

   With MTP the restored turn is text-identical; draft acceptance of that turn
   is 0.89 against 0.96 in one process (code prompt, thinking off), because no
   drafter state is saved and the draft context refills as the turn runs.

Properties and limits of v1, as measured:

- **The measured SWA restored state is row-exact, not
  logit-exact.** `tests/test-state-restore-swa-exact.cpp` (Gemma-4 E2B, window
  512, f16, q8_0 and turbo3_tcq KV) reads K/V rows back by position from both caches.
  After a whole-sequence restore and after the resume route (range append plus
  the held-cells tail), across a wrapped ring of 2185 tokens, each saved row
  equals the live row bit for bit. The ranged route includes every older held
  row, not just the shared attended window. Required restore arms must succeed,
  and nonfinite logits fail the test. Two live runs give bit-equal logits, so nothing in
  the comparison is run-to-run noise. The logits of the same eight next tokens
  still differ from live (f16 max KLD 7e-9, top-1 8/8), and the control shows why:
  below the window, where a restore is otherwise logit-exact, the same exact
  rows placed 200 cells further along differ by as much (max KLD 2e-6, top-1
  8/8). A restore compacts the sequence to the front of the cache while the
  live ring has wrapped, so the rows sit at other cell indices and the
  attention kernels reduce over them in another order and padding. In the
  server, the restart, sleep/wake and rewind continuations of the 9k-token SWA
  conversation are token-identical with f16 KV; with TCQ 3-bit KV the turns
  after a restart or a wake diverge in text (rewind and the first turn do
  not). Dense and hybrid continuations were identical in the measured cases,
  not proof for every restore path.
- A prefix install on a model with a partial part lands only on a tail-state
  position, so it needs an `early` or `turn` state inside the smaller context.
- Every save of a hybrid conversation rewrites the frontier and turn states
  (two partial images), whatever the number of new tokens.
- A restart with fewer slots keeps the entries it had no slot for (§4) and, with
  `--cache-ram`, serves them from the host cache (§7). What is saved of such a
  conversation is still only what a slot held at a save: turns made while it
  lived in the host cache alone reach the store when it is next in a slot at a
  save (P4 closes that).
- Conversations that live only in the host prompt cache at shutdown are not
  saved (P4).
- Slots with a QSA index are skipped with a reason.
  Dynamic VBR supports consecutive- and shared-position media (§11). Not measured for media:
  a chunk without a content id, and a turn made after a prefix restore that ended
  at an image.

## 11. Dynamic VBR route (P3, as built)

Under `-ctk vbr` the library sequence state refuses (§3), and the bytes of a
cache whose tiers move cannot be cut into token ranges. The entry therefore
carries one object instead of chunks and tail states: the VBR artifact that the
VBR host prompt cache already captures, validates and imports. Resume owns the
file and the entry; the VBR artifact library owns every byte inside it and every
decision about it. One envelope family, two install routes, no downgrade from
one to the other: an artifact entry is never read by the fixed route, and a
chunked entry is never offered to the import.

**Entry.** The manifest holds `artifact` (an object record of kind 3: `p1 =
n_tokens`, `gen`, `bytes`, `xxh3`, `prefix_digest`, `producer`) and
`sequence_epoch`, with `chunks[]` and `tail_states[]` empty; the validator
accepts either shape, never both, and an artifact only together with a non-zero
epoch. The object `v-<gen>` has the 64-byte object header of §2.1 followed by
the artifact's own self-verifying wire format, streamed in both directions: it
is never held in one buffer on the way to or from disk (limit 256 GiB, checked
against the declared size, not allocated from it).

**Key.** The resume key of §5 gains the build label (`llama_commit()`) when
dynamic VBR is active: the artifact format and the degrade tables are not
versioned apart from the build, so v1 is same-build only. A different build
lists nothing to install and leaves the entries alone.

**Execution identity and epoch.** An artifact is bound to the execution
identity and the sequence epoch of its capture. Both outlive the process with
the store: the first server of a namespace writes its identity to
`execution-identity` (first writer wins, later servers adopt it), and the epoch
counter starts above every epoch an entry still holds, so no epoch is handed
out twice. A namespace whose identity cannot be read or written runs without
persistence (`execution_identity_unavailable`).

**Capture.** Exact capture of the idle slot through the VBR artifact library's
prepare / transfer / publish, with no tenant and no host-cache admission, then
the library's export into the object stream. A slot that is not idle is `slot_busy`;
a capture the library refuses is `capture_refused` with its status and phase.
A save whose tokens, length and epoch equal the entry's keeps the artifact
(`artifact_kept`, no bytes written, 28 ms): under one epoch the same tokens are
the state the artifact was taken from. Otherwise the previous entry is
uncommitted and removed before the new artifact is written, so the disk never
holds two. A crash inside that window loses the conversation on disk; that is
the chosen side of the tradeoff (loss of the affected cache over twice the
disk). A rewind in a slot re-saves into the slot's entry.

**Install.** Into an empty slot of an empty cache: the clear, the idle boundary
(`breathe`, which releases the watermark a warmup or a cleared prompt leaves),
the library's ingest from the object stream and its import with the same
publish pair the host cache uses. It installs whole or the slot starts cold;
the outcome carries the library's decision (`native_import`, `live_rebased`,
`downward_rebase`) or its refusal statuses. The import never retiers the
target, and transcodes artifact bytes by at most one rung; an artifact more
than one rung below what the target holds is `precision_refused` (measured: an
artifact saved under a 120 MB KV budget against an unconstrained target), and
one the target cannot hold is `state_rejected` with `destination=invalid`
(the reverse). The same budget on both sides restores at every depth
measured. After a `downward_rebase` the greedy continuation can differ from a
server that never restarted, because one rung of transcoding is lossy; the
native and live-rebased imports measured token-identical.

**The envelope is the library's.** It carries the companions the capture had:
recurrent state, and the drafter or accelerator state of a server that runs
one. "No drafter state" is a property of the fixed route only. Measured with
the MTP head of a 27B: three companions, identical tokens, acceptance
unchanged (0.91–0.95).

**Every slot of the cache, one image.** An artifact is an image of the whole
pool up to its watermark with the placement of one sequence, and the library's
import takes an empty cache, not an empty slot ("empty import is a whole-child
contract"). The rows of the other slots are already inside that image, so the
route saves the image once and, for every other slot, where its rows lie:

- *Pool entry.* The most recently used slot whose capture succeeds writes the
  artifact (`v-<gen>`, kind 3) as above.
- *Placed entry.* Every other live slot writes a placement object (`p-<gen>`,
  kind 4) and, on a hybrid model, one tail state at its frontier. The manifest
  names the image it lies in: `pool_entry`, `pool_generation`, `pool_xxh3`. The
  payload is little-endian: `u32 magic 'RSPL'`, `u32 version = 1`, `u32 count`,
  then per placement `u32 child_id, stream_index, computation_frontier,
  n_cells` and per cell `u32 physical_cell, i32 logical_position, ext_x,
  ext_y`. That is 16 bytes per token per attention child; the recurrent tail
  state of a 4B hybrid is about 52 MB.
- *Install.* One empty import of the pool entry. The placed entries that match
  its three pool fields and get a slot go in as co-residents: validation adds
  their rows to the authorized runs, the image is installed with those rows
  under their own sequence ids, and the generation tracker stamps them. The
  decision is `live_rebased` (or `downward_rebase`), never `native_import`.
  Each co-resident slot is then established as on the fixed route (tokens from
  its ledger, the tail state into the recurrent child).
- *Pricing.* The destination is priced with the rows of every placed entry of
  the image, whether or not its sequence joins this import (`unowned_cells`
  for those that do not), because the published watermark is the artifact's.
- *Refusal.* If the library refuses the import with co-residents, it is retried
  once alone; the co-residents report `pool_refused` and start cold.
- *Fewer slots.* A placed entry that gets no slot is `no_free_slot` and stays
  in the store. It is dropped the next time the image is rewritten, and a
  placed entry whose image is gone is dropped at install (`pool_missing`). A
  placed entry is never installed alone (`pool_not_installed`).
- *Unchanged restart.* When no slot changed and the group still names the same
  image, every entry is kept (`artifact_kept`, no bytes written).
- *Rewrite.* A changed primary rewrites the image. Before it is written, the
  other members' artifact entries and the placed entries of the old image leave
  the disk, so the store never holds two images of one cache.

A cache with a child that holds one sequence only (the sliding-window child of
an iSWA model, or a QSA index) has no image to share: the library refuses the
capture while other sequences are live. The save passes are terminal (shutdown,
or sleep before the context is destroyed), so there the other slots leave the
cache and the most recently used conversation is saved alone (`"sole": true`;
the others report `pool_unshared`). Multi-slot restore therefore covers dense
and hybrid models; an iSWA model restores its most recent conversation.

An idle capture in flight holds the transfer ring, so a save pass cancels and
drains it first; without that a save that follows a request by a few
milliseconds is `capture_refused` (`transfer_failed`, ring unavailable).

**The live slot is what gets saved.** With one slot, the VBR host cache's idle
capture normally publishes the conversation to host memory and then clears the
live slot. The projected package it is left in has no wire form (unit ids in the
projected domains, Merkle roots in place of payloads), so a save after an idle
moment would find nothing to capture. With a persistent resume active the idle
capture still publishes but leaves the source live, as it already does in a
multi-slot unified cache; a returning conversation then restores through the
library's occupied-replacement route, which needs room for both conversations in
the context. The cost is that an idle server keeps its cache mapped.

**The host cache goes through a slot.** A conversation that lives in the VBR
host cache alone is a projected package, which no file can hold. It is saved
and brought back by the two doors the VBR artifact library already has, with no file kind
of its own:

- *Save.* After the slots are saved, the hosted states of the running
  execution identity and adapter configuration are listed (no media; not an
  earlier state of a saved slot or of another hosted state; the newest up to
  the entry bound, `max(8, 4 x slots)`). The most recently used slot is the
  stage, the other slots leave the cache, and the entries of all of them are
  held. Each hosted conversation is restored into the stage by the library's
  automatic restore, replacing what the stage holds, and saved as an ordinary
  pool entry (`"hosted": true` in the event) with a last-used time older than
  every slot's, in the order of the host cache. A refused replacement is
  retried once into a cleared stage; a refusal after that is
  `host_restore_refused` and the entry the conversation had from an earlier
  pass, if any, stays.
- *Load.* The first artifact entry, newest first, is the pool entry of the
  cache and takes its slot with its co-residents. Every other artifact entry
  is a hosted one: it is installed into an empty staging slot (the install
  above), published by the library's idle capture run to completion, and the
  stage is cleared, oldest first so that the host cache's own order returns.
  The outcome is `installed_host`; an entry that is not published stays in the
  store (`host_cache_rejected`). This runs before the pool entry is
  installed, the cache being empty in between.
- *Cost.* One artifact per hosted conversation, and one restore and capture
  of each at save, one install and capture at load (measured on the 0.6B
  dense, 353 MB each: about 2.9 s per conversation either way). An image
  covers the cache up to its watermark, so a conversation restored above the
  one it replaced costs up to twice its size on disk.

**Needs the artifact store.** The store exists with the VBR host cache. Under
`--cache-ram 0` a save is `unsupported_artifact` and the server warns at start
that nothing is persisted. Media is supported by the exact live-slot artifact
route and automatic host capture/restore (§4a), including its media ledger.

Media regression coverage on RTX 3090: Gemma 4 E2B image conversations over two
restarts, both below and above the sliding window, including an ~11k-token
mixed-tier F16/T4 layout at an 80 MiB budget. All continuations matched the
uninterrupted control, with the full saved prefix reused. An audio transcription
and two follow-ups also matched across restarts. Two image conversations saved
and restored together under `-np 2 --kv-unified` preserved both the primary
artifact ledger and its placed co-resident. The short-window reuse boundary has
a unit regression: `pos_min == 0` must not trigger a cold replay before the
sliding window fills.

Measured (RTX 3090, NVMe; 0.6B dense, 4B hybrid, 27B with MTP): restart ×2,
sleep/wake ×2, rewind, live restore through the slot action and a degraded
cache restore token-identical to one process that never stopped. 353 MB dense
artifact: save 2.6 s, install 2.9 s; 255–280 MB hybrid artifact of 6–6.7k
tokens: save 1.8 s (capture 0.7 s), install 2.0 s. The remainder is hashing:
the artifact's encode and decode hash the payload about 13 times, which is why
`llama_sha256` uses the SHA extensions on x86 (1.8 GB/s against 0.32 GB/s; the
same save took 13.6 s before).
