#include "../src/llama-kv-cells.h"

#include <cstdio>
#include <cstdlib>

static void expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-kv-cell-position-index: %s\n", message);
        std::abort();
    }
}

static void add_cell(
        llama_kv_cells & cells,
        uint32_t         cell,
        llama_pos        pos,
        llama_token      token,
        llama_seq_id     seq) {
    cells.pos_set(cell, pos);
    llama_kv_cell_ext ext = {};
    ext.tok = token;
    cells.ext_set(cell, ext);
    cells.seq_add(cell, seq);
}

int main() {
    llama_kv_cells cells;
    cells.resize(12);

    expect(cells.seq_pos_tok_le(0, 100) == LLAMA_TOKEN_NULL, "empty index returned a token");

    add_cell(cells, 1, 2, 102, 0);
    add_cell(cells, 7, 5, 705, 0);
    add_cell(cells, 3, 5, 305, 0);
    add_cell(cells, 9, 9, 909, 1);

    expect(cells.seq_pos_tok_le(0, 1) == LLAMA_TOKEN_NULL, "lookup crossed the lower boundary");
    expect(cells.seq_pos_tok_le(0, 4) == 102, "nearest earlier position was not selected");
    expect(cells.seq_pos_tok_le(0, 5) == 705, "repeated position did not select the highest physical cell");
    expect(cells.seq_pos_tok_le(1, 9) == 909, "sequence ownership was ignored");
    expect(cells.seq_pos_count_before(0, 6) == 3, "repeated positions were not counted independently");

    cells.seq_rm(7, 0);
    expect(cells.seq_pos_tok_le(0, 5) == 305, "removed repeated-position cell remained indexed");

    cells.pos_add(3, 3);
    expect(cells.seq_pos_tok_le(0, 5) == 102, "shift left a stale position entry");
    expect(cells.seq_pos_tok_le(0, 8) == 305, "shifted position was not re-indexed");

    cells.seq_add(3, 2);
    expect(cells.seq_pos_tok_le(2, 8) == 305, "sequence copy was not indexed");
    cells.seq_keep(3, 2);
    expect(cells.seq_pos_tok_le(0, 8) == 102, "sequence keep left removed ownership indexed");
    expect(cells.seq_pos_tok_le(2, 8) == 305, "sequence keep removed retained ownership");

    cells.pos_div(3, 2);
    expect(cells.seq_pos_tok_le(2, 4) == 305, "position division was not re-indexed");

    llama_kv_cells snapshot = cells.cp({ 1, 3, 9 });
    llama_kv_cells restored;
    restored.resize(6);
    restored.set(std::vector<uint32_t>{ 0, 2, 5 }, snapshot);
    expect(restored.seq_pos_tok_le(0, 2) == 102, "restored sequence-0 index is incomplete");
    expect(restored.seq_pos_tok_le(2, 4) == 305, "restored repeated metadata is incomplete");
    expect(restored.seq_pos_tok_le(1, 9) == 909, "restored sequence-1 index is incomplete");

    std::puts("KV cell position index test PASSED");
    return 0;
}
