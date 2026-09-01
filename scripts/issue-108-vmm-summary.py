#!/usr/bin/env python3

import glob
import json
import os
import re
import sys


def read_meta(path):
    result = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            key, sep, value = line.rstrip("\n").partition("=")
            if sep:
                result[key] = value
    return result


def read_rates(path):
    pp = tg = None
    if not os.path.exists(path):
        return pp, tg
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("n_prompt", 0):
                pp = row.get("avg_ts")
            if row.get("n_gen", 0):
                tg = row.get("avg_ts")
    return pp, tg


def read_diagnostics(path):
    totals = {key: 0 for key in ("new_chunks", "create_us", "map_us", "access_us", "memset_sync_us", "total_us")}
    calls = 0
    if not os.path.exists(path):
        return calls, totals
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "GGML_VBR_VMM_DIAG" not in line or "event=map" not in line:
                continue
            calls += 1
            fields = dict(re.findall(r"([a-z_]+)=([^ ]+)", line))
            for key in totals:
                try:
                    totals[key] += int(fields.get(key, "0"))
                except ValueError:
                    pass
    return calls, totals


def fmt(value):
    return "" if value is None else f"{float(value):.6f}"


if len(sys.argv) != 2:
    raise SystemExit("usage: issue-108-vmm-summary.py OUTPUT_DIR")

root = sys.argv[1]
print("label\tcommit_kb\tpp_tok_s\ttg_tok_s\tmap_calls\tnew_chunks\tcreate_ms\tmap_ms\taccess_ms\tmemset_sync_ms\ttotal_ms\tstatus")
for meta_path in sorted(glob.glob(os.path.join(root, "*.meta.txt"))):
    meta = read_meta(meta_path)
    stem = meta_path[:-len(".meta.txt")]
    pp, tg = read_rates(stem + ".jsonl")
    calls, totals = read_diagnostics(stem + ".stderr.log")
    print("\t".join((
        meta.get("label", os.path.basename(stem)),
        meta.get("commit_kb", ""),
        fmt(pp),
        fmt(tg),
        str(calls),
        str(totals["new_chunks"]),
        f'{totals["create_us"] / 1000:.3f}',
        f'{totals["map_us"] / 1000:.3f}',
        f'{totals["access_us"] / 1000:.3f}',
        f'{totals["memset_sync_us"] / 1000:.3f}',
        f'{totals["total_us"] / 1000:.3f}',
        meta.get("status", "missing"),
    )))
