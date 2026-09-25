#!/usr/bin/env python3
"""Private factorial/crossover screen. One GPU, serial fresh servers, cold requests."""
import json
import os
import pathlib
import statistics
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parent
round_name = sys.argv[1] if len(sys.argv) > 1 else 'a'
configs = [('baseline', 0, False), ('mmvq', 0, True), ('fa4', 4, False),
           ('both4', 4, True), ('both1', 1, True), ('both2', 2, True)]
if round_name in ('b', 'locked-b'):
    configs.reverse()
for name, cols, mmvq in configs:
    env = os.environ.copy()
    env.pop('GGML_CUDA_MMVQ_MAX_N', None)
    env['LD_PRELOAD'] = str(root/'mtp-fixed-fa.so')
    env['MTP_FIXED_FA_COLS'] = str(cols)
    if mmvq:
        env['GGML_CUDA_MMVQ_MAX_N'] = '8'
    out = root/'results'/f'mtp-perf-{round_name}-{name}'
    subprocess.run([sys.executable, str(root/'mtp-serving-fidelity.py'),
        '--bin', str(root/'build-cuda/bin/llama-server'),
        '--model', '/root/models/apollo-gsq-rco/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf',
        '--out', str(out), '--prompt', 'code', '--repeats', '6'], check=True, env=env)
    for arm in ['plain', 'mtp']:
        records = [json.loads((out/f'{arm}-code-{i}.json').read_text()) for i in range(6)]
        rates = [r['timings']['predicted_per_second'] for r in records[1:]]
        print(json.dumps({'config': name, 'round': round_name, 'arm': arm,
            'median_tps': statistics.median(rates), 'rates': rates,
            'accept': [(r['timings'].get('draft_n_accepted'), r['timings'].get('draft_n')) for r in records],
        }), flush=True)
