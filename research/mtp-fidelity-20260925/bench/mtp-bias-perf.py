#!/usr/bin/env python3
"""Before/after fusion with native and batch-consistent arithmetic controls."""
import json
import os
import pathlib
import statistics
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parent
tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
order = ['native', 'fused', 'fused', 'native'] if tag in ('v3', 'v4') else ['native', 'consistent', 'fused', 'fused', 'consistent', 'native']
for index, name in enumerate(order):
    env = os.environ.copy()
    before = name != 'fused'
    env['LD_LIBRARY_PATH'] = str(root/('baseline-bin-mtp-consistency' if before else 'build-cuda/bin'))
    env['LD_PRELOAD'] = str(root/'mtp-fixed-fa.so')
    env['MTP_FIXED_FA_COLS'] = '0' if name == 'native' else '4'
    env.pop('GGML_CUDA_MMVQ_MAX_N', None)
    if name != 'native':
        env['GGML_CUDA_MMVQ_MAX_N'] = '8'
    out = root/'results'/f'mtp-bias-perf-{tag}-{index}-{name}'
    subprocess.run([sys.executable, str(root/'mtp-serving-fidelity.py'),
        '--bin', str(root/'build-cuda/bin/llama-server'),
        '--model', '/root/models/apollo-gsq-rco/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf',
        '--out', str(out), '--prompt', 'code', '--repeats', '6'], check=True, env=env)
    for arm in ['plain', 'mtp']:
        records = [json.loads((out/f'{arm}-code-{i}.json').read_text()) for i in range(6)]
        rates = [r['timings']['predicted_per_second'] for r in records[1:]]
        print(json.dumps({'config': name, 'index': index, 'arm': arm,
            'median_tps': statistics.median(rates), 'rates': rates,
            'accept': [(r['timings'].get('draft_n_accepted'), r['timings'].get('draft_n')) for r in records],
        }), flush=True)
