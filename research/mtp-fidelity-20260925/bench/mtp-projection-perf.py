#!/usr/bin/env python3
"""Private before/after serving test; no production controls are added."""
import json
import os
import pathlib
import statistics
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parent
tag = sys.argv[1]
anchor = None
for index, name in enumerate(['native', 'consistent', 'candidate', 'candidate', 'consistent', 'native']):
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(root / ('build-cuda/bin' if name == 'candidate' else 'baseline-bin-mtp-consistency'))
    env['LD_PRELOAD'] = str(root / 'mtp-fixed-fa.so')
    env['MTP_FIXED_FA_COLS'] = '0' if name == 'native' else '4'
    env.pop('GGML_CUDA_MMVQ_MAX_N', None)
    if name != 'native':
        env['GGML_CUDA_MMVQ_MAX_N'] = '8'
    out = root / 'results' / f'mtp-projection-perf-{tag}-{index}-{name}'
    subprocess.run([sys.executable, str(root / 'mtp-serving-fidelity.py'),
        '--bin', str(root / 'build-cuda/bin/llama-server'),
        '--model', '/root/models/apollo-gsq-rco/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf',
        '--out', str(out), '--prompt', 'code', '--repeats', '6'], check=True, env=env)
    summary = json.loads((out / 'summary.json').read_text())
    if anchor is None:
        anchor = summary['plain']['code']['text']
    for arm in ['plain', 'mtp']:
        assert summary[arm]['code']['repeat_exact']
        assert summary[arm]['code']['text'] == anchor
        records = [json.loads((out / f'{arm}-code-{i}.json').read_text()) for i in range(6)]
        rates = [r['timings']['predicted_per_second'] for r in records[1:]]
        print(json.dumps({'config': name, 'index': index, 'arm': arm,
            'median_tps': statistics.median(rates), 'rates': rates,
            'accept': [(r['timings'].get('draft_n_accepted'), r['timings'].get('draft_n')) for r in records],
        }), flush=True)
