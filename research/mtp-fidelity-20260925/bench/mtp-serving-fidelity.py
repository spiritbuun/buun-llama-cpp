#!/usr/bin/env python3
"""Cold-request greedy comparison, with repeated same-mode anchors. Not a task benchmark."""
import argparse
import json
import os
import pathlib
import signal
import subprocess
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--bin', required=True)
p.add_argument('--model', required=True)
p.add_argument('--out', required=True)
p.add_argument('--kv', default='f16')
p.add_argument('--port', type=int, default=8143)
p.add_argument('--probs', action='store_true')
p.add_argument('--prompt', choices=['code', 'prose', 'math'])
p.add_argument('--repeats', type=int, default=2)
a = p.parse_args()
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
url = f'http://127.0.0.1:{a.port}'
prompts = {
    'code': 'Write a Python quicksort function. Output only the code in a single code block, no explanation.',
    'prose': 'In two paragraphs, explain why people sometimes remember an event differently even when they saw it together.',
    'math': 'A shop discounts an 80 dollar jacket by 25 percent and then adds 8 percent sales tax. What is the final price? Explain your calculation briefly.',
}
if a.prompt:
    prompts = {a.prompt: prompts[a.prompt]}

def http(path, body=None):
    req = urllib.request.Request(url+path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)

summary = {}
for arm in ('plain', 'mtp'):
    cmd = [a.bin, '-m', a.model, '-ngl', '99', '-fa', 'on', '-c', '4096', '-np', '1',
           '-b', '512', '-ub', '512', '-ctk', a.kv, '-ctv', a.kv, '--fit', 'off',
           '--cache-ram', '0', '--host', '127.0.0.1', '--port', str(a.port), '-lv', '4',
           '--chat-template-kwargs', '{"enable_thinking":false}']
    if arm == 'mtp':
        cmd += ['--spec-type', 'draft-mtp', '--draft-max', '3']
    (out/f'{arm}-command.json').write_text(json.dumps(cmd, indent=2))
    with (out/f'{arm}.log').open('w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ)
        try:
            deadline = time.monotonic()+300
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'{arm} startup exited {proc.returncode}')
                try:
                    if http('/health').get('status') == 'ok':
                        break
                except Exception:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('startup')
                time.sleep(.5)
            summary[arm] = {}
            for name, prompt in prompts.items():
                formatted = http('/apply-template', {
                    'messages': [{'role': 'user', 'content': prompt}],
                    'chat_template_kwargs': {'enable_thinking': False},
                })['prompt']
                prompt_tokens = http('/tokenize', {'content': formatted, 'add_special': True, 'parse_special': True})['tokens']
                (out/f'{arm}-{name}-prompt.json').write_text(json.dumps({'text': formatted, 'tokens': prompt_tokens}))
                texts = []
                for repeat in range(a.repeats):
                    result = http('/v1/chat/completions', {
                        'messages': [{'role': 'user', 'content': prompt}],
                        'temperature': 0, 'seed': 1234, 'max_tokens': 192,
                        **({'logprobs': True, 'top_logprobs': 5} if a.probs else {}),
                        'cache_prompt': False, 'return_tokens': True,
                        'repeat_penalty': 1, 'presence_penalty': 0, 'frequency_penalty': 0,
                        'chat_template_kwargs': {'enable_thinking': False},
                    })
                    (out/f'{arm}-{name}-{repeat}.json').write_text(json.dumps(result, indent=2))
                    if not result.get('choices'):
                        raise RuntimeError(f'invalid response: {result}')
                    message = result['choices'][0]['message']
                    text = (message.get('reasoning_content') or message.get('reasoning') or '') + (message.get('content') or '')
                    if not text:
                        raise RuntimeError('empty completion')
                    texts.append(text)
                repeat_exact = all(text == texts[0] for text in texts)
                summary[arm][name] = {'repeat_exact': repeat_exact, 'text': texts[0]}
                print(json.dumps({'arm': arm, 'prompt': name, 'repeat_exact': repeat_exact}), flush=True)
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f'{arm} exit {proc.returncode}')
for name in prompts:
    x, y = summary['plain'][name]['text'], summary['mtp'][name]['text']
    lcp = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
    summary[name] = {'cross_mode_exact': x == y, 'common_characters': lcp}
    print(json.dumps({'prompt': name, **summary[name]}), flush=True)
(out/'summary.json').write_text(json.dumps(summary, indent=2))
