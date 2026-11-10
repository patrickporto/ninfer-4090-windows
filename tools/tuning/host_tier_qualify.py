#!/usr/bin/env python3
"""Host-tier qualification: force demotion to the Host tier and compare greedy resumes.

usage: qualify.py PORT OUT.json
Four ~5K-token conversations against a 16K KV capacity with no extra device state slots push
earlier conversations to the Host tier; resuming each must hit the cache (cached_tokens > 0)
and the greedy continuations are recorded for cross-mode comparison.
"""
import json, sys, time, urllib.request, random

port, out = int(sys.argv[1]), sys.argv[2]
url = f"http://127.0.0.1:{port}/v1/chat/completions"

def chat(messages, max_tokens=48):
    body = json.dumps({"model": "qwen3.8-27b", "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t = time.time()
    with urllib.request.urlopen(urllib.request.Request(url, body, {"content-type": "application/json"}), timeout=600) as r:
        j = json.load(r)
    return j["choices"][0]["message"]["content"], j["usage"], time.time() - t

words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu".split()
def doc(seed):
    rng = random.Random(seed)
    lines = [f"Registro {i}: " + " ".join(rng.choice(words) for _ in range(14)) for i in range(260)]
    return f"Documento {seed}.\n" + "\n".join(lines)

convs = {k: [{"role": "user", "content": doc(k) + "\n\nResuma o documento em uma frase."}] for k in range(4)}
result = {"first": {}, "resume": {}}
for k, msgs in convs.items():
    text, usage, dt = chat(msgs)
    msgs.append({"role": "assistant", "content": text})
    result["first"][k] = {"text": text, "usage": usage, "seconds": round(dt, 3)}
for k, msgs in convs.items():
    msgs.append({"role": "user", "content": "Qual é a primeira palavra do Registro 7?"})
    text, usage, dt = chat(msgs)
    result["resume"][k] = {"text": text, "usage": usage, "seconds": round(dt, 3)}
json.dump(result, open(out, "w"), indent=1, ensure_ascii=False)
for phase in ("first", "resume"):
    for k, r in result[phase].items():
        u = r["usage"]
        print(phase, k, "prompt", u["prompt_tokens"], "cached", u.get("prompt_tokens_details", {}).get("cached_tokens"), f"{r['seconds']}s", repr(r["text"][:60]))
