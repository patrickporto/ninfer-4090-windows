#!/usr/bin/env python3
"""Concurrent long-conversation cache-pressure repro: one thread per conversation, several turns each.
usage: cache_pressure_repro.py PORT OUT.json SIZE[,SIZE...] TURNS [MAX_TOKENS]
Prints per-turn prompt/cached tokens; resumes (turn > 0) should hit the cache."""
import json, sys, time, urllib.request, random, threading
port, out = int(sys.argv[1]), sys.argv[2]
sizes, turns = [int(s) for s in sys.argv[3].split(",")], int(sys.argv[4])
max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 32768
url = f"http://127.0.0.1:{port}/v1/chat/completions"
words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu".split()
TOK_PER_LINE = 34.0  # calibrated below on first reply
def text(seed, tokens):
    rng = random.Random(seed)
    return "\n".join(f"R{seed}-{i}: " + " ".join(rng.choice(words) for _ in range(15)) for i in range(int(tokens / TOK_PER_LINE) + 1))
def chat(msgs):
    body = json.dumps({"model": "qwen3.8-27b", "messages": msgs, "max_tokens": max_tokens, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t = time.time()
    with urllib.request.urlopen(urllib.request.Request(url, body, {"content-type": "application/json"}), timeout=3600) as r:
        j = json.load(r)
    return j["choices"][0]["message"]["content"][:200], j["usage"], time.time() - t
log, lock, t0 = [], threading.Lock(), time.time()
def run(k, size):
    msgs = [{"role": "system", "content": f"Conversa {k}. Responda sempre apenas: ok"}, {"role": "user", "content": text(1000 + k, size)}]
    for turn in range(turns):
        if turn:
            msgs.append({"role": "user", "content": text(k * 100 + turn, 1500)})
        reply, u, dt = chat(msgs)
        msgs.append({"role": "assistant", "content": reply})
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        row = dict(t=round(time.time() - t0, 1), conv=k, turn=turn, prompt=u["prompt_tokens"], cached=cached,
                   hit=round(100 * cached / max(1, u["prompt_tokens"]), 1), seconds=round(dt, 1))
        with lock:
            log.append(row); print(json.dumps(row), flush=True)
th = [threading.Thread(target=run, args=(k, s)) for k, s in enumerate(sizes)]
for i, x in enumerate(th): x.start(); time.sleep(1)
for x in th: x.join()
json.dump(log, open(out, "w"), indent=1)
later = [r for r in log if r["turn"] > 0]
print("SUMMARY turns>0:", len(later), "zero-hit:", sum(r["hit"] < 5 for r in later),
      "median hit:", sorted(r["hit"] for r in later)[len(later) // 2] if later else None,
      "total s:", round(time.time() - t0, 1))
