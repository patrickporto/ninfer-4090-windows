#!/usr/bin/env python3
"""Code-generation speed + correctness bench for speculative backends.

usage: codebench.py PORT LABEL
Each problem asks for one Python function; the reply's code block runs against hidden asserts
in an isolated interpreter (python3 -I, temp dir, timeout). Speed is decode tok/s from the
server's timings; runs at concurrency 1 and 4.
"""
import json, re, subprocess, sys, tempfile, threading, time, urllib.request, os

PORT, LABEL = int(sys.argv[1]), sys.argv[2]
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

P = [
 ("is_palindrome(s)", "returns True if s reads the same backwards ignoring case and non-alphanumerics",
  "assert is_palindrome('A man, a plan, a canal: Panama'); assert not is_palindrome('race a car'); assert is_palindrome('')"),
 ("merge_intervals(xs)", "merges overlapping [start,end] intervals and returns them sorted",
  "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]; assert merge_intervals([[1,4],[4,5]])==[[1,5]]; assert merge_intervals([])==[]"),
 ("lru_cache_ops(capacity, ops)", "simulates an LRU cache; ops is a list of ('get',k) or ('put',k,v); returns the list of results of get ops (-1 when missing)",
  "assert lru_cache_ops(2,[('put',1,1),('put',2,2),('get',1),('put',3,3),('get',2),('put',4,4),('get',1),('get',3),('get',4)])==[1,-1,-1,3,4]"),
 ("roman_to_int(s)", "converts a Roman numeral string to an integer",
  "assert roman_to_int('MCMXCIV')==1994; assert roman_to_int('LVIII')==58; assert roman_to_int('IV')==4"),
 ("longest_common_subsequence(a, b)", "returns the length of the longest common subsequence of two strings",
  "assert longest_common_subsequence('abcde','ace')==3; assert longest_common_subsequence('abc','def')==0; assert longest_common_subsequence('','a')==0"),
 ("topo_sort(n, edges)", "returns a topological order of nodes 0..n-1 given directed edges (u,v) meaning u before v, or None if there is a cycle",
  "o=topo_sort(4,[(0,1),(1,2),(0,3)]); assert sorted(o)==[0,1,2,3] and o.index(0)<o.index(1)<o.index(2) and o.index(0)<o.index(3); assert topo_sort(2,[(0,1),(1,0)]) is None"),
 ("parse_csv_line(line)", "parses one CSV line with double-quoted fields that may contain commas and doubled quotes, returning the list of fields",
  "assert parse_csv_line('a,b,c')==['a','b','c']; assert parse_csv_line('\"x,y\",z')==['x,y','z']; assert parse_csv_line('\"he said \"\"hi\"\"\",1')==['he said \"hi\"','1']; assert parse_csv_line('a,,b')==['a','','b']"),
 ("dijkstra(graph, src)", "graph is a dict node->list of (neighbor, weight); returns dict of shortest distances from src to every reachable node",
  "g={'a':[('b',1),('c',4)],'b':[('c',2),('d',5)],'c':[('d',1)],'d':[]}; assert dijkstra(g,'a')=={'a':0,'b':1,'c':3,'d':4}"),
 ("eval_rpn(tokens)", "evaluates a Reverse Polish Notation expression with + - * / where division truncates toward zero",
  "assert eval_rpn(['2','1','+','3','*'])==9; assert eval_rpn(['4','13','5','/','+'])==6; assert eval_rpn(['10','6','9','3','+','-11','*','/','*','17','+','5','+'])==22"),
 ("flatten(d)", "flattens a nested dict into dot-joined keys, e.g. {'a':{'b':1}} -> {'a.b':1}",
  "assert flatten({'a':{'b':1,'c':{'d':2}},'e':3})=={'a.b':1,'a.c.d':2,'e':3}; assert flatten({})=={}"),
 ("count_islands(grid)", "counts 4-connected islands of '1' in a grid given as a list of strings",
  "assert count_islands(['11000','11000','00100','00011'])==3; assert count_islands([])==0; assert count_islands(['1'])==1"),
 ("next_permutation(xs)", "returns the lexicographically next permutation of the list xs, or the sorted list if xs is the last permutation",
  "assert next_permutation([1,2,3])==[1,3,2]; assert next_permutation([3,2,1])==[1,2,3]; assert next_permutation([1,1,5])==[1,5,1]"),
]

def prompt(sig, desc):
    return (f"Write a Python function `{sig}` that {desc}. Include a short docstring and type hints, "
            "then briefly explain the algorithm and its complexity. Put the code in one ```python block.")

def call(sig, desc, out, key):
    body = {"model": "qwen3.8-27b", "max_tokens": 900, "temperature": 0,
            "messages": [{"role": "user", "content": prompt(sig, desc)}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.load(r)
    out[key] = {"text": j["choices"][0]["message"]["content"], "usage": j["usage"],
                "timings": j.get("timings", {}), "wall": time.time() - t0}

def check(text, tests):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    if not m: return False
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "t.py")
        open(f, "w").write(m[0] + "\n\n" + tests + "\n")
        try:
            return subprocess.run([sys.executable, "-I", f], cwd=d, capture_output=True, timeout=20).returncode == 0
        except subprocess.TimeoutExpired:
            return False

res = {}
for conc in (1, 4):
    out = {}
    t0 = time.time()
    for b in range(0, len(P), conc):
        ths = [threading.Thread(target=call, args=(P[i][0], P[i][1], out, i)) for i in range(b, min(b + conc, len(P)))]
        for t in ths: t.start()
        for t in ths: t.join()
    wall = time.time() - t0
    toks = sum(v["usage"]["completion_tokens"] for v in out.values())
    dec = [v["timings"].get("predicted_per_second", 0) for v in out.values()]
    passed = [check(out[i]["text"], P[i][2]) for i in range(len(P))]
    res[conc] = {"aggregate_tok_s": toks / wall, "decode_tok_s_mean": sum(dec) / len(dec),
                 "pass": sum(passed), "n": len(P), "passed": passed,
                 "texts": [out[i]["text"] for i in range(len(P))]}
    print(f"{LABEL} C={conc}: pass {sum(passed)}/{len(P)} | per-request decode {sum(dec)/len(dec):.1f} tok/s | aggregate {toks/wall:.1f} tok/s | {toks} tok", flush=True)
json.dump(res, open(f"code-{LABEL}.json", "w"), indent=1)
