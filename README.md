# NInfer-4090

NInfer-4090 runs **Qwen3.8-27B** on one 24 GB NVIDIA GeForce RTX 4090, optionally helped by a
second GPU. It is an `sm_89` port of [NInfer-3090](https://github.com/Don-Chad/ninfer-3090),
which derives from [Neroued/ninfer](https://github.com/Neroued/ninfer), a specialized
C++20/CUDA inference engine. The engine loads the official groupwise `.ninfer` artifact, serves
OpenAI- and Anthropic-compatible APIs, and supports paged KV, compatible-prefix reuse, CUDA
Graphs, MTP and DFlash2 speculative decoding, reasoning-effort control, and ReplaySSM state
transactions.

This repository adds, on top of the 4090 port:

- **Two-GPU execution:** pipeline (PP2) and tensor (TP2) parallelism across mixed
  sm_89/sm_120 pairs without P2P ([docs/multi-gpu.md](docs/multi-gpu.md)).
- **Second GPU as context-cache tier:** `--host-tier-device N` keeps demoted KV and state in an
  idle GPU's VRAM instead of pinned RAM.
- **Four concurrent lanes** at the full 262K context, with fair access for API calls that
  wait on the engine.
- **PDF input** (Linux builds) and **Windows + Linux release builds** published by CI on every
  push to `main` (see [Releases](#releases)).
- A measured **KV-format and speculative-decoding study** that changed the recommended KV mode
  to `rk4v4` ([docs/rtx4090-tuning.md](docs/rtx4090-tuning.md)).

Blackwell-only NVFP4/W4A4 execution is unavailable on the 4090; the engine uses the same
groupwise-int path as the 3090 base. The Qwen3.6-35B-A3B target is inherited but untested here.

## Recommended configuration

Measured on an RTX 4090 with an RTX 5070 Ti as the idle second GPU (2026-11). This is the
configuration running in production:

```bash
CUDA_VISIBLE_DEVICES=<rtx4090>,<second-gpu> ninfer-serve models/qwen3_8_27b.ninfer \
  --host 127.0.0.1 --port 11434 \
  --max-context 262144 --kv-capacity 262144 \
  --max-concurrency 4 --device-state-slots 2 \
  --host-state-slots 8 --host-kv-mib 8192 --host-tier-device 1 \
  --max-pending-requests 16 --pending-timeout-ms 600000 --prefill-chunk 1024 \
  --kv-dtype rk4v4 --spec mtp --draft-tokens 3 --lm-head-draft \
  --preserve-thinking --vision --vision-max-tokens 4096
```

Without a second GPU, drop `--host-tier-device 1` (the cache tier then lives in pinned RAM;
`--host-kv-mib 2048 --host-state-slots 4` keeps it at about 2.8 GiB).

Why each choice (details and raw numbers in [docs/rtx4090-tuning.md](docs/rtx4090-tuning.md)):

| Choice | Measured reason |
|---|---|
| `--kv-dtype rk4v4` | Same 266K ceiling as `rk4v4-e8`, about half its loss on code (dNLL vs bf16 +0.0018 against +0.0033), same speed. 8-bit formats cost 30-50% of the context for no measurable gain. |
| MTP3 over DFlash2 | DFlash2 K=5 is 12-22% faster on code but leaves only 74K of KV with four lanes and vision (168K with two lanes). |
| 4 lanes | 210 tok/s aggregate on the code benchmark against 144 tok/s for one lane; the KV pool is shared, so one session still reaches 256K. |
| `--device-state-slots 2` | The default (4) leaves the four-lane 256K profile about 100 MB short. |
| `--host-tier-device 1` | Restores from the 5070 Ti tier run at 12.3 GB/s (pinned RAM: 13.4) and frees about 9 GiB of pinned RAM. |

| Workload (this configuration) | Result |
|---|---|
| Code generation, 1 request | 144 tok/s, 11/12 tasks pass |
| Code generation, 4 concurrent | 210 tok/s aggregate |
| Resume a 7.3K-token conversation from the cache tier | 0.5 s (re-prefill: 2.5 s) |
| KV quality, `rk4v4` vs `bf16`, 1M-token corpus at 64K windows | PPL 4.414 vs 4.419 |

## Releases

Every push to `main` builds and publishes a prerelease on GitHub Releases:

- **Windows x64** (MSVC): Single and Pipeline placement; no Tensor placement (NCCL is
  Linux-only) and no PDF input.
- **Linux x64** (Ubuntu 24.04, NCCL 2.29.7 bundled): Single, Pipeline and Tensor placement,
  PDF input. It links the distribution's FFmpeg 6 and poppler-glib; other distributions
  should build from source.

Both target `sm_89` and `sm_120`. Each release has a runtime archive (`ninfer`, `ninfer-serve`,
topology probe) and a `-tools` archive (`ninfer-perplexity`, `ninfer_bench`) that extracts into
the same folder.

## Earlier measurements on the RTX 4090

Conditions: single request, greedy decoding, CUDA Graphs on, INT8 KV, `--prefill-chunk 1024`,
official 16.96 GiB Qwen3.8-27B artifact. The code-generation decode row and the prefill rows
are measured from the `ninfer-serve` `/metrics` counters (computed prefill only); the other
decode rows use the `ninfer` CLI.

| Test | Result |
|---|---|
| Decode, code generation, MTP3 | **148.6 tok/s** at 81.0% draft acceptance |
| Decode, bench corpus, MTP3 | 106.5 tok/s at 48.7% acceptance |
| Decode, no speculation | 50.5 tok/s |
| Decode at 128K depth, no speculation | 39.6 tok/s |
| 64K needle-in-a-haystack | exact answer, 1,849 tok/s prefill |
| 128K needle-in-a-haystack | exact answer, 1,561 tok/s prefill |
| Vision, chart reading | 3 of 3 oracle facts, 22 ms vision tower |
| Ops test suite | 78 of 78 runnable tests pass on `sm_89` |

MTP acceptance, and with it the decoded rate, tracks how predictable the output is: structured
code accepts about 81% of draft tokens, the mixed bench corpus about 49%.

The default later moved from INT8 KV to the E8 4-bit KV mode, and in 2026-11 to `rk4v4`
(same capacity, lower loss; see [Recommended configuration](#recommended-configuration)). Both
4-bit modes serve the model's full native 262,144-token context on this card. Retrieval stays exact through 260K
(single-needle, 5-needle, and exact-code-detail probes), MTP acceptance at depth is unchanged,
and the costs against the INT8 numbers above are a 5.7% decode tax and 1-2% of prefill; see
[Quick start](#text-only-full-262k-native-context-e8-4-bit-kv-default) for the measured deltas.

For scale: llama.cpp on the same card decodes the Qwen3.8-27B `UD-Q4_K_XL` GGUF at about
46 tok/s in a 144K-context configuration where the MTP buffers do not fit. The upstream engine
on an RTX 5090 measures 172 tok/s on the same code-generation prompts with a 400 W power cap
(the upstream README quotes about 200), so this card lands within 14% of it under MTP.

### Depth sweep against llama.cpp

Both engines were measured on the same card. llama.cpp build 10358 ran `llama bench` on the
`UD-Q4_K_XL` GGUF (16.68 GiB) with q8_0 KV cache, flash attention, and `-ub 1024 -b 4096`,
which matches its deployed configuration, on 2026-08-15. The NInfer side was re-measured on
2026-08-17 on the deployed E8 262K configuration through the `/metrics` counters; the
llama.cpp configuration did not change between the dates. Two caveats: the artifacts differ
by about 2% in size, and `llama bench` is a bare kernel loop while the NInfer numbers
include the full server path.

Marginal rates at depth:

| Depth | llama.cpp pp2048 | llama.cpp tg32 | NInfer decode, no speculation |
|---:|---:|---:|---:|
| 0 | 3,024 tok/s | 45.9 tok/s | 50.4 tok/s |
| 32K | 2,327 | 42.0 | - |
| 64K | 1,866 | 38.6 | - |
| 128K | 1,336 | 33.1 | 42.1 |
| 256K | no entry | no entry | 36.6 |

Wall time to prefill one full prompt (llama.cpp integrated from the marginal rates, NInfer
measured):

| Prompt | llama.cpp | NInfer |
|---:|---:|---:|
| 32K | 12.5 s (2,630 tok/s) | 14.5 s (2,027 tok/s) |
| 64K | 28.3 s (2,317 tok/s) | 31.7 s (1,857 tok/s) |
| 128K | 70.4 s (1,862 tok/s) | 74.5 s (1,581 tok/s) |
| 192K | no entry | 127.9 s (1,381 tok/s) |
| 256K | no entry | 191.7 s (1,228 tok/s) |

The llama.cpp prefill lead narrows with depth. Server-measured, it prefills a 64K prompt in
28.7 s against 31.7 s (a 10% lead) and a 128K prompt in 71.6 s against 74.5 s (4%); the
server path costs llama.cpp 2-4% over the bare-loop estimates above. Everything past its
144K ceiling is NInfer-only. Decode inverts the shallow picture. NInfer leads by 10%
shallow and by 27% at 128K without speculation, and the MTP3 gap grows with depth:

| Workload | llama.cpp `draft-mtp` | NInfer MTP3 (E8) |
|---|---:|---:|
| Code, shallow | 118.8 tok/s at 85.9% acceptance | 142.9 tok/s at 78.0% |
| Prose, 64K depth | 55.5 tok/s at 45.3% | 86.1 tok/s at 42.3% |
| Prose, 128K depth | 42.3 tok/s at 45.4% | 77.5 tok/s at 41.6% |
| Prose, 256K depth | no entry | 65.4 tok/s at 41.1% |
| Code, 256K depth | no entry | 91.2 tok/s at 72.1% |

The NInfer rows in this table use the 2026-08-17 generated corpora; acceptance on them runs
a few points below the 2026-08-15 payloads (code 78% against 81%), which accounts for the
difference from the headline 148.6 tok/s. The llama.cpp MTP rows required a reduced
131,584-token context; the draft buffers push VRAM
to 23.8 of 24 GiB, and the deployed 144K llama.cpp configuration cannot fit them at all.
NInfer serves 172,032 tokens with MTP in the same VRAM at INT8 KV, and the full native
262,144 with the E8 4-bit KV default. Acceptance matches per content type, so the decode gap
is engine time, not draft quality.

Full configurations, method, and raw numbers:
[NInfer against llama.cpp](docs/llamacpp-comparison.md).

## Quick start (Linux)

Requirements: an RTX 4090, a recent NVIDIA driver, Docker with the NVIDIA Container Toolkit.

Build the image and download the model once:

```bash
docker build --tag ninfer-4090:sm89 .
NINFER_MODEL_DIR="$PWD/models" bash scripts/download-qwen38.sh
```

Then start one of the three profiles. The API is available at `http://127.0.0.1:8080/v1`.

The profiles as written run one generation slot. More lanes are measured and worthwhile:
each lane costs a few hundred MiB (state pools plus CUDA-graph allowance) while the KV page
pool stays shared, so a lone session still uses the full context. Two lanes decode at roughly
1.5x aggregate throughput; four lanes (with `--device-state-slots 2` to fit 256K and vision)
reach 210 tok/s aggregate on code against 144 tok/s for one. Prefill still serializes across
lanes, so a deep cold prefill delays the other lanes' first token.

When clients edit recent conversation history (a re-serialized reply, tool results
folded into the previous turn, an updated agent memory block), the server proposes a
private long anchor at each of the last N message boundaries of every prompt
(`--auto-long-anchors N`, on by default at the `--max-long-anchors-per-continuation`
cap of 2) and re-prefills from the anchor below the edit instead of from zero.
Coverage reaches back exactly as many message boundaries as the retention cap:
when the anchor set is full the shallowest is evicted, so an edit deeper than the
cap still re-prefills from zero. Raise `--max-long-anchors-per-continuation` and
`--auto-long-anchors` together for deeper reach; each retained anchor holds one GDN
state image (about 147 MiB of host memory on Qwen3.8-27B), so size `--host-state-slots`
for `continuations x (2 + anchors)`. The old `--turn-checkpoints` ring is retired
and ignored; see [docs/turn-checkpoint-ring.md](docs/turn-checkpoint-ring.md).

Extra requests beyond the slots wait in the admission queue, and the queue deadline
defaults to 30 seconds. A deep prefill can hold a slot longer than that, so
parallel agent clients would fail with `request_queue_timeout`. The
`--pending-timeout-ms 600000` line raises the deadline to 10 minutes. On a
streaming request the timeout arrives as an in-band SSE error event after HTTP 200;
a client that does not parse error events sees a stream that ends without a
`finish_reason`. See [docs/serving.md](docs/serving.md) for the full queue
contract.

### Text-only, full 262K native context (4-bit KV, default)

The rotated 4-bit KV modes (`rk4v4`, and the E8 lattice variant `rk4v4-e8`, ported from
[UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090); see
[the fork comparison](docs/udp-fork-comparison.md)) fit the model's entire native
262,144-token context on 24 GB with about 1.4 GiB to spare. `rk4v4` is the default since
2026-11: same capacity and speed as `rk4v4-e8`, lower loss
([measurements](docs/rtx4090-tuning.md#kv-cache-format)):

```bash
docker run --rm --gpus all --publish 8080:8080 \
  --volume "$PWD/models:/workspace/models:ro" \
  ninfer-4090:sm89 \
  ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 262144 --kv-capacity 262144 \
  --max-concurrency 1 --max-pending-requests 16 \
  --pending-timeout-ms 600000 \
  --prefill-chunk 1024 --kv-dtype rk4v4 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --preserve-thinking
```

Measured for `rk4v4-e8` against INT8 KV on an earlier build: identical MTP acceptance at 111K depth
(78.8% vs 78.4%), a 5.7% decode tax (126.6 vs 134.2 tok/s on a shallow greedy code
probe), prefill within 1-2% at matched depth, and exact single-needle, 5-needle, and
code-detail retrieval through 260K tokens.

### Text-only, 168K context (INT8 KV, maximum precision)

```bash
docker run --rm --gpus all --publish 8080:8080 \
  --volume "$PWD/models:/workspace/models:ro" \
  ninfer-4090:sm89 \
  ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 172032 --kv-capacity 172032 \
  --max-concurrency 1 --max-pending-requests 16 \
  --pending-timeout-ms 600000 \
  --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --preserve-thinking
```

### With vision, full 262K context (4-bit KV)

The vision scratchpad defaults to 8192 tokens (`--vision-max-tokens`, ported from
the same fork as the E8 KV modes) instead of the former hardcoded 32768. The
smaller scratchpad frees about 1.5 GiB, so the full native context fits next to
vision on 4-bit keys:

```bash
docker run --rm --gpus all --publish 8080:8080 \
  --volume "$PWD/models:/workspace/models:ro" \
  ninfer-4090:sm89 \
  ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 262144 --kv-capacity 262144 \
  --max-concurrency 1 --max-pending-requests 16 \
  --pending-timeout-ms 600000 \
  --prefill-chunk 1024 --kv-dtype rk4v4 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --preserve-thinking
```

The vision tower encodes one image at a time, so the scratchpad bounds the size
of each image, not the number of images or the conversation depth. One
1024x1024 image costs 1026 vision tokens, and the default admits a single image
of up to about 2880x2880 pixels. All images in one request share the upstream
aggregate budget of `min(--max-context, 32768)` vision tokens. Agent clients
send every earlier image again with each turn, so that aggregate is what a long
conversation with screenshots uses up. The server rejects an image over the
scratchpad, or a request over the aggregate, with `media_budget_exceeded`
before the request reaches the encoder. For large single images or dense video,
raise the limit with `--vision-max-tokens`. Each additional 1024 tokens of
scratchpad costs about 62 MiB of VRAM.

### The tradeoff

KV precision, vision, and maximum context trade against each other on a 24 GB card
(earlier single-lane measurements; `rk4v4` sizes exactly like `rk4v4-e8`; the four-lane
capacity of every format is in [docs/rtx4090-tuning.md](docs/rtx4090-tuning.md#capacity)):

| Profile | KV mode | Context | KV runtime | Startup slack |
|---|---|---:|---:|---:|
| Text-only, MTP3 | `rk4v4-e8` | 262144 (256K) | 5.08 GiB | 1.37 GiB |
| Text-only, MTP3 | `rk2v4-e8` | 262144 (256K) | 4.01 GiB | 2.43 GiB |
| Text-only, MTP3 | `int8` | 172032 (168K) | 6.31 GiB | 136 MiB |
| With `--vision`, MTP3 | `rk4v4-e8` | 262144 (256K) | 5.41 GiB | 780 MiB |
| With `--vision` (32K scratchpad), MTP3 | `rk2v4-e8` | 262144 (256K) | 5.85 GiB | 329 MiB |
| With `--vision` (32K scratchpad), MTP3 | `rk4v4-e8` | 212992 (208K) | 6.06 GiB | 108 MiB |
| With `--vision` (32K scratchpad), MTP3 | `int8` | 98304 (96K) | - | ~1 GiB |

262,144 is the model's own context limit, so `rk2v4-e8` (2-bit keys, 96.2% cosine)
buys no additional context over `rk4v4-e8` in the text-only profile - only slack.
That slack is what pays for vision. With the former hardcoded 32,768-token vision
scratchpad, vision cost about 2.1 GiB (1.83 GiB of runtime buffers plus a
0.28 GiB tower): INT8 could only afford it at 96K, 4-bit keys topped out at
212992, and only 2-bit keys fit the full 262,144. The default 8192-token
scratchpad cuts the cost to about 0.6 GiB, and the full native 262,144 now fits
alongside vision on 4-bit keys with 780 MiB of slack. The vision
modes answer a two-swatch color oracle exactly at temperature 0, including with
the image buried under 52,700 tokens of text on `rk2v4-e8`. `rk2v4-e8` also passes
the text retrieval gates (single-needle at 260K, 5-needle at 118K, exact code
details at 168K) at a 10% decode tax (120.5 tok/s on the shallow code probe). The
INT8 text-only ceiling is near 176K: 172032 starts, and 196608 is rejected at
startup with a byte-exact deficit. The server validates memory before it listens,
so an oversized context fails fast instead of at request time.

For a native build, follow the [Linux build guide](docs/rtx-3090-linux.md) with
`CMAKE_CUDA_ARCHITECTURES=89` (the default in this fork). The build requires CUDA 12.8 or newer,
GCC 13, and CMake 3.28 or newer; the Docker image builds with CUDA 13.1.

## What this fork changes

- **`sm_89` retarget.** The CMake architecture pin, the runtime compute-capability check, and the
  NVFP4 stub gate now select `sm_89`. Most SM86 kernel schedules run unmodified on Ada; the
  INT8 attention prefill schedule is retuned (below).
- **Ada-retuned INT8 attention prefill.** The SM120 schedule spills registers on Ada and pays the
  consumer half-rate penalty for f32-accumulate HMMA. Arch-gated for `sm_89`: the full
  128-register budget, eight paired producer warps over `Bc` column halves with one named-barrier
  exchange per key tile, byte-permute V dequantization (bit-identical), and fp16-accumulated PV
  tiles folded into the fp32 running accumulator each tile. The kernel gains 30% at 64K depth
  (109 to 143 TFLOP/s on the `d256-h24-kv4` INT8 append shape); serve prefill gains 5-7% at
  88K-128K. Needle-in-a-haystack retrieval stays exact at both depths and all 84 suite tests
  pass, which bounds the fp16-accumulation numerics change.
- **Causal-tile partitioned key-block traversal.** Interior key blocks (wholly below the causal
  diagonal for the whole CTA tile) run a separate instantiation of the key-block body: KV stages
  with unconditional copies and the softmax drops its masking selects; boundary blocks keep the
  exact masked path. The idea comes from the
  [UDPSendToFailed fork](https://github.com/UDPSendToFailed/ninfer-4090) (c5f70526),
  re-implemented inside the retuned schedule above. Kernel: 144 to 165 TFLOP/s at 32K-224K
  context on the INT8 append shape (-12 to -13% latency), register count unchanged, bit-exact.
  End-to-end this is bounded by the attention wall share of this hybrid-GDN model: about +1%
  serve prefill at 51K on INT8 KV, within noise on the E8 modes, whose staging time is dominated
  by lattice decode rather than the removed guards.
- **`/v1/models` reports `context_window`.** Clients without access to a llama.cpp `/props` or a
  vLLM `max_model_len` can size prompts from the models payload.
- **llama.cpp-compatible `timings` on chat completions.** Responses and final stream chunks carry
  a top-level `timings` block (`prompt_n`/`predicted_n`, per-second rates, `ttft_ms`, `cache_n`,
  `draft_n`/`draft_n_accepted`), so proxies such as llama-swap show per-request prefill and decode
  rates, MTP draft acceptance, and prefix-cache hits. Contributed by the
  [shantanusingh16 fork](https://github.com/shantanusingh16/ninfer-4090) of this repository.
- **`GET /metrics`.** Prometheus counters under llama.cpp-compatible names
  (`llamacpp:prompt_tokens_total`, `llamacpp:prompt_seconds_total`,
  `llamacpp:tokens_predicted_total`, `llamacpp:tokens_predicted_seconds_total`,
  `llamacpp:requests_processing`, `llamacpp:requests_deferred`), so existing scrapers read this
  server without changes. Processing/deferred occupancy is reserved before prompt preparation
  or engine submission and held through response release, so accepted work cannot disappear from
  metrics while queued. Prompt tokens count only computed prefill; prefix-cache hits are excluded,
  as in llama.cpp. Additional `ninfer:` series report request totals, prefix-cache hits, and MTP
  draft/acceptance totals.
- **`GET /slots`.** A llama.cpp-shaped slot table read from the engine's real lane state: busy
  slots report their request's prompt and reused-prefix sizes, idle retained slots report the
  resident session's depth and its identifying `session_digest`. Truthful per-slot attribution
  holds at any `--max-concurrency`.
- **Slot session save/restore.** `--slot-save-path DIR` (off by default) enables llama.cpp-style
  `POST /slots/{id}?action=save|restore|erase`: one idle slot's complete resident session -
  paged Text and MTP KV, GDN linear-attention state, rewrite checkpoint, long anchors, and
  prefix identity - moves to or from disk, and a restored slot reuses the cache across server
  restarts instead of re-prefilling (a 6.9k-token session restores in about 0.1 s against a
  multi-second reprefill).
  Sessions are identified by a stable `session_digest`; chat completions carry `id_slot` and the
  digest next to `timings`, and `save`/`erase` accept an `if_digest` precondition checked
  atomically, so a client always persists exactly the session it means. A restored session is
  reusable from its endpoint, its rewrite checkpoint, or any retained long anchor; the GDN
  state cannot rewind below the deepest retained checkpoint, and the DFlash backend is not
  supported. Details in [docs/serving.md](docs/serving.md).
- **Reuse-aware lane choice.** When prefix reuse ties (typically zero for a fresh session),
  admission picks the lane whose occupation costs least to replace - an empty lane before any
  retained session, then the shallowest - so a burst request no longer evicts a deep resident
  session while a free lane exists.
- **Automatic long anchors.** `--auto-long-anchors N` (default: the
  `--max-long-anchors-per-continuation` cap) has the server propose a private long anchor at
  each of the last N message boundaries of every prompt. Upstream's long anchors exist only
  where a client places an explicit `PrivateLongAnchor` marker, which no OpenAI or Anthropic
  request can express, so without this flag a rewrite deeper than the last assistant reply
  has no reuse candidate at all and re-prefills from token zero. With it, the prompt restores
  at the anchor below the edit. A full anchor set replaces its shallowest entry, so the
  retained anchors track the most recent boundaries and coverage reaches back exactly the
  cap: an edit deeper than `--max-long-anchors-per-continuation` boundaries still re-prefills
  from zero, so raise the cap and this flag together (and `--host-state-slots` with them) for
  deeper history. Measured on agent traffic, 94% of consecutive prompts are pure appends and
  96.5% of the remaining history rewrites are two messages deep or less, so the default cap of
  2 covers 99.8% of turns. The rare deeper edits replace the whole history from message one or
  two, where no anchor can help. Anchors ride the existing catalog, pressure planner and slot
  snapshots. This replaces the retired `--turn-checkpoints` ring
  ([docs/turn-checkpoint-ring.md](docs/turn-checkpoint-ring.md)).
- **Auto-save on eviction.** `--auto-save-evicted` (off by default, requires
  `--slot-save-path`) spills an involuntarily evicted session - endpoint, rewrite checkpoint
  and long anchors included - back to the slot file it was last saved to or restored from,
  before the eviction destroys it. Rotating more sessions than slots then loses nothing: the
  next restore recovers the session at its latest frontier. Explicit `erase` never auto-saves.
  Two rules keep a spill from losing data. First, a slot file is bound to at most one slot at
  a time: the most recent `save` or `restore` of a path owns it, and every other slot that held
  the same path is unbound. A stale copy of a session, left behind when a restore retains its
  source, therefore cannot write the file when it is evicted. Second, a spill never rolls a
  file back: a spill with fewer tokens than the file already holds is refused and logged as
  `slot auto-save SKIPPED`, while an explicit `save` always wins. Without these rules a
  two-day-old copy of a live session once overwrote its 78k-token file, and the client resumed
  the rolled-back state.
- **Planner diagnostics in the request JSONL.** `--request-log-jsonl FILE` records, per
  request, the reuse path the planner chose (`prefix_reuse_path`), the prefix tokens it reused,
  and the materialization search behind the choice: `stop_reason`, `budget_exhausted`,
  `selected_maximal_fallback`, `targets_evaluated`, and `best_reuse_prompt_tokens`, the most
  reuse any candidate offered. That last field separates the two causes of a cold prefill. A
  value of `0` means that no reuse candidate existed, so the cause sits upstream of the planner:
  a missing anchor or a changed prefix. A large value beside `prefix_reuse_path=root` means
  that a candidate existed and the planner rejected it, which points at the search itself.
  `/metrics` carries only the llama.cpp-compatible subset, so this file is the only place these
  fields appear. Field reference in [docs/serving.md](docs/serving.md).
- **NVFP4-A4 test gating.** The A4 activation tests skip on hardware without FP4 tensor cores
  instead of aborting. The full remaining suite passes on the RTX 4090.
- **E8 lattice KV quantization (ported).** The `rk8v4`/`rk4v4`/`rk4v4-e8`/`rk2v4-e8` KV modes
  and the 262K-to-1M visible-keys envelope lift from the
  [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) sibling fork,
  merged under this fork's retuned `sm_89` attention prefill schedule. The E8 codec verifies
  bit-exactly against the upstream microbenchmark (96.155% / 98.678% cosine); their 1 GiB
  CUDA-graph allowance bump was deliberately not taken (it would evict the INT8 168K profile).
  Method and measurements in [docs/udp-fork-comparison.md](docs/udp-fork-comparison.md).
- **Configurable vision scratchpad (ported).** `--vision-max-tokens` comes from the same fork
  and sizes the vision encode workspace (default 8192 tokens, formerly hardcoded 32768). This
  fork additionally wires the processor's single-item budget to the same limit, so an
  over-limit image fails as `media_budget_exceeded` instead of reaching an undersized encoder.
  The aggregate budget over all images in a request stays at upstream's 32768 tokens.

- **Two-GPU pipeline and tensor parallelism.** `--parallel pipeline|tensor --devices A,B` runs
  Qwen3.8-27B across two GPUs, including mixed sm_89/sm_120 pairs without P2P. TP2 decodes
  ~30% faster than one RTX 4090 without MTP (66 against 50.5 tok/s); MTP3 on one 4090 remains
  faster. Details and limits in [docs/multi-gpu.md](docs/multi-gpu.md).
- **Second GPU as context-cache tier.** `--host-tier-device N` backs the Host StateImage and
  Host KV tiers with an idle GPU's VRAM instead of pinned RAM. Tier copies use
  `cudaMemcpyDefault`, so they need no P2P; greedy output and cache hits match the pinned tier.
- **Fair engine access.** API calls that need the engine lock (memory summary, cancellation)
  yield-wait at most 50 ms instead of starving behind back-to-back decode rounds.
- **PDF input.** Chat requests may attach PDFs; text is extracted locally (poppler, Linux builds).
- **Windows and Linux release CI.** `.github/workflows/release.yml` builds both platforms for
  `sm_89` + `sm_120` and publishes a prerelease on every push to `main`.

## Known limits on the RTX 4090

- Prefill trails llama.cpp by 16-24% on full 32K-128K prompts under matched conditions (see
  the depth sweep above). The rate is flat across `--prefill-chunk` 1024 to 2688, so the
  chunk size is not the lever. With the attention schedule retuned, the remaining gap sits in
  the custom quantized GEMMs, which run about 10% below cuBLAS on Ada. Decode is where this
  engine leads.
- Keep `--prefill-chunk` at 2688 or below. This fork carries measured `sm_89` cooperative
  residency tables (the former hard abort above chunk 1024 is fixed), and chunks through 2688
  stay on split-K. Larger chunks route to the unsplit schedule, which is marginally less
  accurate at its onset (about 1e-5 relative).
- Up to four lanes are measured on the 4090 (see [Recommended configuration](#recommended-configuration));
  more are untested here.
- Prefill is strictly serialized across lanes with no chunk-level interleaving, and decode
  starves while any prefill runs: a short request submitted behind a 31k-token cold prefill
  measured a 13.5 s first token. Concurrency pays off for decode and for per-lane resident
  prefixes, not for prefill fairness.
- DFlash2 fits only 74K of KV with four lanes and vision (168K with two lanes); see
  [the tuning study](docs/rtx4090-tuning.md#speculative-decoding-mtp3-vs-dflash2).
- The limits of the base engine apply: one process, one model, bounded FIFO admission, no
  weight offload. Two-GPU placement is limited to the PP2/TP2 modes in
  [docs/multi-gpu.md](docs/multi-gpu.md), and `--host-tier-device` requires Single placement.

## Artifact

| Model | Artifact | Revision | Size | SHA-256 |
|---|---|---|---:|---|
| Qwen3.8-27B | [official NInfer groupwise artifact](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `3526913004b1` (2026-08-14, container v2) | 16.96 GiB | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
| Qwen3.8-27B + DFlash2 weights | same repository | `dc370fb6295a` (2026-09-06, container v2) | 19.03 GiB | `0634abb07024221de141456cf04a42ab74b18bc38e1b781c6eb2e062a467eec3` |

The download scripts fetch the pinned `3526913004b1` revision, which is the artifact this fork is
validated with. The artifact is architecture-independent; the model card's RTX 5090 requirement
describes the upstream engine, not the file. The `dc370fb6295a` revision adds the DFlash2 draft
weights and loads on the same engine (`--spec dflash2 --draft-tokens 5` to use them; K=5 is the
fastest measured on code that still serves four lanes, at reduced context).

**Do not download the `main` revision.** Since 2026-09-15 it is a container-v3 artifact with a
maintained jinja chat template. This fork reads container v2 only and rejects v3 with
`artifact magic is not NInfer v2`; v3 support arrives with the next upstream catch-up.

## Reasoning effort

Qwen3.8-27B has three trained reasoning depths plus an off switch. OpenAI Chat Completions
accepts a top-level `reasoning_effort` field (`low`, `medium`, `xhigh`) and a top-level
`enable_thinking` boolean; hidden reasoning returns separately as `message.reasoning_content`.
Only those three levels are accepted, plus `none` to turn thinking off. `high`, `minimal`, and
`max` are rejected as `reasoning_effort_not_supported`, so a client that offers a `high` setting
must map it to `xhigh`. A token budget for reasoning is separate from the effort level. It is
set only through the Anthropic Messages path (`thinking.budget_tokens`) or server-wide with
`--default-thinking-budget N`; the OpenAI paths have no field for it. Without a budget,
reasoning is bounded only by the request's `max_tokens`, which is what the model card
recommends, and the `model_thinking_tokens` field of the request JSONL reads zero, because that
counter runs only under a budget. The `chat_template_kwargs` request field of llama.cpp is not
supported and is rejected. For the CLI, pass `--reasoning-effort` or `--no-thinking`. Sampling
defaults come from the model card and switch with the thinking mode: `temperature=1.0`,
`top_p=0.95`, `top_k=20` in thinking mode; `temperature=0.7`, `top_p=0.80`, `top_k=20`,
`presence_penalty=1.5` in non-thinking mode.

## Serving APIs

OpenAI Chat Completions, OpenAI Responses with streaming and local continuation state, Anthropic
Messages, prompt-rendered function tools with parsed tool calls, compatible-prefix reuse, and
JSONL request logs. See [HTTP serving](docs/serving.md) and [CLI usage](docs/cli.md).

## Upstream and credits

- [Neroued/ninfer](https://github.com/Neroued/ninfer) - the engine, developed for the RTX 5090
  (`sm_120a`).
- [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) - the SM86 compatibility layer,
  ReplaySSM integration, and Qwen3.8 runtime support this fork builds on. Its
  [v0.6.1 release notes](RELEASE_NOTES_0.6.1.md) describe the inherited state.
- [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) - a sibling
  RTX 4090 port from the same 3090 base. The rotated and E8-lattice KV-cache quantization
  modes (`rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8`), the E8 codecs, and the 1M visible-keys
  envelope are their work, cherry-picked here with authorship preserved. The full 262K
  default profile exists because of it; see
  [the fork comparison](docs/udp-fork-comparison.md).
- [jram4/ninfer-4090](https://github.com/jram4/ninfer-4090) - an earlier RTX 4090 port of a July
  2026 snapshot. Its Ada dispatch tuning targets a kernel organization that upstream has since
  replaced, so this fork starts from the current 3090 base instead.

## Support

NInfer is a personal project that I develop out of interest. If you find it useful and would like
to support its continued development, you can [support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

Apache License 2.0. See [LICENSE](LICENSE).
