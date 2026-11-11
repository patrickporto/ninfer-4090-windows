# RTX 4090 tuning study (2026-11)

Measurements behind the [recommended configuration](../README.md#recommended-configuration).
Hardware: RTX 4090 (compute) + RTX 5070 Ti (idle second GPU), Ryzen 9 9950X3D, 32 GB RAM,
Linux, CUDA 13.2, official `3526913004b1` artifact (and `dc370fb6295a` for DFlash2). Every
serving measurement uses the production flags unless stated: `--max-concurrency 4
--device-state-slots 2 --prefill-chunk 1024 --spec mtp --draft-tokens 3 --lm-head-draft
--preserve-thinking --vision --vision-max-tokens 4096`.

## KV cache format

### Capacity

Largest `--kv-capacity` that passes runtime planning on the 4090 with the production flags
(binary search in 64-token page groups; planning precedes weight loading, so it is exact):

| `--kv-dtype` | Max KV tokens | vs `rk4v4-e8` |
|---|---:|---:|
| `bf16` | 72,448 | -73% |
| `int8` | 140,544 | -48% |
| `rk8v4` | 185,536 | -32% |
| `rk4v4` | 272,896 | same |
| `rk4v4-e8` | 272,896 | - |
| `rk2v4-e8` | 356,864 | +31% |

`k8v4` and `nvfp4` are sm_120a-only. Only the 4-bit-key formats fit the model's full 262,144
context next to four lanes and vision.

### Quality

`ninfer-perplexity` on the full `ninfer-ppl-1m-v1` corpus (1,044,557 scored tokens, 16
windows) at `--context 65536 --stride 32768`, with `--token-nll` exports paired token by token
against `bf16`. ± is two standard errors of the paired mean.

| `--kv-dtype` | PPL | NLL | dNLL vs bf16 | Code NLL | Code dNLL vs bf16 |
|---|---:|---:|---:|---:|---:|
| `bf16` | 4.4189 | 1.48590 | - | 0.41718 | - |
| `rk8v4` | 4.4190 | 1.48592 | +0.00002 ± 0.00030 | 0.41819 | +0.00101 ± 0.00043 |
| `int8` | 4.4207 | 1.48630 | +0.00040 ± 0.00022 | 0.41721 | +0.00003 ± 0.00033 |
| **`rk4v4`** | 4.4139 | 1.48476 | -0.00114 ± 0.00039 | 0.41899 | +0.00181 ± 0.00053 |
| `rk4v4-e8` | 4.4514 | 1.49321 | +0.00732 ± 0.00047 | 0.42048 | +0.00330 ± 0.00065 |
| `rk2v4-e8` | 4.6419 | 1.53512 | +0.04922 ± 0.00100 | 0.44986 | +0.03267 ± 0.00143 |

`rk4v4` has the same capacity as `rk4v4-e8` with about half its code-domain loss and no
measurable aggregate loss (its small aggregate lead over `bf16` is noise, not a gain). The
8-bit formats buy almost nothing more and cost 30-50% of the context. `rk2v4-e8` loses clearly.

### Speed

Same decode rate for both 4-bit formats on the code benchmark below (MTP3, 256K):

| `--kv-dtype` | 1 request | 4 concurrent (aggregate) | Passed |
|---|---:|---:|---:|
| `rk4v4-e8` | 145.5 tok/s | 210.5 tok/s | 11/12 |
| `rk4v4` | 144.3 tok/s | 210.2 tok/s | 11/12 |

## Speculative decoding: MTP3 vs DFlash2

`tools/tuning/codebench.py`: twelve Python function tasks at temperature 0, each answer's code
block run against hidden asserts in an isolated interpreter; per-request decode rate from the
server `timings`. Production flags, KV capacity set to the largest that fits (capped at 262,144).

| Backend | KV capacity | 1 request | 4 concurrent (aggregate) | Passed |
|---|---:|---:|---:|---:|
| MTP3 (`rk4v4`) | 262,144 | 144.3 tok/s | 210.2 tok/s | 11/12 |
| DFlash2 K=3 | 75,648 | 150.6 (+4%) | 209.3 (0%) | 11/12 |
| DFlash2 K=5 | 74,560 | 176.5 (+22%) | 234.4 (+12%) | 11/12 |
| DFlash2 K=7 | 71,680 | 186.8 (+29%) | not measured | 11/12 |

DFlash2 is faster on code, but its draft weights (+2.1 GiB) and buffers come out of the KV
budget. DFlash2 K=5 capacity by serving shape:

| Shape | Max KV tokens |
|---|---:|
| 4 lanes + vision | 74,560 |
| 4 lanes, no vision | 95,040 |
| 2 lanes + vision | 167,936 |
| 2 lanes, no vision | 188,352 |
| 1 lane, no vision | 229,440 |

With concurrent agent sessions of 10-25K tokens each, 74K shared across four lanes forces
constant eviction, so production stays on MTP3. DFlash2 K=5 with two lanes (`--spec dflash2
--draft-tokens 5 --max-concurrency 2 --kv-capacity 167936`) is the code-speed alternative
when two concurrent sessions are enough. Greedy DFlash2 text differs from MTP3 (the
verify path is not bit-exact), which the pass rate above does not penalize. Session
save/restore (`--slot-save-path`) does not support DFlash.

## Concurrency

Four lanes fit next to 256K context, MTP3 and vision with `--device-state-slots 2` (the
default of 4 extra device state slots is about 100 MB short). The KV page pool is shared, so
a single session still reaches the full context. On the code benchmark four lanes give 210 tok/s
aggregate against 144 tok/s for one; on 300-token essays about 145-174 tok/s aggregate. An
external API call waiting on the engine lock (memory summary, cancellation) yields to the
worker within 50 ms instead of starving behind back-to-back decode rounds.

## Second GPU as context-cache tier

`--host-tier-device N` puts the Host StateImage and Host KV tiers in the VRAM of an idle GPU
instead of pinned RAM (Single placement only). Copies use `cudaMemcpyDefault` and work without
P2P; the driver stages through host memory.

| Path | Bandwidth |
|---|---:|
| tier GPU -> 4090 (restore) | 12.3 GB/s |
| pinned RAM -> 4090 | 13.4 GB/s |

Qualification (`tools/tuning/host_tier_qualify.py`): four ~7.3K-token conversations on a 16K
KV pool force demotion; resuming restores 7,241-7,359 cached tokens in 0.46-0.54 s against
about 2.5 s of re-prefill, with identical greedy text and cache hits for the pinned and GPU
tiers. With `--host-kv-mib 8192 --host-state-slots 8` on the 5070 Ti, pinned host memory
drops from 9.2 GiB to near zero.

## Concurrent long agent sessions

Production showed agents of 150-230K tokens re-prefilling from 0% cache every turn (2-5 minutes
each) while other agents queued. `tools/tuning/cache_pressure_repro.py` reproduces it: several
long conversations run concurrently, each appending about 1.5K tokens per turn, with the
agents' `max_tokens` of 32,768. Two mechanisms were found:

- **Planner time budget.** Fitting a large continuation searches for which cached conversations
  to demote to the Host tier. The search had 10 ms when other requests were runnable (50 ms
  otherwise, first grant 5 ms); when it ran out it fell back to evicting every cached
  conversation (request log: `search_budget_exhaustions`, `maximal_fallback_selections`,
  `private_owners_evicted`). A lost cache costs 30-280 s of re-prefill, so the limits are now
  150 ms / 400 ms with a 50 ms first grant.
- **Whole-pool requests.** A request reserves prompt + `max_tokens` (capped at the pool), so a
  230K prompt with `max_tokens` 32,768 needs the entire 262K device pool and every other
  conversation must leave the device at once. No tier size or budget kept the cache in that
  case; capping the client context (128K in opencode) avoids it.

Five sessions of 120/110/90/60/30K tokens, four turns each, production flags with
`--host-kv-mib 8192 --host-state-slots 8`:

| Build | Resumes at 0% cache | Median hit | Wall time |
|---|---:|---:|---:|
| 10 ms / 50 ms budget | 4 of 15 | 97.6% | 287 s |
| 150 ms / 400 ms budget | 0 of 15 | 98.5% | 190 s |

Greedy outputs and cache hits on the Host tier qualification are identical to the previous
build. Splitting demotions into 16K-token extents (against Host arena fragmentation) did not
help and was not kept. The run filled all 8 Host state slots while the Host KV stayed under
6 GiB, so production uses `--host-state-slots 16 --host-kv-mib 10240`.

## Multi-GPU compute

Pipeline (PP2) and tensor (TP2) parallelism across the 4090 and 5070 Ti are covered in
[multi-gpu.md](multi-gpu.md). Without MTP, TP2 decodes 66 tok/s against 50.5 on the 4090
alone, but production MTP3 on one 4090 (124-145 tok/s) remains faster, and the second GPU is
more useful as the cache tier.
