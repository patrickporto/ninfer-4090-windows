# Multi-GPU: pipeline (PP2) and tensor (TP2) parallelism

Qwen3.8-27B can run across two GPUs, including mixed architectures such as an RTX 4090
(sm_89) with an RTX 5070 Ti (sm_120). The GPUs do not need P2P; activations and reductions
go through pinned host memory.

| Mode | Flag | Artifact | Where it helps |
|---|---|---|---|
| Single | (default) | `qwen3_8_27b.ninfer` | One GPU holds the model; fastest with MTP |
| Pipeline (PP2) | `--parallel pipeline` | `qwen3_8_27b.ninfer` | Model or context does not fit one GPU |
| Tensor (TP2) | `--parallel tensor` | `qwen3_8_27b.tp2.ninfer` | Faster plain decode (no MTP yet) |

```sh
# PP2: layers [0,36) on GPU 0, [36,64) on GPU 1
ninfer-serve qwen3_8_27b.ninfer --parallel pipeline --devices 0,1 --pipeline-split 36
# TP2: both GPUs compute every layer; needs a NCCL build (Linux release) and the TP2 artifact
ninfer-serve qwen3_8_27b.tp2.ninfer --parallel tensor --devices 0,1 --kv-dtype rk4v4-e8
```

Rank 0 (the first device) owns embedding, LM head and sampling. Build both architectures with
`-DCMAKE_CUDA_ARCHITECTURES="89-real;120a-real"`. TP2 needs `-DNINFER_ENABLE_NCCL=ON`
(Linux only; Windows builds offer Single and Pipeline).

## Current limits

- TP2 is eager text generation: MTP, vision, CUDA graphs, context cache and checkpoints are
  rejected at startup. KV storage must be `bf16` or `rk4v4-e8`; prefill chunk at most 1024.
- PP2 does not yet support MTP or vision either.
- Decode batches above 8 rows fall back from the A16 to the A8 local profile in TP2.

## Measured (RTX 4090 + RTX 5070 Ti, PCIe without P2P)

4000-token prompt, 256 generated tokens, one request:

| Configuration | Prefill tok/s | Decode tok/s |
|---|---|---|
| Single, graphs, rk4v4-e8, MTP3 (production) | 3720 | 124 |
| Single, no MTP | 3746 | 50.5 |
| PP2, no MTP | 2521 | 46.1 |
| TP2, no MTP | 1700 | 66 |

TP2 decode spends about 11 ms per token in rank-local compute and about 2 ms in the 128 FP32
all-reduces. `NINFER_TP_PROFILE=1` prints a per-rank breakdown every 64 passes.

## Numerical contract

Every rank-local Op is qualified against an FP64 oracle on sm_89 and sm_120. Model level:
PP2 must match Single in aggregate NLL (mean |dNLL| <= 0.01) and in greedy generation on
margin-certified fixtures. TP2 changes the kernel route of most ops from layer 0, so it is held
to Single's own route-change envelope (Single A16 vs Single A8 on the same fixture) and may not
regress aggregate NLL. See `tests/parallel/`.

## Second GPU as context-cache tier

In Single placement an idle second GPU can hold the context cache's Host tier (Host
StateImages and Host KV) instead of pinned RAM:

```bash
CUDA_VISIBLE_DEVICES=<4090>,<5070ti> ninfer-serve model.ninfer --host-tier-device 1 \
    --host-kv-mib 8192 --host-state-slots 8 ...
```

Demotion and restore copies use `cudaMemcpyDefault`, so they work without P2P (the driver stages
through host memory). Measured on RTX 4090 + RTX 5070 Ti: tier-to-compute copies run at
12.3 GB/s versus 13.4 GB/s from pinned RAM, and resuming a 7.2K-token conversation from the tier
takes 0.46-0.54 s versus 2.5 s of re-prefill. Greedy outputs and cache hits match the pinned tier
(`--host-tier-device` requires Single placement and a device other than `--device`).
