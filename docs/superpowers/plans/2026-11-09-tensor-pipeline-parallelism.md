# Tensor and Pipeline Parallelism Implementation Plan

> **For agentic workers:** Execute the milestones in order with the `executing-plans` skill; use `subagent-driven-development` only if delegation is authorized. Checkboxes describe future implementation, not work completed in this planning session. Do not commit without an explicit request.

**Goal:** Add correct, measured, single-node PP2 and TP2 inference for the registered Qwen3.8-27B groupwise-int target, preserving the public Engine, state transactions, and serving semantics.

**Architecture:** Keep one Engine coordinator and one logical Program; introduce explicit per-device execution shards, resource reservations, and communication owned below the Engine scheduling boundary. Deliver contiguous layer pipeline placement first, then head/channel tensor sharding; retain the existing single-device execution route as the default and reference product route.

**Tech Stack:** C++/CUDA, existing `.ninfer` artifacts and packed integer kernels, optional NCCL for TP collectives, CMake/CTest, independent CPU FP64 mathematical and exact codec oracles.

---

## Status, scope, and recommendation

The planning phase is complete. The user subsequently authorized implementation after the current fork qualification. No GPU probing, conversion or multi-device implementation was performed during planning. It was prepared against the local `integrate/windows-improvements` workspace on 2026-11-09. That workspace is changing concurrently; re-read the named contracts and final kernel interfaces before implementation. The Ada projection, attention-launch and device-scheduling experiments were subsequently rejected by whole-inference measurements and restored to the previous implementation. Preserve the selected baseline, existing PDF/JSON work, and retained NLL scoring/export; do not reapply rejected CUDA experiments as a prerequisite.

Reported deployment context: RTX 4090 24 GB (SM89), Ryzen 9950X3D, 32 GB host RAM, Manjaro/CUDA 13.2; Qwen3.8-27B groupwise artifact approximately 18.2 GB; service on port 11434 using 256K context, E8 rk4v4 KV, MTP3 with lm-head drafting, PDF/Vision, and XGrammar JSON output. The RTX 5070 Ti 16 GB (SM120) is a possible **future** second device, explicitly excluded from current deployment optimization. These are supplied context, not measurements made for this plan.

Recommendation: implement **PP2 for capacity first**, with a measured contiguous layer split and explicit transfer fallback. Qualify TP2 next on two real devices, preferably a matched pair for the first performance comparison; test the heterogeneous pair separately. Do not buy hardware or promise tokens/s based on this design. At concurrency one, PP adds an activation transfer and executes stages sequentially; its primary benefit is distributing residency. TP can reduce weight-read and compute work per rank, but adds roughly two reductions per decoder layer and waits for the slower rank. A mixed 4090/5070 Ti pair with unverified PCIe/P2P behavior is not an established latency improvement.

| Mode | Distributed state | Communication in a decoder pass | Initial use |
|---|---|---|---|
| Single GPU | All weights, KV, GDN local | None | Default; measure current optimized baseline |
| PP2 | Whole contiguous physical layers per stage | One hidden-state boundary transfer plus token/control return | More capacity; eventual throughput across independent rows/chunks |
| TP2 | Projection channels, heads, KV heads, GDN state heads | Two hidden-width SUM reductions per layer; input/control broadcast | Potential latency/prefill gain after topology and kernel qualification |
| TP×PP | TP groups assigned to pipeline stages | Both costs | Future ≥4-device extension, not a two-device MVP |

Do not include MoE expert parallelism, multi-node execution, dynamic placement, preemption, or a generic model graph. Initially reject unsupported registered targets/formats in multi-GPU mode with a precise startup error. Single-GPU support remains unchanged. NVFP4, DFlash/DFlash2, and MoE multi-GPU support require separate qualification and are explicitly outside initial delivery.

## Evidence and current ownership

Read these authorities before implementation: `AGENTS.md`, `docs/maintainer/engine-architecture.md`, `resource-scheduling-and-context-cache.md`, `paged-kv-cache.md`, `storage-layouts.md`, `tensor-formats.md`, `op-development.md`, `qwen3.6-27b-model.md`, and `qwen3.8-27b-artifact.md` in the same maintainer directory.

| Existing source | Actual assumption and required evolution |
|---|---|
| `include/ninfer/types.h` (`EngineOptions`) | One `int device`; memory summaries are aggregate. Add explicit placement and per-device reporting. |
| `src/runtime/engine/engine.cpp` | `initialize_device` and Engine implementation own one `DeviceContext`. Construct the device group before target materialization. |
| `src/core/device.h`, `device.cu` | Each context owns a device, compute/transfer streams, and device properties. Preserve its authority; every rank must use its own `DeviceExecutionView`. |
| `src/core/arena.h`, `arena.cu`, `decode_graph.h` | Allocations/graphs rely on current CUDA context; allocation ownership lacks an explicit device in the public primitive. Add owner-device lifetime and stream validation. Audit events/timers during destruction on another host thread. |
| `src/artifact/materializer.h`, `materializer.cpp` | A `MaterializedArtifact` has one arena and one device pointer per object. Add device-qualified physical bindings, without moving model semantics into artifact framing. |
| `src/targets/qwen3_6_27b/impl/load/bindings.*`, `variant.*`, `config.h` | Exact package owns storage and execution leaves; it must own the registered distributed weight/head bindings. |
| `src/targets/qwen3_6/impl/runtime/program.h`, `program_impl.h`, `schedule.h`, `instance.h` | Program/schedule carry one context and monolithic workspace/state. Keep one transaction owner, give it explicit physical shards. |
| `src/targets/qwen3_6/impl/runtime/text_context_impl.h` | `run_layers` traverses every physical layer; attention/GDN/output and post-mixer calls assume global geometry. Introduce bounded stage ranges and separately typed local TP geometry. |
| `src/targets/qwen3_6/impl/runtime/workspace_recipe.h`, `layouts_impl.h`, `text_context.h` | Compile-time full dimensions size projections/workspaces. Compute local geometry without accidentally retaining full-sized replicated buffers. |
| `src/targets/qwen3_6/impl/runtime/logical_kv_store.h`, `state_image_store.h`, `resource_projection.h`, `pressure_planner.h` | Resource feasibility, checkpoint and physical KV ownership must become device vectors while keeping logical lanes separate from compact rows. |
| `src/runtime/engine/engine_core.h` | Sole coordinator for accepted-prefix model commit, Frontend commit, output publication and failure. Retain that ordering across all ranks. |
| `src/targets/qwen3_6/impl/runtime/mtp_impl.h`, `decode_impl.h`, `graph_impl.h`, `session_snapshot_impl.h` | MTP rollback, graph profiles and snapshot frontiers currently cover one device. Extend to complete rank sets. |
| `src/targets/qwen3_6/impl/frontend/frontend.cpp` | XGrammar matcher/mask and output preview/commit stay single-owner; ranks do not maintain competing grammar state. |
| `src/product/document/pdf.cpp`, `src/serve/document_input.cpp` | PDF acquisition/Poppler remains product-side. Transfer model-ready vision activations, never move document handling into GPU workers. |

The architecture currently explicitly scopes execution to one GPU. Multi-GPU is a requested product extension: update its active authority alongside implementation, rather than treating a new placement helper as permission to bypass admission or publication contracts.

## Implementation evidence (2026-11-09)

- M0 placement and independent FP64/codec references are implemented and reviewed. Both CPU executables have no CUDA runtime link dependency.
- M1 device ownership, bounded copy/fallback and executor queues passed nine focused tests on the explicitly selected 4090/5070 Ti pair, including post-submit failure cleanup, blocked-producer failure and destructor FIFO drain. Spec and code-quality reviews passed after fixing default-stream identity validation before device binding. Engine ownership of the group remains part of M2 integration.
- The topology probe passed all twenty directional/size/policy cases. CUDA reports no directional P2P access on this pair; the selected path is pinned host staging. PCIe negotiated widths were x8 (4090) and x4 (5070 Ti); speed snapshots and raw timing distributions are in `local-test/parallel/topology.json`. These measurements include host enqueue, synchronization and possible clock/activity effects; they are not distributed inference benchmarks.
- M2 selective whole-object loading is implemented and reviewed. Independent synthetic ownership tests cover all 63 cuts, fused/split groupwise bindings and malformed plans. The opt-in real-artifact test passed stage-0 loading on the 4090, stage-1 loading on the 5070 Ti, rejection of missing owned data and unchanged full-model loading. Sparse views retain global weight indices; state execution and reservation integration is still pending.
- A separate mixed build targets `89-real;120a-real`, retaining the groupwise/A16 compatibility routes. A standalone FP8 MMA smoke test executed cubins on both GPUs with PTX JIT disabled. Full Ops compilation and model qualification are pending. Generic `120` cannot assemble the existing architecture-specific FP8 instruction.
- Production runs from the frozen `runtime-current` release on the 4090. No partial parallel runtime has been deployed. Loading tests briefly stopped the service and restarted that same verified release automatically.

## Mathematical and storage design

### Fixed geometry and valid partitions

The package `config.h` defines 64 physical decoder layers, hidden width 5120, MLP width 17408, 24 query heads, 4 KV heads, head dimension 256, and rotary dimension 64. Full attention occupies physical layers 3,7,…,63 (zero-based); the other 48 layers use GDN. GDN has 16 Q/K heads and 48 V heads of dimension 128; each Q/K head serves three V heads. Use physical layer numbers for PP and explicit local lookup tables for full-attention/GDN indices; never infer a local GDN index from a stage-relative layer number.

For TP2 each rank owns 12 Q heads, 2 KV heads, 8 GDN Q/K heads, 24 GDN V heads, and 8704 MLP channels. Partition complete GQA groups (6 Q per KV) and complete GDN groups (3 V per Q/K); Q/K normalization and RoPE stay inside a complete head. TP4 is geometrically plausible but not part of the MVP. Reject TP3 and TP>4 initially instead of silently replicating KV or unevenly slicing heads. `5120`, `6144`, and `17408` have TP2 partitions aligned to K128; use actual shapes and storage validation rather than a general claim that every tensor is splittable.

### TP equations and collective placement

Use column vectors below; weights have logical `[N,K]` storage. Hidden `x` is replicated on TP ranks at each sublayer boundary.

```text
MLP rank r:
  u_r = W_up[r_rows, :] x
  g_r = W_gate[r_rows, :] x
  z_r = silu(g_r) * u_r
  p_r = W_down[:, r_columns] z_r
  y   = x + SUM_r p_r

Attention rank r:
  Q_r, K_r, V_r, gate_r = projections of the assigned complete heads
  a_r = gated_attention(Q_r, K_r, V_r, local_KV)
  p_r = W_output[:, r_head_columns] a_r
  y   = x + SUM_r p_r

GDN rank r:
  q_r,k_r,v_r,z_r,a_r,b_r = projections for complete assigned head groups
  local convolution -> local recurrent delta-rule state -> headwise gated norm
  p_r = W_output[:, r_value_columns] local_output
  y   = x + SUM_r p_r
```

Column-parallel input projections produce locally consumable channels: **no all-gather** between gate/up and down, or between head-local attention/GDN and output. Row-parallel output/down projections produce a full hidden-width partial output: **SUM all-reduce** it before the following global hidden RMSNorm or sublayer. Add the residual exactly once after reduction. Existing `linear_add` fusions must not add replicated residual on every rank and then reduce, which would multiply it by TP degree. Bias, if present in any future supported leaf, follows the same once-only rule. If an Op stores BF16 output, define the distributed semantic boundary explicitly: FP32 local partials, FP32 SUM, then the declared BF16 output/residual boundary; do not silently introduce BF16 rounding of each partial.

RMSNorm over replicated 5120-wide hidden requires no collective. Headwise Q/K and GDN output norm remain local. Sharding hidden across ranks would require additional norm reductions and layout changes; exclude sequence parallelism from this plan.

There are 128 hidden-width reductions in one ordinary pass through 64 layers (mixer plus MLP), excluding embedding/head/control and speculation. For B=1, a FP32 hidden reduction payload is 5120×4=20,480 bytes; latency at that message size matters more than a large-message bandwidth headline. Prefill/verify uses `T×5120×4` bytes per reduction. Account for collective algorithm traffic separately from payload size.

For MVP, run embedding, final norm/full output head, sampling and MTP on one designated owner (last PP stage / TP rank 0), and broadcast the resulting input hidden or accepted token/control as needed. Do not replicate a full output head merely for convenience: inventory its owner memory explicitly. Later vocabulary sharding may all-gather full logits in vocabulary order for exact existing sampling, or implement mathematically complete global logsumexp/top-p. Local argmax/top-k alone is insufficient for general sampling, target log probabilities, grammar masks and MTP acceptance. Padded vocabulary rows remain unsampleable. Do not add vocabulary sharding until head residency or measured head time justifies it.

### Packed weight preservation

The current groupwise artifact uses `row-split-k128-v1`; Q4/Q5 use G64 scales and W8 uses G32. Q5 has distinct base, high-bit and FP16 scale planes; fused projections can have noncontiguous semantic row segments. A raw byte cut of an object is not a legal tensor shard.

PP uses whole existing objects and views on the owning stage: no new artifact format is necessary. TP output-row slices can copy independent plane spans while preserving original scale words. TP input-column slices of `down` and output projections need a compact K-shard layout, with whole quantization groups, new local K padding and correct plane offsets. Generate these **offline** from the existing packed words with an exact codec transform; do not dequantize/requantize or perform runtime weight repacking. Treat fused `[gate,up]`, Q/K/gate/V and GDN Q/K/V/Z segments explicitly, preserving aliases.

Proposed artifact extension for TP: one `.ninfer` container retains the existing registered model/weight identity and adds a versioned, package-validated placement inventory for degree 2. Each physical weight shard declares source semantic object, rank, logical row/column selection, local shape/layout, and exact payload. Nonsharded objects declare their owner. The converter reads the selected source `.ninfer`, emits all required shards once, and compares their independently decoded values against the exact selections from the source. Reject missing/overlapping shards, illegal group boundaries, conflicting aliases, wrong geometry, unsupported versions, and inconsistent degree before GPU allocation. Avoid an external JSON manifest as a second product artifact; `.ninfer` remains the product entry. Metadata provenance records the source identity/digest and partition recipe. Do not carry both a complete unsharded weight copy and all TP shards in device memory.

## Execution, resources, and failure contracts

Use one in-process Engine coordinator and one fixed executor thread per selected physical device for eager PP/TP. Only the coordinator mutates request state, logical Program transactions, ResourceManager and Frontend. Executors consume immutable commands carrying monotonic unit ID, rank, physical layer range, compact row identities, positions, and expected collective sequence; they own their CUDA context and execute physical work. Program owns all shard allocations and lifetimes, with no independently scheduled nested Engines.

Proposed public configuration, replacing the old scalar `device` across owned callers in the same change:

```cpp
struct ParallelOptions {
    std::vector<int> devices{0}; // unique CUDA ordinals, flattened [pp_stage][tp_rank]
    std::uint32_t tensor_degree = 1;
    std::uint32_t pipeline_degree = 1;
    std::vector<std::uint32_t> pipeline_layer_ends; // exclusive; final element 64
};
// EngineOptions::parallel; existing request/sampling API stays unchanged.
```

Proposed CLI/serve options: `--devices 0,1 --tensor-parallel 1 --pipeline-parallel 2 --pipeline-layer-ends 36,64`; the cut is illustrative, not a recommended placement. `--devices 0,1 --tensor-parallel 2 --pipeline-parallel 1` selects TP2 and requires the TP2 artifact. Device count must equal TP×PP; reject duplicate ordinals. Default `{0},1,1` preserves single-GPU operation. During MVP accept only (1,1), (1,2), (2,1); reject (2,2) until hybrid is qualified. Do not keep competing `--device` aliases after migrating project-owned callers, following repository policy. Report UUID, ordinal, SM, stage/head ranges, measured transport path, allocations and context limits in startup summaries, not user chat output.

Startup checks: validate artifact and placement first; obtain per-device capability/available memory; verify compiled kernel support for every assigned device and dtype (SM120 does not make NVFP4 legal on SM89); validate layer coverage, all shard shapes, communication availability, rank order and buffer capacities; initialize communication and scratch resources before graphs; prove full per-device active reservation before accepting requests. Memory feasibility is a vector inequality, not `sum(free VRAM) >= model bytes`:

```text
for every device d:
  weights[d] + state[d] + kv_reservation[d] + workspace_peak[d]
  + transfer_buffers[d] + graph_allowance[d] + communication_scratch[d]
  + startup/runtime_margin[d] <= usable_vram[d]
```

Include MTP backend KV, GDN snapshots/replay, prefix replicas, vision peak, output-head/logit staging and exact B=1..8 profile resources. PP partitions KV by owned full layers and GDN by owned recurrent layers; TP partitions KV by head and recurrent state by complete head groups. E8 block/codec alignment must remain head-local and be qualified with local head counts. One request's context ceiling is the minimum feasible frontier over its ranks, not the sum of their token capacities. Host spill and checkpoint staging share a bounded host-memory budget; 32 GB RAM rules out blindly holding whole model copies per rank.

Reserve/prepare all shards first; if any preparation fails, unwind all prepared reservations without publishing. A logical checkpoint is valid only when every shard reports the same accepted frontier. One `PendingBatch` covers all physical stages/ranks; an accepted-prefix decision, RNG state, MTP decision and grammar progress are authoritative only after the entire unit is committed. Saved slots need a versioned complete shard inventory or a canonical reassembled representation; initially require matching placement on restore and reject mismatches before overwriting a live slot. Placement-independent restore can be a later explicit codec task.

Cancellation is observed at stable unit boundaries. Never cancel one rank's collective while peers wait for it: finish the scheduled collective/unit, apply the same zero-accept cancellation decision to every shard, then release. A recoverable pre-dispatch validation failure creates no GPU work. An asynchronous CUDA/NCCL failure latches Engine unhealthy, stops admissions, fails all affected handles once, aborts all communicators, drains/joins executors when possible and destroys device resources on their owners. No partial checkpoint/output is published. The existing aborting `CUDA_CHECK` path must be considered explicitly: communication/executor paths need error-returning wrappers to report failure to Engine; do not claim graceful recovery while reachable branches call process abort. If a driver operation cannot be bounded/recovered in-process, mark the process unusable and require restart; never hang request completion indefinitely.

NCCL rank ordering is fixed for lifetime; each communicator has one executor owner. Concurrent groups use the same order on every rank, including inactive-row handling. Timeouts are communication-health failures, distinct from ordinary request cancellation. Independent communicator launches on the same GPU must be serialized. These rules follow NVIDIA's [communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html) and [thread ownership](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html) contracts.

## Pipeline schedule and heterogeneous placement

PP cuts only between complete decoder blocks. Prefer cuts at multiples of four as initial candidates to distribute full attention and GDN regularly, but evaluate all legal boundaries if measurements favor them. Enumerate cuts 1..63; reject each violating a per-device reservation, then minimize measured stage bottleneck for throughput or stage-sum plus boundary transfer for single-request latency. Include long-context attention cost, embedding/Vision cost, final head and MTP ownership; a simple split by layer count, VRAM ratio or GPU marketing FLOPS is insufficient. Compare both device orders and final-head placement.

Stage A sends the post-MLP hidden tensor, positions/MRoPE metadata and compact row map to stage B. Keep BF16 hidden boundaries unchanged. B returns selected token/control to the coordinator and next-token owner. At hidden width 5120 the main boundary payload is 10 KiB per token, or 10 MiB for T=1024, excluding metadata and extra speculative inputs. Preallocate bounded double buffers, use transfer-completion events, and never reuse a buffer until its consuming stage completes. For no-P2P systems support bounded pinned-host staging with explicit D2H completion before H2D and H2D completion before compute. This is a transfer mode, not a CPU model-inference path.

At concurrency one, decode token n+1 depends on the sampled result of token n; do not claim that two pipeline stages decode successive tokens concurrently. For p balanced stages and m independent microbatches, ideal fill/drain utilization is `m/(m+p-1)`; it is only a model and excludes imbalance, transfers and token feedback. m=1,p=2 gives 50% aggregate stage utilization. MTP verification can increase token width within a pass, but is not arbitrary independent future-token work.

First PP runs one complete existing execution unit through stages, preserving the scheduler. Later, subdivide one frozen compact decode batch into independent request-row microbatches; do not admit extra work mid-round or change FIFO fairness. For prompt chunk pipelining, preserve per-stage causal KV/GDN order: stage A may process chunk k+1 only after its own k state is ready, and B consumes k before k+1. Bound speculative state and chunk buffers. Benchmark B=1,2,4,8 and chunked prefill separately; more microbatches can lose GEMM efficiency and increase latency.

MTP MVP starts disabled only for the first internal milestone. Production multi-GPU readiness requires the user's MTP3 + lm-head draft route: keep draft layer/head and backend KV on the final/owner device; deliver required embedding inputs there and return draft tokens to target verification. All target stages verify identical candidate positions; all GDN replay and KV truncation use the same accepted prefix. Test first rejection, partial acceptance, all accepted, grammar-shortened nonterminal prefix, EOS/stop, cancelled row and forced-control suffix. Preserve RNG advancement exactly according to accepted semantics, not rank count. DFlash2's taps at physical layers 5,19,33,47,61 would need additional feature transfers and local-context transactions; reject that backend for initial multi-GPU mode instead of assuming MTP support covers it.

Vision runs on the first stage for PP and an explicit owner for TP; scatter final visual embeddings into the logical hidden stream before distribution. Include its transient VRAM in placement and preserve MRoPE coordinates and text/vision token accounting. PDF ingestion remains on CPU/product side. XGrammar masks and sampling stay on the designated logits owner, and accepted preview/commit remains in the existing Engine/Frontend path. No change to OpenAI/Anthropic response schemas is required solely to expose device placement.

## CUDA Graph policy

First PP/TP milestones use eager execution, explicitly rejecting graph=true for multi-GPU until graph support is qualified. Do not silently ignore the flag. Single-GPU graph behavior stays intact.

Next, capture local **compute-only** graph segments per device, with transfers/collectives outside captured segments and ordered by explicit events. Prebuild exact batch/verify/local-head profiles and stable buffers; graph keys include placement, rank, local shape, KV mode and speculative mode. Do not copy a graph instantiated on one device onto another. Validate eager/graph state and output equivalence at all supported profile boundaries.

Capturing NCCL inside graphs is a separate optimization, not required to ship TP. NVIDIA currently warns of multi-GPU-per-process graph deadlocks, particularly single-thread launch; captured operation and replay participation must be uniform across ranks. Prefer one GPU per process if full collective capture becomes necessary; that would add an internal process transport and failure-supervision milestone. Do not assume per-rank threads alone eliminate all graph hazards or enable collective capture by default. Source: [NCCL CUDA Graph requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html).

## Qualification and future topology measurements

These commands are a future procedure, **not executed for this plan**, and must run in a maintenance window after selecting devices explicitly. Do not stop the current service merely to populate this plan.

```sh
rtk nvidia-smi -L
rtk nvidia-smi topo -m
rtk nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version --format=csv
rtk proxy lspci -tv
```

Add `tools/parallel_topology_probe.cu` and target `ninfer_parallel_topology_probe`; future invocation: `rtk ./build/ninfer_parallel_topology_probe --devices 0,1 --json profiles/parallel-topology.json`. It must query both directions of `cudaDeviceCanAccessPeer`, actual negotiated PCIe width/speed (using supported system queries), CUDA/driver/NCCL versions, and available memory; measure event-ordered peer-copy and pinned-host paths at 10 KiB, 20 KiB, 160 KiB, 10 MiB and 20 MiB. Test with peer access unavailable and report actual chosen transport. NCCL benchmark representative FP32 all-reduce payloads, startup and tail latency, not only peak GB/s. No NVLink or working P2P is assumed for a 4090. CUDA's [peer-access API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html) requires an explicit capability query and directional enablement. Distinguish a CUDA capability bit from an empirically usable/faster route; do not change IOMMU/ACS/BIOS configuration as part of discovery.

NCCL send/receive calls require matching count/type and compatible progress grouping; use that if adopting NCCL PP transport later, following [NVIDIA point-to-point semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html). PP's initial CUDA-copy transport avoids making NCCL mandatory for users who only need layer placement.

Numerical qualification uses independently decoded stored scales/codes and naive FP64 dense/attention/GDN equations, not agreement with another GPU kernel. Exact tests cover shard coverage, aliases, bit planes, BF16 transfer bits, positions, compact-row mapping and logical frontier/commit decisions. Use signed alternating and cancellation-heavy data, zeros, tails, maximal finite representable test values and G64/K128 boundaries. Test T=1,2,16,17,128,1024 and B=1,2,4,8; include actual model dimensions and at least one complete 3-GDN/1-attention cycle.

Freeze numeric limits before accepting optimized implementations. Use existing Op-specific limits from the relevant tests as the starting authority; never relax them merely to make TP pass. For new FP32 reduction primitives, compare against FP64 using a componentwise absolute scale bound `gamma_n * sum(abs(partials))`, where `gamma_n = n*u/(1-n*u)` and u is FP32 unit roundoff, with a documented small operation-count multiplier for the chosen algorithm. Include near-zero cancellation cases; relative error alone is invalid. For full quantized projections, separate existing local-kernel error from reduction/cast error and qualify the final declared output directly against FP64. BF16 boundary comparison uses the existing Op contract, with both absolute/RMS error reported; no universal end-to-end token equality requirement under changed reduction order.

**Cross-architecture precision contract (approved by the user, 2026-11-09).** Mixed SM89/SM120 placements keep every Op bound to its frozen independent FP64 budget, but model-level agreement with Single is judged in aggregate rather than per token. BF16/TF32 tensor-core accumulation order is implementation-defined across architectures, and diagnosis showed 1-ULP BF16 KKT differences amplified by TF32 operand rounding inside GDN even when each GDN route passes its FP64 oracle. Frozen limits for PP/TP vs Single on the fixed 1,100-token corpus: mean |dNLL| <= 0.01 nats, |mean signed dNLL| <= 0.005 nats (perplexity ratio within about 0.5%), RMS <= 0.05, at most 0.5% of tokens beyond 0.25 nats, and a hard per-token cap of 1.0 nats; nonfinite scores fail. Greedy equality remains required on margin-certified fixtures. Same-architecture partition tests remain bit-exact.

Model-level qualification: compare per-token log probabilities/NLL, perplexity and final hidden/logit errors on the same fixed token corpus, tokenizer, context windows and precision settings; define corpus tolerances from baseline repeatability plus the validated Op error budget before optimization. Greedy token equality is required on deterministic fixtures with a clear logit margin, not on arbitrarily close logits. Stochastic tests check seed/RNG ownership and acceptance distributions/semantics rather than expecting universal cross-topology bit identity. JSON must parse and satisfy the supplied schema; malformed schemas, cancellation and forced stops must preserve error contracts. Numerical testing must include E8 rk4v4 local-head cache append/read, long-context boundaries and MTP replay. A single ordinary text completion is insufficient evidence.

## Implementation milestones and reviewable tasks

All files listed as **new** are proposals, not existing APIs. Register tests with the established `ninfer_add_test` / `ninfer_add_op_test` helpers in `tests/CMakeLists.txt`; missing hardware must return 77 rather than pretend to pass. The CPU oracle executable must run without CUDA. New test executable names below are deliberate contracts for implementation.

### M0 — Freeze scope, oracle and placement contract

Files: modify `include/ninfer/types.h`, `docs/maintainer/engine-architecture.md`; new `src/targets/qwen3_6_27b/impl/parallel_plan.h`, `parallel_plan.cpp`, `tests/parallel/test_plan.cpp`, `tests/parallel/test_math_oracle.cpp`.

- [x] Add CPU-only failing tests for duplicate devices, unsupported degree/profile, noncovering/overlapping physical layer ranges and TP2 head mappings. Assert that stage [36,64) contains 7 full-attention and 21 GDN layers, and TP2 maps each KV head to exactly six Q heads.
- [x] Implement immutable package placement descriptors with half-open ranges and typed local/global head indices; validate before allocation.
- [x] Add naive CPU FP64 reference for the three equations above, independently decode packed source values, and compare emulated TP SUM and PP layer composition to the unsplit equations. Test a residual of 7 with rank partials 2 and 3 yields 12, not 19.
- [x] Record numerical budgets and unsupported backend errors; update the single-GPU authority to describe the explicitly scoped extension.
- [x] Run `rtk ctest --test-dir build -R '^ninfer_parallel_(plan|math_oracle)_test$' --output-on-failure`; expect passing CPU tests. At this milestone no GPU is required and no performance claim is valid.

### M1 — Device lifetime and communication foundation

Files: modify `src/core/device.*`, `arena.*`, `decode_graph.*`, `src/runtime/engine/engine.cpp`, `src/CMakeLists.txt`; new `src/core/device_group.h`, `device_group.cpp`, `peer_transfer.h`, `peer_transfer.cu`, `src/targets/qwen3_6/impl/runtime/rank_executor.h`, `rank_executor.cpp`, `tests/parallel/test_device_group.cpp`, `test_peer_transfer.cpp`.

- [x] Write owner-device lifetime tests: allocate on device A, switch current device to B, move/destroy buffer/event/graph, and verify correct cleanup; test asynchronous transfer dependency and buffer reuse.
- [x] Implement fixed device group and command queues with one context per executor. Preserve coordinator-only logical mutation and bind every allocation/destruction to its owner.
- [x] Implement bounded peer copy and pinned fallback with exact byte preservation, explicit directional capabilities and endpoint completion. Inject a failed endpoint and prove both buffers are released once.
- [x] Add optional NCCL discovery/linking behind a build option for TP, with a clear startup error if TP is requested in a build without it. Never download/install NCCL from runtime code.
- [x] Run `rtk ctest --test-dir build -R '^ninfer_parallel_(device_group|peer_transfer)_test$' --output-on-failure` on two **distinct real GPUs** later. On one GPU, skip the hardware tests. Two virtual ranks on one GPU can test a CPU/mock scheduler but must never join one NCCL communicator; NVIDIA explicitly does not support duplicate-device ranks.

### M2 — PP selective loading and eager text/scoring

Files: modify `src/artifact/materializer.*`, `src/targets/qwen3_6_27b/impl/load/bindings.*`, family `program.h`, `program_impl.h`, `text_context_impl.h`, `workspace_recipe.h`, `layouts_impl.h`; new family `pipeline_schedule.h`, `pipeline_schedule.cpp`, `tests/parallel/test_pipeline.cpp`.

- [x] Write artifact tests proving each physical decoder weight is resident only on its stage and owner-only embedding/head/vision/MTP weights are counted explicitly; reject an invalid device budget before loading.
- [x] Implement selective whole-object PP materialization and local full/GDN index maps, preserving immutable target weight views and existing artifact identity.
- [x] Bound `run_layers` to a physical stage range, disable cross-stage next-projection hints, and transfer post-block hidden using stable buffers. Keep exact stage-local phase and KV positions.
- [x] Add the complete per-device reservation vector and joint prepare/abort transition; test a second-stage OOM after first-stage preparation leaves no active lane or leaked claim.
- [x] Qualify text prefill, ordinary decode and CausalScoring through public Engine, graphs/speculation disabled. Use deterministic mini-model oracle fixtures plus the real artifact on explicitly selected GPUs.
- [x] Run `rtk ctest --test-dir build -R '^ninfer_parallel_pipeline_test$' --output-on-failure`. Milestone exit: usable eager PP2 text and scoring, explicitly not full production feature parity.
  - 2026-11-09 exit evidence: `local-test/parallel/pipeline-engine-aggregate-direct.log` on 4090+5070 Ti passes the cross-architecture contract (1,099 tokens: mean |dNLL| 0.00202, signed 0.00065, RMS 0.0153, 1 token > 0.25, max 0.353), bit-exact repeated suffix window, 16 margin-certified greedy rows, cancellation, B=2..8 batching and second-rank fault rollback. GDN prefill T=1024 uses the existing direct recurrence; the SIMT T=1024 control profile was rejected (bit-identical controls across GPUs but model max 1.078 and +0.23 ms/layer).

### M3 — PP state transactions and current serving features

Files: family `logical_kv_store.h`, `state_image_store.h`, `resource_projection.h`, `pressure_planner.h`, `mtp_impl.h`, `session_snapshot_impl.h`, `vision_context_impl.h`; `src/runtime/engine/engine_core.h`; new `tests/parallel/test_transactions.cpp`, `test_engine_features.cpp`.

- [ ] Add fault-injection tests at transfer completion, provisional state, accepted-prefix prepare and resource release. Assert output publication occurs only after every shard commits and that a failed shard makes Engine unhealthy exactly once.
- [ ] Implement coordinated checkpoint/frontier, cancellation and slot save/restore inventory; include host spill limits and inactive prefix reuse.
- [ ] Bring up MTP3/lm-head owner route, exact accepted-prefix rollback of GDN/Main/backend KV, and grammar-shortened output. Test first rejection, all accepted, partial accepted, cancelled compact row, stop and forced control.
- [ ] Bring up Vision embedding transfer with original MRoPE; run PDF/text-image requests and JSON-schema generation via public Engine/serving contracts. Keep Poppler acquisition unchanged.
- [ ] Run `rtk ctest --test-dir build -R '^ninfer_parallel_(transactions|engine_features)_test$' --output-on-failure` plus affected existing frontend, PDF and schema tests. Exit: PP2 feature parity for the stated deployment in eager mode; unsupported DFlash2 is rejected at startup.

### M4 — Offline TP2 artifacts and local-shape Ops

Files: new `tools/convert/qwen3_8_27b/shard.py`, `tests/artifact/test_parallel_shards.py`; modify `tools/artifact/layouts.py`, package bindings and `docs/maintainer/qwen3.8-27b-artifact.md`; new `src/targets/qwen3_6_27b/impl/parallel_weights.h`; modify affected `src/ops/linear*`, `attn_input_proj`, `gdn_input_proj`, `gdn_gating_proj` and family workspace recipes only where local geometry requires it.

- [x] Write exact shard round-trip tests for Q4/Q5/W8 base/high/scales, row selection from fused semantic segments, and K128/G64 alignment. Assert independently decoded joined shards equal source decoded values exactly, including physical padding.
- [x] Implement offline packed-word slicing and one-container placement inventory; expose `python -m tools.convert.qwen3_8_27b.shard --input SOURCE.ninfer --output DESTINATION.ninfer --tensor-parallel 2`. The command must reject input/output equality and validate the full plan before creating output.
- [x] Bind local shapes in package-owned payloads without runtime repacking. Add separate non-residual FP32 partial-output Op contracts needed by row-parallel down/output projections.
- [x] Qualify attention 12Q/2KV, GDN 8QK/24V, MLP 8704 local geometry and local E8 cache against the independent oracle at real shapes and existing dispatch boundaries. Do not infer that global-shape kernels handle smaller heads automatically.
- [x] Run the shard unit module with the repository-selected Python 3.11 interpreter and `rtk ctest --test-dir build -R '^ninfer_parallel_local_ops_test$' --output-on-failure` after registering that executable. Exit: exact artifacts and independently correct local Op routes, without a distributed performance claim.
  - 2026-11-09 exit evidence: `ninfer_parallel_tensor_{mlp,gdn,attention,projection}_test` pass on both 4090 (sm_89) and 5070 Ti (sm_120); TP2 artifact conversion and independent oracle passed earlier (`tp2-conversion-real.log`, `tp2-oracle-real.log`). Local GDN controls (24 heads) use the existing per-row SIMT kernels templated on head count (GEMV T=1, fused norm T<=42, SmallTSplit10 otherwise); the 16-head MMA tiles cannot cover 24 heads. Local recurrence keeps the chunked route: a 10-seed sweep showed the qualified 16/48 route exceeds the frozen worst-element bound on 2/10 seeds (8/24: 4/10), always one element at <=1.16x the bound with relative L2 at ~92% of budget, so local GDN tests apply the approved isolated-outlier contract (<=1e-5 of elements, <=2x bound, relative L2 strict). Gated-norm and control fixtures were aligned with the authority test ranges.

### M5 — TP2 eager collectives, transaction parity and measured choice

Files: new `src/core/collectives.h`, `collectives.cpp`, family `tensor_parallel_schedule.h`, `tensor_parallel_schedule.cpp`, `tests/parallel/test_collectives.cpp`, `test_tensor_parallel.cpp`; modify package `variant.*`, family text schedule and workspace, runtime failure integration.

- [ ] Implement FP32 SUM collectives with fixed count/type/order and timeout/error propagation; test zero/alternating/cancellation-heavy partials against FP64 and detect divergent sequence IDs before dispatch.
- [ ] Wire the two reductions per layer, once-only residual and BF16 boundary; broadcast owner-produced embedding/token/control. Preserve owner-only sampling/grammar/RNG and full-head scoring semantics.
- [ ] Apply the same MTP, Vision, checkpoint, prefix, cancellation and failure tests used for PP with TP2 local state. Confirm no full KV/GDN state duplication hidden in the memory report.
- [ ] Run `rtk ctest --test-dir build -R '^ninfer_parallel_(collectives|tensor_parallel|transactions|engine_features)_test$' --output-on-failure` with TP2 placement explicitly set by the test fixture, then public Engine real-artifact NLL qualification.
- [ ] Benchmark the same workloads on single GPU, PP2 and TP2. Choose defaults from the results; retaining single GPU for low-latency concurrency-one inference is a valid outcome.
  - 2026-11-09 implementation (source only, not yet built or qualified): opt-in `ParallelMode::Tensor` through the public Engine for `qwen3.8-27b/groupwise-int-tp2`, eager text prefill/decode/CausalScoring. The PP two-rank unit (`pipeline_execution.h`) gains a TP mode reusing the rank-1 physical stage, paired transactions and quarantine; `tensor_execution.h` owns the NCCL unit (128 coordinator-prepared FP32 SUM tickets per pass, one per mixer and one per MLP, residual added once by the new exact `tensor_residual_add` BF16 boundary Op). Rank-local KV/GDN layouts use `TensorLocalConfig`; owner embedding is broadcast BF16, final norm/head/sampling stay on rank 0. New local `attn_input_proj` [3584,5120] AllowA8 profile (oracle case in `ninfer_parallel_tensor_attention_test`). Graphs, MTP, Vision, context cache, checkpoints, non-BF16/E8 KV and prefill_chunk > 1024 are rejected at startup; NCCL build required. Qualification target: `ninfer_parallel_tensor_engine_test`.

### M6 — Local graphs, pipeline overlap, public configuration and release

Files: family `graph_impl.h`, `decode_impl.h`, pipeline schedule; `apps/cli/options.cpp`, `src/serve/serve_options.*`, `include/ninfer/types.h`, project-owned benchmark/tool callers; `docs/cli.md`, `docs/serving.md`, active maintainer references; new `tests/parallel/test_graphs.cpp`, `test_pipeline_overlap.cpp`.

- [ ] Migrate all scalar-device callers and parsing to the proposed placement API, with negative tests for malformed lists and unsupported combinations. Report per-device memory and stage assignments at startup.
- [ ] Capture compute-only local graphs and order uncaptured communication explicitly. Test exact B/profile transitions, graph/eager outputs, and repeated teardown/startup. Keep collective capture disabled.
- [ ] Add within-round row microbatching and bounded ordered prefill chunk overlap; assert unchanged compact membership and FIFO/fairness behavior. Compare overlap enabled/disabled for B=1,2,4,8 and do not enable a losing policy automatically.
- [ ] Run the new graph/overlap tests and affected existing CLI/serve/public API tests; build affected targets with `rtk cmake --build build -j`. Run `rtk git diff --check` and update active docs with supported combinations, explicit limitations and measured scope.
- [ ] Exit: PP2/TP2 are opt-in, documented and qualified for the registered groupwise target; current single-GPU defaults and serving behavior remain valid. Full NCCL graph capture and TP×PP await distinct measured justification and qualification.

## Benchmark decision record and costs

For each mode record GPU UUID/order, driver/CUDA/NCCL, compiled SMs, chosen P2P or staged transport, artifact identity/partition recipe, prompt tokens, output limit, B, context/cache state, KV dtype, speculative settings and graph mode. Reuse the same tokenizer/input tokens, warmup policy and sampling settings. Measure cold startup separately from warmed inference; cache hits separately from misses. Cover short prompt, prefill 1K/8K/32K, and long-context decode up to the requested 256K only where reservation proves feasibility. Include ordinary and MTP3 and one representative PDF/Vision input. Record TTFT, median/p95 inter-token latency, completion latency, aggregate output tokens/s, prompt tokens/s, accepted draft tokens, per-device peak VRAM, host RSS/pinned memory, transfer/reduction time and stage idle time.

Use synchronized wall time through the public Engine for end-to-end claims; device events only attribute local phases. Report median and dispersion across repeats and stop when the alternatives can be distinguished; do not claim a kernel-only speedup as a serving speedup. Use actual service B=1 as the latency decision workload. A mode that merely fits a larger context is a capacity result, even if throughput falls. The smallest device, replicated owner resources, communication scratch and usable PCIe topology constrain the result; 24+16 GB does not equal one transparent 40 GB device.

Main engineering cost is not NCCL wiring: it is explicit multi-device state ownership, full-shape assumptions in fused kernels, exact packed artifact slicing, per-device admission proofs, rollback and graph qualification. PP limits changes to layer boundaries and offers the earliest reviewable result. TP adds new numerical boundaries and about 128 reductions/pass; heterogeneous compute and host-staged communication can erase its savings. Hybrid on two devices adds no useful independent topology beyond PP2 or TP2; postpone until there are enough devices and evidence.

Planning is complete and the user has now authorized implementation after qualification of the preceding features. Multi-device tests may use the two explicitly selected GPUs in the implementation phase; the default serving deployment remains the qualified single-GPU baseline until a replacement is qualified. No parallel speedup has been measured yet.
