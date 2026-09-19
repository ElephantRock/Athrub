# A1 FP32 shared-context scaling plan (Issue #15, protocol v0.1)

Frozen **before** any GPU execution. The machine-readable authoritative copy of
every constant is `configs/a1_fp32_scaling.v0.1.json`; this document is its
human-readable record. This campaign is the narrower FP32-only follow-up to the
Issue #11 disposition; Issue #3 retains the original FP32+BF16 scope
historically unchanged. BF16 is explicitly out of scope: plain shared BF16 is
UNSUPPORTED on this runtime under contract v0.2, and this plan measures only
the numerically validated FP32 shared-context path.

**Freeze status: fully frozen (all P0 review items transcribed verbatim).**

## Binding

| Item | Value |
|---|---|
| Precision | FP32 only |
| Frozen A0 oracle manifest | sha256 `fcd3cb6b3e56d747fa82b823353fa6eb893adface51ff2985190aa9f261834b5` |
| Attention policy | efficient_sdpa / expand_kv_heads / q_to_kv_ratio 2 / fallback disallowed |
| Frozen at commit | `b894981922e441b8e570f7be41d5ab79fec285d8` |

## Measurement grids

```text
K grid            2, 4, 8, 16, 32, 64, 128, 255
prefix grid       128, 512, 1024, 2048, 4096, 8192   (units; candidate_units = 16)
```

The 8192-prefix FP32 regime sits above the known WDDM-spill boundary observed
on this workstation; cell feasibility is established by the chunked feasibility
probe below, and an infeasible cell is recorded as infeasible, never silently
dropped.

## Chunking and feasibility (ratified policy)

Feasibility chunk candidates per cell: `1, 2, 4, 8, 16, 32, 64, 128`, plus a
full-K probe for K=255. A chunk size is **safe** iff all of:

```text
peak allocated <= 90% of physical CUDA VRAM
peak reserved  <= 95% of physical CUDA VRAM
no OOM or execution error
no observed WDDM/shared-memory spill
```

- Attribution cells run four paths: **flat-sequential, flat-batched,
  shared-sequential, shared-batched**, where both batched paths use the same
  chunk size `min(max_safe_flat, max_safe_shared, K)` established by the
  feasibility probe on that cell.
- The primary comparison (optimized/chunked flat vs chunked shared) lets each
  path use **its own largest safe chunk**.
- Shared chunking semantics: the prefix is computed **exactly once per
  request** and reused across all continuation chunks; candidate logits are
  restored to original candidate order; softmax is applied **once** over the
  full K-candidate set.

## Attribution matrix

```text
prefixes          128, 512, 1024
K                 2, 8, 32
paths             flat-sequential, flat-batched, shared-sequential, shared-batched
```

This separates ordinary batching improvements from shared-computation effects,
per the Phase 1 requirement that optimized-flat serve as a control.

## Execution order

```text
cell order seed   271828 (deterministic shuffled cell order)
path order        alternating flat/shared execution order
```

## Cadence

```text
primary cells     3 warmups / 10 repeats
anchor suite      10 warmups / 50 repeats
semantic suite    5 warmups / 20 repeats
```

## Anchor suite and substitution rule

```text
(p512, K32)   (p512, K128)
(p1024, K16)  (p1024, K64)
(p2048, K16)  (p2048, K64)
```

Substitution rule (verbatim from the Issue #15 review):

- If an anchor is resource-infeasible, substitute the largest lower K from the
  frozen K grid at the same prefix, chosen only from the resource preflight
  before timing.
- Do not duplicate an already-selected anchor at that prefix; step down again.
- If no unique lower-K substitute exists, mark that anchor unavailable.
- If fewer than 4 unique anchors remain feasible, the campaign may report
  measurements but may not issue a strong/conditional A2 performance verdict.

## Semantic confirmation suite

Uses the already-frozen Issue #11 holdout manifest
(sha256 `b7945ef0edc5b2e7d3c71d27bae0c46647dcce02ff612f5620d887bf81761632`):
exactly the **24 records whose metadata has `suite == "semantic"`** (they
follow the 18 synthetic rows in that manifest).

## Correctness gate (FP32) — scope: every feasible measured cell

Every measured path must satisfy, against the reference flat backend on its
cell:

```text
max |Δp| < 1e-5   and   exact argmax
```

**Performance from a path is admissible only if it preserves FP32 equivalence
on that cell** — this applies to every feasible measured cell, not only
attribution and anchor cells. A performance number produced by a failing path
is void. FP32 shared-context equivalence is already established (development
matrix 9/9; Issue #11 holdout 42/42); this gate re-verifies it for the
chunked/optimized execution forms before their timings are admitted.

## Primary comparison

```text
optimized/chunked flat  vs  chunked shared   (independent largest-safe chunk sizes)
```

Reference flat and unchunked shared remain diagnostic context.

## Metrics and artifacts

Authoritative metric set (verbatim from the Issue #15 review):

```text
latency:            p50, p90, p95, p99, mean
throughput:         requests/s, candidates/s, speedup
token/work:         exact tokenizer counts; flat logical token positions;
                    shared logical token positions; logical-reduction ratio
timing:             prefix timing where available; continuation timing where
                    available; model timing where available
memory/execution:   peak allocated VRAM; peak reserved VRAM; chunk size;
                    model-call counts
correctness:        max/mean absolute logit delta; max/mean absolute
                    probability delta; total variation; KL divergence;
                    argmax agreement
provenance:         Athrub execution commit; frozen A0 binding/hashes;
                    effective attention policy; environment; hardware;
                    driver/CUDA; PyTorch
```

Required interpretation clause:

```text
Logical token-position reduction is theoretical accounting, not measured FLOPs.
```

Artifacts follow the established pattern: provenance (execution git SHA, raw
manifest hashes, frozen A0 binding, environment, effective attention policy),
per-cell JSONL with full measurement records, feasibility-probe records, and a
summary. Nothing in this campaign may be quoted as an A1 correctness result,
and BF16 rows do not exist.

## A2 interpretation rules (verbatim from the Issue #15 review)

Historical Phase-1 gate:

```text
correctness
AND
(
    >= 2x effective throughput
    OR
    >= 50% independently measured computational-work reduction
)
with acceptable VRAM.
```

The `>=50%` path may be invoked **only if profiler/kernel evidence supplies an
independent measured work proxy. Logical-position reduction alone does not
qualify.**

Compute the **geometric-mean candidates/s speedup across the final unique
anchor set**, then interpret:

```text
>= 2.0x        strong
1.5x..(<2.0x)  conditional, only if measured work reduction is clear
< 1.5x         runtime bottleneck rather than architectural-speedup evidence
```

Fewer than four unique feasible anchors **disables both the strong and
conditional A2 performance verdicts**, regardless of the geometric mean.

## Status

No GPU preflight has been run. P1 (harness implementation), the resource
preflight, and any performance execution remain blocked pending final freeze
review.
