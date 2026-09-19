# A1 FP32 shared-context scaling plan (Issue #15, protocol v0.1)

Frozen **before** any GPU execution. The machine-readable authoritative copy of
every constant is `configs/a1_fp32_scaling.v0.1.json`; this document is its
human-readable record. This campaign is the narrower FP32-only follow-up to the
Issue #11 disposition; Issue #3 retains the original FP32+BF16 scope
historically unchanged. BF16 is explicitly out of scope: plain shared BF16 is
UNSUPPORTED on this runtime under contract v0.2, and this plan measures only
the numerically validated FP32 shared-context path.

**Freeze status: amended after P0 review; three items remain pending literal
transcription from the Issue #15 review (anchor cells + substitution rule,
full metric set, A2 interpretation rules) — see the final section.**

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

Per invocation: latency, peak allocated/reserved CUDA bytes, logical token
position accounting (flat and shared), shared-path stage latencies. Artifacts
follow the established pattern: provenance (execution git SHA, raw manifest
hashes, frozen A0 binding, environment, effective attention policy), per-cell
JSONL with full measurement records, feasibility-probe records, and a summary.
Nothing in this campaign may be quoted as an A1 correctness result, and BF16
rows do not exist.

## Pending literal transcription from the Issue #15 review

The following were ratified and posted to Issue #15 but their exact values have
not yet been transmitted to the local assistant; the freeze is incomplete
without them and they must be transcribed verbatim before P1:

1. The six exact anchor cells and the substitution rule.
2. The full required metric set (beyond the four listed above).
3. The A2 interpretation rules.

No GPU preflight has been run. P1 (harness implementation), the resource
preflight, and any performance execution remain blocked pending final freeze
review.
