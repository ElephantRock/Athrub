# A1 FP32 shared-context scaling plan (Issue #15, protocol v0.1)

Frozen **before** any GPU execution. The machine-readable authoritative copy of
every constant is `configs/a1_fp32_scaling.v0.1.json`; this document is its
human-readable record. This campaign is the narrower FP32-only follow-up to the
Issue #11 disposition; Issue #3 retains the original FP32+BF16 scope
historically unchanged. BF16 is explicitly out of scope: plain shared BF16 is
UNSUPPORTED on this runtime under contract v0.2, and this plan measures only
the numerically validated FP32 shared-context path.

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

Full scaling grid: every (prefix, K) cell measured on the primary comparison.
The 8192-prefix FP32 regime sits above the known WDDM-spill boundary observed
on this workstation; cell feasibility must be established by the P1 resource
preflight, and an infeasible cell is recorded as infeasible, never silently
dropped.

## Attribution matrix

```text
prefixes          128, 512, 1024
K                 2, 8, 32
```

The attribution cells run every execution path required by the Issue #15
attribution matrix so that ordinary batching/chunking improvements are not
misattributed to shared computation (Phase 1 defines optimized-flat as a
required control for exactly this reason).

## Cadence

```text
primary cells     3 warmups / 10 repeats
anchor suite      10 warmups / 50 repeats
```

## Correctness gate (FP32)

Every optimized/chunked path must satisfy, against the reference flat backend
on its cell:

```text
max |Δp| < 1e-5   and   exact argmax
```

on all attribution cells and the anchor suite. A performance number produced by
a path that fails the correctness gate is void. FP32 shared-context
equivalence is already established (development matrix 9/9 and Issue #11
holdout 42/42); this gate re-verifies it for the chunked/optimized execution
forms before their timings are admitted.

## Primary comparison

```text
optimized/chunked flat  vs  chunked shared
```

Reference flat and unchunked shared remain available as diagnostic context;
the Phase 1 performance claim is about the optimized forms.

## Metrics and artifacts

Per invocation: latency, peak allocated/reserved CUDA bytes, logical token
position accounting (flat and shared), shared-path stage latencies. Artifacts
follow the established pattern: provenance (execution git SHA, raw manifest
hash, frozen A0 binding, environment, effective attention policy), per-cell
JSONL with full measurement records, and a summary with aggregates. All
latency evidence in this campaign is performance evidence — there is no
adjudication — but nothing here may be quoted as an A1 correctness result, and
BF16 rows do not exist.

## A2 decision logic

Per Issue #15: the shared-vs-flat wall-clock and work-reduction results of
this campaign feed the A2 gate decision (whether to proceed from frozen
reference execution toward trainable shared computation). The campaign reports
measurements; the A2 decision is recorded separately.

## Open items requiring review before P1 (harness implementation)

The following are pinned by the Issue #15 protocol text and must be confirmed
in review before any harness code or GPU execution; this freeze deliberately
does not invent them:

1. Anchor suite cell list.
2. Chunking specification for optimized/chunked flat and chunked shared.
3. The exact attribution-matrix execution-path set (per the Phase 1
   four-path definition).
4. The infeasible-cell rule for the 8192-prefix FP32 regime.

No GPU preflight has been run. P1 (harness implementation) and any performance
execution remain blocked pending review of this freeze.
