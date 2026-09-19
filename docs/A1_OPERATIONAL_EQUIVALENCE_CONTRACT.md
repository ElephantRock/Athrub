# A1 operational-equivalence contract v0.2 (Issue #11)

Predeclared and frozen **before** any holdout GPU execution. The machine-readable
authoritative copy of every constant is
`configs/a1_operational_equivalence.contract.v0.2.json`; this document is its
human-readable record. Nothing in this contract was derived from holdout
results — the holdout has not been executed. The nine Issue #9 oracle-relative
rows are the development set and are quarantined from threshold selection.

## Roles

```text
F = flat FP32                  semantic/numerical oracle
G = shared-context FP32        architectural equivalence control
A = flat BF16                  diagnostic baseline; A does NOT gate B
B = shared-context BF16        operational candidate under test
mixed prefix                   excluded (Issue #9 rejected it generally)
```

## Precondition (FP32 architectural control)

G ↔ F must satisfy max |Δp| < 1e-5 with exact argmax on every valid holdout row.
This re-establishes, on the holdout, the FP32 equivalence already proven on the
development matrix.

## BF16 operational gates (every valid row; no failure budget)

```text
B ↔ F  max |Δp| ≤ 1e-2
B ↔ F  TV      ≤ 2e-2
```

## Decision rule and the oracle ambiguity radius

The **oracle ambiguity radius** is 2e-2. It is not a tunable near-tie threshold:
it is mathematically coupled to the per-candidate probability bound εp = 1e-2,
because a top-1 ordering can reverse only when the oracle probabilities of the
two candidates differ by at most 2εp.

```text
rows outside the ambiguity region:  exact B ↔ F argmax required
rows inside the ambiguity region:   B's prediction must lie within F's
                                    ambiguity set (candidates whose F
                                    probability is within the radius of F's
                                    top-1)
```

Required reporting: exact B↔F argmax agreement across all 42 rows (even where
ambiguity membership permits a different winner); the fraction of rows inside
and outside the ambiguity region; oracle decision regret
`p_F(top1) − p_F(pred_B)` for every permitted flip; KL divergence and centered
logit deltas reported but not gated.

## Holdout suite (42 fresh rows, frozen)

- 18 synthetic shape cells: prefix units {96, 224, 384} × candidate count
  {3, 6, 12} × candidate units {8, 24}. These avoid every development cell
  ({128, 512, 1024} × {2, 4, 8, 16} × {16}) and stay below the WDDM-spill
  long-prefix FP32 regime.
- 24 semantic rows: three fresh generator-v0.1 records from each of the eight
  A0 workload families at fixed seed **314159** (distinct from the A0
  training/validation seeds 1729/2718), candidate range 2–8 (generator
  defaults; flagged for reviewer ratification — this is the one parameter the
  Issue #11 proposal did not pin).
- Fixed ordering: the 18 synthetic rows (prefix ascending, then candidate
  count, then candidate units) followed by the 24 semantic rows in generation
  order.
- Exact manifest: `experiments/a1_operational_equivalence/holdout_manifest.v0.2.jsonl`
  (42 JSONL request records; deterministic regeneration verified),
  SHA-256 `b7945ef0edc5b2e7d3c71d27bae0c46647dcce02ff612f5620d887bf81761632`.
  Generator: `experiments/a1_operational_equivalence/generate_holdout_manifest.py`
  (includes guards asserting no overlap with development cells, unique request
  ids, exactly 3 records per family, and seed distinctness).

## Binding

| Item | Value |
|---|---|
| Frozen A0 oracle manifest | sha256 `fcd3cb6b3e56d747fa82b823353fa6eb893adface51ff2985190aa9f261834b5` |
| Contract frozen at commit | `5442809ba039bc9136eef1be7bc28ba274ff355d` (main, post PR #12) |
| Attention policy | CANONICAL_A0_POLICY: efficient_sdpa / expand_kv_heads / ratio 2 / no fallback |
| Request grouping | one DecisionRequest per score invocation, identical across F, G, A, B |

## Methodology (binding)

- **A does not gate B.** Flat BF16 measures how much error BF16 itself
  introduces; shared BF16 succeeds or fails directly against FP32 semantic
  truth.
- **No post-hoc adjustment.** If B fails any valid holdout row, plain shared
  BF16 remains unsupported on this runtime. Any new execution policy requires a
  newly predeclared contract and a fresh holdout.
- **Resource-only preflight** is allowed before the run to establish shape
  feasibility; it may not inspect decision results or replace cases. If a
  declared shape is infeasible, stop and freeze a revised manifest before
  seeing acceptance metrics.
- Holdout execution is authorized only after this contract and manifest are
  merged to main.
