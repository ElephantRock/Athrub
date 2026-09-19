# A1 operational-equivalence result (Issue #11, contract v0.2)

Final adjudicated result of the frozen 42-row holdout executed under
`configs/a1_operational_equivalence.contract.v0.2.json` (raw-byte manifest SHA
`b7945ef0edc5b2e7d3c71d27bae0c46647dcce02ff612f5620d887bf81761632`) by
`benchmarks/a1_operational_holdout.py`.

## Verdict

```text
FP32 shared architecture control     SUPPORTED   (G<->F: 42/42 rows within 1e-5, exact argmax)
plain shared-context BF16            UNSUPPORTED on this runtime
Issue #3 performance                 remains blocked pending the next architecture decision
```

Per the frozen contract there is no failure budget: 17 of 42 rows violate the
predeclared BF16 gates, so plain shared BF16 is unsupported regardless of the
passing rows. No threshold was adjusted and no row was discarded.

## Adjudicated numbers

| Item | Value |
|---|---|
| FP32 precondition | 42/42 pass |
| B↔F max \|Δp\| gate (≤ 1e-2) | FAIL — 17 rows exceed; worst 3.794e-2 (`shape-p384-k3-c24`) |
| B↔F TV gate (≤ 2e-2) | FAIL — 13 rows exceed; worst 4.002e-2 (`shape-p224-k6-c8`) |
| Exact B↔F argmax | 41/42 (0.976) |
| Argmax flips | 1: `shape-p96-k12-c24` |
| Permitted flips (inside 2e-2 radius) | 0 |
| Unpermitted flips (outside radius) | 1: `shape-p96-k12-c24`, oracle margin 2.205e-2, regret 2.205e-2 |
| Rows inside ambiguity region | 4 (all synthetic K=12); outside 38 |
| Failing rows | 15/18 synthetic (p96 6/6, p224 6/6, p384 3/6) + 2/24 semantic (`deadline-fit`, `closest-target`) |

Failure pattern (recorded without causal claim): synthetic failures
concentrate at small prefixes; the shared-BF16 distribution error relative to
the FP32 oracle occupies the same 2–4e-2 band observed in the Issue #9
development set, above the predeclared 1e-2 tolerance.

## Protocol fidelity

Two earlier provisional runs are preserved (gitignored
`artifacts/a1-operational-holdout-v0.2-run1-provisional/` and
`-run2-provisional/`). Run 1 held the FP32 pair resident through BF16
adjudication (a release-helper binding bug); run 2 added the post-release
memory proof, which exposed a second stray binding. The adjudicated run
records post-release CUDA memory of **8.5 MB allocated / 20 MB reserved**
after the FP32 pair is deleted, proving the pair is gone before the BF16 pair
loads. All three runs — two memory regimes — produced bitwise-identical
decision outputs and metrics, so the verdict does not depend on the protocol
defects; the defects are documented because adjudication required a clean run.

Provenance (in `provenance.json`): execution git SHA, raw manifest hash,
frozen A0 reference manifest SHA, canonical attention policy, environment
(torch 2.11.0+cu128, CUDA 12.8, RTX 3080 Ti, driver 616.64, Python 3.11.14),
one-request-per-invocation grouping in frozen order.

## Artifact integrity (adjudicated run, `artifacts/a1-operational-holdout-v0.2/`)

| Artifact | sha256 |
|---|---|
| `provenance.json` | `af0efbd5bc235eab724497e6df3bf8f41bce5c2f272fc1410b2661ca180277fb` |
| `phase0_remeasure.json` | `238cfd1e18a5c5c553c3e7bcd31096cac2f427af0d757452df469ad9f3b33639` |
| `fp32_precondition.jsonl` | `1260eaf4c524ae53be07a388652e61f49cee83885e4fb888e2c7e4dcf5e816fe` |
| `rows.jsonl` | `b873253438192655563598a11debaa3a59fa1a5206e73b57b27312ef6b1a1018` |
| `summary.json` | `761d32ad7e03a446b0bd3d626c885ea9048522029c38b72b4a398e062ed08fae` |

`phase0_remeasure.json` latencies (including the spill-adjacent G reserved
figure) are resource-gate evidence only and must not appear in Phase 1
performance claims.
