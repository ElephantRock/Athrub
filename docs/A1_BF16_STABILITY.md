# A1 BF16 shared-context numerical stability (Issue #9 research record)

This document records the Issue #9 investigation: why shared-context BF16 execution
failed the original A1 flat-vs-shared equivalence gates, where the divergence
originates, which workarounds were tested and rejected, and the resulting research
disposition. Issue #2's original gates remain historically unchanged: FP32
flat-vs-shared equivalence passed; the BF16 numerical (1e-3) and 100%-argmax gates
failed and were not relaxed.

## Provenance

| Item | Value |
|---|---|
| Frozen oracle manifest | `artifacts/a0-reference-v0.1-run1/reference_manifest.json`, sha256 `fcd3cb6b3e56d747fa82b823353fa6eb893adface51ff2985190aa9f261834b5` |
| Athrub source commit at investigation | `a2332d40f53a081cd919cf7ef5108e4de8c8c1b0` (main, post PR #10) |
| Attention policy | `CANONICAL_A0_POLICY`: efficient_sdpa, expand_kv_heads, q_to_kv_ratio 2, fallback disallowed (`src/athrub/attention_runtime.py`) |
| Workstation | RTX 3080 Ti 12 GiB (sm_86), driver 616.64, torch 2.11.0+cu128, CUDA 12.8, transformers 4.57.6, Python 3.11.14, Windows 11 Pro 25H2 |
| Diagnostic harness | `experiments/a1_bf16_stability/` (tracked); artifacts under gitignored `artifacts/a1-bf16-stability/` |
| SharedContextBackend | **unmodified throughout** — all diagnostics are external to the backend |

## Experiment sequence and conclusions

1. **Original BF16 equivalence failure** (A1 correctness matrix, PR #10 evidence):
   FP32 flat-vs-shared passed 9/9 (max |Δp| 5.07e-6); BF16 failed — max |Δp| 6.24e-2,
   4/9 argmax flips, none near-tied; failure bitwise-stable across reruns.
   Artifacts: `artifacts/a1-equivalence-v0.1-original/` (checksums alongside).

2. **Layerwise localization** (`layerwise_p128k2.json`, script `01`): on BF16
   `p128×k2`, prefix hidden states diverge from the **first transformer layer**
   (layer 1; layer 0 embeddings bitwise identical). Flat batch-2 full-path (A),
   prefix-only batch-1 (B), and prefix-only batch-2 (C) all differ pairwise;
   A↔B and A↔C saturate at max |Δ| = 32 (one BF16 ulp at activation-outlier
   magnitude) while B↔C stays much smaller. The flat batch's two rows are bitwise
   identical over the prefix at every layer: one well-defined flat BF16 prefix
   representation exists; no prefix-only execution reproduces it.

3. **First-block sub-operation localization** (`first_block_trace_p128k2.json`,
   script `02`): embeddings and input RMSNorm are bitwise identical; the first
   divergence appears in the **linear projections** — q_proj diverges for A↔B but
   matches bitwise for A↔C, k/v_proj the reverse. This is the signature of
   per-shape GEMM kernel selection (flaten M = batch×length differs per execution)
   choosing different reduction orders. The primary mechanism is sequence-shape
   GEMM numerics, not SDPA. A specific cuBLAS algorithm-selection mechanism is
   plausible but was not directly proven (no kernel profiling, by decision).

4. **Execution-shape controls** (`shape_controls_p128k2.json`, script `02`):
   E (batch 2 × length 598, duplicated identical suffix) matches A **bitwise at
   every layer, both rows** — suffix semantics are numerically irrelevant to the
   prefix; execution shape alone determines the BF16 prefix representation.
   D (batch 1 × full length) diverges more than B (max 64 vs 32): the exact batch
   shape is also required. Conclusion: reproducing the flat BF16 prefix bitwise
   requires executing the flat shape itself; single-prefix BF16 shape-matching is
   refuted.

5. **Single-cell mixed-prefix success** (`mixed_prefix_p128k2.json`, script `03`):
   M = FP32 prefix (frozen BF16-valued weights promoted) → BF16-cast KV → view
   branch → BF16 continuation/head reproduced A's probabilities bitwise on
   `p128×k2` via an exact common-mode logit shift (−0.25 on both logits) that
   softmax cancels. Deterministic. Operational-candidate verdict at cell scale:
   PASS (diagnostic).

6. **Full mixed-prefix rejection** (`mixed_prefix_suite.json`, script `04`):
   on the 9-row controlled suite, M↔A passes only 2/9 (max |Δp| up to 1.07e-1,
   one argmax flip); common-mode cancellation does not generalize (centered
   residuals up to 0.41); failures correlate with prefix length; M is not
   uniformly closer to flat BF16 than plain shared BF16. The simple FP32-prefix
   workaround is rejected. `mixed_precision_operational_candidate: FAIL`.

7. **Research disposition**: exact flat-BF16 reproduction is not the operational
   target going forward. The flat-BF16 trajectory is partly an execution-shape
   artifact; chasing it risks optimizing for one runtime/kernel realization
   rather than Athrub's decision semantics. Follow-up is Issue #11: define a
   precision-coherent BF16 operational criterion against the FP32 semantic
   oracle, predeclared before a fresh holdout suite. The oracle-relative
   development table (`oracle_relative_dev_table.json`, script `05`) computes
   A/B/M vs F per row with F-based near-tie margins and ΔE = E_B − E_A; those
   nine rows are the **development set** and must not be turned into thresholds.

## Oracle-relative development-set summary (Issue #11 input)

| Path vs F (flat FP32) | worst max \|Δp\| | worst TV | worst KL | F-argmax preserved |
|---|---|---|---|---|
| A = flat BF16 | 3.54e-2 | 4.31e-2 | 5.34e-3 | 7/9 |
| B = shared BF16 | 4.17e-2 | 4.17e-2 | 4.68e-3 | 5/9 |
| M = mixed prefix | 8.67e-2 | 8.67e-2 | 1.56e-2 | 7/9 |

- ΔE = E_B − E_A is mixed-sign across rows (5 positive / 4 negative; magnitudes
  ≤ 2.2e-2): sharing adds comparatively small, sometimes negative incremental
  distribution error beyond BF16's intrinsic oracle gap.
- No row is near-tied from F at the 2e-3 margin threshold; smallest F margin is
  2.46e-3 (`p128×k8`).
- A and M flip the same two F-decisions (`p128×k2`, `p1024×k8`); B flips four.
- M does not dominate A or B (largest worst-case distribution errors of the three).

## Artifact integrity

All artifacts live under `artifacts/a1-bf16-stability/` (gitignored; copy
alongside the frozen A0 archive for durability). Exact SHA-256 digests at record
time — the repository itself carries these so the evidence bundle remains
identifiable without the gitignored files:

| Artifact | sha256 |
|---|---|
| `layerwise_p128k2.json` | `aa34e5bf496b4881d5dad3fa5452870b5ff3a4118d6d5311f63ac4296be0ef77` |
| `first_block_trace_p128k2.json` | `4146be6df5fc2adf48346e0a96377e44c0891bef746e57fd60c28e0a4ca26c5d` |
| `shape_controls_p128k2.json` | `7ffcad94e89c475736dc94c0c877ab690d82779dfc28acb5c2842657d6cee074` |
| `mixed_prefix_p128k2.json` | `021c2d2366c13b8103e63e6d7027074ba90a716fb3a22339b6fe9455562399cc` |
| `mixed_prefix_suite.json` | `7b1fd8818cbfee5d3a32bf7af773f9a60983b134054b1be36ff21448cd251756` |
| `oracle_relative_dev_table.json` | `c8a967589c7470ba0b4cebf55d3bd55e278d3584b58bcc1026d0f5c615bbca9c` |

`ARTIFACT_SHA256.txt` in the same gitignored directory duplicates these digests
for on-disk verification.
