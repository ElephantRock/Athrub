# A1 BF16 stability diagnostics (Issue #9)

Research-evidence harness for the BF16 shared-context numerical-stability
investigation. See `docs/A1_BF16_STABILITY.md` for the full experiment record,
provenance, and conclusions. `SharedContextBackend` is not modified by anything
in this directory.

All GPU scripts verify the frozen A0 manifest hashes before execution and run
under `CANONICAL_A0_POLICY`. Artifacts are written to gitignored
`artifacts/a1-bf16-stability/`. Run from the repository root with the project
virtual environment, e.g. `python experiments/a1_bf16_stability/01_layerwise_prefix_decomposition.py`.

| Script | Experiment | Artifact |
|---|---|---|
| `01_layerwise_prefix_decomposition.py` | layerwise prefix decomposition (A/B/C + cross-row identity) | `layerwise_p128k2.json` |
| `02_first_block_and_shape_controls.py` | first-block sub-operation trace; D/E execution-shape controls | `first_block_trace_p128k2.json`, `shape_controls_p128k2.json` |
| `03_mixed_prefix_cell.py` | single-cell mixed-precision prefix (M vs A/B/F) | `mixed_prefix_p128k2.json` |
| `04_mixed_prefix_suite.py` | full 9-row controlled mixed-prefix suite | `mixed_prefix_suite.json` |
| `05_oracle_relative_table.py` | oracle-relative development table vs F (CPU; reads the suite artifact) | `oracle_relative_dev_table.json` |

The nine rows produced by `04`/`05` are the development set for the Issue #11
criterion; thresholds must be predeclared against a fresh holdout suite, not
derived from them.
