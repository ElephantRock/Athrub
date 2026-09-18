# Shared-context experiment

This directory is the working area for Athrub Phase 1.

Performance/equivalence claims here use the frozen `Athrub A0 Reference v0.1` defined by `docs/A0_REFERENCE.md`. The reference substrate is a replaceable experimental dependency, not Athrub architecture identity.

The experiment must preserve a strict separation between:

- **reference execution** — canonical full candidate-path computation
- **optimized flat execution** — batching/packing changes without context sharing
- **shared-context execution** — common context computed once, candidate continuations branched from shared state
- **shared-packed execution** — shared context plus packed candidate continuation

## Required outputs

Each experiment run should emit machine-readable artifacts containing:

- implementation name
- Athrub git revision
- frozen substrate/tokenizer revisions or hashes
- decision-head hash
- precision
- workload shape
- exact token counts
- latency samples
- throughput
- memory measurements
- logits
- probabilities
- numerical comparison against reference

Identity-bearing acquisition details for the reference substrate remain in private research provenance. Research result files and checkpoints are intentionally excluded from Git; publish stable summaries or identity-neutral release artifacts separately when they become part of the project record.

## First implementation task

Freeze the meaningful A0 reference, then exercise `FlatReferenceBackend` and `SharedContextBackend` behind the same `athrub.backends.DecisionBackend` contract. The benchmark and comparison layers must not know which execution strategy is being measured.
