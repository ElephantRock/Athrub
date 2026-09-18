# Shared-context experiment

This directory is the working area for Athrub Phase 1.

The experiment must preserve a strict separation between:

- **reference execution** — canonical full candidate-path computation
- **optimized flat execution** — batching/packing changes without context sharing
- **shared-context execution** — common context computed once, candidate continuations branched from shared state
- **shared-packed execution** — shared context plus packed candidate continuation

## Required outputs

Each experiment run should emit machine-readable artifacts containing:

- implementation name
- git revision
- model/tokenizer revision
- precision
- workload shape
- exact token counts
- latency samples
- throughput
- memory measurements
- logits
- probabilities
- numerical comparison against reference

Research result files and checkpoints are intentionally excluded from Git; publish stable summaries or release artifacts separately when they become part of the project record.

## First implementation task

Implement the reference backend behind `athrub.backends.DecisionBackend`, then implement shared-context execution behind the same contract. The benchmark and comparison layers must not know which execution strategy is being measured.
