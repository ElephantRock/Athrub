# Athrub A0 adaptation

This directory contains the executable workflow used to create `Athrub A0 Reference v0.1` from private/local training inputs.

A0 training is not the final Athrub architecture. Its sole purpose is to produce a meaningful, frozen correctness reference for the A1 shared-computation experiment.

## Data contract

Training and validation files use JSONL. Every line is one bounded decision:

```json
{"request_id":"case-1","state":"...","question":"...","candidates":["A","B","C"],"label":1}
```

or a soft target distribution:

```json
{"request_id":"case-2","state":"...","question":"...","candidates":["A","B"],"target_distribution":[0.8,0.2]}
```

Exactly one of `label` or `target_distribution` is required. Runtime validation additionally requires soft targets to match candidate count and sum to one. See `schemas/a0_decision_record.schema.json`.

## Training sequence

The executable implements the A0 gate literally:

```text
frozen reference substrate
        ↓
train scalar decision head
        ↓
validate against chance baseline
        ↓
if gate passes
        ↓
brief low-LR full adaptation
        ↓
export frozen substrate + tokenizer + decision head
```

The scalar head and categorical cross-entropy are intentionally simple. A0 must not introduce a second architecture research question.

## Run

Install the optional reference-loading dependency:

```bash
pip install -e '.[reference]'
```

Copy `configs/a0_adaptation.example.json` to a private/local configuration, fill in immutable substrate/tokenizer revisions and local dataset paths, then run:

```bash
python experiments/a0/train_reference.py --config configs/private/a0_adaptation.json
```

`configs/private/` and `provenance/private/` are gitignored so identity-bearing acquisition/source information is not accidentally committed.

## Outputs

The trainer writes under the configured `output_dir`:

```text
substrate/              frozen substrate snapshot
tokenizer/              frozen tokenizer snapshot
decision_head.pt        trained scalar head
training_config.json    exact executed configuration
training_history.json   per-stage train/validation metrics
environment.json        runtime environment
training_summary.json   gate result and executed stages
```

These artifacts are inputs to `benchmarks/freeze_a0_reference.py`, which produces the public identity-neutral reference manifest.

## Warm-up gate

The initial automated gate compares validation accuracy against the mean random-choice probability `mean(1 / K)` plus the configured margin. This is only a minimum non-triviality check. The A0 freeze decision must additionally report the task-appropriate quality, NLL, Brier score where applicable, and documented trivial/random/majority controls required by `docs/A0_REFERENCE.md`.

## Reproducibility

The caller owns dataset ordering and source provenance. The trainer records the configured seed and uses deterministic per-epoch Python shuffling from that seed. Canonical A0 freezing still requires immutable hashes of the dataset manifest, training configuration, model/tokenizer/head artifacts, benchmark outputs, and environment.
