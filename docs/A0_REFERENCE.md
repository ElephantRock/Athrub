# Athrub A0 Reference v0.1

## Purpose

A0 exists to create one meaningful, frozen correctness reference for Phase 1 shared-computation experiments.

The reference substrate is an experimental dependency, not the Athrub architecture.

Athrub identity is defined by its bounded decision contract, serialization, decision scoring semantics, shared-computation research program, evaluation discipline, and later training/runtime work. The temporary substrate used to establish A0 must remain replaceable.

## Public / private provenance boundary

The public repository must remain identity-neutral with respect to external research substrates.

Public Athrub artifacts record:

- content hashes
- immutable revision identifiers
- opaque private-provenance identifiers
- parameter count / architecture class where useful
- Athrub code revision
- tokenizer artifact hash
- decision-head hash
- codec version
- training/data/config hashes
- benchmark/environment hashes

Source/vendor/product acquisition details belong in private research provenance, not in Athrub architecture vocabulary, model names, benchmark titles, or product positioning.

The canonical public manifest must validate against:

```text
schemas/reference_manifest.schema.json
```

## Reference architecture

A0 uses a conventional causal Transformer reference substrate near the 0.6B scale, a fixed tokenizer, the current `TextDecisionCodec`, and one scalar linear decision head.

For candidate `c_i`:

```text
shared state + question + c_i -> final hidden state -> scalar logit z_i
```

Candidate logits for one request are normalized together:

```text
p_i = softmax(z)_i
```

The complete logit vector and complete probability vector are reference semantics.

## Training sequence

A0 adaptation is intentionally small and boring.

### Stage 1 — head warm-up

- freeze the complete reference substrate
- train only the scalar decision head
- use categorical cross-entropy
- start with bounded candidate counts between 2 and 16
- use programmatically verifiable and independently labeled decisions wherever practical

The warm-up establishes whether the fixed representation already provides useful decision signal.

### Stage 2 — brief full adaptation

If Stage 1 exceeds trivial/random controls, unfreeze the substrate and perform a short low-learning-rate decision adaptation run.

This run is not intended to create the final Athrub architecture. Its only purpose is to create a meaningful frozen decision model whose execution can be compared under flat and shared-context computation.

The initial planning envelope is:

```text
50k–200k bounded decisions
1–3 epochs maximum
categorical cross-entropy
2–16 candidates during A0 training
```

These are planning bounds, not benchmark claims.

## Freeze gate

A reference becomes `Athrub A0 Reference v0.1` only when all of the following are immutable and hashed together:

1. Athrub source commit.
2. Reference substrate artifact and immutable revision.
3. Tokenizer artifact and immutable revision.
4. Trained scalar decision-head checkpoint.
5. `TextDecisionCodec` source/version.
6. Training configuration.
7. Dataset manifest.
8. Random seed and precision.
9. Canonical semantic reference benchmark output.
10. Canonical shape-controlled benchmark output.
11. Runtime/environment manifest.
12. Public identity-neutral reference manifest.

Mutable aliases such as `main` or `latest` are not valid canonical evidence.

## Quality gate

The A0 artifact must be meaningful enough that Phase 1 tests real decision behavior rather than random-head mechanics.

At minimum, record:

- accuracy or task-appropriate decision quality against independently known targets
- negative log-likelihood
- Brier score where probability targets permit it
- trivial/random/majority control performance
- per-task-family results

No fixed universal quality threshold is declared yet. The freeze decision must document why the selected checkpoint is non-trivial and stable enough to serve as the Phase 1 correctness oracle.

## A1 handoff

After A0 is frozen, A1 changes no weights, tokenizer, serialization, candidate ordering, decision head, or normalization semantics.

A1 compares the same frozen reference through:

```text
flat complete-path execution
        vs
shared-context execution
```

The first claim remains narrow:

> Can Athrub remove repeated shared-context computation while preserving the complete decision distribution?

Only after this is established may Athrub proceed toward trainable shared computation.
