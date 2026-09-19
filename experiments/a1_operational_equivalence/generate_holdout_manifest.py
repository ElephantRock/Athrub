"""Generate the frozen Issue #11 holdout request manifest (42 rows).

Deterministic. 18 synthetic shape cells (prefix {96,224,384} x K {3,6,12} x
candidate units {8,24}) followed by 24 semantic rows: three fresh generator-v0.1
records from each of the eight A0 workload families at seed 314159 (distinct
from the A0 training/validation seeds 1729/2718). The script refuses to emit a
manifest that overlaps the Issue #11 development cells or reuses a development
seed. The output JSONL is committed; its SHA-256 is recorded in the frozen
contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from athrub.a0_data import GENERATOR_VERSION, generate_decisions
from athrub.contracts import DecisionRequest
from athrub.workloads import synthetic_request

MANIFEST_PATH = Path("experiments/a1_operational_equivalence/holdout_manifest.v0.2.jsonl")
HOLDOUT_PREFIXES = (96, 224, 384)
HOLDOUT_CANDIDATE_COUNTS = (3, 6, 12)
HOLDOUT_CANDIDATE_UNITS = (8, 24)
SEMANTIC_SEED = 314159
SEMANTIC_RECORDS = 24
SEMANTIC_CANDIDATE_RANGE = (2, 8)

DEVELOPMENT_PREFIXES = (128, 512, 1024)
DEVELOPMENT_CANDIDATE_COUNTS = (2, 4, 8, 16)
DEVELOPMENT_CANDIDATE_UNITS = (16,)
DEVELOPMENT_SEEDS = (1729, 2718)


def build_requests() -> list[DecisionRequest]:
    requests: list[DecisionRequest] = []
    for prefix in HOLDOUT_PREFIXES:
        for count in HOLDOUT_CANDIDATE_COUNTS:
            for units in HOLDOUT_CANDIDATE_UNITS:
                requests.append(
                    synthetic_request(
                        request_id=f"shape-p{prefix}-k{count}-c{units}",
                        prefix_units=prefix,
                        candidate_units=units,
                        candidate_count=count,
                    )
                )
    decisions = generate_decisions(
        count=SEMANTIC_RECORDS,
        seed=SEMANTIC_SEED,
        candidate_min=SEMANTIC_CANDIDATE_RANGE[0],
        candidate_max=SEMANTIC_CANDIDATE_RANGE[1],
    )
    for index, decision in enumerate(decisions):
        requests.append(
            DecisionRequest(
                request_id=decision.request_id,
                state=decision.state,
                question=decision.question,
                candidates=list(decision.candidates),
                metadata={
                    "suite": "semantic",
                    "task_family": decision.task_family,
                    "generator_version": GENERATOR_VERSION,
                    "seed": SEMANTIC_SEED,
                    "generation_index": index,
                },
            )
        )
    return requests


def verify_no_development_overlap(requests: list[DecisionRequest]) -> None:
    for request in requests:
        if request.metadata.get("suite") == "semantic":
            assert request.metadata["seed"] not in DEVELOPMENT_SEEDS
            assert SEMANTIC_CANDIDATE_RANGE[0] <= len(request.candidates) <= SEMANTIC_CANDIDATE_RANGE[1]
            continue
        prefix = request.metadata["prefix_units"]
        count = len(request.candidates)
        units = request.metadata["candidate_units"]
        assert prefix not in DEVELOPMENT_PREFIXES, request.request_id
        assert count not in DEVELOPMENT_CANDIDATE_COUNTS, request.request_id
        assert units not in DEVELOPMENT_CANDIDATE_UNITS, request.request_id
    identifiers = [request.request_id for request in requests]
    assert len(identifiers) == len(set(identifiers)) == 42
    families = [r.metadata["task_family"] for r in requests if r.metadata.get("suite") == "semantic"]
    family_counts = Counter(families)
    assert len(families) == SEMANTIC_RECORDS
    assert len(family_counts) == 8
    assert all(count == 3 for count in family_counts.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(MANIFEST_PATH))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    requests = build_requests()
    verify_no_development_overlap(requests)

    def serialize() -> str:
        lines = []
        for request in requests:
            payload = {
                "request_id": request.request_id,
                "state": request.state,
                "question": request.question,
                "candidates": list(request.candidates),
                "metadata": dict(request.metadata),
            }
            lines.append(json.dumps(payload, sort_keys=True))
        return "\n".join(lines) + "\n"

    digest = hashlib.sha256(serialize().encode("utf-8")).hexdigest()
    if args.verify_only:
        existing = Path(args.output).read_text(encoding="utf-8")
        print(json.dumps({"manifest_sha256": digest, "matches_existing": existing == serialize()}, indent=2))
        return
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialize(), encoding="utf-8")
    print(json.dumps({"output": str(output), "requests": len(requests), "manifest_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
