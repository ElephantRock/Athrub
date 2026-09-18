from __future__ import annotations

import json
from pathlib import Path

from athrub.a0_data import GENERATORS, dataset_summary, generate_decisions, write_decision_jsonl
from athrub.a0_training import load_decision_jsonl


def test_generator_is_deterministic() -> None:
    first = generate_decisions(count=32, seed=1234, candidate_min=2, candidate_max=6)
    second = generate_decisions(count=32, seed=1234, candidate_min=2, candidate_max=6)

    assert first == second
    assert [decision.request_id for decision in first] == [decision.request_id for decision in second]


def test_generator_covers_all_families_and_valid_labels() -> None:
    decisions = generate_decisions(
        count=len(GENERATORS) * 4,
        seed=99,
        candidate_min=2,
        candidate_max=8,
    )

    assert {decision.task_family for decision in decisions} == set(GENERATORS)
    assert all(0 <= decision.label < len(decision.candidates) for decision in decisions)
    assert all(len(set(decision.candidates)) == len(decision.candidates) for decision in decisions)


def test_generated_jsonl_loads_through_training_contract(tmp_path: Path) -> None:
    decisions = generate_decisions(count=24, seed=55, candidate_min=2, candidate_max=5)
    path = tmp_path / "generated.jsonl"
    write_decision_jsonl(path, decisions)

    loaded = load_decision_jsonl(path)

    assert len(loaded) == len(decisions)
    for source, training in zip(decisions, loaded, strict=True):
        assert source.request_id == training.request_id
        assert source.candidates == training.candidates
        assert training.target_distribution[source.label] == 1.0
        assert sum(training.target_distribution) == 1.0


def test_dataset_summary_is_machine_readable() -> None:
    decisions = generate_decisions(count=40, seed=101, candidate_min=2, candidate_max=7)
    summary = dataset_summary(decisions)

    assert summary["examples"] == 40
    assert summary["candidate_min"] >= 2
    assert summary["candidate_max"] <= 7
    json.dumps(summary)
