"""Athrub-native programmatic supervision for the first A0 reference run.

These generators provide independently computable bounded-decision targets without
embedding any external project identity. They are an A0 bootstrap corpus, not evidence
of broad domain generalization.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GENERATOR_VERSION = "0.1"


@dataclass(frozen=True, slots=True)
class GeneratedDecision:
    request_id: str
    task_family: str
    state: str
    question: str
    candidates: tuple[str, ...]
    label: int

    def to_record(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_family": self.task_family,
            "state": self.state,
            "question": self.question,
            "candidates": list(self.candidates),
            "label": self.label,
        }


def _stable_id(family: str, seed: int, index: int) -> str:
    payload = f"{GENERATOR_VERSION}:{family}:{seed}:{index}".encode("utf-8")
    return f"a0-{family}-{hashlib.sha256(payload).hexdigest()[:16]}"


def _shuffle_labeled(
    rng: random.Random, candidates: Sequence[str], correct_index: int
) -> tuple[tuple[str, ...], int]:
    indexed = list(enumerate(candidates))
    rng.shuffle(indexed)
    shuffled = tuple(value for _, value in indexed)
    label = next(position for position, (source_index, _) in enumerate(indexed) if source_index == correct_index)
    return shuffled, label


def _unique_ints(rng: random.Random, count: int, lower: int, upper: int) -> list[int]:
    if upper - lower + 1 < count:
        raise ValueError("integer range is too small for unique sampling")
    return rng.sample(range(lower, upper + 1), count)


def _largest_value(rng: random.Random, request_id: str, candidate_count: int) -> GeneratedDecision:
    values = _unique_ints(rng, candidate_count, -500, 500)
    names = [f"option {chr(65 + index)}" for index in range(candidate_count)]
    state = "Observed values: " + "; ".join(
        f"{name} = {value}" for name, value in zip(names, values, strict=True)
    ) + "."
    correct = max(range(candidate_count), key=values.__getitem__)
    candidates, label = _shuffle_labeled(rng, names, correct)
    return GeneratedDecision(
        request_id=request_id,
        task_family="largest-value",
        state=state,
        question="Which option has the largest observed value?",
        candidates=candidates,
        label=label,
    )


def _closest_target(rng: random.Random, request_id: str, candidate_count: int) -> GeneratedDecision:
    target = rng.randint(-200, 200)
    offsets = _unique_ints(rng, candidate_count, 1, 80)
    signs = [rng.choice((-1, 1)) for _ in offsets]
    values = [target + sign * offset for sign, offset in zip(signs, offsets, strict=True)]
    names = [f"setting {index + 1}" for index in range(candidate_count)]
    state = f"Target value: {target}. Candidate settings: " + "; ".join(
        f"{name} produces {value}" for name, value in zip(names, values, strict=True)
    ) + "."
    distances = [abs(value - target) for value in values]
    correct = min(range(candidate_count), key=distances.__getitem__)
    candidates, label = _shuffle_labeled(rng, names, correct)
    return GeneratedDecision(
        request_id=request_id,
        task_family="closest-target",
        state=state,
        question="Which setting produces the value closest to the target?",
        candidates=candidates,
        label=label,
    )


def _capacity_fit(rng: random.Random, request_id: str, candidate_count: int) -> GeneratedDecision:
    demands = _unique_ints(rng, candidate_count, 5, 180)
    capacity = rng.randint(min(demands), max(demands))
    if not any(value <= capacity for value in demands):
        capacity = min(demands)
    names = [f"load {index + 1}" for index in range(candidate_count)]
    state = f"Available capacity is {capacity} units. Proposed loads: " + "; ".join(
        f"{name} requires {demand}" for name, demand in zip(names, demands, strict=True)
    ) + "."
    feasible = [index for index, demand in enumerate(demands) if demand <= capacity]
    correct = max(feasible, key=demands.__getitem__)
    candidates, label = _shuffle_labeled(rng, names, correct)
    return GeneratedDecision(
        request_id=request_id,
        task_family="capacity-fit",
        state=state,
        question="Which load uses the most capacity without exceeding the limit?",
        candidates=candidates,
        label=label,
    )


def _deadline_fit(rng: random.Random, request_id: str, candidate_count: int) -> GeneratedDecision:
    durations = _unique_ints(rng, candidate_count, 5, 180)
    minutes_left = rng.randint(min(durations), max(durations))
    names = [f"plan {index + 1}" for index in range(candidate_count)]
    state = f"There are {minutes_left} minutes available. Plan durations: " + "; ".join(
        f"{name} takes {duration} minutes"
        for name, duration in zip(names, durations, strict=True)
    ) + "."
    feasible = [index for index, duration in enumerate(durations) if duration <= minutes_left]
    correct = max(feasible, key=durations.__getitem__)
    candidates, label = _shuffle_labeled(rng, names, correct)
    return GeneratedDecision(
        request_id=request_id,
        task_family="deadline-fit",
        state=state,
        question="Which plan is the longest one that still finishes within the available time?",
        candidates=candidates,
        label=label,
    )


def _route_cost(rng: random.Random, request_id: str, candidate_count: int) -> GeneratedDecision:
    times = _unique_ints(rng, candidate_count, 10, 150)
    risks = _unique_ints(rng, candidate_count, 1, 40)
    risk_weight = rng.randint(2, 8)
    names = [f"route {chr(65 + index)}" for index in range(candidate_count)]
    scores = [time + risk_weight * risk for time, risk in zip(times, risks, strict=True)]
    while len(set(scores)) != len(scores):
        risks = _unique_ints(rng, candidate_count, 1, 40)
        scores = [time + risk_weight * risk for time, risk in zip(times, risks, strict=True)]
    state = (
        f"Use total cost = travel time + {risk_weight} × risk. Routes: "
        + "; ".join(
            f"{name}: time {time}, risk {risk}"
            for name, time, risk in zip(names, times, risks, strict=True)
        )
        + "."
    )
    correct = min(range(candidate_count), key=scores.__getitem__)
    candidates, label = _shuffle_labeled(rng, names, correct)
    return GeneratedDecision(
        request_id=request_id,
        task_family="route-cost",
        state=state,
        question="Which route has the lowest total cost under the stated rule?",
        candidates=candidates,
        label=label,
    )


def _efficiency(rng: random.Random, request_id: str, candidate_count: int) -> GeneratedDecision:
    inputs = _unique_ints(rng, candidate_count, 5, 80)
    outputs = _unique_ints(rng, candidate_count, 20, 300)
    names = [f"process {index + 1}" for index in range(candidate_count)]
    ratios = [output / input_ for output, input_ in zip(outputs, inputs, strict=True)]
    while len({round(value, 12) for value in ratios}) != len(ratios):
        outputs = _unique_ints(rng, candidate_count, 20, 300)
        ratios = [output / input_ for output, input_ in zip(outputs, inputs, strict=True)]
    state = "Process measurements: " + "; ".join(
        f"{name}: output {output}, input {input_}"
        for name, output, input_ in zip(names, outputs, inputs, strict=True)
    ) + "."
    correct = max(range(candidate_count), key=ratios.__getitem__)
    candidates, label = _shuffle_labeled(rng, names, correct)
    return GeneratedDecision(
        request_id=request_id,
        task_family="efficiency",
        state=state,
        question="Which process has the highest output-to-input ratio?",
        candidates=candidates,
        label=label,
    )


def _sequence_next(rng: random.Random, request_id: str, candidate_count: int) -> GeneratedDecision:
    start = rng.randint(-50, 50)
    step = rng.choice([value for value in range(-12, 13) if value != 0])
    length = rng.randint(4, 7)
    sequence = [start + step * index for index in range(length)]
    correct_value = sequence[-1] + step
    distractor_offsets = _unique_ints(rng, candidate_count - 1, 1, 30)
    distractors = [correct_value + rng.choice((-1, 1)) * offset for offset in distractor_offsets]
    while len(set(distractors + [correct_value])) != candidate_count:
        distractors = [correct_value + rng.choice((-1, 1)) * offset for offset in distractor_offsets]
    raw_candidates = [str(correct_value), *(str(value) for value in distractors)]
    candidates, label = _shuffle_labeled(rng, raw_candidates, 0)
    return GeneratedDecision(
        request_id=request_id,
        task_family="sequence-next",
        state="Sequence: " + ", ".join(str(value) for value in sequence) + ".",
        question="Assuming the constant step continues, which value comes next?",
        candidates=candidates,
        label=label,
    )


def _constraint_choice(rng: random.Random, request_id: str, candidate_count: int) -> GeneratedDecision:
    capacities = _unique_ints(rng, candidate_count, 20, 150)
    latencies = _unique_ints(rng, candidate_count, 5, 120)
    costs = _unique_ints(rng, candidate_count, 10, 200)
    names = [f"candidate {index + 1}" for index in range(candidate_count)]

    chosen_seed = rng.randrange(candidate_count)
    minimum_capacity = max(1, capacities[chosen_seed] - rng.randint(0, 10))
    maximum_latency = latencies[chosen_seed] + rng.randint(0, 10)
    feasible = [
        index
        for index in range(candidate_count)
        if capacities[index] >= minimum_capacity and latencies[index] <= maximum_latency
    ]
    if not feasible:
        feasible = [chosen_seed]
    correct = min(feasible, key=costs.__getitem__)
    state = (
        f"Requirements: capacity at least {minimum_capacity}; latency at most {maximum_latency}. "
        "Candidates: "
        + "; ".join(
            f"{name}: capacity {capacity}, latency {latency}, cost {cost}"
            for name, capacity, latency, cost in zip(
                names, capacities, latencies, costs, strict=True
            )
        )
        + "."
    )
    candidates, label = _shuffle_labeled(rng, names, correct)
    return GeneratedDecision(
        request_id=request_id,
        task_family="constraint-choice",
        state=state,
        question="Among candidates meeting both requirements, which has the lowest cost?",
        candidates=candidates,
        label=label,
    )


Generator = Callable[[random.Random, str, int], GeneratedDecision]

GENERATORS: dict[str, Generator] = {
    "largest-value": _largest_value,
    "closest-target": _closest_target,
    "capacity-fit": _capacity_fit,
    "deadline-fit": _deadline_fit,
    "route-cost": _route_cost,
    "efficiency": _efficiency,
    "sequence-next": _sequence_next,
    "constraint-choice": _constraint_choice,
}


def generate_decisions(
    *,
    count: int,
    seed: int,
    candidate_min: int = 2,
    candidate_max: int = 8,
    families: Sequence[str] | None = None,
) -> list[GeneratedDecision]:
    if count < 1:
        raise ValueError("count must be positive")
    if candidate_min < 2 or candidate_max < candidate_min:
        raise ValueError("invalid candidate-count range")
    selected = tuple(families or GENERATORS.keys())
    if not selected:
        raise ValueError("at least one task family is required")
    unknown = set(selected) - set(GENERATORS)
    if unknown:
        raise ValueError(f"unknown task families: {sorted(unknown)}")

    rng = random.Random(seed)
    output: list[GeneratedDecision] = []
    for index in range(count):
        family = selected[index % len(selected)]
        candidate_count = rng.randint(candidate_min, candidate_max)
        request_id = _stable_id(family, seed, index)
        output.append(GENERATORS[family](rng, request_id, candidate_count))
    return output


def write_decision_jsonl(path: str | Path, decisions: Sequence[GeneratedDecision]) -> None:
    if not decisions:
        raise ValueError("decisions must not be empty")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision.to_record(), sort_keys=True) + "\n")


def dataset_summary(decisions: Sequence[GeneratedDecision]) -> dict[str, Any]:
    if not decisions:
        raise ValueError("decisions must not be empty")
    family_counts = Counter(decision.task_family for decision in decisions)
    position_counts = Counter(decision.label for decision in decisions)
    return {
        "generator_version": GENERATOR_VERSION,
        "examples": len(decisions),
        "family_counts": dict(sorted(family_counts.items())),
        "label_position_counts": {str(key): value for key, value in sorted(position_counts.items())},
        "candidate_min": min(len(decision.candidates) for decision in decisions),
        "candidate_max": max(len(decision.candidates) for decision in decisions),
    }
