"""Three scientific invariants; integration is checked by running the pipeline."""

import pytest

from common import cfr_weights, pair_metrics
from prepare import normalize, split_pairs


def row(prompt, preference=-2, left="a", right="b"):
    return dict(
        domain="general",
        context=[{"role": "user", "content": prompt}],
        response1=left,
        response2=right,
        overall_preference=preference,
    )


def test_preference_direction_and_duplicate_disagreements():
    pairs, _ = normalize(
        [
            row("one"),
            row("two", 2),
            row("tie", 0),
            row("conflict"),
            row("conflict", 2),
            row("one", 2, "b", "a"),
        ]
    )
    assert [(p["context"][0]["content"], p["label"]) for p in pairs] == [
        ("one", 1),
        ("two", -1),
    ]


def test_prompt_splits_and_official_holdout_ownership():
    development, _ = normalize([row(str(i)) for i in range(30)] + [row("sealed")])
    test, _ = normalize([row("sealed")])
    config = {
        "seed": 42,
        "data": {
            "validation_fraction": 0.2,
            "train_prompts": 100,
            "validation_prompts": 100,
            "test_prompts": 100,
        },
    }
    parts = split_pairs(development, test, config)
    prompts = {key: {p["prompt_id"] for p in rows} for key, rows in parts.items()}
    assert not prompts["train"] & prompts["validation"]
    assert not (prompts["train"] | prompts["validation"]) & prompts["test"]
    assert sum(len(rows) for rows in parts.values()) == 31


def test_reverse_pair_cfr_balances_observed_table_and_undefined_gap():
    directions = [1, 1, 1, -1]
    labels = [1, 1, 1, 1]
    weights = cfr_weights(directions, labels)
    assert sum(weights) == pytest.approx(4)
    assert sum(weights[:3]) == pytest.approx(weights[3])
    assert weights == pytest.approx(
        cfr_weights([-d for d in directions], [-y for y in labels])
    )
    assert pair_metrics([1, 1], [1, 1], [1, 1])["absolute_gap"] is None
