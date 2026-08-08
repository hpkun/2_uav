"""Pure, read-only helpers for the v12 mixed-policy role diagnosis.

The runner deliberately keeps these helpers independent of the trainer and
optimizer.  They only define the slot source maps and paired statistics used
by ``scripts/diagnose_v12_mixed_policy_roles.py``.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


TEAM_AGENT_IDS_V12 = ("red_0", "red_1", "red_2", "red_3")
AGENT_TO_ROLE_V12 = {
    "red_0": "support",
    "red_1": "combat_1",
    "red_2": "combat_2",
    "red_3": "combat_3",
}

COMBOS_V12_MIXED: tuple[tuple[str, dict[str, str]], ...] = (
    ("M0_all_rule", {"red_0": "rule", "red_1": "rule", "red_2": "rule", "red_3": "rule"}),
    ("M1_all_learned", {"red_0": "learned", "red_1": "learned", "red_2": "learned", "red_3": "learned"}),
    ("M2_learned_support_rule_combats", {"red_0": "learned", "red_1": "rule", "red_2": "rule", "red_3": "rule"}),
    ("M3_rule_support_learned_combats", {"red_0": "rule", "red_1": "learned", "red_2": "learned", "red_3": "learned"}),
    ("M4_rule_support_learned_combat_1", {"red_0": "rule", "red_1": "learned", "red_2": "rule", "red_3": "rule"}),
    ("M5_rule_support_learned_combat_2", {"red_0": "rule", "red_1": "rule", "red_2": "learned", "red_3": "rule"}),
    ("M6_rule_support_learned_combat_3", {"red_0": "rule", "red_1": "rule", "red_2": "rule", "red_3": "learned"}),
    ("M7_learned_support_learned_combat_1", {"red_0": "learned", "red_1": "learned", "red_2": "rule", "red_3": "rule"}),
    ("M8_learned_support_learned_combat_2", {"red_0": "learned", "red_1": "rule", "red_2": "learned", "red_3": "rule"}),
    ("M9_learned_support_learned_combat_3", {"red_0": "learned", "red_1": "rule", "red_2": "rule", "red_3": "learned"}),
)


def validate_source_map(source_map: Mapping[str, str]) -> None:
    """Validate one fixed four-slot source map without touching runtime state."""
    if tuple(source_map.keys()) != TEAM_AGENT_IDS_V12 and set(source_map) != set(TEAM_AGENT_IDS_V12):
        raise ValueError("source map must contain red_0..red_3 exactly")
    if any(value not in {"rule", "learned"} for value in source_map.values()):
        raise ValueError("source map values must be 'rule' or 'learned'")


def validate_all_combinations(combos: Sequence[tuple[str, Mapping[str, str]]] = COMBOS_V12_MIXED) -> None:
    names = []
    for name, source_map in combos:
        validate_source_map(source_map)
        names.append(name)
    if names != [f"M{i}_{name.split('_', 1)[1]}" for i, name in enumerate(names)]:
        raise ValueError("combination names must preserve M0..M9 ordering")
    if len(names) != 10 or len(set(names)) != 10:
        raise ValueError("exactly ten unique combinations are required")


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant paired outcomes."""
    n = int(b) + int(c)
    if n <= 0:
        return 1.0
    k = min(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return float(min(1.0, 2.0 * tail))


def paired_bootstrap(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    samples: int = 10000,
    seed: int = 7,
) -> dict[str, float]:
    """Reproducible paired bootstrap for ``b-a`` with a percentile CI."""
    if len(values_a) != len(values_b):
        raise ValueError("paired samples must have equal length")
    if not values_a:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    if int(samples) <= 0:
        raise ValueError("samples must be positive")
    delta = np.asarray(values_b, dtype=np.float64) - np.asarray(values_a, dtype=np.float64)
    if not np.isfinite(delta).all():
        raise ValueError("paired values must be finite")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(delta), size=(int(samples), len(delta)))
    bootstrap_means = delta[indices].mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci_high": float(np.quantile(bootstrap_means, 0.975)),
    }


def practical_equivalence(delta_ci: tuple[float, float], threshold: float) -> str:
    """Classify a paired CI against a symmetric diagnostic equivalence interval."""
    low, high = (float(delta_ci[0]), float(delta_ci[1]))
    bound = abs(float(threshold))
    if low >= bound:
        return "materially_better"
    if high <= -bound:
        return "materially_worse"
    if low >= -bound and high <= bound:
        return "practical_equivalent"
    return "inconclusive"


def select_targeted_seeds(
    rows: Sequence[Mapping[str, Any]],
    *,
    category: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Select only seeds satisfying a named, pre-registered contrast.

    This helper intentionally never fills missing categories with convenient
    seeds.  An empty result means the contrast was not present in the data.
    """
    by_key = {(str(row["checkpoint"]), str(row["combo"]), int(row["episode_seed"])): row for row in rows}
    seeds = sorted({int(row["episode_seed"]) for row in rows})
    selected: list[dict[str, Any]] = []
    for seed in seeds:
        rule = by_key.get(("best", "M0_all_rule", seed))
        if rule is None or not bool(rule.get("task_win", False)):
            continue
        if category == "support":
            contrast = by_key.get(("best", "M2_learned_support_rule_combats", seed))
            if contrast is not None and not bool(contrast.get("task_win", False)):
                selected.append({"category": "A_support", "episode_seed": seed})
        elif category == "combat":
            contrast = by_key.get(("best", "M3_rule_support_learned_combats", seed))
            if contrast is not None and not bool(contrast.get("task_win", False)):
                selected.append({"category": "B_combat", "episode_seed": seed})
        elif category.startswith("combat_"):
            slot = category.rsplit("_", 1)[-1]
            combo = {"1": "M4_rule_support_learned_combat_1", "2": "M5_rule_support_learned_combat_2", "3": "M6_rule_support_learned_combat_3"}[slot]
            contrast = by_key.get(("best", combo, seed))
            if contrast is not None and (not bool(contrast.get("task_win", False)) or int(contrast.get("red_attack_kills", 0)) < int(rule.get("red_attack_kills", 0))):
                selected.append({"category": f"C_combat_{slot}", "episode_seed": seed})
        elif category == "best_final":
            best = by_key.get(("best", "M1_all_learned", seed))
            final = by_key.get(("final", "M1_all_learned", seed))
            if best is not None and final is not None and int(best.get("red_attack_kills", 0)) > 0 and int(final.get("red_attack_kills", 0)) == 0:
                selected.append({"category": "D_best_final", "episode_seed": seed})
        else:
            raise ValueError(f"unknown targeted category: {category}")
    return selected[: int(limit)]


__all__ = [
    "TEAM_AGENT_IDS_V12",
    "AGENT_TO_ROLE_V12",
    "COMBOS_V12_MIXED",
    "validate_source_map",
    "validate_all_combinations",
    "exact_mcnemar_pvalue",
    "paired_bootstrap",
    "practical_equivalence",
    "select_targeted_seeds",
]
