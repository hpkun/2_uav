from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from algorithm.happo.dbm_rgaa import DBM_RGAA_METHOD, build_method_actors
from algorithm.happo.evaluation import evaluate_actors
from env.mavuav import load_environment_config
from tools.audit_tacm_last_blue_cleanup import (
    cleanup_summary, parse_args, run_episode, select_trace_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def v310():
    return deepcopy(load_environment_config(ROOT / "configs/env_v310.yaml"))


def test_diagnostic_policy_episode_matches_formal_evaluator_exactly():
    actors = build_method_actors(method_variant=DBM_RGAA_METHOD, training_seed=5)
    loaded = {
        "actors": actors, "environment_config": v310(), "training_seed": 5,
        "sampled_steps": 1_000_000,
        "checkpoint": ROOT / "synthetic/checkpoint_final.pt",
    }
    audited = run_episode(
        loaded, episode=0, environment_seed=3000, action_seed=4000, device="cpu",
    )["episode"]
    formal = evaluate_actors(
        actors, v310(), 1, "main", seed=3000, device="cpu",
        deterministic=False, action_seed=4000,
    )[0]
    for audit_key, formal_key in (
        ("outcome", "outcome"), ("episode_return", "episode_return"),
        ("episode_length", "episode_length"), ("red_attack_kills", "red_attack_kills"),
        ("blue_attack_kills", "blue_attack_kills"), ("mav_survived", "mav_survived"),
        ("red_uav_survivors", "red_uav_survivors"),
    ):
        assert audited[audit_key] == formal[formal_key]
    assert audited["implementation_violation_count"] == 0


def test_cleanup_summary_detects_geometry_streak_selector_and_attrition():
    episode = {
        "checkpoint": "run/checkpoint_final.pt", "training_seed": 5, "episode": 2,
        "environment_seed": 3002, "action_seed": 4002, "outcome": "draw",
        "episode_length": 75, "red_attack_kills": 3, "blue_attack_kills": 1,
        "mav_survived": True, "red_uav_survivors": 1, "third_kill_step": 50,
        "fourth_kill_step": None, "remaining_steps_after_third_kill": 25,
        "third_to_fourth_kill_steps": None, "last_blue_after_third": "Blue4",
    }
    steps = []
    pairs = []
    for step, distance, ata, aa, streak in (
        (50, 3500.0, 20.0, 60.0, 0),
        (51, 2500.0, 20.0, 60.0, 1),
        (52, 2400.0, 20.0, 60.0, 2),
        (53, 800.0, 20.0, 60.0, 0),
    ):
        steps.append({
            "decision_step": step, "UAV1_alive": True, "UAV2_alive": False,
            "UAV3_alive": False, "UAV1_router_p1": 0.7, "UAV2_router_p1": None,
            "UAV3_router_p1": None, "UAV1_router_mode": 1,
            "UAV2_router_mode": None, "UAV3_router_mode": None,
            "blue_tracking_targets": '{"Blue4":"MAV"}',
        })
        pairs.append({
            "decision_step": step, "agent": "UAV1", "blue": "Blue4",
            "uav_alive": True, "distance_m": distance,
            "distance_gate": 1000 <= distance <= 3000, "ATA_gate": ata < 30,
            "AA_gate": aa < 90, "full_gate": 1000 <= distance <= 3000,
            "effective_attack_streak": streak, "selector_matches_blue": True,
            "team_visible": True, "direct_visible": True, "datalink_visible": False,
            "action_saturated": False,
        })
    result = {"episode": episode, "steps": steps, "pairs": pairs}
    row = cleanup_summary(result)
    assert row is not None
    assert row["alive_uavs_at_third"] == 1
    assert row["remaining_steps_after_third_kill"] == 25
    assert row["maximum_attack_streak"] == 2
    assert row["streak2_then_gate_lost_count"] == 1
    assert row["overshoot_pair_steps"] == 1
    assert row["selector_alignment"] == 1.0
    assert row["last_blue_tracks_MAV_fraction"] == 1.0


def test_trace_selection_keeps_post_third_or_last_fifteen_only():
    base_pairs = [
        {"decision_step": step, "agent": "UAV1", "blue": "Blue1"}
        for step in range(1, 76)
    ]
    steps = [
        {
            "decision_step": step, "alive_red": "MAV;UAV1", "alive_blue": "Blue1",
            "red_kill_targets": "", "blue_kill_targets": "", "boundary_deaths": "",
            "blue_tracking_targets": "{}", "mav_alive": True, "mav_R_threat": 0.0,
        }
        for step in range(1, 76)
    ]
    third = {
        "episode": {"third_kill_step": 60, "outcome": "draw", "red_attack_kills": 3,
                    "episode_length": 75},
        "steps": steps, "pairs": base_pairs,
    }
    assert min(row["decision_step"] for row in select_trace_rows(third)) == 60
    low_kill = {
        "episode": {"third_kill_step": None, "outcome": "draw", "red_attack_kills": 2,
                    "episode_length": 75},
        "steps": steps, "pairs": base_pairs,
    }
    assert min(row["decision_step"] for row in select_trace_rows(low_kill)) == 61


def test_cli_caps_diagnostic_at_thirty_episodes(tmp_path):
    checkpoints = [str(tmp_path / f"seed{seed}.pt") for seed in (5, 7, 9)]
    args = parse_args([*checkpoints, "--output-dir", str(tmp_path / "out")])
    assert args.episodes == 30 and args.env_seed_start == 3000 and args.action_seed == 4000
    try:
        parse_args([*checkpoints, "--episodes", "31", "--output-dir", str(tmp_path / "out")])
    except SystemExit:
        pass
    else:
        raise AssertionError("31 episodes must be rejected")
