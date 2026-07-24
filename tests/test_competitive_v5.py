import numpy as np
from uav_combat.mappo.trainer import _add_matchup_metadata, new_funnel, summarize_competitive_records
from scripts.train_mappo import score


def record(reason, outcome, length=10):
    return {"reason": reason, "outcome": outcome, "length": length, "returns": np.zeros(2), "funnels": {"red": new_funnel(), "blue": new_funnel()}}


def test_competitive_metrics_separate_kills_outcomes_and_other_causes():
    rows = [
        record("red_kill", "red", 5),
        record("blue_kill", "blue", 7),
        record("altitude_boundary", "red", 9),
        record("xy_boundary", "blue", 11),
        record("collision", "draw", 13),
        record("mutual_kill", "draw", 15),
        record("max_steps", "draw", 17),
    ]
    result = summarize_competitive_records(rows)
    assert result["red_outcome_wins"] == 2 and result["blue_outcome_wins"] == 2 and result["draws"] == 3
    assert result["red_kills"] == result["blue_kills"] == 1
    assert result["combat_decisive_rate"] == 2 / 7
    assert result["non_draw_rate"] == 4 / 7
    assert result["red_boundary_losses"] == result["blue_boundary_losses"] == 1
    assert result["boundary_rate"] == 2 / 7
    assert result["altitude_boundary_rate"] == result["xy_boundary_rate"] == 1 / 7
    assert result["collision_count"] == result["mutual_kill_count"] == result["max_steps_count"] == 1
    assert result["mean_episode_length"] == 11


def test_boundary_outcome_win_is_not_combat_success():
    result = summarize_competitive_records([record("altitude_boundary", "red")])
    assert result["red_outcome_win_rate"] == 1
    assert result["red_kill_rate"] == result["blue_kill_rate"] == result["combat_decisive_rate"] == 0
    assert result["blue_boundary_loss_rate"] == 1


def test_blue_matchup_metadata_does_not_require_color_inference():
    result = {"overall": summarize_competitive_records([record("blue_kill", "blue"), record("xy_boundary", "red")])}
    result = _add_matchup_metadata(result, "blue_vs_zero")
    assert result["matchup"] == "blue_vs_zero" and result["learned_side"] == "blue"
    assert result["learned_kills"] == 1 and result["learned_kill_rate"] == .5
    assert result["learned_boundary_losses"] == 1 and result["learned_boundary_loss_rate"] == .5
    assert result["opponent_kills"] == 0


def test_competitive_best_prefers_kills_then_safety_then_length():
    evaluation = lambda combat, boundary, collision, length: {"overall": {"combat_decisive_rate": combat, "boundary_rate": boundary, "collision_rate": collision, "mean_episode_length": length}}
    assert score(evaluation(.1, .9, 0, 500)) > score(evaluation(0, 0, 0, 1))
    assert score(evaluation(.2, .1, .1, 500)) > score(evaluation(.2, .2, .1, 1))
    assert score(evaluation(.2, .1, .1, 100)) > score(evaluation(.2, .1, .1, 200))
