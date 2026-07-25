"""Tests for paper_coupled_team_v2 reward components."""
from pathlib import Path
import numpy as np
import pytest
from uav_combat.rewards import (
    coupled_attack_advantage,
    approach_progress_reward,
    soft_boundary_risk,
    friendly_separation_risk,
    head_on_collision_risk,
    madsac_segmented_reward,
)
from uav_combat.models import AircraftState
from uav_combat.geometry import compute_pairwise_geometry

CONFIG = Path(__file__).parents[1] / "configs" / "homogeneous_3v3.yaml"


def state(x=0, y=0, z=-3000, v=150, theta=0, psi=0, alive=True):
    return AircraftState(x, y, z, v, theta, psi, alive)


class TestOldRewardUnchanged:
    def test_madsac_values_unchanged(self):
        r = madsac_segmented_reward(state(x=1000), state(x=0), "red", None, None)
        assert np.isclose(r["reward_threat"], -0.15)


class TestApproachProgress:
    def test_initial_head_on_approach_positive(self):
        """Red at (-750,0) approaching blue at (+750,0) -> closing distance."""
        prev_r = state(x=-765, z=-3000, psi=0)
        curr_r = state(x=-750, z=-3000, psi=0)
        prev_b = state(x=750, z=-3000, psi=np.pi)
        curr_b = state(x=735, z=-3000, psi=np.pi)
        val = approach_progress_reward(prev_r, curr_r, prev_b, curr_b, 1000.0, 30.0)
        # Red is closing with blue (both moving toward each other)
        assert val > 0, f"approach should be positive: {val}"

    def test_far_stationary_no_approach(self):
        r = state(x=0); b = state(x=2000, psi=np.pi)
        val = approach_progress_reward(r, r, b, b, 1000.0, 30.0)
        assert abs(val) < 1e-9

    def test_moving_away_negative(self):
        """Red and blue both moving away: distance increases -> negative approach."""
        prev_r = state(x=0, psi=0); curr_r = state(x=2, psi=0)
        prev_b = state(x=2000, psi=0); curr_b = state(x=2005, psi=0)
        # Both heading +x, blue ahead of red. Distance increases -> closing negative.
        # Distance > 1000 threshold, so approach_progress_reward is active.
        val = approach_progress_reward(prev_r, curr_r, prev_b, curr_b, 1000.0, 30.0)
        assert val <= 0, f"moving away should be negative or zero: {val}"


class TestCoupledAttackAdvantage:
    def test_ideal_tail_attack_near_1(self):
        """Red 600m behind blue, both heading same direction."""
        r = state(x=0, psi=0); b = state(x=600, psi=0)
        score = coupled_attack_advantage(r, b, 600.0, 450.0, 0.5236, 1.0472)
        assert 0.8 < score <= 1.0, f"tail attack should be near 1: {score}"

    def test_head_on_lower_than_tail(self):
        r = state(x=0, psi=0); b = state(x=600, psi=np.pi)
        head_on = coupled_attack_advantage(r, b, 600.0, 450.0, 0.5236, 1.0472)
        b_tail = state(x=600, psi=0)
        tail = coupled_attack_advantage(r, b_tail, 600.0, 450.0, 0.5236, 1.0472)
        assert tail > head_on, f"tail={tail:.4f} should > head_on={head_on:.4f}"

    def test_far_distance_reduces_score(self):
        r = state(x=0, psi=0); b = state(x=600, psi=0)
        close = coupled_attack_advantage(r, b, 600.0, 450.0, 0.5236, 1.0472)
        b_far = state(x=3000, psi=0)
        far = coupled_attack_advantage(r, b_far, 600.0, 450.0, 0.5236, 1.0472)
        assert close > far

    def test_threat_penalty_when_blue_tail_attacks_red(self):
        """Blue behind red: blue can attack red."""
        b = state(x=0, psi=0); r = state(x=600, psi=0)
        threat = coupled_attack_advantage(b, r, 600.0, 450.0, 0.5236, 1.0472)
        assert threat > 0.8


class TestSoftBoundary:
    def test_monotonic_toward_boundary(self):
        r1 = soft_boundary_risk(state(x=16000), 20000, 20000, 500, 6000, 0.8, 750)
        r2 = soft_boundary_risk(state(x=18000), 20000, 20000, 500, 6000, 0.8, 750)
        r3 = soft_boundary_risk(state(x=19500), 20000, 20000, 500, 6000, 0.8, 750)
        assert r1["total_risk"] < r2["total_risk"] < r3["total_risk"]

    def test_monotonic_toward_altitude_boundary(self):
        """alt=600 is closer to ground than alt=1000 -> more risk."""
        r_high = soft_boundary_risk(state(z=-4000), 20000, 20000, 500, 6000, 0.8, 750)  # alt=4000, safe
        r_low  = soft_boundary_risk(state(z=-600), 20000, 20000, 500, 6000, 0.8, 750)   # alt=600, near min
        assert r_low["total_risk"] > r_high["total_risk"]


class TestFriendlySeparation:
    def test_close_friendly_penalty(self):
        own = state(x=0); mates = [state(x=10)]
        risk = friendly_separation_risk(own, mates, 200.0, 30.0)
        assert risk > 0.1

    def test_safe_distance_no_penalty(self):
        own = state(x=0); mates = [state(x=500)]
        risk = friendly_separation_risk(own, mates, 200.0, 30.0)
        assert risk == 0.0


class TestHeadOnRisk:
    def test_tail_chase_no_head_on(self):
        r = state(x=0, psi=0); b = state(x=200, psi=0)
        risk = head_on_collision_risk(r, b, 300.0, 0.5236)
        assert risk == 0.0, "tail chase should not trigger head-on"

    def test_close_head_on_triggers(self):
        r = state(x=0, psi=0); b = state(x=200, psi=np.pi)
        risk = head_on_collision_risk(r, b, 300.0, 0.5236)
        assert risk > 0.1, f"head-on at 200m should trigger: {risk}"


class TestV2RewardValues:
    def test_kill_reward_exact_20(self):
        """Verify config kill_reward = 20."""
        import yaml
        with open(CONFIG) as f: cfg = yaml.safe_load(f)
        assert cfg["reward_v2"]["kill_reward"] == 20.0

    def test_attack_death_penalty_exact_20(self):
        import yaml
        with open(CONFIG) as f: cfg = yaml.safe_load(f)
        assert cfg["reward_v2"]["attack_death_penalty"] == 20.0

    def test_boundary_death_penalty_exact_30(self):
        import yaml
        with open(CONFIG) as f: cfg = yaml.safe_load(f)
        assert cfg["reward_v2"]["boundary_death_penalty"] == 30.0

    def test_collision_death_penalty_exact_25(self):
        import yaml
        with open(CONFIG) as f: cfg = yaml.safe_load(f)
        assert cfg["reward_v2"]["collision_death_penalty"] == 25.0

    def test_complete_elimination_bonus_exact_20(self):
        import yaml
        with open(CONFIG) as f: cfg = yaml.safe_load(f)
        assert cfg["reward_v2"]["complete_elimination_bonus"] == 20.0

    def test_max_steps_penalty_exact_5(self):
        import yaml
        with open(CONFIG) as f: cfg = yaml.safe_load(f)
        assert cfg["reward_v2"]["max_steps_penalty"] == 5.0

    def test_dense_range(self):
        """Dense reward must be in [-0.15, 0.05]."""
        import yaml
        with open(CONFIG) as f: cfg = yaml.safe_load(f)
        assert cfg["reward_v2"]["dense_reward_min"] == -0.15
        assert cfg["reward_v2"]["dense_reward_max"] == 0.05

    def test_random_states_finite_and_in_range(self):
        """For random states, all reward components are finite."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            r = state(x=rng.uniform(-20000, 20000), y=rng.uniform(-20000, 20000),
                      z=rng.uniform(-6000, -500), v=rng.uniform(100, 250),
                      psi=rng.uniform(-np.pi, np.pi))
            b = state(x=rng.uniform(-20000, 20000), y=rng.uniform(-20000, 20000),
                      z=rng.uniform(-6000, -500), v=rng.uniform(100, 250),
                      psi=rng.uniform(-np.pi, np.pi))
            s1 = coupled_attack_advantage(r, b, 600, 450, 0.5236, 1.0472)
            assert np.isfinite(s1) and 0 <= s1 <= 1
            s2 = approach_progress_reward(r, r, b, b, 1000, 30)
            assert np.isfinite(s2) and -1 <= s2 <= 1
            s3 = soft_boundary_risk(r, 20000, 20000, 500, 6000, 0.8, 750)
            assert np.isfinite(s3["total_risk"]) and 0 <= s3["total_risk"] <= 2
            s4 = friendly_separation_risk(r, [b], 200, 30)
            assert np.isfinite(s4) and 0 <= s4 <= 1
            s5 = head_on_collision_risk(r, b, 300, 0.5236)
            assert np.isfinite(s5) and 0 <= s5 <= 1
