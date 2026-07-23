import numpy as np
import pytest

from uav_combat.combat import SimplifiedAttackModel, situation_score
from uav_combat.models import AircraftState


def state(x=0, y=0, psi=0):
    return AircraftState(x, y, -3000, 150, 0, psi)


@pytest.fixture
def attack_model():
    return SimplifiedAttackModel(100, 1000, np.deg2rad(30), np.pi / 2)


def test_attack_envelope_and_each_rejection(attack_model):
    attacker = state()
    assert attack_model.can_attack(attacker, state(x=500))
    assert not attack_model.can_attack(attacker, state(x=50))
    assert not attack_model.can_attack(attacker, state(x=1100))
    assert not attack_model.can_attack(state(psi=np.pi / 2), state(x=500))
    assert not attack_model.can_attack(attacker, state(x=500, psi=np.pi))


def test_situation_score_range_and_ordering():
    favorable = situation_score(state(), state(x=600), 600, 600)
    head_on = situation_score(state(), state(x=600, psi=np.pi), 600, 600)
    diverging = situation_score(state(psi=np.pi), state(x=600), 600, 600)
    assert 0 <= favorable <= 1 and 0 <= head_on <= 1 and 0 <= diverging <= 1
    assert favorable > head_on and favorable > diverging

