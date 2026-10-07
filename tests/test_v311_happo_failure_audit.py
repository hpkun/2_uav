"""Read-only observer tests: real physics/combat, no training or long evaluation."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from tools import audit_v311_happo_failure as audit
from algorithm.happo.evaluation import evaluate_actors


@pytest.fixture
def config():
    return audit.load_environment_config(audit.ROOT/"configs/env_v311.yaml")


@pytest.mark.parametrize("distance,ata,aa,expected", [(1000,0,0,True),(3000,0,0,True),
    (999,0,0,False),(3001,0,0,False),(2000,30,0,False),(2000,0,90,False),(2000,29,89,True)])
def test_exact_reward_gate_boundaries(distance,ata,aa,expected,config):
    from types import SimpleNamespace
    g=SimpleNamespace(distance=distance,ata=np.deg2rad(ata),aa=np.deg2rad(aa))
    assert audit.inside(g,config) == expected


def test_scripted_real_streak_reward_and_multi_target(config):
    one=audit.scripted(config)
    assert [r["real_streak"] for r in one]==[1,2,3]
    assert one[-1]["event"]==config["reward"]["blue_kill"]
    assert one[-1]["deaths"]=={"Blue1":"red_attack"}
    all_kills=audit.scripted(config,4)
    assert all_kills[-1]["event"]==400 and all_kills[-1]["terminal"]==100
    assert all_kills[-1]["team_reward"]==500
    loss=audit.scripted(config,4,"UAV2")
    assert loss[0]["event"]==-10 and loss[0]["deaths"]=={"UAV2":"boundary"}
    assert sum(r["event"]+r["terminal"] for r in loss)==490


def test_visibility_boundary_has_datalink_and_no_sensing_reward_jump(config):
    rows=audit.visibility_boundary(config)
    assert [r["direct"] for r in rows]==[True,True,False]
    assert [r["datalink"] for r in rows]==[False,False,True]
    assert all(r["target"]=="Blue1" for r in rows)
    assert all(any(abs(v)>0 for v in r["enemy_geometry"]) for r in rows)
    assert max(abs(rows[i+1]["process"]-rows[i]["process"]) for i in range(2))<.02


def test_episode_increment_weighting():
    rows=[dict(sampled_steps="2048",completed_episodes="2",mean_red_attack_kills="0",red_win_rate="0",entropy="3",critic_loss="2",curriculum_alpha="0",**{f"actor_{i}_loss":"0" for i in range(4)}),
          dict(sampled_steps="4096",completed_episodes="5",mean_red_attack_kills="1",red_win_rate="0",entropy="4",critic_loss="2",curriculum_alpha="0",**{f"actor_{i}_loss":"0" for i in range(4)})]
    phases,_=audit.phase_rows(rows,"fixture")
    assert phases[0]["episodes"]==5
    assert phases[0]["mean_red_attack_kills"]==.6


def test_range_times_are_not_gate_times(config):
    data=audit.static_geometry(config,3)
    u=[r for r in data["nominal"] if r["agent"]=="UAV2" and r["blue"]=="Blue2"][0]
    assert 5<u["straight_line_5km_s"]<6
    assert u["aa_deg"]>170 and u["aa_change_to_threshold_deg"]>80
    assert audit.range_time([8000,6000,0],[-550,0,0],5000) is None


def test_matched_reward_ranking_is_explicit_hypothetical(config):
    rows={r["scenario"]:r for r in audit.preferences(config)["rows"]}
    assert [rows[k]["undiscounted"] for k in "ABCDE"]==[0,100,300,500,490]
    assert rows["F"]["undiscounted"]==0
    assert rows["D"]["discounted"]>rows["E"]["discounted"]>rows["C"]["discounted"]>rows["B"]["discounted"]>rows["A"]["discounted"]


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA policy replay required")
def test_cuda_observer_same_as_official_stochastic_evaluator(config):
    torch.manual_seed(7)
    actors=audit.IndependentActors(hidden_dim=16).cuda().eval()
    params={k:v.clone() for k,v in actors.state_dict().items()}
    records,agents,_,_=audit.replay(actors,config,1,"main",1000,2000)
    formal=evaluate_actors(actors,config,1,"main",1000,"cuda",deterministic=False,action_seed=2000)
    for k,v in formal[0].items():
        assert records[0][k]==v
    assert {r["agent"] for r in agents}==set(audit.RED_IDS[1:])
    assert all(torch.equal(params[k],v) for k,v in actors.state_dict().items())


def test_counterfactual_does_not_mutate_original(config):
    old=deepcopy(config)
    cf=deepcopy(config); cf["sensing"]["UAV_range"]=8000
    env=audit.Env(cf); env.reset(seed=1000)
    assert config==old and env.config["sensing"]["UAV_range"]==8000


def test_observer_excludes_dead_attacker_with_stale_counter(config):
    env=audit.Env(config); env.reset(seed=0)
    env.entities["UAV1"].state.alive=False
    env._attack_streak[("UAV1","Blue1")]=2
    observations=audit.install_observer(env)
    env._resolve_attacks()
    assert all(pair[0]!="UAV1" for pair in observations[-1]["pairs"])
    assert not any(event["attacker"]=="UAV1" for event in observations[-1]["events"])


def test_tree_equality_checks_optimizer_tensor_contents():
    a={"state":{1:{"exp_avg":torch.tensor([1.,2.])}},"steps":np.array([1,2])}
    assert audit.same_tree(a,deepcopy(a))
    b=deepcopy(a); b["state"][1]["exp_avg"][1]=3
    assert not audit.same_tree(a,b)
