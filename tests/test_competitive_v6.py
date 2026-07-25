import numpy as np
import pytest
from uav_combat.mappo import trainer as module
from uav_combat.mappo.trainer import competitive_score,evaluate_paired_policies,new_funnel,summarize_competitive_records

def record(reason,outcome,red_id="a",blue_id="b",scenario="tail_chase",length=10):
    winner=red_id if outcome=="red" else blue_id if outcome=="blue" else None
    loser=blue_id if winner==red_id else red_id if winner==blue_id else None
    return {"reason":reason,"outcome":outcome,"length":length,"returns":np.zeros(2),"funnels":{"red":new_funnel(),"blue":new_funnel()},"red_policy_id":red_id,"blue_policy_id":blue_id,"policy_a_team":"red" if red_id=="a" else "blue","policy_b_team":"red" if red_id=="b" else "blue","winner_policy":winner,"loser_policy":loser,"scenario":scenario,"rear_team":None}

def test_metrics_separate_policy_color_kill_and_boundary():
    rows=[record("red_kill","red"),record("blue_kill","blue"),record("xy_boundary","red"),record("altitude_boundary","blue"),record("collision","draw")]
    result=summarize_competitive_records(rows)
    assert result["red_kills"]==result["blue_kills"]==1 and result["combat_decisive_rate"]==.4
    assert result["policy_a_kills"]==result["policy_b_kills"]==1
    assert result["policy_a_boundary_losses"]==result["policy_b_boundary_losses"]==1
    assert result["policy_a_role_kill_gap"]==result["policy_b_role_kill_gap"]==.2

def test_paired_evaluation_reuses_seed_scenario_rear_and_swaps_colors(monkeypatch):
    calls=[]
    def fake(red,blue,red_id,blue_id,env_config,seed,scenario,rear):
        calls.append((red_id,blue_id,seed,scenario,rear))
        return record("max_steps","draw",red_id,blue_id,scenario)
    monkeypatch.setattr(module,"_run_episode",fake)
    result=evaluate_paired_policies(lambda *x:0,lambda *x:0,"unused",6,seed=100)
    assert calls[0][0:2]==("a","b") and calls[1][0:2]==("b","a")
    for i in range(0,6,2):assert calls[i][2:]==calls[i+1][2:]
    assert result["overall"]["episodes"]==6
    assert sum(r["red_policy_id"]=="a" for r in result["records"])==3
    with pytest.raises(ValueError,match="even"):evaluate_paired_policies(lambda *x:0,lambda *x:0,"unused",3)

def evaluation(worst,a,b,combat=.5,a_boundary=0,b_boundary=0,a_gap=0,b_gap=0,collision=0):
    overall={"worst_scenario_combat_decisive_rate":worst,"policy_a_kill_rate":a,"policy_b_kill_rate":b,"min_policy_kill_rate":min(a,b),"paired_combat_decisive_rate":combat,"policy_a_boundary_loss_rate":a_boundary,"policy_b_boundary_loss_rate":b_boundary,"policy_a_role_kill_gap":a_gap,"policy_b_role_kill_gap":b_gap,"collision_rate":collision}
    return {"overall":overall,"by_scenario":{"x":{"combat_decisive_rate":worst}}}

def test_v6_score_prioritizes_scenarios_then_bilateral_skill():
    assert competitive_score(evaluation(.2,.2,.2))>competitive_score(evaluation(.1,.9,.9))
    assert competitive_score(evaluation(.2,.2,.2))>competitive_score(evaluation(.2,.4,0))
    assert competitive_score(evaluation(.2,.2,.2,a_boundary=.1))>competitive_score(evaluation(.2,.2,.2,a_boundary=.2))
