"""Second-stage audit: exact resolver observation, no trainer/updates."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from algorithm.happo.evaluation import evaluate_actors
from tools import audit_v311_attack_geometry as a
from tools import audit_v311_happo_failure as old


@pytest.fixture
def config():
    return a.load_environment_config(a.ROOT/"configs/env_v311.yaml")


@pytest.mark.parametrize("d,ata,aa,gate", [(1000,0,0,True),(3000,0,0,True),
    (999,0,0,False),(3001,0,0,False),(2000,30,0,False),(2000,0,90,False),
    (2000,29,89,True),(2000,0,180,False)])
def test_literal_gate_real_resolver_thresholds(config,d,ata,aa,gate):
    flags=a.conditions(d,np.deg2rad(ata),np.deg2rad(aa),config["combat"])
    assert all(flags)==gate
    g=SimpleNamespace(distance=d,ata=np.deg2rad(ata),aa=np.deg2rad(aa))
    assert all(flags)==old.inside(g,config)


def test_AA_reference_direction_and_radian_units(config):
    from env.models import AircraftState
    u=AircraftState(0,0,6000,275,0,0)
    parallel=AircraftState(2000,0,6000,275,0,0)
    headon=AircraftState(2000,0,6000,275,0,np.pi)
    assert a.geometry(u,parallel).aa==0
    assert a.geometry(u,headon).aa==np.pi
    assert a.geometry(u,headon).ata==0
    assert not all(a.conditions(2000,0,np.pi,config["combat"]))


def test_actual_multiattacker_multitarget_streak(config):
    env=a.Env(config); env.reset(seed=0)
    for aid,e in env.entities.items(): e.state.alive=aid in ("MAV","UAV1","UAV2","Blue1","Blue2")
    for aid in ("UAV1","UAV2"):
        st=env.entities[aid].state; st.x=0; st.y=0; st.h=6000; st.psi=0; st.theta=0
    for bid in ("Blue1","Blue2"):
        st=env.entities[bid].state; st.x=2000; st.y=0; st.h=6000; st.psi=0; st.theta=0
    for expected in (1,2):
        events,deaths=env._resolve_attacks()
        assert not events and not deaths
        assert all(env._attack_streak[(u,b)]==expected for u in ("UAV1","UAV2") for b in ("Blue1","Blue2"))
    events,deaths=env._resolve_attacks()
    assert len(events)==4 and len(deaths)==2
    assert all(env._attack_streak[(u,b)]==0 for u in ("UAV1","UAV2") for b in ("Blue1","Blue2"))


def test_phase_is_preaction_not_kill_transition():
    assert [a.phase(n) for n in (4,3,2,1)]==["P0","P1","P1","P2"]


def test_sensitivity_changes_one_variable(config):
    original=deepcopy(config["combat"])
    for _,field,value,c in a.envelope_specs(original):
        if field in ("none","supplement_combination"): continue
        changed=[k for k in c if c[k]!=original[k]]
        assert changed in ([],["distance" if field.startswith("distance_") else field])
        if field=="distance_min": assert c["distance"][1]==3000
        if field=="distance_max": assert c["distance"][0]==1000
    assert config["combat"]==original


def test_offline_streak_pair_identity_and_gaps(config):
    def row(step,aid="UAV1",d=2000):
        return dict(environment_seed=1000,agent=aid,blue="Blue1",episode_step=step,
                    distance_m=d,ATA_deg=0,AA_deg=0)
    rows=[row(1),row(2),row(3),row(4,d=4000),row(5),row(7)]
    assert [s for _,_,s in a.offline_streaks(rows,config["combat"])]==[1,2,3,0,1,1]


def test_scope_preserves_P1_to_P2_real_streak_continuity(config):
    rows=[]
    for step in (1,2,3):
        rows.append(dict(environment_seed=1000,agent="UAV1",blue="Blue1",episode_step=step,
                         distance_m=2000,ATA_deg=0,AA_deg=0,distance_gate=1,ATA_gate=1,AA_gate=1,
                         full_gate=1,attack_streak=step,streak_before=step-1,
                         phase="P1" if step<3 else "P2",outcome="red",UAV_speed=275,Blue_speed=275,
                         closure_mps=0,boundary_margin_m=5000,direct_visible=1,datalink_visible=0,
                         team_visible=1,action_saturation=0,remaining_steps=72,optimistic_range_time_s=0,Blue_target="UAV1"))
    _,sensitivity,_,_,_=a.compute_statistics(rows,config["combat"],dict(training_seed=1))
    r=next(r for r in sensitivity if r["scope"]=="P2_win" and r["setting"]=="official")
    assert r["episodes_streak_ge3"]==1 and r["first_qualifying_step_mean"]==3


def test_recover_P2_exact_counts_not_product_of_marginals(tmp_path):
    source=tmp_path/"source";source.mkdir(); output=tmp_path/"output";output.mkdir()
    episodes=[];agents=[]
    for seed in (1,2,3):
        episodes.append(dict(run=f"s{seed}",training_seed=seed,sampled_steps=2000000,episode=0,P2_transitions=5,outcome="draw"))
        for aid in a.UAVS:
            agents.append(dict(run=f"s{seed}",sampled_steps=2000000,episode=0,agent=aid,P2_geometry_samples=5,
                               P2_distance_gate_fraction=.6,P2_angle_gate_fraction=.4,P2_full_gate_fraction=.2,P2_max_real_streak=1))
    a.table(source/"replay_episodes.csv",episodes);a.table(source/"replay_agents.csv",agents)
    result=a.recover_population_p2(source,output)
    r=next(r for r in result if r["training_seed"]==1 and r["scope"]=="P2_draw")
    assert r["L0_alive_pair"]==15 and r["L1_distance"]==9 and r["L6_ATA_AND_AA"]==6 and r["L7_full_gate"]==3
    assert r["episodes_max_streak_ge1"]==1 and r["episodes_max_streak_ge3"]==0


def test_saved_sparse_trace_plot_missing_pairs_are_gaps(tmp_path):
    row=dict(episode_step=1,phase="P2",environment_seed=1000,winner="draw")
    for aid in (*a.RED_IDS,*a.BLUE_IDS):
        row.update({f"{aid}_{k}":0. for k in ("x","y","pre_x","pre_y","heading_deg")})
        row[f"{aid}_alive"]=1;row[f"{aid}_death_cause"]=None
    for aid in a.UAVS:
        row.update({f"{aid}_action_{d}":0. for d in range(3)})
        for bid in a.BLUE_IDS:
            row[f"{aid}_{bid}_distance_m"]=None
            row[f"{aid}_{bid}_full_gate"]=None
    a.plot_case([row],tmp_path,1)
    assert len(list(tmp_path.glob("*.png")))==7


def test_matched_selection_prefers_both_long_P2_draws():
    rows=[]
    for e,durations in ((1000,(20,0)),(1001,(40,50)),(1002,(30,55))):
        for s in (1,2,3):
            rows.append(dict(sampled_steps="2000000",training_seed=str(s),environment_seed=str(e),
                             action_seed=str(e+1000),episode=str(e-1000),outcome="red" if s==1 else "draw",
                             P2_transitions="0" if s==1 else str(durations[s-2])))
    assert a.choose_case(rows)["environment_seed"]==1001


@pytest.mark.parametrize("condition",["hold1","ATA45","AA120","range4km","horizon100","sensor8km"])
def test_counterfactual_no_original_mutation(config,condition):
    before=deepcopy(config); cf=a.diagnostic_config(config,condition)
    assert config==before and cf!=config
    # Validated real environment accepts each isolated in-memory diagnostic.
    env=a.Env(cf); env.reset(seed=1000)
    assert config==before


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA-only replay")
def test_cuda_replay_matches_evaluator_rng_parameters_and_reproducibility(config,tmp_path):
    torch.manual_seed(7)
    actors=a.IndependentActors(hidden_dim=16).cuda().eval()
    params={k:v.clone() for k,v in actors.state_dict().items()}
    cpu_rng=torch.get_rng_state().clone(); cuda_rng=[r.clone() for r in torch.cuda.get_rng_state_all()]
    checkpoint=tmp_path/"test.pt"; torch.save({"actors":actors.state_dict()},checkpoint); before=a.sha(checkpoint)
    result,rows,pairs=a.replay_episode(actors,config,1000,2000)
    assert torch.equal(cpu_rng,torch.get_rng_state())
    assert all(torch.equal(x,y) for x,y in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    formal=evaluate_actors(actors,config,1,"main",1000,"cuda",deterministic=False,action_seed=2000)[0]
    assert result==formal
    r2,rows2,pairs2=a.replay_episode(actors,config,1000,2000)
    assert result==r2 and rows==rows2 and pairs==pairs2
    assert len(rows)==result["episode_length"]
    assert all(torch.equal(params[k],v) for k,v in actors.state_dict().items())
    assert a.sha(checkpoint)==before
    assert all(r["attack_streak"]==(r["streak_before"]+1 if r["full_gate"] else 0) for r in pairs)
    stats=a.compute_statistics(pairs,config["combat"],dict(training_seed=1,population="test"))
    assert len(stats[0])==44 and len(stats[1])==88
