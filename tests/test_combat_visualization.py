from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithm.happo.networks import IndependentActors
from algorithm.happo.evaluation import evaluate_recurrent_actors
from algorithm.happo.recurrent import RecurrentIndependentActors
from algorithm.modules.hrta import HRTAIndependentActors
from algorithm.modules.structured_uniform import StructuredUniformIndependentActors
from env.mavuav import ENTITY_IDS, ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, load_environment_config
from tools.combat_visualization import (HETERO_COMBAT_TRACE_SCHEMA_VERSION, STYLES, episode_cube_ranges,
                                        interpolate_trace_for_visualization,
                                        shortest_angle_interpolate, validate_raw_trace)
from tools.record_combat_episode import _event_rows, record_episode
from tools.render_combat_episode import render_episode
from tools.render_combat_episode_interactive import APP_JS, render_interactive
from tools.replay_policy import infer_method_display_name, load_replay_actors


def synthetic_trace(frames: int = 3):
    kin = np.zeros((frames, 5, 6), dtype=np.float64)
    for f in range(frames):
        for i in range(5): kin[f, i] = [f*1000+i*100, i*200, 5000+f*100, 200+i*20, .1*f, np.deg2rad(179 if f == 0 else -179)]
    alive = np.ones((frames, 5), dtype=bool); alive[-1, 3] = False
    n = frames - 1
    return {"kinematics":kin,"alive":alive,"steps":np.arange(frames),"time_s":np.arange(frames,dtype=float),
            "red_actions":np.zeros((n,3,3),np.float32),"team_reward":np.zeros(n),"team_situation":np.zeros(n),
            "event_reward":np.zeros(n),"terminal_reward":np.zeros(n),
            "minimum_friendly_red_distance":np.full(n,500.),"red_safe_distance_violation":np.zeros(n,bool)}


def synthetic_metadata():
    config=load_environment_config(None)
    return {"trace_schema_version":1,"decision_dt":1.,"algorithm":"HAPPO","evaluation_profile":"main",
            "blue_target_mode":"nearest","entity_types":{"MAV":"MAV","UAV1":"UAV","UAV2":"UAV","Blue1":"Blue","Blue2":"Blue"},
            "entity_teams":{x:("red" if i<3 else "blue") for i,x in enumerate(ENTITY_IDS)},
            "aircraft_specs":config["aircraft_specs"],"battlefield":config["battlefield"],
            "events":[{"trace_frame":2,"time_s":2.,"type":"attack","attacker":"MAV","target":"Blue1"},
                      {"trace_frame":2,"time_s":2.,"type":"attack","attacker":"UAV1","target":"Blue1"},
                      {"trace_frame":2,"time_s":2.,"type":"death","entity":"Blue1","cause":"red_attack"}],
            "outcome":"red","mav_survived":True,"red_uav_survivors":2,"blue_survivors":1,
            "red_attack_kills":1,"blue_attack_kills":0,"episode_return":12.5,"episode_length":2}


def write_trace(path: Path):
    path.mkdir(); np.savez_compressed(path/"episode_trace.npz",**synthetic_trace())
    (path/"metadata.json").write_text(json.dumps(synthetic_metadata()),encoding="utf-8")


def recurrent_checkpoint(path: Path, *, hidden_dim: int = 12, recurrent_hidden_dim: int = 9):
    architecture={"observation_dim":OBS_DIM,"encoder_dim":hidden_dim,
                  "recurrent_hidden_dim":recurrent_hidden_dim,"head_dim":hidden_dim,"action_dim":3}
    actors=RecurrentIndependentActors(hidden_dim=hidden_dim,recurrent_hidden_dim=recurrent_hidden_dim)
    payload={"environment_version":ENVIRONMENT_VERSION,"observation_dim":OBS_DIM,
             "global_state_dim":GLOBAL_STATE_DIM,"actor_variant":"recurrent",
             "method_variant":"baseline","actor_architecture":architecture,
             "trainer_config":{"actor_variant":"recurrent","method_variant":"baseline",
                               "hidden_dim":hidden_dim,"recurrent_hidden_dim":recurrent_hidden_dim},
             "environment_profile":"learnability","actors":actors.state_dict(),
             "rollout_state":{"actor_hidden_states":np.ones((1,3,recurrent_hidden_dim),np.float32),
                              "actor_recurrent_masks":np.ones((1,3),np.float32)}}
    torch.save(payload,path)
    return actors,payload


def test_schema_entity_order_altitude_positive_up_and_shapes():
    assert ENTITY_IDS == ("MAV","UAV1","UAV2","Blue1","Blue2")
    trace=synthetic_trace();validate_raw_trace(trace)
    assert trace["kinematics"].shape==(3,5,6) and trace["alive"].shape==(3,5)
    assert trace["steps"][0]==0 and trace["kinematics"][0,0,2]==5000
    assert {STYLES[x]["color"] for x in ("MAV","UAV1","UAV2")} == {"#c5163a"}
    assert {STYLES[x]["color"] for x in ("Blue1","Blue2")} == {"#4169e1"}


def test_shortest_angle_and_interpolation_no_future_leakage():
    mid=shortest_angle_interpolate(np.deg2rad(179),np.deg2rad(-179),.5)
    assert abs(abs(np.rad2deg(mid))-180)<1e-6
    trace=synthetic_trace(2);trace["alive"][1,0]=False
    visual=interpolate_trace_for_visualization(trace,1.,.25)
    assert np.allclose(visual["kinematics"][2,0,:4],(trace["kinematics"][0,0,:4]+trace["kinematics"][1,0,:4])/2)
    assert visual["kinematics"][2,0,2]>0 # h stays positive-up, never -h
    assert visual["alive"][:-1,0].all() and not visual["alive"][-1,0]
    assert visual["raw_step"].tolist()==[0,0,0,0,1]


def test_episode_cube_ranges_are_equal_grounded_and_contain_all_finite_positions():
    trace=synthetic_trace(2);kin=trace["kinematics"]
    kin[0,:,0]=np.linspace(-4000,4000,5);kin[1,:,0]=np.linspace(-4000,4000,5)
    kin[0,:,1]=np.linspace(-3000,5000,5);kin[1,:,1]=np.linspace(-3000,5000,5)
    kin[0,:,2]=np.linspace(4000,6000,5);kin[1,:,2]=np.linspace(6000,8000,5)
    trace["alive"][1,4]=False
    ranges=episode_cube_ranges(kin,trace["alive"])
    spans=[ranges[axis][1]-ranges[axis][0] for axis in ("x","y","z")]
    assert ranges["z"][0]==0.0 and np.allclose(spans,spans[0])
    assert np.isclose(ranges["span"],spans[0]) and ranges["span"]==10.0
    positions=kin[:,:,:3]/1000
    assert positions[:,:,0].min()>ranges["x"][0] and positions[:,:,0].max()<ranges["x"][1]
    assert positions[:,:,1].min()>ranges["y"][0] and positions[:,:,1].max()<ranges["y"][1]
    assert positions[:,:,2].min()>=ranges["z"][0] and positions[:,:,2].max()<ranges["z"][1]
    assert ranges["z"][1]-positions[:,:,2].max()>=1.0


def test_episode_cube_uses_finite_dead_positions_and_never_offsets_altitude():
    trace=synthetic_trace(2);trace["alive"][1,3]=False
    trace["kinematics"][1,3,:3]=[9000,7000,8500]
    ranges=episode_cube_ranges(trace["kinematics"],trace["alive"])
    assert ranges["x"][1]>9.0 and ranges["y"][1]>7.0
    assert ranges["z"]==[0.0,ranges["span"]]
    assert trace["kinematics"][1,3,2]==8500


def test_event_mapping_keeps_attack_pairs_and_boundary_safety():
    info={"attack_events":[{"attacker":"MAV","target":"Blue1"},{"attacker":"UAV1","target":"Blue1"}],
          "killed_ids":["Blue1","UAV2"],"death_causes":{"Blue1":"red_attack","UAV2":"boundary"},
          "red_safe_distance_violation":True,"minimum_friendly_red_distance":83.}
    events=_event_rows(info,4,4.)
    assert all(e["trace_frame"]==4 for e in events)
    assert [(e["attacker"],e["target"]) for e in events if e["type"]=="attack"]==[("MAV","Blue1"),("UAV1","Blue1")]
    assert not any("killer" in e for e in events)
    assert {e.get("cause") for e in events} >= {"red_attack","boundary"}
    assert any(e["type"]=="red_separation_warning" for e in events)


@pytest.mark.parametrize("variant",["vanilla","hrta","structured_uniform"])
def test_policy_loader_variants(tmp_path,variant):
    architecture={"entity_dim":32,"role_dim":8,"fusion_hidden_dim":64,"action_dim":3}
    if variant=="vanilla": actors=IndependentActors(hidden_dim=16);config={"hidden_dim":16,"actor_variant":"vanilla"};arch=None
    elif variant=="hrta": actors=HRTAIndependentActors(**architecture);config={"actor_variant":"hrta"};arch=architecture
    else: actors=StructuredUniformIndependentActors(**architecture);config={"actor_variant":"structured_uniform"};arch=architecture
    payload={"environment_version":ENVIRONMENT_VERSION,"observation_dim":OBS_DIM,"global_state_dim":GLOBAL_STATE_DIM,
             "actor_variant":variant,"method_variant":"baseline","actor_architecture":arch,"trainer_config":config,"actors":actors.state_dict()}
    path=tmp_path/f"{variant}.pt";torch.save(payload,path);loaded=load_replay_actors(path)
    assert loaded.actor_variant==variant and loaded.actors.training is False
    assert set(loaded.actors.state_dict())==set(actors.state_dict())


def test_recurrent_policy_loader_contract_and_method_name(tmp_path):
    path=tmp_path/"recurrent.pt";actors,payload=recurrent_checkpoint(path)
    loaded=load_replay_actors(path)
    assert loaded.actor_variant=="recurrent" and loaded.method_variant=="baseline"
    assert loaded.method_display_name=="R-HAPPO" and loaded.actor_architecture==payload["actor_architecture"]
    assert isinstance(loaded.actors,RecurrentIndependentActors) and loaded.actors.training is False
    assert set(loaded.actors.state_dict())==set(actors.state_dict())
    assert loaded.hidden_states is None and loaded.recurrent_masks is None


def test_recurrent_policy_loader_rejects_invalid_contracts(tmp_path):
    path=tmp_path/"recurrent.pt";_,base=recurrent_checkpoint(path)
    cases=[]
    nonbaseline={**base,"method_variant":"agp"};cases.append((nonbaseline,"only method_variant='baseline'"))
    missing={**base};missing.pop("actor_architecture");cases.append((missing,"architecture metadata"))
    extra={**base,"actor_architecture":{**base["actor_architecture"],"extra":1}};cases.append((extra,"architecture metadata"))
    wrong_obs={**base,"actor_architecture":{**base["actor_architecture"],"observation_dim":OBS_DIM+1}};cases.append((wrong_obs,"architecture dimensions"))
    wrong_head={**base,"actor_architecture":{**base["actor_architecture"],"head_dim":13}};cases.append((wrong_head,"encoder_dim must equal head_dim"))
    wrong_action={**base,"actor_architecture":{**base["actor_architecture"],"action_dim":2}};cases.append((wrong_action,"architecture dimensions"))
    wrong_hidden={**base,"trainer_config":{**base["trainer_config"],"recurrent_hidden_dim":10}};cases.append((wrong_hidden,"recurrent_hidden_dim mismatch"))
    for index,(payload,message) in enumerate(cases):
        invalid=tmp_path/f"invalid_{index}.pt";torch.save(payload,invalid)
        with pytest.raises(RuntimeError,match=message):load_replay_actors(invalid)


def test_recurrent_adapter_history_reset_and_agent_masks(tmp_path):
    torch.manual_seed(23);path=tmp_path/"recurrent.pt";recurrent_checkpoint(path)
    adapter=load_replay_actors(path);adapter.reset_episode()
    obs0={aid:np.linspace(-.2,.2,OBS_DIM,dtype=np.float32)+index*.01
          for index,aid in enumerate(("MAV","UAV1","UAV2"))}
    obs1={aid:value+.05 for aid,value in obs0.items()}
    first=adapter.actions(obs0)
    first_next=[state.clone() for state in adapter.next_hidden_states]
    assert all(torch.count_nonzero(state)>0 for state in first_next)
    adapter.after_step([1,1,1],False)
    assert all(torch.equal(state,first_next[index]) for index,state in enumerate(adapter.hidden_states))
    second=adapter.actions(obs1)
    manual=[]
    with torch.no_grad():
        for index,aid in enumerate(("MAV","UAV1","UAV2")):
            action,_,_=adapter.actors.actors[index].sample_step(
                torch.as_tensor(obs1[aid]).unsqueeze(0),first_next[index],torch.ones(1),deterministic=True)
            manual.append(action.squeeze(0).numpy())
    assert np.allclose(second,np.asarray(manual))
    adapter.after_step([1,0,1],False)
    assert torch.count_nonzero(adapter.hidden_states[1])==0 and adapter.recurrent_masks[:,0].tolist()==[1,0,1]
    assert torch.count_nonzero(adapter.hidden_states[0])>0 and torch.count_nonzero(adapter.hidden_states[2])>0
    adapter.actions(obs1);blue_death_next=[state.clone() for state in adapter.next_hidden_states]
    adapter.after_step([1,1,1],False)
    assert all(torch.equal(state,blue_death_next[index]) for index,state in enumerate(adapter.hidden_states))
    assert adapter.recurrent_masks[:,0].tolist()==[1,1,1]
    adapter.actions(obs1);adapter.after_step([1,1,1],True)
    assert all(torch.count_nonzero(state)==0 for state in adapter.hidden_states)
    assert torch.count_nonzero(adapter.recurrent_masks)==0
    reset_first=adapter.actions(obs0)
    assert np.array_equal(first,reset_first)


def test_policy_loader_rejects_contract_and_unknown(tmp_path):
    base={"environment_version":"old","observation_dim":OBS_DIM,"global_state_dim":GLOBAL_STATE_DIM,"actors":{}}
    p=tmp_path/"bad.pt";torch.save(base,p)
    with pytest.raises(RuntimeError,match="environment contract"):load_replay_actors(p)
    base.update(environment_version=ENVIRONMENT_VERSION,actor_variant="mystery",trainer_config={}) ;torch.save(base,p)
    with pytest.raises(RuntimeError,match="unsupported actor architecture for replay"):load_replay_actors(p)


def test_method_names():
    assert infer_method_display_name("vanilla","baseline")=="HAPPO"
    assert infer_method_display_name("vanilla","agp_curriculum")=="HAPPO-AGP-Curriculum"
    assert infer_method_display_name("hrta")=="HAPPO-HRTA"
    assert infer_method_display_name("structured_uniform")=="HAPPO-Structured-Uniform"
    assert infer_method_display_name("recurrent")=="R-HAPPO"


def test_deterministic_short_recording(tmp_path):
    torch.manual_seed(7);actors=IndependentActors(hidden_dim=16)
    payload={"environment_version":ENVIRONMENT_VERSION,"observation_dim":OBS_DIM,"global_state_dim":GLOBAL_STATE_DIM,
             "actor_variant":"vanilla","method_variant":"baseline","trainer_config":{"hidden_dim":16},
             "environment_profile":"learnability","actors":actors.state_dict()}
    ckpt=tmp_path/"model.pt";torch.save(payload,ckpt);adapter=load_replay_actors(ckpt)
    cfg=load_environment_config(None);cfg["simulation"]["max_decision_steps"]=2
    m1=record_episode(adapter,ckpt,tmp_path/"a",profile="learnability",blue_mode="nearest",seed=424242,env_config=cfg)
    m2=record_episode(adapter,ckpt,tmp_path/"b",profile="learnability",blue_mode="nearest",seed=424242,env_config=cfg)
    with np.load(tmp_path/"a"/"episode_trace.npz") as a,np.load(tmp_path/"b"/"episode_trace.npz") as b:
        for key in ("red_actions","kinematics","alive"): assert np.array_equal(a[key],b[key])
        assert a["kinematics"].shape[0]==m1["episode_length"]+1
    assert m1["events"]==m2["events"] and m1["outcome"]==m2["outcome"]
    assert m1["episode_role"]=="qualitative_visualization_only" and not m1["used_for_quantitative_metrics"]


def test_recurrent_recording_matches_evaluator_and_repeats(tmp_path):
    torch.manual_seed(29);checkpoint=tmp_path/"recurrent.pt";_,payload=recurrent_checkpoint(checkpoint)
    adapter=load_replay_actors(checkpoint)
    cfg=load_environment_config(None);cfg["simulation"]["max_decision_steps"]=2
    m1=record_episode(adapter,checkpoint,tmp_path/"a",profile="learnability",blue_mode="nearest",
                      seed=424242,env_config=cfg)
    m2=record_episode(adapter,checkpoint,tmp_path/"b",profile="learnability",blue_mode="nearest",
                      seed=424242,env_config=cfg)
    records=evaluate_recurrent_actors(adapter.actors,cfg,1,"nearest","learnability",seed=424242,device="cpu")
    with np.load(tmp_path/"a"/"episode_trace.npz") as a,np.load(tmp_path/"b"/"episode_trace.npz") as b:
        for key in ("red_actions","kinematics","alive"):assert np.array_equal(a[key],b[key])
    assert m1["events"]==m2["events"] and m1["outcome"]==m2["outcome"]
    assert all(m1[key]==value for key,value in records[0].items())
    assert m1["algorithm"]=="R-HAPPO" and m1["actor_variant"]=="recurrent"
    assert m1["actor_architecture"]==payload["actor_architecture"]


def test_preview_and_standalone_html(tmp_path):
    d=tmp_path/"episode";write_trace(d)
    result=render_episode(d,preview=d/"preview.png",mp4=False)
    assert Path(result["preview"]).stat().st_size>1000
    out=render_interactive(d)
    html=out.read_text(encoding="utf-8")
    assert out.stat().st_size>1_000_000 and "<script src=" not in html.lower() and "scatter3d" in html
    for token in ("Play","Pause","Previous Frame","Next Frame","Restart","slider","speed","Trail","Headings","Labels","Death markers","Attack lines","Reset Camera","Episode View","Full Battlefield","uirevision","Plotly.newPlot","Plotly.restyle"):
        assert token in html
    assert "Adjusting camera" not in html and "temporarily hidden" not in html
    payload_text=html.split("<script>const PAYLOAD=",1)[1].split(";</script>",1)[0]
    payload=json.loads(payload_text)
    spans=[payload["episode_ranges"][axis][1]-payload["episode_ranges"][axis][0] for axis in ("x","y","z")]
    assert payload["episode_ranges"]["z"][0]==0.0 and np.allclose(spans,spans[0])
    assert "NaN" not in payload_text and "// APP_JS_START" in html and "// APP_JS_END" in html
    node=shutil.which("node")
    if node:
        js=html.split("// APP_JS_START",1)[1].split("// APP_JS_END",1)[0]
        path=tmp_path/"application.js";path.write_text("// APP_JS_START"+js,encoding="utf-8")
        subprocess.run([node,"--check",str(path)],check=True,capture_output=True,text=True)


def test_app_js_uses_time_based_event_visibility():
    assert "t-Number(e.time_s)>=-1e-9" in APP_JS
    assert "Number(e.time_s) <= t + 1e-9" in APP_JS
    assert "uirevision" in APP_JS and "frame === n-1" in APP_JS
    assert APP_JS.count("'scene.xaxis.autorange':false")==1
    assert "aspectmode:'cube'" in APP_JS and "type:'mesh3d'" in APP_JS
    render_body=APP_JS.split("async function renderFrame(frameIndex) {",1)[1].split("function requestRender()",1)[0]
    assert "scene.xaxis.range" not in render_body and "Plotly.relayout" not in render_body
    assert "setEpisodeView" in APP_JS and "setFullBattlefieldView" in APP_JS
    assert "options('resetCamera').onclick=()=>Plotly.relayout(graph,{'scene.camera':P.default_camera})" in APP_JS


def test_app_js_decouples_logical_playback_rendering_and_camera_interaction(tmp_path):
    for token in ("logicalFrame", "renderedFrame", "cameraInteracting", "renderBusy", "renderPending",
                  "requestAnimationFrame", "frameAtOrBeforeTime", "beginCameraInteraction",
                  "endCameraInteraction", "pointerdown", "pointerup", "pointercancel", "wheelEndTimer"):
        assert token in APP_JS
    assert "update().then(()=>" not in APP_JS and "setTimeout(tick" not in APP_JS
    assert "renderEpoch" not in APP_JS and "hidePending" not in APP_JS
    assert "hideDynamicCombatTraces" not in APP_JS and "dynamicTraceIndices" not in APP_JS
    assert "cameraOverlay" not in APP_JS
    scheduler_body=APP_JS.split("async function runCombatMutationScheduler() {",1)[1].split("function requestRender()",1)[0]
    assert "const target=logicalFrame, completed=await renderFrame(target)" in scheduler_body
    assert "if(completed)renderedFrame=target;else renderPending=true" in scheduler_body
    assert "if(cameraInteracting)break" in scheduler_body
    request_body=APP_JS.split("function requestRender() {",1)[1].split("function logicalVisualTime",1)[0]
    assert "if(cameraInteracting || renderBusy)return" in request_body and "renderFrame(" not in request_body
    begin_body=APP_JS.split("function beginCameraInteraction() {",1)[1].split("function endCameraInteraction()",1)[0]
    end_body=APP_JS.split("function endCameraInteraction() {",1)[1].split("async function setView",1)[0]
    assert "Plotly." not in begin_body and "Plotly." not in end_body
    assert "cameraInteracting=true" in begin_body
    assert "cameraInteracting=false" in end_body and "renderedFrame!==logicalFrame" in end_body
    assert "if(!cameraInteracting)updatePanels(logicalFrame)" in APP_JS
    assert APP_JS.count("await Plotly.restyle(graph") == 6  # 5 render batches + one ground-view update
    assert "dragmode:'orbit'" in APP_JS and "scrollZoom:true" in APP_JS

    node=shutil.which("node")
    if node:
        function=APP_JS.split("function frameAtOrBeforeTime(time) {",1)[1].split("function updatePanels",1)[0]
        source=("const P={time_s:[0,0.1,0.2,0.3,0.4]};const n=P.time_s.length;"
                "function frameAtOrBeforeTime(time) {"+function+
                "if(frameAtOrBeforeTime(.25)!==2||frameAtOrBeforeTime(.4)!==4||frameAtOrBeforeTime(9)!==4)process.exit(1);")
        path=tmp_path/"playback_helper.js";path.write_text(source,encoding="utf-8")
        subprocess.run([node,str(path)],check=True,capture_output=True,text=True)
