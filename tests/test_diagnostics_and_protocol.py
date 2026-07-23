from pathlib import Path
from types import SimpleNamespace
import inspect
import numpy as np
from uav_combat.controller import TargetStateController
from uav_combat.dynamics import PointMassDynamics
from uav_combat.environment import HomogeneousAirCombatEnv
from uav_combat.integrator import RK4Integrator
from uav_combat.models import AircraftState
from uav_combat.mappo.trainer import new_funnel,update_funnel
from scripts.evaluate_rule_baselines import run_matchup
from scripts.train_mappo import phase_spec
import scripts.finalize_fixed_diagnostics as finalizer

CONFIG=Path(__file__).parents[1]/"configs/homogeneous_1v1.yaml"


def test_zero_action_preserves_level_heading_pitch_and_speed(spec):
    controller=TargetStateController(); dynamics=PointMassDynamics(); integrator=RK4Integrator(0.1); state=AircraftState(0,0,-3000,150,0,0)
    for _ in range(100):
        _,control=controller.control_from_action(state,np.zeros(3),spec); state=integrator.step(state,control,dynamics,spec)
    assert np.allclose([state.v,state.theta,state.psi],[150,0,0],atol=1e-12)


def test_actual_rates_and_tracking_errors_use_dynamics_derivative():
    env=HomogeneousAirCombatEnv(CONFIG); env.reset(9,"tail_chase","red"); old={aircraft.aircraft_id:aircraft.state.copy() for aircraft in env.aircraft}; _,_,_,_,info=env.step({"red_0":np.array([.4,-.2,.7]),"blue_0":np.zeros(3)})
    for aircraft in env.aircraft:
        diagnostics=info["control_diagnostics"][aircraft.aircraft_id]; derivative=env.dynamics.derivatives(old[aircraft.aircraft_id],info["controls"][aircraft.aircraft_id])
        assert np.array_equal([diagnostics["actual_acceleration"],diagnostics["actual_pitch_rate"],diagnostics["actual_yaw_rate"]],derivative[3:6])
        for label,commanded,actual in (("acceleration","clipped_acceleration","actual_acceleration"),("pitch_rate","clipped_pitch_rate","actual_pitch_rate"),("yaw_rate","clipped_yaw_rate","actual_yaw_rate")):
            error=diagnostics[commanded]-diagnostics[actual]; assert diagnostics[f"{label}_tracking_error"]==error; assert diagnostics[f"{label}_tracking_absolute_error"]==abs(error)


def test_joint_gates_require_same_step_and_violation_margins():
    combat={"attack_distance_min":100.0,"attack_distance_max":1000.0,"attack_ata_max":np.pi/6,"attack_aa_max":np.pi/2}; funnel=new_funnel()
    update_funnel(funnel,SimpleNamespace(distance=500.0,ata=np.pi/3,aa=0.0),combat,False); update_funnel(funnel,SimpleNamespace(distance=1200.0,ata=0.0,aa=0.0),combat,False)
    assert funnel["ever_within_attack_distance"] and funnel["ever_satisfy_ata"] and funnel["ever_satisfy_aa"]
    assert not funnel["ever_distance_and_ata"] and not funnel["ever_full_attack_envelope"] and funnel["ever_ata_and_aa"]
    assert funnel["minimum_distance_violation"]==0 and funnel["minimum_ata_violation"]==0 and funnel["minimum_aa_violation"]==0
    expected=min((np.pi/6)/np.pi,200/1000); assert np.isclose(funnel["minimum_combined_violation"],expected)


def test_full_attack_envelope_matches_environment_attack():
    env=HomogeneousAirCombatEnv(CONFIG); env.reset(scenario_name="fixed"); red,blue=env.aircraft; red.state.x=0; red.state.psi=0; blue.state.x=500; blue.state.psi=0
    _,_,_,_,info=env.step({"red_0":np.zeros(3),"blue_0":np.zeros(3)}); funnel=new_funnel(); update_funnel(funnel,info["geometries"]["red_0"],env.config["combat"],info["attacks"]["red_0"])
    assert funnel["ever_full_attack_envelope"]==info["attacks"]["red_0"]==True


def test_rule_baseline_single_episode_per_scenario_runs():
    result=run_matchup(CONFIG,"pursuit","zero",episodes_per_scenario=1,seed=1234); assert result["overall"]["episodes"]==3 and set(result["by_scenario"])=={"tail_chase","offset_head_on","crossing"}


def test_stage_checkpoint_names_and_finalizer_is_read_only():
    assert phase_spec("straight_tail_chase")[2:]==("straight_best.pt","straight_final.pt")
    assert phase_spec("pursuit_tail_chase")[2:]==("pursuit_tail_best.pt","pursuit_tail_final.pt")
    assert phase_spec("pursuit_all_scenarios")[2:]==("fixed_best.pt","fixed_final.pt")
    assert "torch.save" not in inspect.getsource(finalizer)


def test_action_and_controller_constants_unchanged(spec):
    controller=TargetStateController(); assert (controller.delta_yaw_max,controller.delta_pitch_max,controller.delta_speed_max)==(np.pi,np.pi/3,50.0)
    assert (spec.nx_min,spec.nx_max,spec.nz_min,spec.nz_max,spec.phi_min,spec.phi_max)==(-1.0,1.0,-3.0,3.0,-np.pi/2,np.pi/2)
