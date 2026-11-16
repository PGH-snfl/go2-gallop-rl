"""Reproduce this collection's checks and comparison from saved evidence only."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaac_task.gallop_core import GallopSettings
from isaac_task.gallop_runtime import checkpoint_source_directory


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


checks={}
def check(name,value):checks[name]=bool(value)

s=read(OUT/'summary.json');m=s['provenance'];cfg=m['gallop_settings']
train=Path(s['train']);play=Path(s['play']);model=train/'model_final.pt'
w=torch.load(model,map_location='cpu',weights_only=True)
check('checkpoint_matches_completed_play',sha(model)==s['checkpoint_sha256']==read(play/'manifest.json')['source_checkpoint']['sha256']=='dc129d38e78654a6559657740b7a46865830ca3cefcf06c908f49cdcef53fea7')
check('completed_2000_updates_lineage_28000',s['training_complete'] and w['iter']==1999 and s['training_status']['policy_updates']==2000 and s['cumulative_iterations']==28000 and s['new_transitions']==12288000)
check('guarded_26000_parent_preserved',m['fine_tuning']['parent_cumulative_iterations']==26000 and sha(Path(m['source_checkpoint']['path']))==m['source_checkpoint']['sha256']=='57607c19b9635ef21567e7a90fdbe75acce1cab532b9f78a44637a937696939d')
check('source_snapshot_and_walking_teacher_verified',checkpoint_source_directory(model,m).is_dir())
check('runtime_core_env_preserved',all(sha(ROOT/'isaac_task'/name)==m[key] for name,key in [('gallop_core.py','gallop_core_sha256'),('gallop_env.py','gallop_env_sha256')]))
check('runtime_config_preserved',GallopSettings.load(ROOT/'configs/gallop_guarded_stability.json').to_dict()==cfg)
check('asset_unchanged',sha(Path(m['asset_path']))==m['asset_sha256'])
check('interface_49_64_12_no_imitation',m['interface']=='unitree_go2_gallop_guarded_49_64_12_v4' and m['guarded_design']['imitation'] is False and not (train/'model_adapted.pt').exists() and w['model_state_dict']['actor.0.weight'].shape[1]==49 and w['model_state_dict']['critic.0.weight'].shape[1]==64 and w['model_state_dict']['actor.6.weight'].shape[0]==12)
with np.load(play/'gallop_trace.npz') as archive:z={key:archive[key] for key in archive.files}
f=z['valid'][:,0]&z['gallop_metric_active'][:,0]
episode=json.loads((play/'episodes.jsonl').read_text().splitlines()[0])
check('completed_play_no_policy_updates',s['play_status']['status']=='completed' and s['play_status']['policy_updates']==0 and s['play_status']['policy_state_unchanged'])
check('finite_trace_single_first_episode',len(z['time'])==1500 and z['valid'][:,0].all() and z['episode_reset'][:,0].sum()==1 and z['episode_reset'][-1,0] and all(np.isfinite(v).all() for v in z.values()))
check('formal_1045_samples_from_9p12_seconds',f.sum()==1045 and np.isclose(z['time'][f][0],9.12))
check('handover_at_8p10_seconds',np.isclose(z['time'][z['startup_stage'][:,0]==3][0],8.10))
check('6000_physics_samples',z['leap_physics_samples'][-1,0]==6000 and (np.diff(z['leap_physics_samples'][:,0])==4).all())
check('no_falls_natural_30s_timeout',episode['age']==1500 and episode['terminated']==0 and episode['timed_out']==1)
check('active_speed_2p5_target_14cm_final_goal_3p0_22cm',cfg['start_speed']==2.5 and cfg['max_speed']==3. and cfg['leap_min_distance']==.22 and np.allclose(z['command'][f,0,0],2.5) and np.allclose(z['stabilization_leap_target'],.14))
check('no_promotions_or_confirmed_speed',s['curriculum']['speed_caps']==[2.5] and s['curriculum']['promotions']==0 and m['highest_confirmed_speed'] is None)
contract=m['guarded_runtime_joint_contract'];original=np.array(contract['original_limits']);physical=np.array(contract['physics_limits']);operating=np.array(contract['target_limits'])
mid=original.mean(-1);radius=np.ptp(original,axis=1)*.45
check('physical_range_original_90percent',np.allclose(physical,np.column_stack((mid-radius,mid+radius)),atol=1e-6))
margin=np.ptp(physical,axis=1)*.05
check('target_range_original_81percent',np.allclose(operating,physical+np.column_stack((margin,-margin)),atol=1e-6))
g=read(OUT/'guarded_play_diagnostics.json');a=g['aggregate']
check('independent_guard_formula_checks_pass',all(g['checks'].values()))
check('command_angles_and_rates_within_limits',a['target_operating_violation_fraction']==a['target_physical_violation_fraction']==a['target_rate_violation_fraction']==0)
check('actual_original_mechanical_range_preserved',a['original_mechanical_violation_fraction']==0)
check('solver_overshoot_reported_below_guard_tolerance',.0035<a['max_actual_new_physical_excess_rad']<.0036 and a['new_physical_violation_fraction_over_guard_tolerance']==0 and a['max_violation_seconds']==0)
check('filter_gate_fails_from_intervention',episode['guarded_limits']==0 and episode['guarded_actual']==0 and episode['guarded_intervention']>cfg['guarded_max_intervention'] and np.isclose(a['filter_intervention_fraction'],episode['guarded_intervention']))
check('angle_clip_is_original_request_statistic',np.isclose(a['raw_angle_exceed_fraction'],episode['action_clip_fraction']) and episode['action_clip_fraction']<.1)
check('feet_pair_mean_gate_passes_despite_fl_individual_excess',episode['guarded_feet']==1 and episode['guarded_front_high']<.1 and episode['guarded_hind_high']<.1 and g['foot_height']['FL']['fraction_over_cap_plus_1mm']>.1)
check('front_width_fails_hind_width_passes',np.isclose(episode['stabilization_front_width_fraction'],250/390) and np.isclose(episode['stabilization_hind_width_fraction'],554/564))
events=list(csv.DictReader((OUT/'play_flight_events.csv').open()))
check('54_flights_52_qualified_35_stable_18_aborted',len(events)==54 and sum(int(r['distance_time_qualified']) for r in events)==52 and sum(int(r['stable_landing']) for r in events)==35 and sum(int(r['recovery_aborted']) for r in events)==18)
check('event_geometry_matches_episode',np.isclose(np.mean([float(r['distance_m']) for r in events]),episode['leap_distance_mean']) and np.isclose(max(float(r['distance_m']) for r in events),episode['leap_distance_max']) and np.isclose(np.mean([float(r['air_seconds']) for r in events]),episode['leap_air_seconds_mean']))
check('landing_gate_fails_without_episode_fall',episode['stabilization_landing']==0 and np.isclose(episode['stabilization_landing_fraction'],35/53) and not episode['terminated'])
st=read(OUT/'stabilization_diagnostics.json')
check('all_failed_gates_reported',set(st['play_failed_gates'])=={'stable','stabilization_width','stabilization_landing','guarded_limits'} and episode['success']==0)
check('no_22cm_flight',st['final_distance_reference']['events']==0 and max(float(r['distance_m']) for r in events)<.22)
check('training_success_zero',st['training']['all_eligible']['episodes']==9671 and st['training']['all_eligible']['successful_episodes']==0)
check('one_play_video_no_training_clips',len(s['videos'])==1 and s['videos'][0]['kind']=='play' and s['videos'][0]['frames']==1499 and s['videos'][0]['duration_seconds']==29.98)
check('collection_no_learning',s['policy_updates_during_collection']==0)

def metrics(report):
    summary=read(report/'summary.json');p=Path(summary['play'])
    h=read(report/'heading_play_diagnostics.json')['formal_gallop'];guard=read(report/'guarded_play_diagnostics.json')
    e=json.loads((p/'episodes.jsonl').read_text().splitlines()[0]);a=guard['aggregate']
    return dict(checkpoint_sha256=summary['checkpoint_sha256'],play=str(p),formal_seconds=h['seconds'],
        command_speed=e['stabilization_target_speed'],actual_vx=h['body_velocity_mean'][0],vx_rmse=h['body_velocity_rmse'][0],
        leap_mean_m=e['leap_distance_mean'],leap_max_m=e['leap_distance_max'],air_mean_seconds=e['leap_air_seconds_mean'],
        stable_landings=e['flight_stable_landings'],aborted_landings=e['flight_aborted_landings'],landing_fraction=e['stabilization_landing_fraction'],
        front_support_width_m=h['foot_widths']['front']['by_contacts']['both_support']['mean_m'],hind_support_width_m=h['foot_widths']['hind']['by_contacts']['both_support']['mean_m'],
        front_width_fraction=e['stabilization_front_width_fraction'],hind_width_fraction=e['stabilization_hind_width_fraction'],
        front_max_height_m=max(guard['foot_height'][k]['p50_p90_p95_max_m'][-1] for k in ('FL','FR')),
        hind_max_height_m=max(guard['foot_height'][k]['p50_p90_p95_max_m'][-1] for k in ('RL','RR')),
        front_high_fraction=e['guarded_front_high'],hind_high_fraction=e['guarded_hind_high'],
        filter_intervention=a['filter_intervention_fraction'],tracking_rmse_rad=a['tracking_rmse_rad'],
        roll_pitch_rms_deg=np.rad2deg(h['roll_pitch_rms_rad']).tolist(),roll_pitch_rate_rms=h['roll_pitch_rate_rms'],
        yaw_rms_deg=float(np.rad2deg(h['yaw_target_rmse_rad'])),max_cross_track_m=h['cross_track_max_abs_from_start_m'],
        actual_physical_max_excess_rad=a['max_actual_new_physical_excess_rad'])

parent=ROOT/'reports/20261103_gallop_guarded_3mps_leap_review'
comparison=dict(parent=metrics(parent),current=metrics(OUT),
    scope='Same nominal 2.5m/s command and cadence, separate deterministic single-episode plays. Seven reward weights differ; not a causal ablation or multi-seed success estimate.')
write(OUT/'guarded_parent_comparison.json',comparison)
check('front_width_improved_but_mean_leap_decreased_and_filter_not_reduced',comparison['current']['front_width_fraction']>comparison['parent']['front_width_fraction'] and comparison['current']['leap_mean_m']<comparison['parent']['leap_mean_m'] and comparison['current']['filter_intervention']>comparison['parent']['filter_intervention'])
not_monitored=[r for r in events if not int(r['recovery_monitored'])]
check('one_5ms_event_excluded_from_recovery_not_pending',len(not_monitored)==1 and np.isclose(float(not_monitored[0]['air_seconds']),.005) and not read(OUT/'play_diagnostics.json')['terminal_recovering'])
parent_settings=read(Path(m['source_checkpoint']['path']).parent/'manifest.json')['gallop_settings']
delta={k for k,v in cfg.items() if parent_settings[k]!=v}
check('only_seven_stability_weights_changed',delta=={'flight_landing_weight','flight_balance_weight','flight_impact_weight','roll_pitch_quiet_weight','stance_width_weight','guarded_height_weight','guarded_projection_weight'})
check('user_requested_modest_width_height_weights',cfg['stance_width_weight']==7 and cfg['guarded_height_weight']==3.5 and cfg['flight_landing_weight']==4)
changed=[]
for name,item in read(OUT/'evidence_manifest.json').items():
    path=Path(name);path=path if path.is_absolute() else ROOT/path
    if sha(path)!=item['sha256']:changed.append(name)
check('original_train_play_inputs_preserved',not changed)
check('official_checkout_clean',not subprocess.run(['git','-C',m['official_repo'],'status','--porcelain'],capture_output=True,text=True,check=True).stdout)
result=dict(all_checks_passed=all(checks.values()),checks=checks,new_policy_updates=0,new_supervised_updates=0,new_physics_rollouts=0,
    runtime_policy_reward_config_preserved=True,scope='Saved evidence validation, not full gait acceptance or new simulation')
write(OUT/'verification.json',result)
print(json.dumps(dict(all_checks_passed=result['all_checks_passed'],checks=len(checks),failed=[name for name,value in checks.items() if not value])))
if not result['all_checks_passed']:raise SystemExit(1)
