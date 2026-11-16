"""Verify guarded result collection; saved files only, no learning/physics."""
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
from isaac_task.gallop_runtime import checkpoint_source_directory
from isaac_task.gallop_core import GallopSettings

def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
checks={}
def check(name,value):checks[name]=bool(value)

s=read(OUT/'summary.json');m=s['provenance'];train=Path(s['train']);play=Path(s['play']);cfg=m['gallop_settings']
model=train/'model_final.pt';w=torch.load(model,map_location='cpu',weights_only=True)
check('model_hash_matches_play',sha(model)==s['checkpoint_sha256']==read(play/'manifest.json')['source_checkpoint']['sha256']=='70ca957fb986b4c04538eea1a78e053db0f1782746ca8ba3311eb7b5578aee92')
check('completed_2000_updates_new_24k_branch',s['training_complete'] and w['iter']==1999 and s['training_status']['policy_updates']==2000 and s['cumulative_iterations']==24000 and s['new_transitions']==12288000)
check('parent_is_pre_imitation_22k',m['fine_tuning']['parent_cumulative_iterations']==22000 and m['source_checkpoint']['sha256']=='c2439fd522a98e5b782e5e08b749aabd01c64f50a73edb6c6ad020e6d2cdbf5a')
check('parent_file_preserved',sha(Path(m['source_checkpoint']['path']))==m['source_checkpoint']['sha256'])
check('saved_sources_and_teacher_verified',checkpoint_source_directory(model,m).is_dir())
check('runtime_core_env_preserved',all(sha(ROOT/'isaac_task'/name)==m[key] for name,key in [('gallop_core.py','gallop_core_sha256'),('gallop_env.py','gallop_env_sha256')]))
check('runtime_settings_preserved',GallopSettings.load(ROOT/'configs/gallop_guarded_2p5.json').to_dict()==cfg)
check('asset_preserved',sha(Path(m['asset_path']))==m['asset_sha256'])
check('interface_49_64_12_without_imitation',m['interface']=='unitree_go2_gallop_guarded_49_64_12_v4' and m['guarded_design']['imitation'] is False and w['model_state_dict']['actor.0.weight'].shape[1]==49 and w['model_state_dict']['critic.0.weight'].shape[1]==64 and w['model_state_dict']['actor.6.weight'].shape[0]==12)
with np.load(play/'gallop_trace.npz') as a:z={k:a[k] for k in a.files}
f=z['valid'][:,0]&z['gallop_metric_active'][:,0]
check('completed_nonlearning_play',s['play_status']['status']=='completed' and s['play_status']['policy_updates']==0 and s['play_status']['policy_state_unchanged'])
check('finite_1500_steps_one_reset',len(z['time'])==1500 and z['valid'][:,0].all() and z['episode_reset'][:,0].sum()==1 and z['episode_reset'][-1,0] and all(np.isfinite(v).all() for v in z.values()))
check('formal_1041_samples_9p2_seconds',f.sum()==1041 and np.isclose(z['time'][f][0],9.2))
check('gallop_handover_at_8p18',np.isclose(z['time'][z['startup_stage'][:,0]==3][0],8.18))
check('6000_physics_samples',z['leap_physics_samples'][-1,0]==6000 and (np.diff(z['leap_physics_samples'][:,0])==4).all())
check('fixed_2p5_and_14cm',cfg['start_speed']==cfg['max_speed']==2.5 and np.allclose(z['command'][f,0,0],2.5) and np.allclose(z['stabilization_leap_target'],.14))
contract=m['guarded_runtime_joint_contract'];original=np.array(contract['original_limits']);physical=np.array(contract['physics_limits']);operating=np.array(contract['target_limits'])
mid=original.mean(-1);radius=np.ptp(original,axis=1)*.45
check('physics_is_original_90percent',np.allclose(physical,np.column_stack((mid-radius,mid+radius)),atol=1e-6))
margin=np.ptp(physical,axis=1)*.05
check('operating_is_original_81percent',np.allclose(operating,physical+np.column_stack((margin,-margin)),atol=1e-6))
g=read(OUT/'guarded_play_diagnostics.json');metrics=g['aggregate']
check('independent_guard_math_all_pass',all(g['checks'].values()))
check('executed_targets_and_rates_respect_limits',metrics['target_operating_violation_fraction']==metrics['target_physical_violation_fraction']==metrics['target_rate_violation_fraction']==0)
check('actual_original_mechanical_range_preserved',metrics['original_mechanical_violation_fraction']==0)
check('small_solver_excursion_reported',.001<metrics['max_actual_new_physical_excess_rad']<.0011 and metrics['new_physical_violation_fraction_over_guard_tolerance']==0 and metrics['max_violation_seconds']==0)
episode=json.loads((play/'episodes.jsonl').read_text().splitlines()[0])
check('no_fall_30s_timeout',episode['age']==1500 and episode['terminated']==0 and episode['timed_out']==1)
check('front_hind_height_gate_pass',episode['guarded_feet']==1 and episode['guarded_front_high']<.01 and episode['guarded_hind_high']<.01)
check('filter_gate_correctly_fails',episode['guarded_limits']==0 and episode['guarded_actual']==0 and episode['guarded_intervention']>cfg['guarded_max_intervention'])
check('front_width_fails_hind_passes',episode['stabilization_front_width_fraction']<.8<episode['stabilization_hind_width_fraction'] and episode['stabilization_front_support_samples']==382 and episode['stabilization_hind_support_samples']==538)
check('landing_gate_correctly_fails',episode['stabilization_landing']==0 and np.isclose(episode['stabilization_landing_fraction'],41/60))
check('raw_requested_angle_clip_is_not_final_violation',np.isclose(metrics['raw_angle_exceed_fraction'],episode['action_clip_fraction']) and episode['action_clip_fraction']<.1)
failed=read(OUT/'stabilization_diagnostics.json')['play_failed_gates']
check('all_failed_gates_reported',set(failed)=={'stable','stabilization_width','stabilization_landing','guarded_limits'} and episode['success']==0)
events=list(csv.DictReader((OUT/'play_flight_events.csv').open()))
check('60_flights_all_qualified_41_stable_19_aborted',len(events)==60 and sum(int(r['distance_time_qualified']) for r in events)==60 and sum(int(r['stable_landing']) for r in events)==41 and sum(int(r['recovery_aborted']) for r in events)==19)
check('one_play_video_no_training_clips',len(s['videos'])==1 and s['videos'][0]['kind']=='play' and s['videos'][0]['frames']==1499 and s['videos'][0]['duration_seconds']==29.98)
check('fixed_curriculum_no_promotion_or_success_claim',s['curriculum']['speed_caps']==[2.5] and s['curriculum']['promotions']==0 and m['highest_confirmed_speed'] is None)
check('training_success_zero',read(OUT/'stabilization_diagnostics.json')['training']['all_eligible']['successful_episodes']==0)

comparison={}
for label,report in [('parent_22k',ROOT/'reports/20261103_gallop_stabilization_review'),('bounded_24k',ROOT/'reports/20261103_gallop_bounded_review'),('guarded_24k',OUT)]:
    summary=read(report/'summary.json');p=Path(summary['play'])
    with np.load(p/'gallop_trace.npz') as a:
        mask=a['valid'][:,0]&a['gallop_metric_active'][:,0];height=a['foot_height'][mask,0]
    comparison[label]=dict(checkpoint_sha256=summary['checkpoint_sha256'],play=str(p),
        front_max_m=float(height[:,:2].max()),hind_max_m=float(height[:,2:].max()),per_foot_p95_m=np.quantile(height,.95,axis=0).tolist())
(OUT/'foot_height_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
check('foot_height_reduction_supported',comparison['guarded_24k']['front_max_m']<.18<comparison['parent_22k']['front_max_m'] and comparison['guarded_24k']['hind_max_m']<comparison['parent_22k']['hind_max_m'])
changed=[]
for name,item in read(OUT/'evidence_manifest.json').items():
    p=Path(name);p=p if p.is_absolute() else ROOT/p
    if sha(p)!=item['sha256']:changed.append(name)
check('original_inputs_preserved',not changed)
check('official_checkout_clean',not subprocess.run(['git','-C',m['official_repo'],'status','--porcelain'],capture_output=True,text=True,check=True).stdout)
result=dict(all_checks_passed=all(checks.values()),checks=checks,new_policy_updates=0,new_supervised_updates=0,new_physics_rollouts=0,
    runtime_policy_reward_config_preserved=True,scope='Saved evidence validation; does not assert full gait acceptance')
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(all_checks_passed=result['all_checks_passed'],checks=len(checks),failed=[k for k,v in checks.items() if not v])))
if not result['all_checks_passed']:raise SystemExit(1)
