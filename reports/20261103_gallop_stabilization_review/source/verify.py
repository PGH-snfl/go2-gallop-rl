"""Check saved stabilization results; no inference, physics or learning."""
import csv
from datetime import datetime,timezone
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

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
def check(name,result):checks[name]=bool(result)

s=read(OUT/'summary.json');train=Path(s['train']);play=Path(s['play']);m=s['provenance'];cfg=m['gallop_settings']
model=train/'model_final.pt';pm=read(play/'manifest.json');source=checkpoint_source_directory(model,m)
check('checkpoint_matches_play_and_collection',sha(model)==s['checkpoint_sha256']==pm['source_checkpoint']['sha256']=='c2439fd522a98e5b782e5e08b749aabd01c64f50a73edb6c6ad020e6d2cdbf5a')
check('frozen_sources_teacher_and_original_asset_intact',source.is_dir() and sha(Path(m['asset_path']))==m['asset_sha256'])
check('parent_checkpoint_intact',sha(Path(m['source_checkpoint']['path']))==m['source_checkpoint']['sha256'])
weights=torch.load(model,map_location='cpu',weights_only=True)
check('2000_updates_and_22000_lineage',int(weights['iter'])+1==2000==s['training_status']['policy_updates'] and m['cumulative_iterations']==22000)
check('49_64_12_interface',weights['model_state_dict']['actor.0.weight'].shape[1]==49 and weights['model_state_dict']['critic.0.weight'].shape[1]==64 and weights['model_state_dict']['actor.6.weight'].shape[0]==12)
check('training_complete_and_transition_count',s['training_complete'] and s['new_transitions']==12288000)
scalar=list(csv.DictReader((OUT/'tensorboard_scalars.csv').open()))
check('all_2000_loss_steps', [int(r['iteration_zero_based']) for r in scalar if r['tag']=='Loss/value_function']==list(range(2000)))
check('completed_play_has_no_updates',s['play_status']['status']=='completed' and s['play_status']['policy_updates']==0 and s['play_status']['policy_state_unchanged'])
with np.load(play/'gallop_trace.npz') as a:z={k:a[k] for k in a.files}
check('finite_1500_policy_steps',len(z['time'])==1500 and all(np.isfinite(v).all() for v in z.values()) and np.allclose(np.diff(z['time']),.02))
check('one_full_episode_timeout_at_30s',z['valid'][:,0].all() and z['episode_reset'][:,0].sum()==1 and z['episode_reset'][-1,0])
formal=z['gallop_metric_active'][:,0]&z['valid'][:,0]
check('formal_1063_samples_from_8p76',formal.sum()==1063 and np.isclose(z['time'][formal][0],8.76))
check('four_startup_stages',set(z['startup_stage'][:,0])=={0,1,2,3} and np.isclose(z['time'][z['startup_stage'][:,0]==3][0],7.74))
check('6000_physics_samples',z['leap_physics_samples'][-1,0]==6000 and (np.diff(z['leap_physics_samples'][:,0])==4).all())
check('saved_speed_and_leap_targets_restored',np.allclose(z['stabilization_target_speed'],2.2) and np.allclose(z['stabilization_leap_target'],.14) and np.allclose(z['command'][formal,0,0],2.2))
check('no_speed_promotions_or_confirmed_stage',s['curriculum']['promotions']==0 and s['curriculum']['speed_caps']==[2.2] and m['highest_confirmed_speed'] is None)
check('play_curriculum_frozen',read(play/'curriculum_summary.json')['events']==[])
rpy=z['rpy'][:,0];rate=z['posture_roll_pitch_rate'][:,0]
delta=np.diff(rpy[:,:2],axis=0);expected=np.arctan2(np.sin(delta),np.cos(delta))/.02
check('temporal_angle_rates_match',np.allclose(rate[1:],expected,atol=1e-4))
dead=np.array([cfg['stabilization_roll_deadband'],cfg['stabilization_pitch_deadband']]);scale=np.array([cfg['stabilization_roll_scale'],cfg['stabilization_pitch_scale']])
pose=np.exp(-np.sum((np.maximum(np.abs(rpy[:,:2])-dead,0)/scale)**2,axis=1))
check('pose_quality_matches',np.allclose(pose,z['posture_pose_quality'][:,0],atol=2e-5))
quiet=np.exp(-np.mean((rate/cfg['roll_pitch_rate_scale'])**2,axis=1))*pose*z['posture_rate_valid'][:,0]
check('quiet_reward_includes_pose',np.allclose(quiet,z['posture_quiet_score'][:,0],atol=2e-5))
wz=z['angular_velocity'][:,0,2];normalized=np.abs(wz/cfg['stabilization_yaw_rate_scale'])
check('yaw_damping_matches',np.allclose(np.where(normalized<1,.5*normalized**2,normalized-.5),z['posture_yaw_damping_cost'][:,0],atol=2e-6))
fractions=[];supports=[]
for i,name in enumerate(('front','hind')):
    c=z['contact'][:,0,2*i:2*i+2];width=z['posture_width'][:,0,i]
    tolerance=np.where(c.all(1),1.,cfg['stance_swing_tolerance_multiplier'])*cfg['stance_width_tolerance']
    error=np.maximum(np.abs(width-cfg[name+'_stance_width'])-tolerance,0)/cfg['stance_width_scale']
    score=1/(1+error**2)*np.where(c.any(1),1.,cfg['stabilization_air_width_weight'])
    check(name+'_dense_width_score_matches',np.allclose(score,z['posture_'+name+'_width_score'][:,0],atol=2e-6))
    selected=width[formal&c.all(1)];supports.append(len(selected))
    fractions.append(float(((selected>=cfg['stabilization_width_min'])&(selected<=cfg['stabilization_width_max'])).mean()))
episode=json.loads((play/'episodes.jsonl').read_text().splitlines()[0])
check('support_counts_match_305_439',supports==[305,439])
check('width_gate_fails_only_hind_pair',fractions[0]>=.8>fractions[1] and np.isclose(fractions[0],episode['stabilization_front_width_fraction']) and np.isclose(fractions[1],episode['stabilization_hind_width_fraction']))
check('calf_scaling_actually_executed',np.allclose(z['executed_action'][formal,0,8:12],z['action'][formal,0,8:12]*.85,atol=1e-6) and np.allclose(z['executed_action'][formal,0,:8],z['action'][formal,0,:8],atol=1e-6))
joint=read(OUT/'joint_target_diagnostics.json')['current']
check('raw_target_clipping_recomputed',np.isclose(joint['overall_clip_fraction'],z['action_clip'][formal,0].mean()))
check('mean_excess_matches_saved_vector',np.isclose(joint['mean_excess_rad'],z['stabilization_target_excess'][formal,0].mean()))
check('excess_huber_matches_saved_cost',np.isclose(joint['comparable_excess_cost'],z['stabilization_excess_cost'][formal,0].mean(),atol=1e-6))
check('tracking_pose_and_landings_pass',all(episode[k] for k in ['tracking','stabilization_pose','stabilization_landing','stabilization_distance','stabilization_measured']))
extra=read(OUT/'stabilization_diagnostics.json')
check('only_stable_and_width_fail',set(extra['play_failed_gates'])=={'stable','stabilization_width'} and episode['success']==0 and episode['action_clip_fraction']>.1)
check('no_fall_in_play',episode['terminated']==0 and episode['timed_out']==1)
events=list(csv.DictReader((OUT/'play_flight_events.csv').open()))
check('59_flights_54_qualified_57_stable_2_aborted',len(events)==59 and sum(int(r['distance_time_qualified']) for r in events)==54 and sum(int(r['stable_landing']) for r in events)==57 and sum(int(r['recovery_aborted']) for r in events)==2)
check('53_meet_current_target_and_stable',sum(int(r['stable_landing']) and int(r['distance_time_qualified']) for r in events)==53)
check('33_meet_final_18cm_reference',extra['final_distance_reference']['events']==33==extra['final_distance_reference']['stable_events'])
check('8304_training_episodes_without_success',extra['training']['all_eligible']['episodes']==8304 and extra['training']['all_eligible']['successful_episodes']==0 and extra['training']['all_eligible']['front_width']['passed_episodes']==0)
videos=list(csv.DictReader((OUT/'videos.csv').open()))
check('one_play_video_1499_frames',len(videos)==1 and videos[0]['kind']=='play' and int(videos[0]['frames'])==1499 and float(videos[0]['duration_seconds'])==29.98)
official=subprocess.run(['git','-C',m['official_repo'],'status','--porcelain'],capture_output=True,text=True,check=True)
check('official_checkout_clean',not official.stdout.strip())
result=dict(verified_at=datetime.now(timezone.utc).isoformat(),all_checks_passed=all(checks.values()),checks=checks,
    new_policy_updates=0,new_simulations=0,raw_checkpoint_and_logs_preserved=True)
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(all_checks_passed=result['all_checks_passed'],checks=len(checks),failed=[k for k,v in checks.items() if not v])))
if not result['all_checks_passed']:raise SystemExit(1)
