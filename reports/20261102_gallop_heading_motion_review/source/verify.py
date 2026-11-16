"""Recheck stored training/play evidence without inference, simulation or PPO."""
from collections import Counter
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
from scripts.analysis.collect_gallop import training_completion


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
checks={};details={}
def check(name,value):checks[name]=bool(value)


s=read(OUT/'summary.json');train=Path(s['train']);play=Path(s['play'])
manifest=read(train/'manifest.json');pm=read(play/'manifest.json');checkpoint=train/'model_final.pt'
source=checkpoint_source_directory(checkpoint,manifest)
check('frozen_source_and_teacher_verified',source.is_dir())
check('checkpoint_matches_play_and_collection',sha(checkpoint)==pm['source_checkpoint']['sha256']==s['checkpoint_sha256'])
check('parent_checkpoint_unchanged',sha(Path(manifest['source_checkpoint']['path']))==manifest['source_checkpoint']['sha256'])
check('teacher_original_unchanged',sha(source/'walking_teacher.pt')==sha(Path(manifest['walking_teacher']['original_checkpoint'])))
check('training_usd_unchanged',sha(Path(manifest['asset_path']))==manifest['asset_sha256'])
weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
check('2000_checkpoint_updates_and_20000_lineage',int(weights['iter'])+1==2000==s['training_status']['policy_updates'] and manifest['cumulative_iterations']==20000)
check('49_64_12_interface',weights['model_state_dict']['actor.0.weight'].shape[1]==49 and weights['model_state_dict']['critic.0.weight'].shape[1]==64 and weights['model_state_dict']['actor.6.weight'].shape[0]==12)
details['learned_heading_column_norms']={key:torch.linalg.vector_norm(weights['model_state_dict'][key][:,-2:],dim=0).tolist() for key in ('actor.0.weight','critic.0.weight')}
check('heading_columns_have_been_trained',all(all(v>0 for v in values) for values in details['learned_heading_column_norms'].values()))
scalar_rows=list(csv.DictReader((OUT/'tensorboard_scalars.csv').open()))
loss=[int(r['iteration_zero_based']) for r in scalar_rows if r['tag']=='Loss/value_function']
reward=[int(r['iteration_zero_based']) for r in scalar_rows if r['tag']=='Train/mean_reward']
check('loss_2000_reward_1994_coverage',loss==list(range(2000)) and reward==list(range(6,2000)))
check('training_completion_true',s['training_complete'] and training_completion(s['training_status'],2000,loss)['complete'])
check('completion_rejects_missing_updates_or_losses',not training_completion(s['training_status'],1999,loss)['complete'] and not training_completion(s['training_status'],2000,loss[:-1])['complete'])
check('correct_transition_count',s['new_transitions']==12288000)
check('play_completed_1500_without_updates',s['play_status']['status']=='completed' and s['play_status']['steps']==1500 and s['play_status']['policy_updates']==0 and s['play_status']['policy_state_unchanged'])
with np.load(play/'gallop_trace.npz') as a:z={k:a[k] for k in a.files}
check('finite_trace_and_time_20ms',all(np.isfinite(v).all() for v in z.values()) and len(z['time'])==1500 and np.allclose(np.diff(z['time']),.02))
check('first_episode_ends_only_at_30s',z['valid'][:,0].all() and z['episode_reset'][:,0].sum()==1 and z['episode_reset'][-1,0])
check('all_startup_stages_completed',set(z['startup_stage'][:,0])=={0,1,2,3})
formal=z['gallop_metric_active'][:,0]&z['valid'][:,0]
check('formal_944_samples_from_11p14s',formal.sum()==944 and np.isclose(z['time'][formal][0],11.14))
check('6000_physics_samples_4_per_step',z['leap_physics_samples'][-1,0]==6000 and (np.diff(z['leap_physics_samples'][:,0])==4).all())
cfg=manifest['gallop_settings'];rpy=z['rpy'][:,0]
yaw=np.arctan2(np.sin(rpy[:,2]-cfg['target_yaw']),np.cos(rpy[:,2]-cfg['target_yaw']))
check('yaw_error_matches_orientation',np.allclose(yaw,z['posture_yaw_error'][:,0],atol=2e-6))
delta=np.diff(rpy[:,:2],axis=0);rate=np.arctan2(np.sin(delta),np.cos(delta))/.02
check('roll_pitch_rate_matches_temporal_difference',np.allclose(rate,z['posture_roll_pitch_rate'][1:,0],atol=1e-4))
check('quiet_score_matches_saved_rate',np.allclose(np.exp(-np.mean((z['posture_roll_pitch_rate'][1:,0]/cfg['roll_pitch_rate_scale'])**2,axis=1)),z['posture_quiet_score'][1:,0],atol=2e-6))
check('first_rate_sample_excluded',not z['posture_rate_valid'][0,0] and z['posture_rate_valid'][1:,0].all() and z['posture_quiet_score'][0,0]==0)
for pair,name in enumerate(('front','hind')):
    width=z['posture_width'][:,0,pair]
    both=z['contact'][:,0,pair*2:pair*2+2].all(1)
    tolerance=np.where(both,1.,cfg['stance_swing_tolerance_multiplier'])*cfg['stance_width_tolerance']
    error=np.maximum(np.abs(width-cfg[name+'_stance_width'])-tolerance,0)/cfg['stance_width_scale']
    check(name+'_width_score_matches_geometry',np.allclose(np.exp(-error**2),z['posture_'+name+'_width_score'][:,0],atol=2e-6))
along=z['posture_along_speed'][:,0];cross=z['posture_cross_speed'][:,0];command=z['command'][:,0,0]
direction=np.arctan2(cross,along);scale=np.maximum(.25,.25*command)
score=np.clip(along/np.maximum(command,.05),0,1)*np.exp(-((along-command)/scale)**2-(direction/cfg['progress_direction_scale'])**2)*((command>.05)&(along>0))
details['forward_score_max_recompute_error']=float(np.max(np.abs(score-z['posture_forward_score'][:,0])))
check('forward_score_matches_real_motion',np.allclose(score,z['posture_forward_score'][:,0],atol=2e-5))
episodes=[json.loads(line) for line in (play/'episodes.jsonl').read_text().splitlines()];episode=episodes[0]
check('single_timeout_without_fall',len(episodes)==1 and episode['timed_out']==1 and episode['terminated']==0 and episode['success']==0)
actual_velocity=np.sqrt(np.mean((z['body_velocity'][formal,0,:2]-z['command'][formal,0,:2])**2,axis=0))
actual_yawrate=float(np.sqrt(np.mean(z['angular_velocity'][formal,0,2]**2)))
check('tracking_failure_is_body_z_angular_speed',actual_velocity[0]<.7 and actual_velocity[1]<.15 and actual_yawrate>.3 and episode['tracking']==0)
check('stable_failure_is_target_clipping',episode['attitude_rms']<cfg['max_attitude_rms'] and episode['action_clip_fraction']>cfg['max_action_clipping'] and episode['stable']==0)
flights=list(csv.DictReader((OUT/'play_flight_events.csv').open()))
check('57_flights_38_qualified_45_stable_12_aborted',len(flights)==57 and sum(int(r['distance_time_qualified']) for r in flights)==38 and Counter(r['status'] for r in flights)=={'stable':45,'aborted':12})
check('27_distance_time_and_stable_landings',sum(int(r['distance_time_qualified']) and int(r['stable_landing']) for r in flights)==27)
check('flight_mean_matches_episode',np.isclose(np.mean([float(r['distance_m']) for r in flights]),episode['leap_distance_mean'],atol=1e-6))
train_rows=[json.loads(line) for line in (train/'episodes.jsonl').read_text().splitlines()]
eligible=[r for r in train_rows if r['eligible']];measured=[r for r in eligible if r['gallop_metric_seconds']>0]
check('8377_training_eligible_8138_measured',len(eligible)==8377 and len(measured)==8138)
check('no_actual_training_tracking_pass_hidden_by_empty_metrics',all(not r['tracking'] for r in measured) and all(r['gallop_metric_seconds']==0 for r in eligible if r['tracking']))
details['reported_training_tracking_passes_without_metrics']=sum(bool(r['tracking']) for r in eligible)
videos=list(csv.DictReader((OUT/'videos.csv').open()))
check('one_29p98s_play_video_no_training_video',len(videos)==1 and videos[0]['kind']=='play' and int(videos[0]['frames'])==1499 and float(videos[0]['duration_seconds'])==29.98)
official=subprocess.run(['git','-C',manifest['official_repo'],'status','--porcelain'],capture_output=True,text=True,check=True)
check('official_checkout_clean',not official.stdout.strip())
result=dict(verified_at=datetime.now(timezone.utc).isoformat(),all_checks_passed=all(checks.values()),checks=checks,details=details,
    new_policy_updates=0,new_simulation_runs=0,raw_checkpoint_and_logs_preserved=True)
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(all_checks_passed=result['all_checks_passed'],checks=len(checks),failed=[k for k,v in checks.items() if not v],details=details)))
if not result['all_checks_passed']:raise SystemExit(1)
