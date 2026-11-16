"""Verify saved bounded-policy evidence; no learning or physics."""
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
from isaac_task.gallop_core import bounded_joint_targets,bounded_limit_cost
from isaac_task.gallop_runtime import checkpoint_source_directory

def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
checks={}
def check(name,value):checks[name]=bool(value)

s=read(OUT/'summary.json');m=s['provenance'];train=Path(s['train']);play=Path(s['play']);cfg=m['gallop_settings']
model=train/'model_final.pt';weights=torch.load(model,map_location='cpu',weights_only=True)
check('completed_2000_updates_24000_lineage',s['training_complete'] and weights['iter']==1999 and s['cumulative_iterations']==24000 and s['new_transitions']==12288000)
check('checkpoint_hash_matches_play',sha(model)==s['checkpoint_sha256']==read(play/'manifest.json')['source_checkpoint']['sha256']=='167504719e5f3b0523893b69bc1a9c12bde5eeefb477a6fc26d1301abccb926d')
check('frozen_core_env_teachers_verified',checkpoint_source_directory(model,m).is_dir())
check('parent_model_preserved',sha(Path(m['source_checkpoint']['path']))==m['source_checkpoint']['sha256'])
check('asset_preserved',sha(Path(m['asset_path']))==m['asset_sha256'])
check('completed_play_without_updates',s['play_status']['status']=='completed' and s['play_status']['policy_updates']==0 and s['play_status']['policy_state_unchanged'])
with np.load(play/'gallop_trace.npz') as a:z={k:a[k] for k in a.files}
f=z['valid'][:,0]&z['gallop_metric_active'][:,0]
check('finite_1500_steps_one_episode',len(z['time'])==1500 and z['valid'][:,0].all() and z['episode_reset'][:,0].sum()==1 and all(np.isfinite(v).all() for v in z.values()))
check('formal_1066_samples',f.sum()==1066 and np.isclose(z['time'][f][0],8.7))
check('6000_physics_samples',z['leap_physics_samples'][-1,0]==6000)
motion=read(OUT/'bounded_motion_diagnostics.json');cur=motion['current'];prior=motion['parent']
limits=torch.tensor([r['soft_limits_rad'] for r in cur['per_joint']],dtype=torch.float32)
mapped=bounded_joint_targets(torch.from_numpy(z['action'][f,0]),limits).numpy()
check('actual_executed_transform_matches',np.allclose(mapped,z['bounded_target'][f,0],atol=2e-6))
names=[r['joint'] for r in cur['per_joint']];calves=torch.tensor([i for i,n in enumerate(names) if '_calf_' in n])
cost,near=bounded_limit_cost(torch.from_numpy(z['bounded_target'][f,0]),limits,calves,cfg['bounded_limit_margin'])
check('near_limit_fraction_recomputed',np.allclose(near.numpy(),z['bounded_near_limit_fraction'][f,0],atol=2e-6))
check('near_limit_cost_recomputed',np.allclose(cost.numpy(),z['bounded_near_limit_cost'][f,0],atol=2e-6))
rate=np.diff(z['bounded_target'][:,0],axis=0)/.02
check('recorded_target_rate_is_real_difference',np.allclose(rate,z['bounded_target_rate'][1:,0],atol=1e-4))
x=np.abs(z['bounded_target_rate'][f,0]/cfg['bounded_target_rate_scale'])
check('target_rate_cost_recomputed',np.allclose(np.where(x<1,.5*x*x,x-.5).mean(-1),z['bounded_target_rate_cost'][f,0],atol=2e-6))
check('target_bounds_pass_hard_actual_bounds_pass',cur['target_soft_limit_violation_fraction']==0 and cur['actual_hard_limit_violation_fraction']==0)
check('actual_soft_overshoot_not_hidden',cur['actual_soft_limit_violation_fraction']>0 and cur['actual_soft_max_excess_rad']>.18)
check('tracking_and_near_telemetry_matches',np.isclose(cur['tracking_rmse_rad'],np.sqrt(np.mean(z['bounded_tracking_error'][f,0]**2))) and np.isclose(cur['near_limit_fraction'],z['bounded_near_limit_fraction'][f,0].mean()))
episode=json.loads((play/'episodes.jsonl').read_text().splitlines()[0])
check('no_fall_timeout',episode['terminated']==0 and episode['timed_out']==1)
check('no_fake_clipping_success',episode['action_clip_fraction']==0 and episode['bounded_limits']==0 and episode['stable']==0 and episode['success']==0)
gates=read(OUT/'stabilization_diagnostics.json')
check('only_stable_and_limit_gates_failed',set(gates['play_failed_gates'])=={'stable','bounded_limits'})
check('no_promotion_at_2p2',s['curriculum']['promotions']==0 and s['curriculum']['speed_caps']==[2.2] and m['highest_confirmed_speed'] is None)
events=list(csv.DictReader((OUT/'play_flight_events.csv').open()))
check('59_flights_56_qualified_55_stable',len(events)==59 and sum(int(r['distance_time_qualified']) for r in events)==56 and sum(int(r['stable_landing']) for r in events)==55)
check('54_distance_time_and_stable',sum(int(r['distance_time_qualified']) and int(r['stable_landing']) for r in events)==54)
check('five_train_one_play_video',len(s['videos'])==6 and sum(r['kind']=='train' for r in s['videos'])==5 and sum(r['kind']=='play' for r in s['videos'])==1)
adapt=read(train/'adaptation/adaptation.json')
check('adaptation_completed_before_ppo',adapt['status']=='completed' and adapt['supervised_updates']==5670 and adapt['ppo_updates']==0 and adapt['final_error']['rmse_rad']<.1)
check('holdout_policy_drift_measured',all(r['final']['rmse_rad']>.2 and r['adapted']['rmse_rad']<.07 for r in motion['policy_drift']['saved_holdout'].values()))
with np.load(OUT/'bounded_command_projection.npz') as p:
    check('offline_projected_angles_bounded',np.all(p['projected']>=p['operating_limits'][:,0]-1e-8) and np.all(p['projected']<=p['operating_limits'][:,1]+1e-8))
    check('offline_projected_rates_bounded',np.all(np.abs(np.diff(p['projected'],axis=0)/.02)<=p['rate_limits']+1e-7))
changed=[]
for name,value in read(OUT/'evidence_manifest.json').items():
    path=Path(name);path=path if path.is_absolute() else ROOT/path
    if sha(path)!=value['sha256']:changed.append(name)
check('raw_input_files_unchanged',not changed)
check('official_checkout_clean',not subprocess.run(['git','-C',m['official_repo'],'status','--porcelain'],check=True,capture_output=True,text=True).stdout)
result=dict(all_checks_passed=all(checks.values()),checks=checks,new_policy_updates=0,new_supervised_updates=0,new_simulations=0,
    runtime_policy_rewards_and_config_unchanged=True,scope='Saved-file audit and offline command projection only')
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(all_checks_passed=result['all_checks_passed'],checks=len(checks),failed=[k for k,v in checks.items() if not v])))
if not result['all_checks_passed']:raise SystemExit(1)
