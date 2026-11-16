"""Verify saved completion, startup stall and unmodified source evidence."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from isaac_task.gallop_runtime import checkpoint_source_directory

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
s=read(OUT/'summary.json');d=read(OUT/'startup_stall_diagnostics.json');m=s['provenance'];train=Path(s['train']);play=Path(s['play'])
w=torch.load(train/'model_final.pt',map_location='cpu',weights_only=True)
parent=Path(m['source_checkpoint']['path']);old=torch.load(parent,map_location='cpu',weights_only=True)
with np.load(play/'gallop_trace.npz') as a:z={k:a[k] for k in a.files}
r=json.loads((play/'episodes.jsonl').read_text().splitlines()[0]);c=m['gallop_settings']
checks={}
def check(name,value):checks[name]=bool(value)
check('model_matches_saved_play',sha(train/'model_final.pt')==s['checkpoint_sha256']==read(play/'manifest.json')['source_checkpoint']['sha256']=='2f8bd83968d3e37e4a12a7fc69951dcfc3db0540c9d003ffa355f0a279233af6')
check('parent_30k_preserved',sha(parent)==m['source_checkpoint']['sha256']=='cc05c078abe28c4c9ac2ce0a8a51af7d006e33268cc6c7f57e15621a85535d7b')
check('4000_updates_34k_lineage',s['training_complete'] and w['iter']==3999 and s['cumulative_iterations']==34000 and s['new_transitions']==24576000)
check('ppo_updates_recorded',s['tensorboard']['Loss/value_function']['count']==4000 and bool(w['optimizer_state_dict']['state']))
check('weights_finite_and_actor_changed',all(torch.isfinite(v).all() for v in w['model_state_dict'].values()) and not torch.equal(w['model_state_dict']['actor.0.weight'],old['model_state_dict']['actor.0.weight']))
check('source_and_teacher_verified',checkpoint_source_directory(train/'model_final.pt',m).is_dir())
check('training_config_3mps_80cm',c['start_speed']==c['max_speed']==3 and c['leap_min_distance']==c['stabilization_min_leap_distance']==.8)
check('nonlearning_play',s['play_status']['status']=='completed' and s['play_status']['policy_updates']==0 and s['play_status']['policy_state_unchanged'])
check('finite_single_1500_step_episode',len(z['time'])==1500 and z['valid'][:,0].all() and all(np.isfinite(v).all() for v in z.values()) and z['episode_reset'][:,0].sum()==1)
check('timeout_without_fall',r['age']==1500 and r['timed_out']==1 and r['terminated']==0)
check('6000_physics_samples_preserved',r['leap_physics_samples']==6000)
check('play_never_reaches_stage3',not (z['startup_stage'][:,0]==3).any() and r['startup_completed']==0 and d['final_stage']==2)
check('formal_segment_is_absent',not z['gallop_metric_active'][:,0].any() and d['formal_samples']==0 and s['rollout']['environments'][0]['evaluated_samples']==0)
check('handover_stalls_at_65percent',np.isclose(d['final_student_mix'],.6551842093467712) and np.isclose(d['final_command_mps'],2.310368537902832))
check('final_command_never_reaches_three',z['command'][:,0,0].max()<3)
tail=z['time']>=28
check('last_2s_mix_is_constant',np.ptp(z['startup_gallop_mix'][tail,0])==0)
check('last_2s_speed_tracking_fails',np.all(np.abs(z['body_velocity'][tail,0,0]-z['command'][tail,0,0])>c['startup_tracking_error']))
check('pose_is_mostly_acceptable',d['segments']['last2s']['pose_pass_fraction']>.98)
check('no_contact_air_in_blend',not (~z['contact'][z['startup_stage'][:,0]==2,0]).all(1).any())
check('formal_zero_fields_are_unmeasured',r['gallop_metric_seconds']==r['leap_events']==r['cycles']==0 and not r['stabilization_measured'])
check('training_episode_partition',d['training_episodes']==19593 and sum(x['episodes'] for x in d['training_windows'])==19593)
check('training_handover_completions',d['handover_completions']==842 and d['measured_episodes']==666)
check('no_handover_after_1145_iterations',d['last_handover_iteration']<1145 and all(x['handover_completions']==0 for x in d['training_windows'] if x['iteration_start']>=1500))
check('last_1000_no_measured_student_gait',all(x['measured_episodes']==x['handover_completions']==x['flight_events']==0 for x in d['training_windows'] if x['iteration_start']>=3000))
check('80cm_not_achieved_in_training',d['full_successes']==0 and np.isclose(d['maximum_recorded_training_leap_m'],.19002962112426758))
check('one_play_video',len(s['videos'])==1 and s['videos'][0]['kind']=='play' and s['videos'][0]['frames']==1499)
evidence=read(OUT/'evidence_manifest.json')
check('all_original_run_files_preserved',all(sha(ROOT/name)==item['sha256'] for name,item in evidence.items()))
result=dict(all_checks_passed=all(checks.values()),checks=checks,policy_updates=0,new_physics_rollouts=0,
    note='Saved results only. No forced handover, new rollout or retraining.')
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(all_checks_passed=result['all_checks_passed'],checks=len(checks),failed=[k for k,v in checks.items() if not v])))
if not result['all_checks_passed']:raise SystemExit(1)
