"""Reproduce integrity and arithmetic checks on preserved recovery results."""
from bisect import bisect_right
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
from isaac_task.gallop_runtime import checkpoint_source_directory,relearn_checkpoint_progress


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
checks={}
def check(name,value):checks[name]=bool(value)

s=read(OUT/'summary.json');m=s['provenance'];train=Path(s['train']);play=Path(s['play']);cfg=m['gallop_settings']
weights=torch.load(train/'model_final.pt',map_location='cpu',weights_only=True)
check('checkpoint_matches_play_and_original',sha(train/'model_final.pt')==s['checkpoint_sha256']==read(play/'manifest.json')['source_checkpoint']['sha256']=='526f221fb8cb4897bbd5931b9575300d8803d2be4763361268128cbd73439207')
check('6000_updates_and_22000_lineage',s['training_complete'] and weights['iter']==5999 and s['cumulative_iterations']==22000 and s['new_transitions']==36864000 and s['training_status']['policy_updates']==6000)
check('loss_iterations_complete',s['tensorboard']['Loss/value_function']['count']==6000)
check('parent_16000_unchanged',m['fine_tuning']['parent_cumulative_iterations']==16000 and sha(Path(m['source_checkpoint']['path']))==m['source_checkpoint']['sha256']=='959457e283918bcc480e71c1d6ce026cffb63feb5a1b9e0834b78f60f09a8200')
check('source_snapshot_and_teacher_verified',checkpoint_source_directory(train/'model_final.pt',m).is_dir())
check('runtime_code_unchanged',all(sha(ROOT/'isaac_task'/name)==m[key] for name,key in [('gallop_core.py','gallop_core_sha256'),('gallop_env.py','gallop_env_sha256')]))
check('config_unchanged',GallopSettings.load(ROOT/'configs/gallop_relearn_recovery.json').to_dict()==cfg)
check('asset_unchanged',sha(Path(m['asset_path']))==m['asset_sha256'])
check('old_v4_gallop_branch_preserved',sha(ROOT/'runs/isaac/train/unitree_go2_gallop/20261103_191405_bea09f/model_final.pt')=='dc129d38e78654a6559657740b7a46865830ca3cefcf06c908f49cdcef53fea7')
progress=relearn_checkpoint_progress(train/'model_final.pt',m)
check('saved_final_goal_is_walking_1p5',progress['level']==m['trained_relearn_level']==7 and progress['speed_cap']==1.5 and m['relearn_design']['plan'][7]['gait']==0)
events=read(train/'curriculum_summary.json')['events'];steps=[e['step'] for e in events]
initial=m['relearn_stage_migration']['level'];matched=0;evidence=read(OUT/'evidence_manifest.json')
for name,record in m['relearn_checkpoints'].items():
    path=train/name
    assert evidence[str(path.relative_to(ROOT))]['sha256']==record['sha256']==sha(path)
    index=bisect_right(steps,record['stage_iterations']*24)-1
    assert record['level']==(events[index]['relearn_level'] if index>=0 else initial)
    matched+=1
check('all_62_checkpoint_phases_and_hashes_match',matched==62)
check('interface_49_64_12',weights['model_state_dict']['actor.0.weight'].shape[1]==49 and weights['model_state_dict']['critic.0.weight'].shape[1]==64 and weights['model_state_dict']['actor.6.weight'].shape[0]==12)
check('play_had_no_updates',s['play_status']['status']=='completed' and s['play_status']['policy_updates']==0 and s['play_status']['policy_state_unchanged'])
with np.load(play/'gallop_trace.npz') as archive:z={k:archive[k] for k in archive.files}
formal=z['valid'][:,0].astype(bool)&z['gallop_metric_active'][:,0].astype(bool)
check('finite_1500_step_single_episode',len(z['time'])==1500 and z['valid'][:,0].all() and all(np.isfinite(a).all() for a in z.values()) and z['episode_reset'][:,0].sum()==1 and z['episode_reset'][-1,0])
check('handover_and_formal_timing',np.isclose(z['time'][z['startup_stage'][:,0]==3][0],7.32) and np.isclose(z['time'][formal][0],8.34) and formal.sum()==1084)
check('6000_physics_samples_preserved',z['leap_physics_samples'][-1,0]==6000 and (np.diff(z['leap_physics_samples'][:,0])==4).all())
check('saved_walking_goal_restored',np.all(z['relearn_level'][:,0]==7) and np.all(z['relearn_gait'][:,0]==0) and np.allclose(z['command'][formal,0,0],1.5))
episode=json.loads((play/'episodes.jsonl').read_text().splitlines()[0])
check('30_second_timeout_no_fall',episode['age']==1500 and episode['terminated']==0 and episode['timed_out']==1)
check('walking_success_is_not_full_gallop',episode['relearn_stage_success']==1 and episode['success']==0 and episode['mature']==0)
check('no_formal_flight_or_leap',episode['leap_events']==0 and episode['flight_fraction']==0 and not (~z['contact'][formal,0]).all(1).any())
vx=z['body_velocity'][formal,0,0].astype(float)
check('speed_metrics_reproduced',np.isclose(vx.mean(),1.5189401013604829) and np.isclose(np.sqrt(np.mean((vx-1.5)**2)),.07464400275384195))
g=read(OUT/'guarded_play_diagnostics.json');a=g['aggregate']
check('guard_and_height_independent_checks',all(g['checks'].values()))
check('zero_requested_angle_overflow',a['raw_angle_exceed_fraction']==0 and episode['relearn_request_fraction']==0)
check('command_angle_and_rate_limits_met',a['target_operating_violation_fraction']==a['target_physical_violation_fraction']==a['target_rate_violation_fraction']==0)
check('saved_actual_joint_ranges_met',a['max_actual_new_physical_excess_rad']==0 and a['original_mechanical_violation_fraction']==0)
check('filter_interventions_are_rate_related',np.isclose(a['filter_intervention_fraction'],a['rate_intervention_fraction']) and np.isclose(a['filter_intervention_fraction'],.0365159901599016))
check('support_width_criteria_passed',episode['stabilization_front_width_fraction']==episode['stabilization_hind_width_fraction']==1 and episode['stabilization_front_support_samples']==304 and episode['stabilization_hind_support_samples']==305)
check('saved_formal_foot_heights_respected',all(v['p50_p90_p95_max_m'][-1]<=v['cap_m'] for v in g['foot_height'].values()))
bounds=np.array(m['guarded_runtime_joint_contract']['target_limits']);q=z['guarded_requested'][:,0]
excess=np.maximum(bounds[:,0]-q,0)+np.maximum(q-bounds[:,1],0)
np.testing.assert_allclose(excess.max(1),z['relearn_request_peak'][:,0],atol=1e-6)
np.testing.assert_allclose((excess>1e-6).mean(1),z['relearn_request_fraction'][:,0],atol=1e-6)
seconds=0.;failed=False;runs=[];failures=[]
for i,peak in enumerate(excess.max(1)):
    active=z['startup_stage'][i,0]==3
    seconds=seconds+.02 if active and peak>z['relearn_request_tolerance'][i,0] else 0.
    failed=failed or (active and seconds>=z['relearn_request_seconds'][i,0]-1e-6)
    runs.append(seconds);failures.append(failed)
np.testing.assert_allclose(runs,z['relearn_request_run'][:,0],atol=1e-6)
np.testing.assert_array_equal(failures,z['relearn_request_failed'][:,0])
check('raw_request_monitor_reproduced',not any(failures))
d=read(OUT/'relearn_training_diagnostics.json');n=read(OUT/'relearn_noise_diagnostics.json');gate=read(OUT/'relearn_gate_diagnostics.json')
check('all_26008_stage_outcomes_reconstructed',d['episodes']==d['independently_reconstructed_stage_successes']==26008)
check('all_144_controller_decisions_replayed',d['controller_decisions_replayed']==len(events)==144)
check('partial_gallop_reached_and_regressed_15_times',d['transitions']['7->8']==d['transitions']['8->7']==15 and d['max_reached_level']==8 and d['visited_gait_strengths']==[0.,.25])
check('no_full_gallop_confirmed',m['highest_confirmed_speed'] is None and all(row['full_gallop_successes']==0 for row in s['relearn']['stages'].values()))
check('balanced_windows_account_for_every_episode',d['accepted_episode_count']==14080 and d['discarded_old_generation_episodes']==8704 and d['repeated_env_episodes']==3224 and 14080+8704+3224==26008)
check('noise_schedule_did_not_grow',np.isclose(n['initial_action_std_mean'],.12) and np.isclose(n['final_action_std_mean'],.08) and n['entropy_coefficient']==0 and np.allclose(weights['model_state_dict']['std'].numpy(),.08))
accepted=gate['stages']['8']['cohorts']['accepted_for_curriculum']
check('stage8_front_cycle_bottleneck',accepted['episodes']==3840 and accepted['stage_successes']==accepted['front_cycle_passes']==19 and accepted['rates']['front_motion']>.95)
check('last_1000_partial_gait_success_zero',gate['last_1000_iterations']['8']['episodes']==1638 and gate['last_1000_iterations']['8']['rates']['relearn_stage_success']==0)
shutdown=read(OUT/'shutdown_preservation.json');backup=Path(shutdown['backup'])
check('target_play_and_launcher_stopped',shutdown['termination']=='killed_after_saved_completion' and not shutdown['child_alive'] and not shutdown['launcher_alive'])
check('signals_escalated_only_after_backup', [e['signal'] for e in shutdown['signals']]==['SIGINT','SIGTERM','SIGKILL'] and all(shutdown['verified_after_stop'].values()))
check('all_19_backup_files_match_before_shutdown',len(shutdown['files'])==19 and all(sha(backup/name)==meta['sha256'] for name,meta in shutdown['files'].items()))
check('one_decodable_play_video',len(s['videos'])==1 and s['videos'][0]['kind']=='play' and s['videos'][0]['frames']==1499 and s['videos'][0]['duration_seconds']==29.98)
check('all_original_train_and_play_files_unchanged',all(sha(ROOT/name)==meta['sha256'] for name,meta in evidence.items()))
check('official_checkout_clean',not subprocess.check_output(['git','-C',m['official_repo'],'status','--porcelain'],text=True))
result=dict(all_checks_passed=all(checks.values()),checks=checks,new_policy_updates=0,new_supervised_updates=0,new_physics_rollouts=0,
    scope='Saved results, source integrity, termination records and independent arithmetic. No new simulation or deployment evaluation.')
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(all_checks_passed=result['all_checks_passed'],checks=len(checks),failed=[k for k,v in checks.items() if not v])))
if not result['all_checks_passed']:raise SystemExit(1)
