"""Integrity and arithmetic checks for saved v5 walking/trend evidence."""
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


def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(1048576),b''):h.update(b)
    return h.hexdigest()
checks={}
def check(name,value):checks[name]=bool(value)

s=read(OUT/'summary.json');m=s['provenance'];train=Path(s['train']);play=Path(s['play']);cfg=m['gallop_settings']
model=train/'model_final.pt';weights=torch.load(model,map_location='cpu',weights_only=True)
check('model_matches_saved_play',sha(model)==s['checkpoint_sha256']==read(play/'manifest.json')['source_checkpoint']['sha256']=='959457e283918bcc480e71c1d6ce026cffb63feb5a1b9e0834b78f60f09a8200')
check('12000_updates_and_16000_lineage_complete',s['training_complete'] and weights['iter']==11999 and s['training_status']['policy_updates']==12000 and s['cumulative_iterations']==16000 and s['new_transitions']==73728000)
check('loss_steps_complete_reward_coverage_explained',s['tensorboard']['Loss/value_function']['count']==12000 and s['reward_scalar_coverage']['count']==11994 and s['reward_scalar_coverage']['first_iteration_zero_based']==6)
check('parent_4000_walking_policy_preserved',m['migration']['parent_cumulative_iterations']==4000 and sha(Path(m['source_checkpoint']['path']))==m['source_checkpoint']['sha256']=='d4bd406a2e0cc97b1370610a3791bf115c31cac8b718056322d95df2f9d192bf')
check('source_and_walking_teacher_preserved',checkpoint_source_directory(model,m).is_dir())
check('runtime_policy_code_preserved',all(sha(ROOT/'isaac_task'/name)==m[key] for name,key in [('gallop_core.py','gallop_core_sha256'),('gallop_env.py','gallop_env_sha256')]))
check('training_configuration_preserved',GallopSettings.load(ROOT/'configs/gallop_relearn.json').to_dict()==cfg)
check('asset_preserved',sha(Path(m['asset_path']))==m['asset_sha256'])
check('v5_interface_and_dimensions',m['interface']=='unitree_go2_gallop_relearn_49_64_12_v5' and weights['model_state_dict']['actor.0.weight'].shape[1]==49 and weights['model_state_dict']['critic.0.weight'].shape[1]==64 and weights['model_state_dict']['actor.6.weight'].shape[0]==12)
progress=relearn_checkpoint_progress(model,m)
check('file_specific_final_progress_matches',progress['level']==m['trained_relearn_level']==1 and progress['speed_cap']==1 and progress['cumulative_iterations']==16000)
events=read(train/'curriculum_summary.json')['events'];steps=[e['step'] for e in events]
evidence=read(OUT/'evidence_manifest.json');matched=0
for name,record in m['relearn_checkpoints'].items():
    p=train/name;item=evidence[str(p.relative_to(ROOT))]
    assert item['sha256']==record['sha256']
    index=bisect_right(steps,record['stage_iterations']*24)-1
    level=events[index]['relearn_level'] if index>=0 else 0
    assert level==record['level'];matched+=1
check('all_122_checkpoint_stage_and_hash_records_match',matched==122)
ps=s['play_status']
check('nonlearning_play_policy_unchanged',ps['status']=='completed' and ps['policy_updates']==0 and ps['policy_state_unchanged'])
with np.load(play/'gallop_trace.npz') as archive:z={k:archive[k] for k in archive.files}
formal=z['valid'][:,0]&z['gallop_metric_active'][:,0]
check('finite_single_episode_1500_steps',len(z['time'])==1500 and z['valid'][:,0].all() and all(np.isfinite(a).all() for a in z.values()) and z['episode_reset'][:,0].sum()==1 and z['episode_reset'][-1,0])
check('student_control_and_metric_timing',np.isclose(z['time'][z['startup_stage'][:,0]==3][0],7.5) and np.isclose(z['time'][formal][0],8.52) and formal.sum()==1075)
check('physics_samples_6000',z['leap_physics_samples'][-1,0]==6000 and (np.diff(z['leap_physics_samples'][:,0])==4).all())
check('walking_goal_restored',np.all(z['relearn_level'][:,0]==1) and np.all(z['relearn_gait'][:,0]==0) and np.allclose(z['command'][formal,0,0],1.))
episode=json.loads((play/'episodes.jsonl').read_text().splitlines()[0])
check('30s_timeout_without_failure',episode['age']==1500 and episode['terminated']==0 and episode['timed_out']==1)
check('walking_stage_pass_is_not_gallop_success',episode['relearn_stage_success']==1 and episode['success']==0 and episode['mature']==0)
check('no_flight_or_leap_credit',episode['leap_events']==0 and episode['flight_fraction']==0 and episode['leap_reward_points']==0)
g=read(OUT/'guarded_play_diagnostics.json');a=g['aggregate']
check('independent_projection_and_height_checks',all(g['checks'].values()))
check('zero_requested_angle_clipping',a['raw_angle_exceed_fraction']==episode['action_clip_fraction']==0 and episode['relearn_request_fraction']==0)
check('executed_target_and_rate_bounds_met',a['target_operating_violation_fraction']==a['target_physical_violation_fraction']==a['target_rate_violation_fraction']==0)
check('saved_actual_joint_positions_within_range',a['max_actual_new_physical_excess_rad']==0 and a['original_mechanical_violation_fraction']==0)
check('remaining_filter_interventions_are_rate_related',np.isclose(a['filter_intervention_fraction'],a['rate_intervention_fraction']) and .08<a['filter_intervention_fraction']<.09)
check('front_and_hind_support_width_pass',episode['stabilization_front_width_fraction']==episode['stabilization_hind_width_fraction']==1 and episode['stabilization_front_support_samples']==392 and episode['stabilization_hind_support_samples']==348)
check('foot_height_caps_respected_in_saved_formal_samples',all(v['p50_p90_p95_max_m'][-1]<=v['cap_m'] for v in g['foot_height'].values()))
# Independently reconstruct every saved request-monitor value from raw angles.
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
check('request_monitor_independently_reproduced',not any(failures))
d=read(OUT/'relearn_training_diagnostics.json');n=read(OUT/'relearn_noise_diagnostics.json')
check('all_66644_stage_outcomes_reconstructed',d['episodes']==d['independently_reconstructed_stage_successes']==66644)
check('all_288_decisions_replayed',d['controller_decisions_replayed']==len(events)==288)
check('repeated_1_to_2_to_1_cycle_verified',d['transitions']=={'0->1':3,'1->0':2,'1->2':46,'2->1':46})
check('no_gallop_stage_reached',d['max_reached_level']==2 and d['visited_gait_strengths']==[0.] and m['highest_confirmed_speed'] is None)
check('accepted_and_late_episodes_separated',d['discarded_old_generation_episodes']==24766 and d['stages']['2']['accepted_for_curriculum']['episodes']==16587 and d['stages']['2']['all_completed']['episodes']==28297)
check('stage2_bottleneck_documented',d['stages']['1']['all_completed']['stage_success_rate']>.85 and d['stages']['2']['all_completed']['stage_success_rate']<.1 and d['stages']['2']['all_completed']['request_termination_rate']>.85)
check('learned_noise_increase_measured',n['final_action_std_mean']/n['initial_action_std_mean']>2.4)
check('noise_run_probability_independently_checked',n['independent_run_probability_check'])
cases={(r['noise_multiplier'],r['goal_index']):r for r in n['cases']}
check('fixed_state_probe_shows_threshold_discontinuity',cases[(1.,1)]['fixed_state_sequence_no_failure_probability']>.99 and cases[(1.,2)]['fixed_state_sequence_no_failure_probability']<.1)
check('noise_counterfactual_is_labeled',bool(n['limitation']) and len(n['cases'])==9)
check('one_play_video_no_training_video',len(s['videos'])==1 and s['videos'][0]['kind']=='play' and s['videos'][0]['frames']==1499 and s['videos'][0]['duration_seconds']==29.98)
check('collection_no_updates',s['policy_updates_during_collection']==0)
check('all_original_run_files_unchanged',all(sha(ROOT/name)==item['sha256'] for name,item in evidence.items()))
check('official_checkout_clean',not subprocess.run(['git','-C',m['official_repo'],'status','--porcelain'],capture_output=True,text=True,check=True).stdout)
result=dict(all_checks_passed=all(checks.values()),checks=checks,new_policy_updates=0,new_supervised_updates=0,new_physics_rollouts=0,
    scope='Saved evidence and analytic fixed-state diagnostics. Not a new simulation or an estimated real-world survival rate.')
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(all_checks_passed=result['all_checks_passed'],checks=len(checks),failed=[k for k,v in checks.items() if not v])))
if not result['all_checks_passed']:raise SystemExit(1)
