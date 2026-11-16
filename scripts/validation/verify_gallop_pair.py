"""Offline verification of pair costs and real checkpoint loading; no physics/PPO."""
import argparse
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from gpd.paths import ROOT,new_run
from isaac_task.gallop_core import GallopSettings,PairBalanceTracker,LeapDistanceTracker,pair_joint_penalty,phase_targets,restore_gallop_policy
from isaac_task.gallop_runtime import checkpoint_source_directory,parent_cumulative_iterations
from scripts.analysis.collect_unitree_baseline import load_yaml


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class ObservationProbe:
    """Static input shapes for loading a policy, not a simulated robot."""
    num_envs=4
    num_actions=12
    def get_observations(self):
        return TensorDict({'policy':torch.zeros(4,47),'critic':torch.zeros(4,62)},batch_size=[4])


def verify_leap_cases(settings):
    settings=replace(settings,metric_warmup_seconds=0.)
    tracker=LeapDistanceTracker(torch.zeros(5,3),settings)
    tracker.begin_step(torch.zeros(5,dtype=torch.bool))
    position=torch.tensor([[0.,0.,.35]]).repeat(5,1)
    q=torch.tensor([[1.,0.,0.,0.]]).repeat(5,1)
    force=torch.full((5,4),100.)
    eligible=torch.tensor([True,True,True,True,False])
    tracker.observe(position,q,force,.005,eligible)
    force.zero_();force[2]=100.
    tracker.observe(position,q,force,.005,eligible)
    steps=round(settings.flight_min_seconds/.005)
    velocities=torch.tensor([1.,.5,1.,-1.,1.])*settings.leap_min_distance/settings.flight_min_seconds
    for i in range(1,steps+1):
        position[:,0]=velocities*(i*.005)
        if i==steps:force.fill_(100.)
        tracker.observe(position,q,force,.005,eligible)
    torch.testing.assert_close(tracker.pending_reward,torch.tensor([1.,.5*settings.leap_partial_credit,0.,0.,0.]),atol=1e-5,rtol=0)
    return {name:dict(distance_m=float(tracker.last_distance[i]),air_seconds=float(tracker.last_seconds[i]),
                      events=int(tracker.events[i]),qualified_events=int(tracker.qualified_events[i]),score=float(tracker.pending_reward[i]))
            for i,name in enumerate(('target_met','half_distance','grounded_motion','backward_flight','forelimb_ineligible'))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/gallop_2mps_pair.json')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    checkpoint=a.checkpoint.resolve();parent=json.loads((checkpoint.parent/'manifest.json').read_text())
    if parent['interface']!='unitree_go2_gallop_47_62_12_v1':raise ValueError('Expected a Gallop checkpoint')
    if parent['action_scale']!=.5:raise ValueError('Unexpected action scale')
    settings=GallopSettings.load(a.config)
    if settings.action_scale!=parent['action_scale']:raise ValueError('Fine-tuning cannot change action units')
    checkpoint_source_directory(checkpoint,parent)
    before=sha(checkpoint)
    agent=load_yaml(checkpoint.parent/'params/agent.yaml')
    runner=OnPolicyRunner(ObservationProbe(),agent,log_dir=None,device='cpu')
    def forbid(*args,**kwargs):raise RuntimeError('Learning forbidden during offline verification')
    runner.learn=forbid;runner.alg.update=forbid
    initialization=restore_gallop_policy(runner,checkpoint)
    initialization['parent_cumulative_iterations']=parent_cumulative_iterations(parent,initialization['parent_stage_iterations'])
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state_dict']
    for key,value in runner.alg.policy.state_dict().items():
        torch.testing.assert_close(value,saved[key],rtol=0,atol=0)
    if runner.alg.optimizer.state or runner.current_learning_iteration!=0:raise ValueError('New stage did not reset Adam/count')
    obs=ObservationProbe().get_observations();obs['policy'][:,6]=settings.start_speed;obs['critic'][:,6]=settings.start_speed
    with torch.inference_mode():
        actions=runner.alg.policy.act_inference(obs);values=runner.alg.policy.evaluate(obs)
    if actions.shape!=(4,12) or not (torch.isfinite(actions).all() and torch.isfinite(values).all()):raise ValueError('Invalid policy outputs')
    reference=phase_targets(torch.zeros(1),torch.tensor([settings.start_speed]),settings)
    leap_cases=verify_leap_cases(settings) if settings.leap_reward_weight>0 else None
    with np.load(a.trace) as archive:d={key:archive[key] for key in archive.files}
    valid=d['valid'][:,0].astype(bool)
    if not valid.any() or np.diff(valid.astype(int)).max()>0:raise ValueError('Trace must have a contiguous first segment')
    dt=float(np.median(np.diff(d['time'])))
    tracker=PairBalanceTracker(torch.tensor(d['joint_pos'][0,:1]),settings)
    rows=[];previous_phase=0.
    for i in np.flatnonzero(valid):
        q=torch.tensor(d['joint_pos'][i,:1]);force=torch.tensor(d['foot_force'][i,:1]);contact=torch.tensor(d['contact'][i,:1])
        phase=float(d['phase'][i,0]);wrapped=torch.tensor([phase<previous_phase])
        previous_q=torch.tensor(d['joint_pos'][max(0,i-1),:1])
        tracker.finish_cycles(wrapped,previous_q)
        active=torch.tensor([bool(d['time'][i]>=settings.metric_warmup_seconds)])
        tracker.observe(q,force,contact,dt,active)
        if active.item():
            joint=pair_joint_penalty(q,torch.tensor(d['joint_reference'][i,:1]),settings).item()
            row=dict(time=float(d['time'][i]),pair_joint_cost=joint,
                     **{'pair_'+key+'_cost':float(value[0]) for key,value in tracker.costs.items()})
            for key,value in tracker.diagnostics.items():
                row['fore_'+key+'_asymmetry']=float(value[0,0]);row['hind_'+key+'_asymmetry']=float(value[0,1])
            rows.append(row)
        previous_phase=phase
    if not rows or not all(np.isfinite(list(row.values())).all() for row in rows):raise ValueError('Nonfinite offline costs')
    after=sha(checkpoint)
    if after!=before:raise ValueError('Parent model changed')
    out=a.output or (lambda path:path.parent/'unitree_go2_gallop'/path.name)(new_run('isaac','check'))
    out.mkdir(parents=True,exist_ok=False)
    with (out/'recorded_pair_costs.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    shutil.copy2(a.config,out/'gallop_config.json')
    source={str(path.relative_to(ROOT)):sha(path) for path in [ROOT/'isaac_task/gallop_core.py',ROOT/'isaac_task/gallop_env.py',ROOT/'isaac_task/gallop_runtime.py',ROOT/'isaac_task/logging.py',ROOT/'scripts/training/gallop_baseline.py',Path(__file__).resolve()]}
    result=dict(status='completed',verification='offline_only',physics_steps=0,policy_updates=0,
                gait_performance_verified=False,cuda_available=torch.cuda.is_available(),
                checkpoint=str(checkpoint),checkpoint_sha256=before,parent_model_unchanged=True,
                initialization=initialization,all_policy_tensors_exact=True,new_optimizer_empty=True,
                policy_probe_finite=True,policy_probe_command=[settings.start_speed,0.,0.],settings=settings.to_dict(),reference_period_at_start_speed=float(reference['period'][0]),
                synthetic_leap_cases=leap_cases,
                recorded_trace=str(a.trace.resolve()),trace_samples=len(rows),completed_pair_cycles=float(tracker.valid_cycles[0]),
                recorded_cost_means={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key.endswith('_cost')},
                source_sha256=source,
                scope='Static policy probes, synthetic leap trajectories and rescoring the stored trace using its own joint references. No Isaac application, PPO update or new dynamics rollout.')
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    (out/'status.json').write_text(json.dumps(dict(status='completed',mode='check',verification='offline_only',policy_updates=0,physics_steps=0),indent=2))
    print(json.dumps(dict(output=str(out.resolve()),parent_model_unchanged=True,initialization=initialization,recorded_cost_means=result['recorded_cost_means'],cuda_available=result['cuda_available']),indent=2))


if __name__=='__main__':main()
