"""Verify guarded fine-tuning and saved requests without learning or physics."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from gpd.paths import ROOT,new_run
from isaac_task.gallop_core import GallopSettings,GuardedJointController,guarded_physics_limits,guarded_foot_height_cost,restore_gallop_policy,stabilization_leap_target,phase_targets
from isaac_task.gallop_runtime import checkpoint_source_directory
from scripts.analysis.collect_unitree_baseline import load_yaml


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/gallop_guarded_2p5.json')
    p.add_argument('--trace',type=Path,default=ROOT/'runs/isaac/play/unitree_go2_gallop/20261103_091200_02fe98/gallop_trace.npz');args=p.parse_args()
    torch.set_num_threads(1);checkpoint=args.checkpoint.resolve();before=sha(checkpoint)
    m=json.loads((checkpoint.parent/'manifest.json').read_text());cfg=GallopSettings.load(args.config)
    source=checkpoint_source_directory(checkpoint,m)
    class Probe:
        num_envs=2;num_actions=12
        def get_observations(self):return TensorDict(dict(policy=torch.zeros(2,49),critic=torch.zeros(2,64)),batch_size=[2])
    runner=OnPolicyRunner(Probe(),load_yaml(checkpoint.parent/'params/agent.yaml'),log_dir=None,device='cpu')
    def forbid(*a,**k):raise RuntimeError('Learning/optimizer steps forbidden in guarded validation')
    runner.learn=forbid;runner.alg.update=forbid;runner.alg.optimizer.step=forbid
    restored=restore_gallop_policy(runner,checkpoint);weights=torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state_dict']
    speeds=torch.tensor([cfg.start_speed,(cfg.start_speed+cfg.max_speed)/2,cfg.max_speed])
    leap_targets=stabilization_leap_target(speeds,cfg)
    checks=dict(compatible_parent=m['interface'] in ('unitree_go2_gallop_heading_49_64_12_v2','unitree_go2_gallop_guarded_49_64_12_v4'),
        exact_actor_critic_noise=all(torch.equal(v,weights[k]) for k,v in runner.alg.policy.state_dict().items()),
        fresh_optimizer=not runner.alg.optimizer.state,iteration_zero=runner.current_learning_iteration==0,
        legacy_delta_actions=not cfg.bounded_actions_enabled and cfg.action_scale==m['action_scale']==.5,
        guarded_stabilization=cfg.guarded_enabled and cfg.stabilization_enabled and cfg.start_speed<=cfg.max_speed and cfg.speed_band==0,
        leap_targets_match_speed_endpoints=bool(torch.allclose(leap_targets,torch.tensor([cfg.stabilization_min_leap_distance,(cfg.stabilization_min_leap_distance+cfg.leap_min_distance)/2,cfg.leap_min_distance]))),
        frozen_parent_sources=source.is_dir())
    asset=json.loads((ROOT/'assets/go2_isaac_contract.json').read_text());robot=load_yaml(checkpoint.parent/'params/env.yaml')['scene']['robot']
    names=asset['native_joint_order'];lookup={j['name']:np.deg2rad([j['lower'],j['upper']]) for j in asset['joints']}
    original=torch.tensor(np.array([lookup[n] for n in names]),dtype=torch.float32)[None]
    physics=guarded_physics_limits(original,cfg.guarded_sim_limit_factor)
    default=torch.tensor([next(v for pattern,v in robot['init_state']['joint_pos'].items() if re.fullmatch(pattern,n)) for n in names])[None]
    hips=torch.tensor([i for i,n in enumerate(names) if '_hip_' in n]);calves=torch.tensor([i for i,n in enumerate(names) if '_calf_' in n])
    guard=GuardedJointController(default,physics,hips,calves,cfg)
    parent_physics=torch.tensor(m['guarded_runtime_joint_contract']['physics_limits'])[None] if m.get('guarded_design') else guarded_physics_limits(original,robot['soft_joint_pos_limit_factor'])
    checks['parent_physical_range_preserved']=bool(torch.allclose(physics,parent_physics))
    checks['default_pose_inside_operating_range']=bool(((default>=guard.lower)&(default<=guard.upper)).all())
    trace=args.trace.resolve();trace_sha=sha(trace)
    with np.load(trace) as z:
        raw=torch.tensor(z['guarded_requested'][:,0]) if 'guarded_requested' in z else default+cfg.action_scale*torch.tensor(z['executed_action'][:,0])
        height=torch.tensor(z['foot_height'][:,0]);time=z['time'].reshape(-1)
    dt=float(np.median(np.diff(time)))
    results=[]
    for request in raw:
        results.append(guard.process(request[None],dt).clone())
    target=torch.cat(results);rate=torch.diff(torch.cat((default,target)),dim=0)/dt
    checks['all_saved_commands_inside_new_range']=bool(((target>=guard.lower-1e-6)&(target<=guard.upper+1e-6)).all())
    checks['all_saved_commands_obey_rate_caps']=bool((rate.abs()<=guard.rate+1e-5).all())
    front,hind,_=guarded_foot_height_cost(height,cfg)
    checks['front_and_hind_excess_height_penalties_nonzero']=bool(front.max()>0 and hind.max()>0)
    checks['airborne_clearance_not_penalized']=all(float(v.max())==0 for v in guarded_foot_height_cost(torch.ones(1,4)*.06,cfg)[:2])
    checks['no_reference_height_conflict']=max(cfg.swing_height_slow,cfg.swing_height_fast)<=min(cfg.guarded_front_height,cfg.guarded_hind_height)
    if cfg.start_speed<cfg.max_speed:
        period=phase_targets(torch.zeros_like(speeds),speeds,cfg)['period']
        air=(1-max(t+d for t,d in zip(cfg.touchdown,cfg.duty)))*period
        checks['command_times_planned_air_covers_leap_targets']=bool((speeds*air>=leap_targets-1e-6).all())
        checks['initial_speed_matches_parent_cap']=abs(cfg.start_speed-m['trained_speed_cap'])<1e-6
    checks['trace_unchanged']=sha(trace)==trace_sha
    checks['parent_and_asset_unchanged']=sha(checkpoint)==before and sha(Path(m['asset_path']))==m['asset_sha256']==asset['isaac_asset_sha256']
    checks['official_checkout_clean']=not subprocess.run(['git','-C',m['official_repo'],'status','--porcelain'],check=True,capture_output=True,text=True).stdout
    run=new_run('isaac','check');out=run.parent/'unitree_go2_gallop'/run.name;out.mkdir(parents=True)
    result=dict(all_checks_passed=all(checks.values()),checks=checks,checkpoint=str(checkpoint),checkpoint_sha256=before,
        settings=cfg.to_dict(),restore=restored,policy_updates=0,supervised_updates=0,physics_rollouts=0,cuda_available=torch.cuda.is_available(),
        limits=dict(native_joint_order=names,original=original[0].tolist(),physics=physics[0].tolist(),target=torch.stack((guard.lower[0],guard.upper[0]),dim=-1).tolist()),
        saved_trace_replay=dict(path=str(trace),sha256=trace_sha,sample_seconds=dt,samples=len(target),mean_front_height_cost=float(front.mean()),mean_hind_height_cost=float(hind.mean()),
            target_change_rmse_rad=float((target-raw).square().mean().sqrt()),note='Old recorded requests filtered offline. Not a new locomotion trajectory.'),
        limitation='New Isaac dynamics and achievement of the speed/leap goals require user-run training/play. Kinematic timing checks do not establish dynamic feasibility.',
        source_sha256={name:sha(ROOT/name) for name in ['isaac_task/gallop_core.py','isaac_task/gallop_env.py','scripts/training/gallop_baseline.py','gpd/rl_mujoco.py']},
        config=dict(path=str(args.config.resolve()),sha256=sha(args.config)),parent_cumulative_iterations=m['cumulative_iterations'])
    (out/'guarded_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),checks=len(checks),all_checks_passed=result['all_checks_passed'],failed=[k for k,v in checks.items() if not v])))
    if not result['all_checks_passed']:raise SystemExit(1)


if __name__=='__main__':main()
