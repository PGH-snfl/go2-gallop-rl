"""Verify the real walking-policy migration and staged plan without learning."""
import argparse
from copy import deepcopy
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
from isaac_task.gallop_core import (GallopSettings,RelearnCurriculum,GuardedJointController,
    guarded_physics_limits,transplant_forward_policy,phase_targets,stabilization_leap_target)
from isaac_task.gallop_runtime import parent_cumulative_iterations
from scripts.analysis.collect_unitree_baseline import load_yaml


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'runs/isaac/train/unitree_go2_forward_only/20261028_160258_54620c/model_final.pt')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/gallop_relearn.json')
    args=parser.parse_args();torch.set_num_threads(1);torch.manual_seed(42)
    checkpoint=args.checkpoint.resolve();before=sha(checkpoint)
    m=json.loads((checkpoint.parent/'manifest.json').read_text());settings=GallopSettings.load(args.config)
    class Probe:
        num_envs=2;num_actions=12
        def __init__(self,a,c):self.a=a;self.c=c
        def get_observations(self):return TensorDict(dict(policy=torch.zeros(2,self.a),critic=torch.zeros(2,self.c)),batch_size=[2])
    agent=load_yaml(checkpoint.parent/'params/agent.yaml')
    old=OnPolicyRunner(Probe(45,60),deepcopy(agent),log_dir=None,device='cpu')
    new=OnPolicyRunner(Probe(49,64),deepcopy(agent),log_dir=None,device='cpu')
    def forbid(*args,**kwargs):raise RuntimeError('Learning is forbidden in the relearning validator')
    for runner in (old,new):
        runner.learn=forbid;runner.alg.update=forbid;runner.alg.optimizer.step=forbid
    old.load(str(checkpoint),load_optimizer=False,map_location='cpu')
    migration=transplant_forward_policy(new,checkpoint,m['upstream_task_audit']['action_scale'],settings.action_scale)
    a=torch.randn(4096,45);c=torch.randn(4096,60);an=a.clone();cn=c.clone()
    an[:,33:45]*=.5;cn[:,48:60]*=.5
    old_obs=TensorDict(dict(policy=a,critic=c),batch_size=[len(a)])
    new_obs=TensorDict(dict(policy=torch.cat((an,torch.randn(len(a),4)),1),critic=torch.cat((cn,torch.randn(len(c),4)),1)),batch_size=[len(a)])
    with torch.inference_mode():
        action_error=(.25*old.alg.policy.act_inference(old_obs)-.5*new.alg.policy.act_inference(new_obs)).abs()
        value_error=(old.alg.policy.evaluate(old_obs)-new.alg.policy.evaluate(new_obs)).abs()
    plan=RelearnCurriculum(settings).plan
    checks=dict(forward_parent=m['interface']=='unitree_go2_forward_only_45_60_12_v1',
        parent_4000_iterations=parent_cumulative_iterations(m,migration['parent_stage_iterations'])==4000,
        physical_action_preserved=float(action_error.max())<1e-5,critic_preserved=float(value_error.max())<1e-5,
        physical_noise_preserved=torch.equal(.25*old.alg.policy.std,.5*new.alg.policy.std),
        four_new_actor_columns_zero=not bool(torch.count_nonzero(new.alg.policy.state_dict()['actor.0.weight'][:,-4:])),
        four_new_critic_columns_zero=not bool(torch.count_nonzero(new.alg.policy.state_dict()['critic.0.weight'][:,-4:])),
        fresh_optimizer=not new.alg.optimizer.state,iteration_zero=new.current_learning_iteration==0,
        staged_guarded_configuration=settings.relearn_enabled and settings.guarded_enabled,
        walking_first=plan[0]['gait']==0 and plan[0]['speed']==1.,
        final_goal=plan[-1]['gait']==1 and plan[-1]['speed']==3. and settings.leap_min_distance==.22,
        request_tolerance_tightens=all(b['request_tolerance']<=a['request_tolerance'] for a,b in zip(plan,plan[1:])),
        request_duration_tightens=all(b['request_seconds']<=a['request_seconds'] for a,b in zip(plan,plan[1:])),
        final_single_step_request_failure=plan[-1]['request_tolerance']==1e-6 and plan[-1]['request_seconds']==.02)
    asset=json.loads((ROOT/'assets/go2_isaac_contract.json').read_text());names=asset['native_joint_order']
    robot=load_yaml(checkpoint.parent/'params/env.yaml')['scene']['robot']
    lookup={j['name']:np.deg2rad([j['lower'],j['upper']]) for j in asset['joints']}
    original=torch.tensor(np.array([lookup[n] for n in names]),dtype=torch.float32)[None]
    physical=guarded_physics_limits(original,settings.guarded_sim_limit_factor)
    default=torch.tensor([next(v for p,v in robot['init_state']['joint_pos'].items() if re.fullmatch(p,n)) for n in names])[None]
    hips=torch.tensor([i for i,n in enumerate(names) if '_hip_' in n]);calves=torch.tensor([i for i,n in enumerate(names) if '_calf_' in n])
    guard=GuardedJointController(default,physical,hips,calves,settings)
    checks['default_pose_valid']=bool(((default>=guard.lower)&(default<=guard.upper)).all())
    previous=default.clone()
    for index in range(1000):
        target=guard.process(torch.full_like(default,100. if index%2 else -100.),.02)
        assert ((target>=guard.lower)&(target<=guard.upper)).all()
        assert ((target-previous).abs()<=guard.rate*.02+1e-6).all();previous=target.clone()
    checks['extreme_commands_angle_and_rate_safe']=True
    speeds=torch.tensor([g['speed'] for g in plan]);period=phase_targets(torch.zeros_like(speeds),speeds,settings)['period']
    air=period*(1-max(t+d for t,d in zip(settings.touchdown,settings.duty)))
    checks['planned_flight_timing_covers_distance']=bool((speeds*air>=stabilization_leap_target(speeds,settings)-1e-6).all())
    checks['parent_asset_unchanged']=sha(checkpoint)==before and sha(m['asset_path'])==m['asset_sha256']==asset['isaac_asset_sha256']
    checks['official_checkout_clean']=not subprocess.run(['git','-C',m['official_repo'],'status','--porcelain'],capture_output=True,text=True,check=True).stdout
    path=new_run('isaac','check');out=path.parent/'unitree_go2_gallop'/path.name;out.mkdir(parents=True)
    result=dict(all_checks_passed=all(checks.values()),checks=checks,parent=str(checkpoint),parent_sha256=before,
        parent_iterations=4000,migration=migration,sample_observations=4096,
        max_physical_action_error_rad=float(action_error.max()),max_critic_error=float(value_error.max()),
        settings=settings.to_dict(),plan=plan,policy_updates=0,supervised_updates=0,physics_rollouts=0,
        cuda_available=torch.cuda.is_available(),
        source_sha256={name:sha(ROOT/name) for name in ['isaac_task/gallop_core.py','isaac_task/gallop_env.py','isaac_task/gallop_runtime.py','scripts/training/gallop_baseline.py','gpd/rl_mujoco.py','configs/gallop_relearn.json']},
        limitation='Policy migration is verified on saved weights and synthetic observations. No new Isaac dynamics or learned gait performance was evaluated.')
    (out/'relearn_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),checks=len(checks),all_checks_passed=result['all_checks_passed'],failed=[k for k,v in checks.items() if not v])))
    if not result['all_checks_passed']:raise SystemExit(1)


if __name__=='__main__':main()
