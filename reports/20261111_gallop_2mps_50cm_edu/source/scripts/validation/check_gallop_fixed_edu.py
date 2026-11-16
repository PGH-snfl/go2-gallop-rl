"""Verify actual50k/46k restoration and fixed2m/s50cm settings without PPO/physics."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from gpd.paths import ROOT,new_run
from isaac_task.gallop_core import GallopSettings,restore_gallop_policy,GuardedJointController,OfficialJointMonitor,FixedGallopGoal
from isaac_task.gallop_runtime import checkpoint_source_directory,staged_checkpoint_progress
from isaac_task.staged_adaptation import StagedSpeedAdaptation
from scripts.analysis.collect_unitree_baseline import load_yaml


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);args=p.parse_args()
    checkpoint=args.checkpoint.resolve();torch.set_num_threads(1);torch.manual_seed(42)
    before=sha(checkpoint);manifest=json.loads((checkpoint.parent/'manifest.json').read_text())
    checkpoint_source_directory(checkpoint,manifest);progress=staged_checkpoint_progress(checkpoint,manifest)
    cfg=GallopSettings.load(ROOT/'configs/gallop_2mps_50cm_edu.json')
    agent=load_yaml(checkpoint.parent/'params/agent.yaml');agent['algorithm']['learning_rate']=1e-4;agent['algorithm']['entropy_coef']=0.
    class Probe:
        num_envs=2;num_actions=12
        def get_observations(self):return TensorDict(dict(policy=torch.zeros(2,49),critic=torch.zeros(2,64)),batch_size=[2])
    old=OnPolicyRunner(Probe(),deepcopy(agent),log_dir=None,device='cpu')
    new=OnPolicyRunner(Probe(),deepcopy(agent),log_dir=None,device='cpu')
    def forbid(*a,**kw):raise RuntimeError('PPO/optimizer updates forbidden')
    for runner in (old,new):runner.learn=forbid;runner.alg.update=forbid;runner.alg.optimizer.step=forbid
    old.load(str(checkpoint),load_optimizer=False,map_location='cpu');info=restore_gallop_policy(new,checkpoint)
    initial={k:v.detach().clone() for k,v in new.alg.policy.state_dict().items()}
    obs=TensorDict(dict(policy=torch.randn(4096,49),critic=torch.randn(4096,64)),batch_size=[4096])
    with torch.inference_mode():
        actor_error=float((old.alg.policy.act_inference(obs)-new.alg.policy.act_inference(obs)).abs().max())
        critic_error=float((old.alg.policy.evaluate(obs)-new.alg.policy.evaluate(obs)).abs().max())
    manager=StagedSpeedAdaptation(new,cfg)
    checks=dict(actor_exact=actor_error==0.,critic_exact=critic_error==0.,initial_std_exact=torch.equal(initial['std'],new.alg.policy.std),
        optimizer_fresh=not new.alg.optimizer.state,iteration_zero=new.current_learning_iteration==0,
        std_frozen=not new.alg.policy.std.requires_grad,entropy_zero=new.alg.entropy_coef==0.,
        mean_actor_trainable=all(p.requires_grad for p in new.alg.policy.actor.parameters()),
        critic_trainable=all(p.requires_grad for p in new.alg.policy.critic.parameters()),
        fixed_goal_enabled=cfg.fixed_goal_enabled,stage_controller_disabled=not cfg.staged_enabled and not cfg.relearn_enabled)
    manager.apply(cfg.staged_noise_ramp_iterations)
    checks['annealed_mean_std16']=abs(float(new.alg.policy.std.mean())-.16)<1e-6
    checks['all_mean_weights_preserved']=all(torch.equal(v,new.alg.policy.state_dict()[k]) for k,v in initial.items() if k!='std')
    goal=FixedGallopGoal(cfg)
    for step in range(1000,6001,1000):
        for _ in range(256):goal.record(True,0,{})
        assert goal.decide(step) is None
    checks.update(fixed2m_goal=goal.cap==2.,fixed50cm_goal=cfg.leap_min_distance==cfg.stabilization_min_leap_distance==.5,
        no_cohorts_or_decisions=goal.events==[],no_confirmed_stage_state=goal.highest_confirmed_speed is None)
    robot=load_yaml(checkpoint.parent/'params/env.yaml')['scene']['robot']
    original=torch.tensor(manifest['guarded_runtime_joint_contract']['original_limits'])[None]
    physics=torch.tensor(manifest['guarded_runtime_joint_contract']['physics_limits'])[None]
    default=torch.zeros(1,12)
    import re
    names=manifest['guarded_runtime_joint_contract']['joint_names']
    default[0]=torch.tensor([next(v for pattern,v in robot['init_state']['joint_pos'].items() if re.fullmatch(pattern,n)) for n in names])
    hips=torch.tensor([i for i,n in enumerate(names) if '_hip_' in n]);calves=torch.tensor([i for i,n in enumerate(names) if '_calf_' in n])
    guard=GuardedJointController(default,physics,hips,calves,cfg);monitor=OfficialJointMonitor(default,calves,cfg)
    checks.update(target_rates_calf12=bool((guard.rate[calves]==12).all()),
        actual_calf_limit15p7=bool(torch.allclose(monitor.velocity[calves],torch.full((4,),15.7))),
        target_inside_original=bool((guard.lower>=original[...,0]).all() and (guard.upper<=original[...,1]).all()),
        unchanged_original_weights=sha(checkpoint)==before)
    assert all(checks.values()),checks
    r=new_run('isaac','check');out=r.parent/'unitree_go2_gallop'/r.name;out.mkdir(parents=True)
    (out/'fixed_edu_verification.json').write_text(json.dumps(dict(checks=checks,all_checks_passed=True,
        checkpoint=str(checkpoint),sha256=before,parent_stage=progress,restored=info,
        new_goal=dict(speed=2.,distance=.5,period=.6,reference_air=.3),initial_std=initial['std'].tolist(),
        target_std_mean=manager.target_mean,actor_max_error=actor_error,critic_max_error=critic_error,
        policy_updates=0,optimizer_steps=0,physics_steps=0),indent=2)+'\n')
    print(json.dumps(dict(output=str(out),checks=len(checks),passed=True)))


if __name__=='__main__':main()
