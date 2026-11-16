"""Check real48k restoration and Gaussian sampling without PPO or physics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from gpd.paths import ROOT,new_run
from isaac_task.gallop_core import GallopSettings,restore_gallop_policy,StagedGallopCurriculum
from isaac_task.gallop_runtime import checkpoint_source_directory,staged_checkpoint_progress
from isaac_task.staged_adaptation import StagedSpeedAdaptation
from scripts.analysis.collect_unitree_baseline import load_yaml


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1);torch.manual_seed(42)
    parent=ROOT/'runs/isaac/train/unitree_go2_gallop/20261111_141724_6a1661'
    checkpoint=parent/'model_final.pt';before=sha(checkpoint)
    manifest=json.loads((parent/'manifest.json').read_text());checkpoint_source_directory(checkpoint,manifest)
    progress=staged_checkpoint_progress(checkpoint,manifest)
    cfg=GallopSettings.load(ROOT/'configs/gallop_staged_speed_adaptation.json')
    agent=load_yaml(parent/'params/agent.yaml');agent['algorithm']['learning_rate']=1e-4
    agent['algorithm']['entropy_coef']=0.
    class Probe:
        num_envs=2;num_actions=12
        def get_observations(self):
            return TensorDict(dict(policy=torch.zeros(2,49),critic=torch.zeros(2,64)),batch_size=[2])
    old=OnPolicyRunner(Probe(),deepcopy(agent),log_dir=None,device='cpu')
    new=OnPolicyRunner(Probe(),deepcopy(agent),log_dir=None,device='cpu')
    def forbid(*args,**kwargs):raise RuntimeError('PPO/optimizer updates forbidden in validation')
    for r in (old,new):r.learn=forbid;r.alg.update=forbid;r.alg.optimizer.step=forbid
    old.load(str(checkpoint),load_optimizer=False,map_location='cpu')
    info=restore_gallop_policy(new,checkpoint)
    parent_state={k:v.clone() for k,v in new.alg.policy.state_dict().items()}
    adaptation=StagedSpeedAdaptation(new,cfg)
    obs=TensorDict(dict(policy=torch.randn(4096,49),critic=torch.randn(4096,64)),batch_size=[4096])
    with torch.inference_mode():
        actor_error=float((old.alg.policy.act_inference(obs)-new.alg.policy.act_inference(obs)).abs().max())
        critic_error=float((old.alg.policy.evaluate(obs)-new.alg.policy.evaluate(obs)).abs().max())
    checks=dict(parent48k=progress['cumulative_iterations']==48000,parent_stage1=progress['index']==1,
        parent_hash_record=before==progress['sha256'],actor_identical=actor_error==0.,critic_identical=critic_error==0.,
        initial_std_preserved=torch.equal(parent_state['std'],new.alg.policy.std),
        fresh_optimizer=not new.alg.optimizer.state,zero_new_iteration=new.current_learning_iteration==0,
        unchanged_adaptive_lr=new.alg.learning_rate==1e-4 and new.alg.schedule==agent['algorithm']['schedule'],
        unchanged_desired_kl=new.alg.desired_kl==agent['algorithm']['desired_kl'],
        entropy_zero=new.alg.entropy_coef==0.,std_frozen=not new.alg.policy.std.requires_grad,
        actor_trainable=all(p.requires_grad for p in new.alg.policy.actor.parameters()),
        critic_trainable=all(p.requires_grad for p in new.alg.policy.critic.parameters()))
    sample=Probe().get_observations();sampling=[]
    with torch.inference_mode():
        for step in range(new.num_steps_per_env*2):
            action=new.alg.act(sample)
            assert torch.allclose(new.alg.transition.actions_log_prob,new.alg.policy.get_actions_log_prob(action))
            assert torch.allclose(new.alg.transition.action_sigma,new.alg.policy.std.expand(2,-1))
            sampling.append(adaptation.std)
    checks['std_constant_within_each_rollout']=all(max(sampling[i:i+new.num_steps_per_env])==min(sampling[i:i+new.num_steps_per_env]) for i in (0,new.num_steps_per_env))
    checks['sample_likelihoods_preserved']=True
    adaptation.apply(cfg.staged_noise_ramp_iterations)
    checks['mean_std_target20']=abs(float(new.alg.policy.std.mean())-.20)<1e-6
    torch.testing.assert_close(new.alg.policy.std/new.alg.policy.std.mean(),parent_state['std']/parent_state['std'].mean())
    checks['joint_std_ratios_preserved']=True
    checks['all_nonstd_state_preserved']=all(torch.equal(v,new.alg.policy.state_dict()[k]) for k,v in parent_state.items() if k!='std')
    control=StagedGallopCurriculum(cfg);control.configure_envs(256);control.restore(progress['index'],progress['highest_confirmed_stage'])
    checks['restored_speed_goal']=control.index==1 and control.goal['speed']==2.75 and control.goal['leap']==.18
    checks['longer_adaptation']=not control.record(True,0,{},0,8999) and control.record(True,0,{},0,9000)
    checks['no_time_only_promotion']=control.decide(9000)['reason']=='waiting_for_cohort' and control.index==1
    old_cfg=manifest['gallop_settings'];actual=cfg.to_dict()
    changed={k:dict(before=old_cfg.get(k),after=v) for k,v in actual.items() if v!=old_cfg.get(k,v)}
    checks['legacy_fields_only_dwell_changed']=set(changed)=={'staged_min_dwell_seconds'}
    checks['speed_gate_unchanged']=cfg.staged_max_vx_rmse==old_cfg['staged_max_vx_rmse']==.25
    checks['safety_limits_preserved']=all(actual[k]==v for k,v in old_cfg.items() if k.startswith('guarded_'))
    checks['reward_weights_preserved']=all(actual[k]==v for k,v in old_cfg.items() if 'weight' in k)
    checks['parent_file_unchanged']=sha(checkpoint)==before
    assert all(checks.values()),checks
    run=new_run('isaac','check');out=run.parent/'unitree_go2_gallop'/run.name;out.mkdir(parents=True)
    result=dict(all_checks_passed=True,checks=checks,parent=str(checkpoint),parent_sha256=before,
        output=str(out),config=actual,restored=info,parent_std_mean=adaptation.initial_mean,
        target_std_mean=adaptation.target_mean,annealed_std=new.alg.policy.std.tolist(),
        actor_max_error=actor_error,critic_max_error=critic_error,changed_parent_fields=changed,
        policy_updates=0,optimizer_steps=0,physics_steps=0,
        note='Actual RSL sampler/checkpoint tested; production PPO update is forbidden. KL update wrapper is exercised by no-op-update unit tests.')
    (out/'speed_adaptation_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),checks=len(checks),all_passed=True)))


if __name__=='__main__':main()
