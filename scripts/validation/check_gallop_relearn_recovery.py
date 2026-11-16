"""Check the completed v5 parent and recovery setup without training or physics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from gpd.paths import ROOT,new_run
from isaac_task.gallop_core import GallopSettings,RelearnCurriculum,RelearnExploration,restore_gallop_policy
from isaac_task.gallop_runtime import relearn_checkpoint_progress,map_relearn_progress,checkpoint_source_directory
from scripts.analysis.collect_unitree_baseline import load_yaml


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1);torch.manual_seed(42)
    parent=ROOT/'runs/isaac/train/unitree_go2_gallop/20261104_144742_917f16'
    checkpoint=parent/'model_final.pt';before=sha(checkpoint)
    manifest=json.loads((parent/'manifest.json').read_text());checkpoint_source_directory(checkpoint,manifest)
    progress=relearn_checkpoint_progress(checkpoint,manifest)
    cfg_path=ROOT/'configs/gallop_relearn_recovery.json';cfg=GallopSettings.load(cfg_path)
    curriculum=RelearnCurriculum(cfg);curriculum.configure_envs(256)
    mapped=map_relearn_progress(progress,manifest,curriculum.plan);curriculum.restore(mapped['level'],mapped.get('highest_confirmed_speed'))
    agent=load_yaml(parent/'params/agent.yaml');agent['algorithm']['learning_rate']=1e-4;agent['algorithm']['entropy_coef']=0.
    class Probe:
        num_envs=2;num_actions=12
        def get_observations(self):
            return TensorDict(dict(policy=torch.zeros(2,49),critic=torch.zeros(2,64)),batch_size=[2])
    old=OnPolicyRunner(Probe(),deepcopy(agent),log_dir=None,device='cpu')
    new=OnPolicyRunner(Probe(),deepcopy(agent),log_dir=None,device='cpu')
    def forbid(*args,**kwargs):raise RuntimeError('Training is forbidden in recovery validation')
    for item in (old,new):item.learn=forbid;item.alg.update=forbid;item.alg.optimizer.step=forbid
    old.load(str(checkpoint),load_optimizer=False,map_location='cpu')
    restored=restore_gallop_policy(new,checkpoint)
    parent_std=old.alg.policy.std.detach().clone()
    schedule=RelearnExploration(new,curriculum)
    obs=TensorDict(dict(policy=torch.randn(4096,49),critic=torch.randn(4096,64)),batch_size=[4096])
    with torch.inference_mode():
        action_error=(old.alg.policy.act_inference(obs)-new.alg.policy.act_inference(obs)).abs()
        value_error=(old.alg.policy.evaluate(obs)-new.alg.policy.evaluate(obs)).abs()
    state=old.alg.policy.state_dict();new_state=new.alg.policy.state_dict()
    previous=manifest['gallop_settings'];actual=cfg.to_dict()
    changed={k:dict(before=previous.get(k),after=v) for k,v in actual.items() if v!=previous.get(k,v)}
    unchanged_keys=('guarded_sim_limit_factor','guarded_target_margin','guarded_hip_rate','guarded_leg_rate',
        'guarded_actual_tolerance','guarded_violation_seconds','guarded_max_intervention',
        'guarded_front_height','guarded_hind_height','stance_width_weight','guarded_height_weight',
        'relearn_final_request_tolerance','relearn_final_request_seconds')
    checks=dict(parent_sha256=before=='959457e283918bcc480e71c1d6ce026cffb63feb5a1b9e0834b78f60f09a8200',
        parent_iterations=progress['cumulative_iterations']==16000,
        parent_level=progress['level']==1,mapped_level=mapped['level']==2,
        preserved_goal=curriculum.goal==manifest['relearn_design']['plan'][progress['level']],
        no_extra_phase_or_action_dimensions=new_state['actor.0.weight'].shape[1]==49 and new_state['critic.0.weight'].shape[1]==64,
        actor_critic_tensors_preserved=all(torch.equal(v,new_state[k]) for k,v in state.items() if k!='std'),
        deterministic_actor_preserved=float(action_error.max())==0.,critic_preserved=float(value_error.max())==0.,
        lowered_initial_std=abs(schedule.std-.12)<1e-8 and bool(torch.all(new.alg.policy.std<parent_std)),
        frozen_std=not new.alg.policy.std.requires_grad,actor_trainable=all(p.requires_grad for p in new.alg.policy.actor.parameters()),
        critic_trainable=all(p.requires_grad for p in new.alg.policy.critic.parameters()),entropy_zero=new.alg.entropy_coef==0.,
        fresh_optimizer=not new.alg.optimizer.state,iteration_zero=new.current_learning_iteration==0,
        learning_rate=new.alg.learning_rate==1e-4,
        limits_and_rewards_preserved=all(actual[k]==previous[k] for k in unchanged_keys),
        old_fields_preserved=all(actual[k]==v for k,v in previous.items()),
        refined_plan=len(curriculum.plan)==31,
        final_strict_request_failure=curriculum.plan[-1]['request_tolerance']==1e-6 and curriculum.plan[-1]['request_seconds']==.02 and curriculum.plan[-1]['request_fraction']==0.)
    # Validate the installed algorithm's sampling interface and stored Gaussian
    # probabilities. Never call update(), optimizer.step() or an environment.
    sample_obs=Probe().get_observations()
    with torch.inference_mode():
        for _ in range(new.num_steps_per_env):new.alg.act(sample_obs)
        checks['rollout_noise_matches_saved_sigma']=bool(torch.allclose(new.alg.transition.action_sigma,torch.full((2,12),.12)))
        checks['sample_log_probabilities_match']=bool(torch.allclose(new.alg.transition.actions_log_prob,
            new.alg.policy.get_actions_log_prob(new.alg.transition.actions)))
    checks['parent_unchanged']=sha(checkpoint)==before
    checks['asset_unchanged']=sha(manifest['asset_path'])==manifest['asset_sha256']
    checks['official_checkout_clean']=not subprocess.check_output(['git','-C',manifest['official_repo'],'status','--porcelain'],text=True)
    run=new_run('isaac','check');out=run.parent/'unitree_go2_gallop'/run.name;out.mkdir(parents=True)
    sources=['isaac_task/gallop_core.py','isaac_task/gallop_env.py','isaac_task/gallop_runtime.py',
        'isaac_task/logging.py','scripts/training/gallop_baseline.py',
        'scripts/validation/check_gallop_relearn_recovery.py','tests/test_gallop_relearn_recovery.py',
        'configs/gallop_relearn_recovery.json']
    result=dict(all_checks_passed=all(checks.values()),checks=checks,parent=str(checkpoint),parent_sha256=before,
        parent_iterations=16000,planned_new_iterations=6000,planned_cumulative_iterations=22000,
        restored=restored,mapped_progress=mapped,parent_std_mean=float(parent_std.mean()),initial_std=schedule.std,
        max_actor_error=float(action_error.max()),max_critic_error=float(value_error.max()),
        original_fields_changed=changed,config=actual,plan=curriculum.plan,
        policy_updates=0,supervised_updates=0,physics_rollouts=0,cuda_available=torch.cuda.is_available(),
        source_sha256={name:sha(ROOT/name) for name in sources},
        limitation='CPU checkpoint and control-logic validation only; no new Isaac dynamics, PPO update, or Gallop performance validation.')
    (out/'recovery_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),checks=len(checks),all_checks_passed=result['all_checks_passed'],failed=[k for k,v in checks.items() if not v])))
    if not result['all_checks_passed']:raise SystemExit(1)


if __name__=='__main__':main()
