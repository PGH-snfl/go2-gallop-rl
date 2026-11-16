"""Rollout-boundary exploration annealing and read-only PPO diagnostics."""
import math

import torch


def gaussian_kl(old_mean,old_std,new_mean,new_std):
    """KL(old||new) per sample, summed across independent action dimensions."""
    if not bool((old_std>0).all() and (new_std>0).all()):
        raise ValueError('Positive Gaussian standard deviations required')
    return (torch.log(new_std/old_std)+(old_std.square()+(old_mean-new_mean).square())/
            (2*new_std.square())-.5).sum(-1).clamp_min(0)


class StagedSpeedAdaptation:
    """Keep Gaussian samples/log probabilities intact within each PPO rollout.

    Freeze state-independent variance and anneal its mean toward the configured
    target while retaining relative per-joint scales. Actor and critic still
    learn normally. No actions, likelihoods, gradients or optimizer steps are
    replaced. Full-rollout KL diagnostics are not the internal scheduler's
    per-minibatch KL, and use deterministic inference without RNG consumption.
    """
    def __init__(self,runner,settings,emit=None):
        self.runner=runner;self.cfg=settings;self.emit=emit;self.calls=0;self.rollout_index=0
        policy=runner.alg.policy
        if getattr(policy,'state_dependent_std',False) or getattr(policy,'is_recurrent',False):
            raise ValueError('Managed exploration requires a feed-forward, state-independent Gaussian')
        if getattr(runner.alg,'is_multi_gpu',False):raise ValueError('This diagnostic currently supports single-GPU training')
        self.parameter=getattr(policy,'log_std',None);self.logarithmic=self.parameter is not None
        if self.parameter is None:self.parameter=policy.std
        self.initial=(self.parameter.detach().exp() if self.logarithmic else self.parameter.detach()).clone()
        if not bool(torch.isfinite(self.initial).all() and (self.initial>0).all()):
            raise ValueError('Invalid parent exploration variance')
        self.initial_mean=float(self.initial.mean());self.target_mean=min(self.initial_mean,settings.staged_noise_target_std)
        self.parameter.requires_grad_(False);runner.alg.entropy_coef=0.
        self.diagnostics=dict(rollout_kl_pre_update=0.,rollout_kl_post_update=0.,rollout_kl_samples=0.)
        self.target_announced=False;self.apply(0)
        if emit:emit('STAGED_ADAPTATION',parent_noise_mean=self.initial_mean,target_noise_mean=self.target_mean,
            ramp_iterations=settings.staged_noise_ramp_iterations,entropy_coef=0.,std_trainable=False,
            update_boundary='before collection; unchanged throughout PPO update')
        original_act=runner.alg.act
        def act(obs):
            if self.calls%runner.num_steps_per_env==0:self.apply(self.calls//runner.num_steps_per_env)
            result=original_act(obs);self.calls+=1
            return result
        runner.alg.act=act
        original_update=runner.alg.update
        def update(*args,**kwargs):
            storage=runner.alg.storage;count=storage.step
            if count!=runner.num_steps_per_env:raise ValueError('Diagnostics require one complete PPO rollout')
            observations=storage.observations[:count].flatten(0,1)
            old_mean=storage.mu[:count].flatten(0,1).detach()
            old_std=storage.sigma[:count].flatten(0,1).detach()
            # RSL clear() only resets counters; these buffers remain readable.
            pre=self._rollout_kl(observations,old_mean,old_std)
            result=original_update(*args,**kwargs)
            post=self._rollout_kl(observations,old_mean,old_std)
            self.diagnostics.update(rollout_kl_pre_update=pre,rollout_kl_post_update=post,
                rollout_kl_samples=float(len(old_mean)))
            return result
        runner.alg.update=update

    def _rollout_kl(self,observations,old_mean,old_std):
        with torch.inference_mode():
            mean=self.runner.alg.policy.act_inference(observations)
            std=(self.parameter.exp() if self.logarithmic else self.parameter).expand_as(mean)
            value=gaussian_kl(old_mean,old_std,mean,std).mean()
            if not bool(torch.isfinite(value)):raise ValueError('Nonfinite rollout KL')
            return float(value)

    def apply(self,rollout_index):
        self.rollout_index=rollout_index
        self.fraction=min(1.,rollout_index/self.cfg.staged_noise_ramp_iterations)
        mean=self.initial_mean+(self.target_mean-self.initial_mean)*self.fraction
        desired=self.initial*(mean/self.initial_mean)
        with torch.no_grad():self.parameter.copy_(desired.log() if self.logarithmic else desired)
        self.std=mean
        if self.fraction==1 and not self.target_announced:
            self.target_announced=True
            if self.emit:self.emit('STAGED_ADAPTATION',noise_mean=mean,rollout_index=rollout_index,
                state='annealing_target_reached',entropy_coef=0.)

    def metrics(self):
        alg=self.runner.alg
        values=dict(noise_std_mean=self.std,noise_ramp_fraction=self.fraction,
            entropy_coef=alg.entropy_coef,learning_rate=alg.learning_rate,
            learning_rate_at_floor=float(alg.learning_rate<=1.00001e-5),**self.diagnostics)
        return {'StagedAdapt/'+k:v for k,v in values.items()}

    def state(self):
        return dict(initial_std=self.initial.tolist(),initial_mean=self.initial_mean,target_mean=self.target_mean,
            last_collected_rollout_index=self.rollout_index,noise_mean=self.std,ramp_fraction=self.fraction,
            ramp_iterations=self.cfg.staged_noise_ramp_iterations,calls=self.calls,entropy_coef=0.,std_trainable=False)
