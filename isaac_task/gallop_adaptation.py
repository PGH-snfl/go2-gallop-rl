"""User-run supervised migration to bounded joint targets, before PPO.

This module has no Isaac imports. The caller supplies an initialized Isaac env.
Only explicit ``adapt_bounded_policy`` calls optimize; offline validation does
not call it. Actor distillation and closed-loop checks are separate from PPO.
"""
import copy
import hashlib
import json
from pathlib import Path
import math

import torch

from isaac_task.gallop_core import bounded_joint_targets,stabilized_student_actions


def legacy_student_targets(actions,default,limits,calves,mix,settings):
    student=stabilized_student_actions(actions,calves,mix,settings)
    return torch.clamp(default+settings.action_scale*student,min=limits[...,0],max=limits[...,1])


def legacy_distillation_targets(actions,walking,default,limits,calves,stage,mix,settings):
    """Match the parent's final clipped target, including blend ordering."""
    student=stabilized_student_actions(actions,calves,mix,settings)
    old=default+settings.action_scale*(walking*(1-mix[:,None])+student*mix[:,None])
    old=torch.where((stage==0)[:,None],default,old)
    old=torch.clamp(old,min=limits[...,0],max=limits[...,1])
    teacher=torch.clamp(default+settings.action_scale*walking,min=limits[...,0],max=limits[...,1])
    desired=(old-teacher*(1-mix[:,None]))/mix[:,None].clamp_min(1e-6)
    return torch.clamp(desired,min=limits[...,0],max=limits[...,1])


def inverse_bounded_targets(target,limits,epsilon=.001):
    lower,upper=limits.unbind(-1)
    normalized=(2*target-lower-upper)/(upper-lower)
    return torch.atanh(normalized.clamp(-1+epsilon,1-epsilon))


def imitation_loss(actor,observations,targets,limits):
    latent=actor(observations)
    desired=inverse_bounded_targets(targets,limits)
    # Inverse-coordinate supervision retains a useful gradient when old
    # targets lie exactly on a boundary; angle loss also measures the task.
    angle=bounded_joint_targets(latent,limits)
    return torch.nn.functional.smooth_l1_loss(latent,desired)+((angle-targets)/(limits[...,1]-limits[...,0])).square().mean()


def angle_error(actor,observations,targets,limits,batch_size):
    differences=[]
    with torch.no_grad():
        for start in range(0,len(observations),batch_size):
            obs=observations[start:start+batch_size].to(limits.device)
            target=targets[start:start+batch_size].to(limits.device)
            differences.append((bounded_joint_targets(actor(obs),limits)-target).abs().cpu())
    error=torch.cat(differences)
    return dict(rmse_rad=float(error.square().mean().sqrt()),p95_rad=float(torch.quantile(error.flatten(),.95)),
        per_joint_rmse_rad=error.square().mean(0).sqrt().tolist(),samples=len(error))


def adaptation_report(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def measured_gait(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    measured=[r for r in rows if r.get('gallop_metric_seconds',0)>=5]
    keys=('pattern','flight','motion','front_load','front_motion','front_cycle')
    return dict(measured_episodes=len(measured),gait_fraction=sum(all(r[k] for k in keys) for r in measured)/max(1,len(measured)))


def adapt_bounded_policy(runner,wrapped,env,checkpoint,out,*,steps=1500,epochs=30,batch_size=4096,
                         learning_rate=1e-4,rounds=2,check_steps=1500,max_rmse=.10,max_p95=.25,emit=None):
    """Collect frozen-parent trajectories, fit the actor, check student dynamics.

    The teacher rollout uses the old target transform and previous-action
    semantics. Holdout consists of whole environments, never random adjacent
    frames. The final check runs the student with its actual new observations.
    Raises before PPO if imitation or basic closed-loop preservation fails.
    """
    if min(steps,epochs,batch_size,rounds,check_steps)<1 or env.num_envs<2:raise ValueError('Adaptation needs >=2 environments and positive counts')
    checkpoint=Path(checkpoint);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    actor=runner.alg.policy.actor;teacher=copy.deepcopy(actor).eval().requires_grad_(False)
    teacher_before={k:v.detach().clone() for k,v in teacher.state_dict().items()}
    state=env.gallop;cfg=state.cfg
    limits=state.robot.data.soft_joint_pos_limits[0].detach().clone()
    default=state.robot.data.default_joint_pos[0].detach().clone()
    if limits.shape!=(12,2) or default.shape!=(12,):raise ValueError('Expected native 12-joint asset order')
    original_episode_path=state.episode_path;original_curriculum=state.curriculum_enabled
    state.curriculum_enabled=False;state.episode_path=out/'teacher_episodes.jsonl';state.bounded_teacher_rollout=True
    observations=[];targets=[];masks=[];validations=[];full_samples=0
    holdout_env=torch.arange(env.num_envs,device=limits.device)%5==0
    # With 2--4 environments keep one environment for independent validation.
    if holdout_env.all():holdout_env[-1]=False
    result=dict(status='collecting',teacher_checkpoint=str(checkpoint.resolve()),teacher_sha256=before,
        supervised_updates=0,ppo_updates=0,teacher_steps=steps,student_check_steps=check_steps,
        holdout='environment IDs divisible by five',physical_target_loss=True,imitation_rounds=rounds,
        student_label_rollouts=rounds-1)
    report=out/'adaptation.json';adaptation_report(report,result)
    def event(name,**values):
        if emit:emit(name,**values)
    def reset_for_stage(path):
        # Do not attribute a truncated previous-stage episode to the new file.
        state.episode_path=None;state.age.zero_();obs,_=wrapped.reset();state.episode_path=path
        return obs
    try:
        obs,_=wrapped.reset()
        with torch.no_grad():
            for step in range(steps):
                x=obs['policy'].detach().clone();action=teacher(x)
                if not torch.isfinite(x).all() or not torch.isfinite(action).all():raise RuntimeError('Nonfinite parent rollout')
                # Fit only the student-active blend/Gallop states. Walking and
                # hold remain controlled by the unchanged 45-input teacher.
                selected=state.startup.mix>0
                target=legacy_distillation_targets(action,state.walking_policy(x),default,limits,
                    state.calf_ids,state.startup.stage,state.startup.mix,cfg)
                observations.append(x.cpu());targets.append(target.cpu())
                full_samples+=int((state.startup.stage==3).sum())
                obs,_,_,_=wrapped.step(action)
                if (step+1)%250==0:event('BOUNDED_TEACHER_COLLECT',step=step+1,total=steps)
                # Store the current student-active and environment holdout mask.
                masks.append(selected.cpu());validations.append(holdout_env.cpu())
        x=torch.cat(observations);y=torch.cat(targets);active=torch.cat(masks);holdout=torch.cat(validations)
        train_ids=torch.where(active&~holdout)[0];test_ids=torch.where(active&holdout)[0]
        if len(train_ids)<batch_size or len(test_ids)<128 or full_samples<128:
            raise RuntimeError('Insufficient blend/Gallop teacher data; increase --distill-steps or environments')
        result.update(status='fitting',training_samples=len(train_ids),holdout_samples=len(test_ids),
            full_gallop_samples=full_samples,initial_error=angle_error(actor,x[test_ids],y[test_ids],limits,batch_size))
        adaptation_report(report,result)
        torch.save(dict(observations=x,targets=y,active=active,holdout=holdout,limits=limits.cpu(),default=default.cpu()),out/'teacher_dataset.pt')
        result['teacher_gait']=measured_gait(out/'teacher_episodes.jsonl')
        if result['teacher_gait']['measured_episodes']<env.num_envs*.5:
            raise RuntimeError('Insufficient completed teacher gait measurements; increase --distill-steps')
        history=[]
        def fit(round_index):
            optimizer=torch.optim.Adam(actor.parameters(),lr=learning_rate)
            best=float('inf');best_actor=None;actor.train()
            for epoch in range(epochs):
                order=train_ids[torch.randperm(len(train_ids))]
                for start in range(0,len(order),batch_size):
                    ids=order[start:start+batch_size]
                    loss=imitation_loss(actor,x[ids].to(limits.device),y[ids].to(limits.device),limits)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite bounded imitation loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
                    result['supervised_updates']+=1
                error=angle_error(actor,x[test_ids],y[test_ids],limits,batch_size)
                if not math.isfinite(error['rmse_rad']):raise RuntimeError('Nonfinite bounded holdout error')
                history.append(dict(round=round_index,epoch=epoch+1,**error))
                event('BOUNDED_IMITATION',round=round_index,epoch=epoch+1,total=epochs,rmse_rad=error['rmse_rad'],p95_rad=error['p95_rad'])
                if error['rmse_rad']<best:
                    best=error['rmse_rad'];best_actor=copy.deepcopy(actor.state_dict())
            actor.load_state_dict(best_actor);actor.eval()
        fit(1)
        # Relabel student-visited observations with the frozen parent. This
        # explicitly covers the changed bounded previous-action distribution.
        for round_index in range(2,rounds+1):
            state.bounded_teacher_rollout=False
            obs=reset_for_stage(out/f'student_label_episodes_{round_index}.jsonl');xs=[];ys=[];ms=[];hs=[]
            with torch.no_grad():
                for step in range(steps):
                    sample=obs['policy'].detach().clone();label=teacher(sample)
                    if not torch.isfinite(sample).all() or not torch.isfinite(label).all():raise RuntimeError('Nonfinite student label rollout')
                    desired=legacy_distillation_targets(label,state.walking_policy(sample),default,limits,
                        state.calf_ids,state.startup.stage,state.startup.mix,cfg)
                    xs.append(sample.cpu());ys.append(desired.cpu());ms.append((state.startup.mix>0).cpu());hs.append(holdout_env.cpu())
                    obs,_,_,_=wrapped.step(actor(sample))
                    if (step+1)%250==0:event('BOUNDED_STUDENT_LABELS',round=round_index,step=step+1,total=steps)
            new_x,new_y,new_active,new_holdout=map(torch.cat,(xs,ys,ms,hs))
            torch.save(dict(observations=new_x,targets=new_y,active=new_active,holdout=new_holdout),out/f'student_dataset_{round_index}.pt')
            x=torch.cat((x,new_x));y=torch.cat((y,new_y));active=torch.cat((active,new_active));holdout=torch.cat((holdout,new_holdout))
            train_ids=torch.where(active&~holdout)[0];test_ids=torch.where(active&holdout)[0]
            fit(round_index)
        result.update(training_samples=len(train_ids),holdout_samples=len(test_ids))
        result.update(history=history,final_error=angle_error(actor,x[test_ids],y[test_ids],limits,batch_size))
        if (not math.isfinite(result['final_error']['rmse_rad']) or not math.isfinite(result['final_error']['p95_rad'])
                or result['final_error']['rmse_rad']>max_rmse or result['final_error']['p95_rad']>max_p95):
            raise RuntimeError('Bounded imitation holdout error too large; PPO has not started. See adaptation/adaptation.json')
        with torch.no_grad():
            policy=runner.alg.policy
            if hasattr(policy,'std'):policy.std.fill_(cfg.bounded_latent_std)
            elif hasattr(policy,'log_std'):policy.log_std.fill_(torch.log(torch.tensor(cfg.bounded_latent_std,device=limits.device)))
            else:raise ValueError('Unknown policy noise representation')
        runner.save(str(out.parent/'model_adapted.pt'))
        # Actual new mapping and actual new previous-action observations.
        state.bounded_teacher_rollout=False
        obs=reset_for_stage(out/'student_check_episodes.jsonl');reached=torch.zeros(env.num_envs,dtype=torch.bool,device=limits.device)
        fell=torch.zeros_like(reached);bounds_ok=True
        with torch.no_grad():
            for step in range(check_steps):
                obs,reward,done,_=wrapped.step(actor(obs['policy']))
                reached|=state.startup.stage==3;fell|=env.reset_terminated
                bounds_ok&=bool(torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
                    and (state.bounded_target>=limits[:,0]-1e-6).all() and (state.bounded_target<=limits[:,1]+1e-6).all())
                if (step+1)%250==0:event('BOUNDED_STUDENT_CHECK',step=step+1,total=check_steps)
        result['student_check']=dict(reached_gallop_fraction=float(reached.float().mean()),
            envs_with_fall_fraction=float(fell.float().mean()),finite_and_within_limits=bounds_ok,
            note='Basic closed-loop preservation check, not full Gallop qualification or sim2sim')
        gait=measured_gait(out/'student_check_episodes.jsonl')
        result['student_check'].update(gait)
        result['student_check']['minimum_gait_fraction']=max(.2,result['teacher_gait']['gait_fraction']-.15)
        if not bounds_ok or result['student_check']['reached_gallop_fraction']<.8 or result['student_check']['envs_with_fall_fraction']>.2:
            raise RuntimeError('Student preservation check failed; PPO has not started. Inspect adaptation/student_check_episodes.jsonl')
        if gait['measured_episodes']<env.num_envs*.5 or gait['gait_fraction']<result['student_check']['minimum_gait_fraction']:
            raise RuntimeError('Student gait/flight/forelimb preservation check failed; PPO has not started. Inspect adaptation/adaptation.json')
        if not all(torch.equal(v,teacher_before[k]) for k,v in teacher.state_dict().items()):raise RuntimeError('Frozen teacher changed')
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=before:raise RuntimeError('Parent checkpoint changed')
        if runner.alg.optimizer.state:raise RuntimeError('PPO optimizer used before adaptation completed')
        result.update(status='completed',teacher_unchanged=True,parent_checkpoint_unchanged=True,ppo_optimizer_fresh=True)
        return result
    except BaseException as error:
        result.update(status='failed',error=str(error));raise
    finally:
        state.bounded_teacher_rollout=False;state.episode_path=original_episode_path;state.curriculum_enabled=original_curriculum
        adaptation_report(report,result)
        # Discard adaptation episodes/counters/curriculum history before PPO.
        if result['status']=='completed':
            state.curriculum=type(state.curriculum)(cfg,state.curriculum.emit)
            env.common_step_counter=0;env.episode_length_buf.zero_();state.age.zero_();state.first.fill_(True)
            state.last_episode={};state.level.zero_()
            wrapped.reset()
