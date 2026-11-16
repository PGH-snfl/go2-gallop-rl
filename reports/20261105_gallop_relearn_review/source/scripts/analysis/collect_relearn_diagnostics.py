"""Saved v5 stage trends, controller replay and conditional action-noise audit."""
import argparse
from bisect import bisect_left
from collections import Counter,defaultdict
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from isaac_task.gallop_core import GallopSettings,RelearnCurriculum
from scripts.analysis.collect_unitree_baseline import load_yaml


def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def csv_rows(p,rows):
    if not rows:return
    with p.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def cohort(rows,cfg,dt):
    if not rows:return dict(episodes=0)
    measured=[r for r in rows if r['gallop_metric_seconds']>=cfg['stabilization_min_metric_seconds']]
    result=dict(episodes=len(rows),measured_episodes=len(measured),
        stage_success_rate=float(np.mean([r['relearn_stage_success'] for r in rows])),
        full_gallop_successes=sum(bool(r['success']) for r in rows),
        request_termination_rate=float(np.mean([r['relearn_request_terminated'] for r in rows])),
        termination_rate=float(np.mean([r['terminated'] for r in rows])),
        timeout_rate=float(np.mean([r['timed_out'] for r in rows])),
        episode_seconds_q10_q50_q90=np.quantile([r['age']*dt for r in rows],[.1,.5,.9]).tolist(),
        criteria={k:float(np.mean([r[k] for r in rows])) for k in ('tracking','front_load','stabilization_pose','stabilization_measured','guarded_feet')})
    result['measured_quantiles']={key:np.quantile([r[key] for r in measured],[.1,.5,.9]).tolist() for key in
        ('relearn_request_fraction','guarded_intervention','action_clip_fraction','guarded_tracking_rmse','vx_rmse')} if measured else {}
    return result


def expected_stage_success(r,goal,cfg,max_age):
    base=bool(r['tracking'] and not r['terminated'] and r['age']>=max_age-1 and r['stabilization_measured']
        and r['stabilization_pose'] and r['guarded_feet'] and r['front_load'] and r['guarded_actual']==0
        and not r['relearn_request_terminated'] and r['relearn_request_fraction']<=float(np.float32(goal['request_fraction'])))
    if goal['gait']==0:return base
    base=base and r['contact_accuracy']>=.5 and r['cycles']>=cfg['min_cycles'] and all(r[k] for k in ('motion','front_load','front_motion','front_cycle'))
    if goal['gait']>=.5:base=base and r['pattern'] and r['flight_fraction']>=.3
    if goal['gait']>=.75:
        n=r['flight_stable_landings']+r['flight_aborted_landings']
        base=base and r['flight_fraction']>=cfg['flight_cycle_fraction'] and r['stabilization_distance'] and n>=cfg['stabilization_min_recoveries'] and r['stabilization_landing_fraction']>=.6
    return bool(r['success']) if goal['gait']>=1-1e-6 else bool(base)


def survival_without_run(probabilities,run_length):
    states=np.zeros(run_length,dtype=float);states[0]=1.
    for probability in probabilities:
        following=np.zeros_like(states);following[0]=states.sum()*(1-probability)
        following[1:]=states[:-1]*probability;states=following
    return float(states.sum())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1);out=args.report.resolve();summary=read(out/'summary.json');m=summary['provenance'];cfg=m['gallop_settings']
    if not cfg.get('relearn_enabled'):raise ValueError('Expected v5 relearning evidence')
    train=Path(summary['train']);play=Path(summary['play']);plan=m['relearn_design']['plan'];dt=summary['policy_dt']
    agent=load_yaml(train/'params/agent.yaml');rollout=agent['num_steps_per_env'];max_age=round(load_yaml(train/'params/env.yaml')['episode_length_s']/dt)
    with (train/'episodes.jsonl').open() as stream:rows=[json.loads(line) for line in stream if line.strip()]
    rows=[r for r in rows if r['eligible']];events=read(train/'curriculum_summary.json')['events'];event_steps=[e['step'] for e in events]
    by_stage=defaultdict(list);accepted=defaultdict(list);windows=defaultdict(list);window_stages=defaultdict(list);csv=[]
    reconstructed=0
    for r in rows:
        level=int(r['relearn_level']);by_stage[level].append(r)
        window=min((int(r['step'])-1)//(1000*rollout),summary['iterations']//1000-1)
        windows[window].append(r);window_stages[(window,level)].append(r)
        previous=bisect_left(event_steps,r['step'])-1;generation=events[previous]['level'] if previous>=0 else 0
        eligible_now=int(r['level'])==generation
        if eligible_now:accepted[level].append(r)
        expected=expected_stage_success(r,plan[level],cfg,max_age)
        if expected!=bool(r['relearn_stage_success']):raise AssertionError(('Stage-success mismatch',r['step'],r['env']))
        reconstructed+=1
        csv.append(dict(step=int(r['step']),iteration=r['step']/rollout,level=level,generation=int(r['level']),
            accepted_by_current_generation=int(eligible_now),stage_success=int(r['relearn_stage_success']),
            request_terminated=int(r['relearn_request_terminated']),terminated=int(r['terminated']),seconds=r['age']*dt,
            measured_seconds=r['gallop_metric_seconds'],request_fraction=r['relearn_request_fraction'],filter_intervention=r['guarded_intervention']))
    csv_rows(out/'relearn_episode_diagnostics.csv',csv)
    # Replay the preserved production curriculum from recorded episode facts.
    curriculum=RelearnCurriculum(GallopSettings(**cfg));index=0;replayed=0
    keys=('tracking','pattern','flight','motion','front_load','front_motion','front_cycle','stable','mature',
          'stabilization_pose','stabilization_width','stabilization_landing','stabilization_distance','stabilization_measured',
          'guarded_limits','guarded_feet','relearn_stage_success')
    for saved in events:
        while index<len(rows) and rows[index]['step']<=saved['step']:
            r=rows[index];curriculum.record(bool(r['relearn_stage_success']),int(r['level']),{k:bool(r[k]) for k in keys});index+=1
        actual=curriculum.decide(saved['step'])
        for key in ('reason','level','episodes','streak','discarded_cross_level','relearn_level'):
            if actual[key]!=saved[key]:raise AssertionError(('Curriculum replay mismatch',key,actual,saved))
        if actual['success_rate']!=saved['success_rate']:raise AssertionError('Rate mismatch')
        replayed+=1
    transitions=[e for e in events if e['relearn_level_before']!=e['relearn_level']]
    transitions_counter=Counter(f"{e['relearn_level_before']}->{e['relearn_level']}" for e in transitions)
    visits=[];start=0;level=0
    for e in transitions:
        visits.append(dict(level=level,entry_iteration=start/rollout,exit_iteration=e['step']/rollout,
            control_seconds=(e['step']-start)*dt,ppo_iterations=(e['step']-start)/rollout,exit_reason=e['reason']))
        start=e['step'];level=e['relearn_level']
    visits.append(dict(level=level,entry_iteration=start/rollout,exit_iteration=summary['iterations'],
        control_seconds=(summary['iterations']*rollout-start)*dt,ppo_iterations=(summary['iterations']*rollout-start)/rollout,exit_reason='training_end'))
    csv_rows(out/'relearn_stage_visits.csv',visits)
    stage_stats={str(k):dict(goal=plan[k],all_completed=cohort(v,cfg,dt),accepted_for_curriculum=cohort(accepted[k],cfg,dt)) for k,v in sorted(by_stage.items())}
    window_stats=[dict(iteration_start=k*1000,iteration_end=(k+1)*1000,all=cohort(v,cfg,dt),
        stages={str(level):cohort(window_stages[(k,level)],cfg,dt) for level in sorted(by_stage) if (k,level) in window_stages}) for k,v in sorted(windows.items())]
    snapshot=[]
    for name,progress in sorted(m['relearn_checkpoints'].items(),key=lambda x:x[1]['stage_iterations']):
        w=torch.load(train/name,map_location='cpu',weights_only=True,mmap=True)
        std=w['model_state_dict']['std'].numpy()
        snapshot.append(dict(checkpoint=name,iteration=progress['stage_iterations'],level=progress['level'],
            action_std_mean=float(std.mean()),action_std_min=float(std.min()),action_std_max=float(std.max()),
            physical_noise_std_mean_rad=float(std.mean()*m['action_scale']),
            learning_rate=w['optimizer_state_dict']['param_groups'][0]['lr']))
    csv_rows(out/'relearn_checkpoint_trends.csv',snapshot)
    parent=torch.load(Path(m['source_checkpoint']['path']),map_location='cpu',weights_only=True,mmap=True)
    initial_std=parent['model_state_dict']['std']*summary['provenance']['migration']['actor_output_and_std_multiplier']
    final=torch.load(train/'model_final.pt',map_location='cpu',weights_only=True,mmap=True);std=final['model_state_dict']['std'].double()
    with np.load(play/'gallop_trace.npz') as archive:z={k:archive[k] for k in archive.files}
    formal=z['valid'][:,0].astype(bool)&z['gallop_metric_active'][:,0].astype(bool)
    if not np.all(z['relearn_gait'][formal,0]==0):raise ValueError('Noise audit currently requires student walking with no calf multiplier')
    bounds=np.array(m['guarded_runtime_joint_contract']['target_limits']);mean=torch.tensor(z['guarded_requested'][formal,0],dtype=torch.float64)
    low=torch.tensor(bounds[:,0]);high=torch.tensor(bounds[:,1]);noise=[]
    cdf=lambda x:.5*(1+torch.erf(x/np.sqrt(2)))
    per_joint=None
    for multiplier in (1.,.75,.5):
        sigma=std*m['action_scale']*multiplier
        for goal_index in (0,1,2):
            g=plan[goal_index];tol=g['request_tolerance']
            each=(1-(cdf((high+tol-mean)/sigma)-cdf((low-tol-mean)/sigma))).clamp(0,1)
            probability=(1-(1-each).prod(-1)).numpy();run=round(g['request_seconds']/dt)
            noise.append(dict(noise_multiplier=multiplier,goal_index=goal_index,request_tolerance=tol,required_steps=run,
                single_action_any_joint_exceed_probability_mean=float(probability.mean()),
                single_action_probability_q10_q50_q90=np.quantile(probability,[.1,.5,.9]).tolist(),
                fixed_state_sequence_no_failure_probability=survival_without_run(probability,run)))
            if multiplier==1 and goal_index==2:
                headroom=torch.minimum(mean-low,high-mean)/(std*m['action_scale'])
                per_joint=[dict(joint=name,exceed_probability_mean=float(each[:,i].mean()),minimum_headroom_in_sigma=float(headroom[:,i].min())) for i,name in enumerate(m['guarded_runtime_joint_contract']['joint_names'])]
    # Independent small enumeration validates the consecutive-event calculation.
    p=[.1,.2,.3,.4];enumerated=0.
    for pattern in itertools.product((0,1),repeat=4):
        if any(pattern[i] and pattern[i+1] for i in range(3)):continue
        enumerated+=np.prod([p[i] if value else 1-p[i] for i,value in enumerate(pattern)])
    assert np.isclose(enumerated,survival_without_run(p,2))
    assert np.isclose(survival_without_run(p,1),np.prod(1-np.array(p)))
    noise_result=dict(formal_seconds=int(formal.sum())*dt,samples=int(formal.sum()),
        initial_action_std_mean=float(initial_std.mean()),final_action_std_mean=float(std.mean()),
        final_physical_std_rad=(std*m['action_scale']).tolist(),entropy_coefficient=agent['algorithm']['entropy_coef'],
        cases=noise,per_joint_stage2=per_joint,independent_run_probability_check=True,
        limitation='Analytic independent Gaussian samples on the same recorded mean requests, with the state trajectory fixed. No state feedback, training rollout or actual survival prediction. Noise multipliers are counterfactual settings only.')
    write(out/'relearn_noise_diagnostics.json',noise_result)
    csv_rows(out/'relearn_noise_cases.csv',noise)
    result=dict(episodes=len(rows),independently_reconstructed_stage_successes=reconstructed,controller_decisions_replayed=replayed,
        stages=stage_stats,windows=window_stats,transitions=dict(transitions_counter),max_reached_level=max(by_stage),
        visited_gait_strengths=sorted({r['relearn_gait'] for r in rows}),
        reason_counts=dict(Counter(e['reason'] for e in events)),final_level=m['trained_relearn_level'],
        discarded_old_generation_episodes=curriculum.discarded,visits=visits,
        note='All completed episodes and episodes accepted by the current curriculum generation are reported separately. Late completions can be discarded after a stage change.')
    write(out/'relearn_training_diagnostics.json',result)
    plot(out,result,noise_result,snapshot,events)
    print(json.dumps(dict(report=str(out),episodes=len(rows),stages={k:v['all_completed']['stage_success_rate'] for k,v in stage_stats.items()},transitions=result['transitions'],replayed=replayed)))


def plot(out,result,noise,snapshot,events):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(14,12),layout='constrained')
    x=np.array([e['step']/24 for e in events]);axes[0,0].step(x,[e['relearn_level'] for e in events],where='post');axes[0,0].set_title('Global walking constraint stage; no Gallop stage reached')
    centers=[(w['iteration_start']+w['iteration_end'])/2 for w in result['windows']]
    for level in result['stages']:
        rates=[w['stages'].get(level,{}).get('stage_success_rate',np.nan) for w in result['windows']]
        failures=[w['stages'].get(level,{}).get('request_termination_rate',np.nan) for w in result['windows']]
        axes[0,1].plot(centers,np.array(rates)*100,label='stage '+level);axes[1,0].plot(centers,np.array(failures)*100,label='stage '+level)
    axes[0,1].axhline(80,color='k',ls='--');axes[0,1].set_title('Stage success / all completed episodes (%)')
    axes[1,0].set_title('Request-limit failure flag / all episodes (%)')
    sx=[r['iteration'] for r in snapshot]
    axes[1,1].plot(sx,[r['action_std_mean'] for r in snapshot]);axes[1,1].axhline(noise['initial_action_std_mean'],color='k',ls='--');axes[1,1].set_title('Learned action-noise std; dashed = migrated parent')
    axes[2,0].semilogy(sx,[r['learning_rate'] for r in snapshot]);axes[2,0].set_title('Saved PPO learning rate')
    for multiplier in (1.,.75,.5):
        subset=[r for r in noise['cases'] if r['noise_multiplier']==multiplier]
        axes[2,1].plot([r['goal_index'] for r in subset],[100*r['fixed_state_sequence_no_failure_probability'] for r in subset],marker='o',label=f'noise x{multiplier:g}')
    axes[2,1].set_title('Fixed-state analytic survival; not a rollout (%)');axes[2,1].set_xticks([0,1,2]);axes[2,1].set_xlabel('Constraint stage')
    for ax in axes.flat:ax.grid(alpha=.2)
    for ax in (axes[0,1],axes[1,0],axes[2,1]):ax.legend()
    for ax in (axes[0,0],axes[0,1],axes[1,0],axes[1,1],axes[2,0]):ax.set_xlabel('PPO iteration')
    fig.suptitle('Relearning trends: walking constraint bottleneck');fig.savefig(out/'relearn_trends.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
