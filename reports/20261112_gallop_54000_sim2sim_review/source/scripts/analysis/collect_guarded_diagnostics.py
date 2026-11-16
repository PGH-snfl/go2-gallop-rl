"""Audit recorded guarded commands, physical limits and foot height penalties."""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from scripts.analysis.collect_unitree_baseline import load_yaml


def read(p):return json.loads(p.read_text())
def write(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def huber(x):return np.where(x<1,.5*x*x,x-.5)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);a=parser.parse_args()
    out=a.report.resolve();summary=read(out/'summary.json');train=Path(summary['train']);play=Path(summary['play'])
    m=summary['provenance'];cfg=m['gallop_settings'];contract=m['guarded_runtime_joint_contract']
    names=contract['joint_names'];original=np.array(contract['original_limits']);physical=np.array(contract['physics_limits'],dtype=np.float32)
    operating=np.array(contract['target_limits'],dtype=np.float32)
    robot=load_yaml(train/'params/env.yaml')['scene']['robot']
    default=np.array([next(v for p,v in robot['init_state']['joint_pos'].items() if re.fullmatch(p,n)) for n in names],dtype=np.float32)
    with np.load(play/'gallop_trace.npz') as z:data={k:z[k] for k in z.files}
    valid=data['valid'][:,0].astype(bool);formal=valid&data['gallop_metric_active'][:,0].astype(bool)
    if not valid.all():raise ValueError('This guarded audit requires a single first episode')
    canonical=[f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
    actual=data['joint_pos'][:,0].reshape(-1,12)[:,[canonical.index(n) for n in names]]
    requested=data['guarded_requested'][:,0];target=data['guarded_target'][:,0]
    dt=np.float32(np.median(np.diff(data['time'])));rates=np.array([cfg['guarded_hip_rate'] if '_hip_' in n else cfg['edu_calf_target_rate'] if cfg.get('edu_limits_enabled',False) and '_calf_' in n else cfg['guarded_leg_rate'] for n in names],dtype=np.float32)
    previous=default.copy();replayed=[];rate_interventions=[];rate_joint=[]
    for request in requested:
        lower=np.maximum(operating[:,0],previous-rates*dt);upper=np.minimum(operating[:,1],previous+rates*dt)
        if not np.all(lower<=upper+1e-6):raise ValueError('Empty feasible command interval')
        next_target=np.clip(request,lower,upper)
        rate_flags=np.abs(request-previous)>rates*dt+1e-6
        rate_interventions.append(rate_flags.mean());rate_joint.append(rate_flags)
        replayed.append(next_target.copy());previous=next_target
    replayed=np.array(replayed);np.testing.assert_allclose(replayed,target,atol=3e-6)
    derivative=np.diff(np.vstack((default,target)),axis=0)/dt
    np.testing.assert_allclose(derivative,data['guarded_target_rate'][:,0],atol=2e-4)
    intervention=np.abs(requested-target)>1e-6
    np.testing.assert_allclose(intervention.mean(-1),data['guarded_intervention'][:,0],atol=2e-6)
    np.testing.assert_allclose(rate_interventions,data['guarded_rate_intervention'][:,0],atol=2e-6)
    weights=np.array([2 if '_calf_' in n else 1 for n in names])
    projection=(huber(np.abs(requested-target)/cfg['guarded_projection_scale'])*weights).mean(-1)
    np.testing.assert_allclose(projection,data['guarded_projection_cost'][:,0],atol=2e-5)
    excess=lambda q,limit:np.maximum(limit[:,0]-q,0)+np.maximum(q-limit[:,1],0)
    actual_excess=excess(actual,physical)
    np.testing.assert_allclose(actual_excess,data['guarded_actual_excess'][:,0],atol=2e-6)
    np.testing.assert_allclose((actual_excess>cfg['guarded_actual_tolerance']).mean(-1),data['guarded_actual_fraction'][:,0],atol=2e-6)
    np.testing.assert_allclose(default+cfg['action_scale']*data['executed_action'][:,0],target,atol=1e-6)
    caps=np.array([cfg['guarded_front_height']]*2+[cfg['guarded_hind_height']]*2)
    raw_height=data['foot_height'][:,0];height=raw_height
    if cfg.get('flight_height_compensation',False):
        height=np.maximum(raw_height-data['guarded_body_rise'][:,0,None],0.)
    cost=huber(np.maximum(height-caps,0)/cfg['guarded_height_scale'])
    np.testing.assert_allclose(height,data['guarded_foot_height'][:,0],atol=1e-6)
    episode=json.loads((play/'episodes.jsonl').read_text().splitlines()[0])
    for sl,name in [(slice(0,2),'front'),(slice(2,4),'hind')]:
        np.testing.assert_allclose(cost[formal,sl].mean(),episode['guarded_'+name+'_cost'],atol=1e-6)
        np.testing.assert_allclose((height[formal,sl]>caps[sl]+.001).mean(),episode['guarded_'+name+'_high'],atol=1e-6)
    aggregate=dict(formal_samples=int(formal.sum()),formal_seconds=float(formal.sum()*dt),
        target_operating_violation_fraction=float((excess(target[formal],operating)>1e-6).mean()),
        target_physical_violation_fraction=float((excess(target[formal],physical)>1e-6).mean()),
        target_rate_violation_fraction=float((np.abs(derivative[formal])>rates+2e-4).mean()),
        original_mechanical_violation_fraction=float((excess(actual[formal],original)>.001).mean()),
        new_physical_violation_fraction_over_1mrad=float((actual_excess[formal]>.001).mean()),
        new_physical_violation_fraction_over_guard_tolerance=float((actual_excess[formal]>cfg['guarded_actual_tolerance']).mean()),
        max_actual_new_physical_excess_rad=float(actual_excess[formal].max()),
        filter_intervention_fraction=float(intervention[formal].mean()),
        request_execution_rmse_rad=float(np.sqrt(np.mean((requested[formal]-target[formal])**2))),
        projection_cost_mean=float(projection[formal].mean()),
        rate_intervention_fraction=float(np.mean(np.array(rate_interventions)[formal])),
        raw_angle_exceed_fraction=float((excess(requested[formal],physical)>1e-6).mean()),
        target_rate_abs_p95=float(np.quantile(np.abs(derivative[formal]),.95)),
        tracking_rmse_rad=float(np.sqrt(np.mean((actual[formal]-target[formal])**2))),
        max_violation_seconds=float(data['guarded_violation_seconds'][formal,0].max()))
    np.testing.assert_allclose(aggregate['tracking_rmse_rad'],episode['guarded_tracking_rmse'],atol=1e-6)
    np.testing.assert_allclose(aggregate['filter_intervention_fraction'],episode['guarded_intervention'],atol=1e-6)
    foot={}
    for i,leg in enumerate(('FL','FR','RL','RR')):
        h=height[formal,i]
        foot[leg]=dict(cap_m=float(caps[i]),body_rise_compensated=cfg.get('flight_height_compensation',False),raw_world_max_m=float(raw_height[formal,i].max()),p50_p90_p95_max_m=np.quantile(h,[.5,.9,.95,1]).tolist(),
            fraction_over_cap=float((h>caps[i]).mean()),fraction_over_cap_plus_1mm=float((h>caps[i]+.001).mean()))
    per_joint=[dict(joint=n,intervention=float(intervention[formal,i].mean()),
        request_rate_limited=float(np.array(rate_joint)[formal,i].mean()),
        actual_max_excess_rad=float(actual_excess[formal,i].max()),tracking_rmse_rad=float(np.sqrt(np.mean((actual[formal,i]-target[formal,i])**2))),
        target_rate_abs_max=float(np.abs(derivative[formal,i]).max()),target_rate_limit=float(rates[i])) for i,n in enumerate(names)]
    write(out/'guarded_play_diagnostics.json',dict(aggregate=aggregate,foot_height=foot,per_joint=per_joint,
        checks=dict(command_projection_reproduced=True,rate_and_cost_reproduced=True,actual_state_excess_reproduced=True,height_cost_reproduced=True),
        limits=contract,note='Saved observations only. Physical solver tolerance and command speed are distinct from actual joint velocity.'))
    rows=[json.loads(line) for line in (train/'episodes.jsonl').read_text().splitlines()]
    eligible=[r for r in rows if r['eligible']];rollout=load_yaml(train/'params/agent.yaml')['num_steps_per_env'];last=summary['iterations']*rollout
    groups=dict(all_eligible=eligible,first_200=[r for r in eligible if r['step']<=200*rollout],
        last_200=[r for r in eligible if r['step']>=last-200*rollout],last_1000=[r for r in eligible if r['step']>=last-1000*rollout])
    result={}
    for label,group in groups.items():
        measured=[r for r in group if r['gallop_metric_seconds']>=cfg['stabilization_min_metric_seconds']]
        result[label]=dict(episodes=len(group),measured_episodes=len(measured),success=sum(bool(r['success']) for r in group),
            terminated_fraction=float(np.mean([r['terminated'] for r in group])) if group else None,
            measured_episodes_with_actual_excess=sum(r['guarded_actual']>0 for r in measured),
            criteria={k:float(np.mean([r[k] for r in group])) for k in ('guarded_limits','guarded_feet','tracking','flight','stabilization_width','stabilization_landing')} if group else {},
            measured_quantiles={k:np.quantile([r[k] for r in measured],[.1,.5,.9]).tolist() for k in ('guarded_intervention','guarded_actual','guarded_front_high','guarded_hind_high','guarded_tracking_rmse')} if measured else {})
    write(out/'guarded_training_diagnostics.json',result)
    with (out/'guarded_joint_diagnostics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_joint[0]));writer.writeheader();writer.writerows(per_joint)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True);t=data['time']
    for i,leg in enumerate(('FL','FR','RL','RR')):axes[0].plot(t,height[:,i]*100,label=leg,alpha=.8)
    axes[0].axhline(caps[0]*100,color='r',ls='--',label='front cap');axes[0].axhline(caps[2]*100,color='k',ls='--',label='hind cap')
    axes[0].set_ylabel('Foot clearance (cm)');axes[0].legend(ncol=6)
    axes[1].plot(t,intervention.mean(-1),label='all filter interventions');axes[1].plot(t,rate_interventions,label='rate interventions',alpha=.7)
    axes[1].axhline(cfg['guarded_max_intervention'],color='r',ls='--');axes[1].set_ylabel('Joint fraction');axes[1].legend()
    axes[2].plot(t,np.sqrt(np.mean((target-actual)**2,axis=1)),label='tracking RMSE');axes[2].set_ylabel('Radians');axes[2].set_xlabel('Seconds');axes[2].legend()
    for ax in axes:ax.axvspan(t[0],t[formal][0],alpha=.1,color='gray');ax.grid(alpha=.2)
    fig.suptitle('Guarded playback: foot height, filtering and joint tracking');fig.tight_layout();fig.savefig(out/'guarded_height_and_intervention.png',dpi=130);plt.close(fig)
    print(json.dumps(dict(aggregate=aggregate,foot_height=foot,eligible_episodes=len(eligible))))


if __name__=='__main__':main()
