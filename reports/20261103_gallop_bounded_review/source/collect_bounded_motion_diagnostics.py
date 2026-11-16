"""Audit saved joint bounds, command rates and policy drift; no physics/PPO."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch

from gpd.paths import ROOT
from scripts.analysis.collect_unitree_baseline import load_yaml


def read(path):return json.loads(path.read_text())


def geometry(train):
    manifest=read(train/'manifest.json');asset=read(ROOT/'assets/go2_isaac_contract.json')
    if asset['isaac_asset_sha256']!=manifest['asset_sha256']:raise ValueError('Asset mismatch')
    cfg=load_yaml(train/'params/env.yaml')['scene']['robot'];names=asset['native_joint_order']
    lookup={j['name']:np.deg2rad([j['lower'],j['upper']]) for j in asset['joints']}
    hard=np.array([lookup[n] for n in names]);mid=hard.mean(-1);radius=np.ptp(hard,axis=1)*.5*cfg['soft_joint_pos_limit_factor']
    soft=np.stack((mid-radius,mid+radius),axis=-1)
    default=np.array([next(v for p,v in cfg['init_state']['joint_pos'].items() if re.fullmatch(p,n)) for n in names])
    return names,hard,soft,default


def project_commands(requested,default,limits,rates,dt):
    """Offline feasibility prototype. Does not predict a robot trajectory."""
    previous=np.clip(default,limits[:,0],limits[:,1]);result=[]
    for target in requested:
        lower=np.maximum(limits[:,0],previous-rates*dt);upper=np.minimum(limits[:,1],previous+rates*dt)
        if not np.all(lower<=upper+1e-10):raise ValueError('Infeasible angle/rate intersection')
        previous=np.clip(target,lower,upper);result.append(previous.copy())
    return np.array(result)


def audit(train,play):
    names,hard,soft,default=geometry(train);manifest=read(train/'manifest.json')
    with np.load(play/'gallop_trace.npz') as data:z={k:data[k] for k in data.files}
    mask=z['valid'][:,0]&z['gallop_metric_active'][:,0];dt=float(np.median(np.diff(z['time'])))
    canonical=[f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
    actual=z['joint_pos'][:,0].reshape(-1,12)[:,[canonical.index(n) for n in names]]
    raw=default+manifest['action_scale']*z['executed_action'][:,0]
    target=np.clip(raw,soft[:,0],soft[:,1])
    if 'bounded_target' in z:np.testing.assert_allclose(target,z['bounded_target'][:,0],atol=2e-6)
    excess=lambda q,limits:np.maximum(limits[:,0]-q,0)+np.maximum(q-limits[:,1],0)
    rate=np.diff(target,axis=0)/dt;valid_rate=mask[1:]&mask[:-1]
    error=target[mask]-actual[mask]
    near=np.minimum(target-soft[:,0],soft[:,1]-target)/np.ptp(soft,axis=1)<.05
    soft_excess=excess(actual[mask],soft);hard_excess=excess(actual[mask],hard)
    requested_excess=excess(target[mask],soft)
    summary=dict(samples=int(mask.sum()),first_second=float(z['time'][mask][0]),
        target_soft_limit_violation_fraction=float((requested_excess>1e-6).mean()),
        max_target_soft_excess_rad=float(requested_excess.max()),near_limit_fraction=float(near[mask].mean()),
        actual_soft_limit_violation_fraction=float((soft_excess>1e-3).mean()),actual_soft_max_excess_rad=float(soft_excess.max()),
        actual_hard_limit_violation_fraction=float((hard_excess>1e-3).mean()),actual_hard_max_excess_rad=float(hard_excess.max()),
        tracking_rmse_rad=float(np.sqrt(np.mean(error**2))),target_rate_abs_p95_rad_s=float(np.quantile(abs(rate[valid_rate]),.95)),
        target_rate_abs_max_rad_s=float(abs(rate[valid_rate]).max()),
        per_joint=[dict(joint=n,tracking_rmse_rad=float(np.sqrt(np.mean(error[:,i]**2))),
            actual_min_max_rad=[float(actual[mask,i].min()),float(actual[mask,i].max())],soft_limits_rad=soft[i].tolist(),hard_limits_rad=hard[i].tolist(),
            near_limit_fraction=float(near[mask,i].mean()),actual_soft_violation_fraction=float((soft_excess[:,i]>1e-3).mean()),
            actual_hard_violation_fraction=float((hard_excess[:,i]>1e-3).mean()),
            target_rate_p95_rad_s=float(np.quantile(abs(rate[valid_rate,i]),.95)),target_rate_max_rad_s=float(abs(rate[valid_rate,i]).max())) for i,n in enumerate(names)],
        note='Targets are end-of-policy-step commands; tracking compares end-of-step actual angles. Actual-limit fractions use 0.001rad tolerance. No motor fault is inferred from tracking error alone.')
    if 'bounded_tracking_error' in z:
        np.testing.assert_allclose(np.sqrt(np.mean(error**2,axis=1)),z['bounded_tracking_error'][mask,0],atol=2e-6)
        summary['torque_saturation_fraction']=float(z['bounded_torque_saturation'][mask,0].mean())
        np.testing.assert_allclose(np.mean(near[mask],axis=1),z['bounded_near_limit_fraction'][mask,0],atol=1e-6)
    return summary,(names,hard,soft,default,z,mask,actual,target)


def actor(path):
    weights=torch.load(path,map_location='cpu',weights_only=True)['model_state_dict'];layers=[]
    keys=sorted([k for k in weights if k.startswith('actor.') and k.endswith('.weight')],key=lambda k:int(k.split('.')[1]))
    for i,key in enumerate(keys):
        w=weights[key];layer=torch.nn.Linear(w.shape[1],w.shape[0]);layer.weight.data.copy_(w);layer.bias.data.copy_(weights[key.replace('weight','bias')]);layers.append(layer)
        if i<len(keys)-1:layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers).eval().requires_grad_(False)


def holdout_drift(train):
    before=actor(train/'model_adapted.pt');after=actor(train/'model_final.pt');result={}
    for name in ('teacher_dataset.pt','student_dataset_2.pt'):
        data=torch.load(train/'adaptation'/name,map_location='cpu',weights_only=True)
        ids=torch.where(data['active']&data['holdout'])[0]
        if len(ids)>8192:ids=ids[torch.linspace(0,len(ids)-1,8192).long()]
        limits=data.get('limits')
        if limits is None:limits=torch.load(train/'adaptation/teacher_dataset.pt',map_location='cpu',weights_only=True)['limits']
        lo,hi=limits.unbind(-1);mid=(lo+hi)/2;half=(hi-lo)/2
        errors={}
        for label,policy in [('adapted',before),('final',after)]:
            differences=[]
            with torch.no_grad():
                for batch in ids.split(512):
                    target=mid+half*torch.tanh(policy(data['observations'][batch]))
                    differences.append((target-data['targets'][batch]).abs())
            error=torch.cat(differences)
            errors[label]=dict(rmse_rad=float(error.square().mean().sqrt()),p95_rad=float(torch.quantile(error.flatten(),.95)))
        result[name]=dict(samples=len(ids),**errors)
    return dict(saved_holdout=result,note='Same stored observations and teacher targets, evenly subsampled to <=8192 states per dataset. Shows function drift during PPO, not a causal locomotion ablation.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--parent-report',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(1)
    out=args.report.resolve();s=read(out/'summary.json');parent=read(args.parent_report/'summary.json')
    train=Path(s['train']);current,details=audit(train,Path(s['play']));prior,_=audit(Path(parent['train']),Path(parent['play']))
    names,hard,soft,default,z,mask,actual,target=details
    margin=.05*np.ptp(soft,axis=1);operating=soft+np.stack((margin,-margin),axis=-1)
    rates=np.array([6. if '_hip_' in n else 16. for n in names]);dt=float(np.median(np.diff(z['time'])))
    projected=project_commands(target,default,operating,rates,dt)
    projected_rate=np.diff(np.vstack((np.clip(default,operating[:,0],operating[:,1]),projected)),axis=0)/dt
    assert np.all(projected>=operating[:,0]-1e-8) and np.all(projected<=operating[:,1]+1e-8)
    assert np.all(np.abs(projected_rate)<=rates+1e-7)
    feasibility=dict(operating_margin_fraction=.05,hip_rate_rad_s=6.,thigh_calf_rate_rad_s=16.,
        angle_and_rate_constraints_satisfied=True,projection_change_rmse_rad=float(np.sqrt(np.mean((projected[mask]-target[mask])**2))),
        changed_joint_command_fraction=float((np.abs(projected[mask]-target[mask])>.001).mean()),
        note='Candidate engineering limits only. Saved commands projected offline; actual motion/reward after filtering is unknown. No runtime policy/control/reward was changed.')
    result=dict(current=current,parent=prior,policy_drift=holdout_drift(train),constraint_feasibility=feasibility,
        policy_updates=0,supervised_updates=0,new_simulations=0)
    (out/'bounded_motion_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'bounded_command_projection.npz',time=z['time'],requested=target,projected=projected,operating_limits=operating,rate_limits=rates)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=(z['time']>=12)&(z['time']<=13)
    fig,axes=plt.subplots(6,2,figsize=(14,17),sharex=True)
    for i,ax in enumerate(axes.flat):
        ax.plot(z['time'][selected],np.rad2deg(target[selected,i]),label='target');ax.plot(z['time'][selected],np.rad2deg(actual[selected,i]),label='actual')
        for limit in soft[i]:ax.axhline(np.rad2deg(limit),color='r',ls='--',alpha=.5)
        ax.set_title(names[i]);ax.set_ylabel('degrees');ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Saved current policy: targets vs actual angles (red: soft limits)');fig.tight_layout();fig.savefig(out/'bounded_joint_tracking.png',dpi=120);plt.close(fig)
    print(json.dumps(dict(target_violation=current['target_soft_limit_violation_fraction'],tracking_rmse=current['tracking_rmse_rad'],parent_tracking_rmse=prior['tracking_rmse_rad'],
        target_rate_p95=current['target_rate_abs_p95_rad_s'],parent_rate_p95=prior['target_rate_abs_p95_rad_s'],drift=result['policy_drift'],feasibility=feasibility)))


if __name__=='__main__':main()
