"""Collect short-training diagnostics and frozen checkpoint comparisons."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gpd.paths import ROOT


def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def aggregate(cases):
    e=[r['episode'] for r in cases];n=len(e)
    events=sum(x['leap_events'] for x in e)
    settled=sum(x['flight_stable_landings']+x['flight_aborted_landings'] for x in e)
    points=sum(x['gallop_metric_seconds'] for x in e)
    def weighted(key):return sum(x[key]*x['gallop_metric_seconds'] for x in e)/points if points else None
    def width(pair):
        count=sum(x['stabilization_'+pair+'_support_samples'] for x in e)
        return sum(x['stabilization_'+pair+'_width_fraction']*x['stabilization_'+pair+'_support_samples'] for x in e)/count if count else None
    vx=[x['mean_vx'] for x in cases if x['mean_vx'] is not None]
    yaw=[x['yaw_rms_deg'] for x in cases if x['yaw_rms_deg'] is not None]
    tilt=[x['roll_pitch_rms_deg'] for x in cases if x['roll_pitch_rms_deg'] is not None]
    return dict(starts=n,full_student_10s=sum(x['formal_seconds']>=10 for x in cases),
        formal_seconds_mean=float(np.mean([x['formal_seconds'] for x in cases])),
        terminations=sum(bool(x['terminated']) for x in e),startup_deadline_failures=sum(bool(x.get('startup_failed')) for x in e),
        full_successes=sum(bool(x['success']) for x in e),mean_vx=float(np.mean(vx)) if vx else None,
        yaw_rms_deg=float(np.sqrt(np.mean(np.square(yaw)))) if yaw else None,
        roll_pitch_rms_deg=np.sqrt(np.mean(np.square(tilt),axis=0)).tolist() if tilt else None,
        vx_rmse=weighted('vx_rmse'),leap_events=int(events),
        leap_distance_mean=sum(x['leap_distance_mean']*x['leap_events'] for x in e)/events if events else None,
        leap_distance_max=max(x['leap_distance_max'] for x in e),
        leap_air_seconds_mean=sum(x['leap_air_seconds_mean']*x['leap_events'] for x in e)/events if events else None,
        landing_fraction=sum(x['flight_stable_landings'] for x in e)/settled if settled else None,
        front_width_fraction=width('front'),hind_width_fraction=width('hind'),
        raw_clip_fraction=weighted('action_clip_fraction'),filter_fraction=weighted('guarded_intervention'),
        actual_guard_failure_fraction=weighted('guarded_actual'),
        forelimb_gate_pass_count=sum(bool(x['front_load']) and bool(x['front_motion']) and bool(x['front_cycle']) for x in e),
        foot_height_gate_pass_count=sum(bool(x['guarded_feet']) for x in e))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evaluation',type=Path)
    p.add_argument('--report',type=Path,default=ROOT/'reports/20261110_gallop_short_pilot');a=p.parse_args()
    out=a.report.resolve();protocol=read(out/'pilot_protocol.json');run=a.evaluation.resolve();manifest=read(run/'manifest.json')
    assert read(run/'status.json')['status']=='completed' and read(run/'status.json')['policies_unchanged']
    cases=read(run/'cases.json');summaries={label:aggregate([x for x in cases if x['label']==label]) for label in manifest['labels']}
    baseline=summaries['parent30k'];g=protocol['directional_acceptance'];decisions={}
    for label,s in summaries.items():
        if label=='parent30k':continue
        both=lambda k:s[k] is not None and baseline[k] is not None
        gates=dict(full_student=s['full_student_10s']/s['starts']>=g['minimum_full_student_10s_fraction'],
            nominal_terminations=s['terminations']<=g['maximum_nominal_terminations'],
            flight_retained=both('leap_distance_mean') and s['leap_distance_mean']>=g['minimum_leap_mean_ratio_to_parent']*baseline['leap_distance_mean'],
            actual_guard=s['actual_guard_failure_fraction'] is not None and s['actual_guard_failure_fraction']<=g['maximum_actual_guard_failure_fraction'],
            raw_clip=s['raw_clip_fraction'] is not None and s['raw_clip_fraction']<=g['maximum_raw_angle_clip_fraction'],
            speed=s['mean_vx'] is not None and abs(s['mean_vx']-protocol['command_mps'])<=g['maximum_mean_speed_error_mps'],
            yaw=s['yaw_rms_deg'] is not None and s['yaw_rms_deg']<=g['maximum_yaw_rms_deg'],
            forelimbs=s['forelimb_gate_pass_count']/s['starts']>=g['minimum_forelimb_gate_fraction'],
            feet=s['foot_height_gate_pass_count']/s['starts']>=g['minimum_foot_height_gate_fraction'],
            roll_pitch=s['roll_pitch_rms_deg'] is not None and all(v<=limit for v,limit in zip(s['roll_pitch_rms_deg'],g['maximum_roll_pitch_rms_deg'])),
            hind_width=both('hind_width_fraction') and s['hind_width_fraction']>=baseline['hind_width_fraction']-g['maximum_other_metric_loss'])
        width_gain=s['front_width_fraction']-baseline['front_width_fraction'] if both('front_width_fraction') else None
        landing_gain=s['landing_fraction']-baseline['landing_fraction'] if both('landing_fraction') else None
        gates['stability_gain']=((width_gain is not None and landing_gain is not None) and
            ((width_gain>=g['minimum_front_width_gain'] and landing_gain>=-g['maximum_other_metric_loss']) or
             (landing_gain>=g['minimum_landing_gain'] and width_gain>=-g['maximum_other_metric_loss'])))
        decisions[label]=dict(directional_pass=bool(all(gates.values())),criteria={k:bool(v) for k,v in gates.items()},
            front_width_gain=width_gain,landing_gain=landing_gain)
    train=ROOT/protocol['train'];tr=rows(train/'episodes.jsonl');windows=[]
    for low,high in [(0,125),(125,250),(250,375),(375,500)]:
        batch=[x for x in tr if low*24<x['step']<=high*24]
        if not batch:continue
        event_count=sum(x['leap_events'] for x in batch)
        windows.append(dict(start_update=low,end_update=high,episodes=len(batch),
            startup_completed_fraction=sum(bool(x['startup_completed']) for x in batch)/len(batch),
            startup_deadline_failures=sum(bool(x.get('startup_failed')) for x in batch),
            measured_5s=sum(x['gallop_metric_seconds']>=5 for x in batch),
            full_successes=sum(bool(x['success']) for x in batch),leap_events=int(event_count),
            mean_leap_m=sum(x['leap_events']*x['leap_distance_mean'] for x in batch)/event_count if event_count else None,
            max_leap_m=max(x['leap_distance_max'] for x in batch)))
    result=dict(train=protocol['train'],evaluation=str(run.relative_to(ROOT)),policy_updates=read(train/'status.json')['policy_updates'],
        summaries=summaries,directional_decisions=decisions,training_windows=windows,
        limitations=['Nominal plane and eight small paired initial perturbations; a pilot, not staged graduation or broad domain validation.',
            'Common updated controller and 18cm goal used for all three policies; this isolates checkpoint change under the pilot task, not individual startup/goal effects.',
            'Training aggregates include noisy randomized episodes and are separate from frozen nominal evaluation.',
            'No 80cm training or feasibility claim; no new Sim2Sim or hardware test.'])
    import torch
    trained=read(train/'manifest.json')
    parent_path=ROOT/protocol['parent']
    parent_state=torch.load(parent_path,map_location='cpu',weights_only=True)['model_state_dict']
    checks=dict(training_completed=read(train/'status.json')['status']=='completed',updates_500=result['policy_updates']==500,
        eval_frozen=read(run/'status.json')['policies_unchanged'],cases_complete=len(cases)==manifest['pairs']*len(manifest['labels']),
        parent_unchanged=hashlib.sha256(parent_path.read_bytes()).hexdigest()==protocol['parent_sha256'])
    for filename,updates in [('model_249.pt',250),('model_final.pt',500)]:
        d=torch.load(train/filename,map_location='cpu',weights_only=True)
        checks[filename+'_updates']=d['iter']+1==updates
        checks[filename+'_finite']=all(torch.isfinite(v).all().item() for v in d['model_state_dict'].values())
        checks[filename+'_actor_changed']=any(not torch.equal(v,parent_state[k]) for k,v in d['model_state_dict'].items() if k.startswith('actor.'))
        checks[filename+'_optimizer_populated']=bool(d['optimizer_state_dict']['state'])
    for name,key in [('gallop_core.py','gallop_core_sha256'),('gallop_env.py','gallop_env_sha256')]:
        checks[name+'_saved_source']=hashlib.sha256((train/'source'/name).read_bytes()).hexdigest()==trained[key]
        checks[name+'_evaluated_source']=manifest['runtime_'+('core' if 'core' in name else 'env')+'_sha256']==trained[key]
    z=np.load(run/'trace.npz');contract=trained['guarded_runtime_joint_contract']
    for i,label in enumerate(manifest['labels']):
        selected=np.arange(i,z['valid'].shape[1],len(manifest['labels']))
        valid=z['valid'][:,selected]
        result['summaries'][label]['actual_guard_excess_max_rad']=float(z['guarded_actual_excess'][:,selected][valid].max())
    limits=np.array(contract['target_limits']);rate=np.array([6. if '_hip_' in n else 16. for n in contract['joint_names']])
    checks['all_executed_targets_within_limits']=bool(((z['guarded_target']>=limits[:,0]-1e-6)&(z['guarded_target']<=limits[:,1]+1e-6)).all())
    checks['all_executed_target_rates_within_limits']=bool((np.abs(z['guarded_target_rate'])<=rate+1e-4).all())
    assert all(checks.values()),checks
    write(out/'verification.json',checks)
    write(out/'results.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=list(summaries)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,(key,title,scale) in zip(axes.flat,[('leap_distance_mean','Event mean flight distance (cm)',100),
        ('mean_vx','Mean forward speed (m/s)',1),('front_width_fraction','Front support width in band (%)',100),
        ('landing_fraction','Stable landing fraction (%)',100),('raw_clip_fraction','Raw requested-angle clip (%)',100),
        ('yaw_rms_deg','Heading RMS (degrees)',1)]):
        values=[summaries[label][key]*scale if summaries[label][key] is not None else np.nan for label in labels]
        ax.bar(labels,values,color=['#666666','#d88c28','#2878ac'][:len(labels)])
        for i,value in enumerate(values):
            if np.isfinite(value):ax.text(i,value,f'{value:.2f}',ha='center',va='bottom')
            else:ax.text(i,.05,'N/A',ha='center',transform=ax.get_xaxis_transform())
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_ylim(bottom=0)
    fig.savefig(out/'pilot_comparison.png',dpi=150);plt.close(fig)
    files=[run/'manifest.json',run/'status.json',run/'cases.json',run/'episodes.jsonl',run/'trace.npz',train/'manifest.json',train/'status.json',train/'episodes.jsonl',out/'pilot_protocol.json']
    files+=list(train.glob('model*.pt'))
    write(out/'input_hashes.json',{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
