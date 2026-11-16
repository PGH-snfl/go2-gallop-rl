"""Saved-state gate and joint-command diagnostics; no inference or simulation."""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from gpd.paths import ROOT
from scripts.analysis.collect_gallop import read,read_rows,CRITERIA,STABILITY_CRITERIA,BOUNDED_CRITERIA
from scripts.analysis.collect_unitree_baseline import load_yaml


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def joint_targets(train,play,penalty_settings):
    cfg=load_yaml(train/'params/env.yaml');asset=read(ROOT/'assets/go2_isaac_contract.json')
    manifest=read(train/'manifest.json')
    if asset['isaac_asset_sha256']!=manifest['asset_sha256']:raise ValueError('Asset contract differs from training')
    names=asset['native_joint_order'];limits={j['name']:np.deg2rad([j['lower'],j['upper']]) for j in asset['joints']}
    hard=np.array([limits[n] for n in names]);mid=hard.mean(1)
    if manifest['gallop_settings'].get('guarded_enabled',False):
        radius=np.ptp(hard,axis=1)*.5*manifest['gallop_settings']['guarded_sim_limit_factor']
        hard=np.stack((mid-radius,mid+radius),axis=-1)
    radius=np.ptp(hard,axis=1)*cfg['scene']['robot']['soft_joint_pos_limit_factor']*.5;lo=mid-radius;hi=mid+radius
    default=np.array([next(value for pattern,value in cfg['scene']['robot']['init_state']['joint_pos'].items() if re.fullmatch(pattern,name)) for name in names])
    with np.load(play/'gallop_trace.npz') as z:
        mask=z['valid'][:,0]&z['gallop_metric_active'][:,0]
        raw=z['guarded_requested'][mask,0] if 'guarded_requested' in z else default+manifest['action_scale']*z['executed_action'][mask,0]
        excess=np.maximum(lo-raw,0)+np.maximum(raw-hi,0);clipped=excess>1e-6
        np.testing.assert_allclose(clipped.mean(1),z['action_clip'][mask,0],atol=1e-7)
        if 'stabilization_target_excess' in z:np.testing.assert_allclose(excess,z['stabilization_target_excess'][mask,0],atol=1e-6)
    e=excess/penalty_settings['stabilization_excess_scale'];huber=np.where(e<1,.5*e**2,e-.5)
    weights=np.array([2. if '_calf_' in name else 1. for name in names])
    rows=[dict(joint=name,clip_fraction=float(clipped[:,i].mean()),
        lower_clip_fraction=float((raw[:,i]<lo[i]-1e-6).mean()),upper_clip_fraction=float((raw[:,i]>hi[i]+1e-6).mean()),
        mean_excess_rad=float(excess[:,i].mean()),max_excess_rad=float(excess[:,i].max())) for i,name in enumerate(names)]
    return dict(samples=int(mask.sum()),overall_clip_fraction=float(clipped.mean()),mean_excess_rad=float(excess.mean()),
        comparable_excess_cost=float(np.mean(huber*weights)),per_joint=rows,
        note='Command saturation, not physical joint-limit violation. Excess cost uses the same current formula for both policies.')


def cohort(rows,cfg):
    measured=[r for r in rows if r['stabilization_measured']]
    result=dict(episodes=len(rows),sufficient_gallop_measurements=len(measured),successful_episodes=sum(bool(r['success']) for r in rows),
        criterion_rates={k:float(np.mean([r[k] for r in rows])) for k in CRITERIA+STABILITY_CRITERIA+
            tuple(k for k in BOUNDED_CRITERIA if all(k in r for r in rows))} if rows else {})
    if rows and 'bounded_limits' in rows[0]:result['bounded_limit_failures']=sum(not r['bounded_limits'] for r in rows)
    for name in ('front','hind'):
        fractions=np.array([r['stabilization_'+name+'_width_fraction'] for r in measured])
        passed=sum(r['stabilization_'+name+'_support_samples']>=cfg['stabilization_min_support_samples'] and
            r['stabilization_'+name+'_width_fraction']>=cfg['stabilization_width_fraction'] for r in rows)
        result[name+'_width']=dict(passed_episodes=passed,measured_fraction_q10_q50_q90_max=np.quantile(fractions,[.1,.5,.9,1]).tolist() if len(fractions) else None)
    result['clip_only_stable_failures']=sum(r['action_clip_fraction']>cfg['max_action_clipping'] and not r['terminated'] and
        r['attitude_rms']<=cfg['max_attitude_rms'] and r['age']>=1499 for r in rows)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--parent-report',type=Path,required=True);args=parser.parse_args()
    out=args.report.resolve();summary=read(out/'summary.json');prior=read(args.parent_report/'summary.json')
    cfg=summary['provenance']['gallop_settings'];train=Path(summary['train']);play=Path(summary['play'])
    rows=[r for r in read_rows(train/'episodes.jsonl') if r['eligible']]
    rollout=load_yaml(train/'params/agent.yaml')['num_steps_per_env'];last=summary['iterations']*rollout
    groups=dict(all_eligible=rows,first_200=[r for r in rows if r['step']<=200*rollout],
        last_200=[r for r in rows if r['step']>=last-200*rollout],last_1000=[r for r in rows if r['step']>=last-1000*rollout])
    current=joint_targets(train,play,cfg);parent=joint_targets(Path(prior['train']),Path(prior['play']),cfg)
    write(out/'joint_target_diagnostics.json',dict(current=current,parent=parent,
        note='Different commanded speeds and episodes; this comparison does not isolate any single reward or calf scaling effect.'))
    episode=read_rows(play/'episodes.jsonl')[0]
    flights=list(csv.DictReader((out/'play_flight_events.csv').open()))
    final=[r for r in flights if float(r['distance_m'])>=cfg['leap_min_distance']-1e-5 and float(r['air_seconds'])>=cfg['flight_min_seconds']-1e-6]
    gates={k:bool(episode[k]) for k in CRITERIA+STABILITY_CRITERIA+tuple(k for k in BOUNDED_CRITERIA if k in episode)}
    result=dict(training={name:cohort(group,cfg) for name,group in groups.items()},
        play_gates=gates,play_failed_gates=[k for k,v in gates.items() if not v],
        bounded_play={k:v for k,v in episode.items() if k.startswith('bounded_')},
        actual_target_distance_m=episode['stabilization_leap_target'],configured_final_distance_m=cfg['leap_min_distance'],
        final_distance_reference=dict(events=len(final),stable_events=sum(int(r['stable_landing']) for r in final),
            note='Reference comparison only: the active qualification target was 14cm, not 18cm.'),
        curriculum=summary['curriculum'],highest_confirmed_speed=summary['provenance'].get('highest_confirmed_speed'),
        note='Training cohorts are stochastic, while the saved play uses deterministic actions. Raw criterion rates and sufficiently measured episodes are reported separately.')
    write(out/'stabilization_diagnostics.json',result)
    print(json.dumps(dict(failed_gates=result['play_failed_gates'],clip=current['overall_clip_fraction'],
        comparable_excess_cost=current['comparable_excess_cost'],parent_excess_cost=parent['comparable_excess_cost'],
        training_front_width_pass=result['training']['all_eligible']['front_width']['passed_episodes'])))


if __name__=='__main__':main()
