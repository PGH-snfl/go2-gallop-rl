"""Summarize saved heading-Gallop evidence; no simulation or learning."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.analysis.collect_gallop import read,read_rows,rollout_metric_mask,training_completion


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def archive(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def rates(z):
    dt=np.diff(z['time'])
    difference=np.diff(z['rpy'][:,0,:2],axis=0)
    change=np.arctan2(np.sin(difference),np.cos(difference))/dt[:,None]
    return np.vstack((np.zeros((1,2)),change))


def segment(z,mask,settings):
    if not mask.any():return None
    rpy=z['rpy'][mask,0];velocity=z['body_velocity'][mask,0];angular=z['angular_velocity'][mask,0]
    eligible=mask.copy();eligible[0]=False
    if 'posture_rate_valid' in z:eligible&=z['posture_rate_valid'][:,0]
    rate=z['posture_roll_pitch_rate'][eligible,0] if 'posture_roll_pitch_rate' in z else rates(z)[eligible]
    error=np.arctan2(np.sin(rpy[:,2]-settings.get('target_yaw',0.)),np.cos(rpy[:,2]-settings.get('target_yaw',0.)))
    initial=z['rpy'][0,0,2]
    drift=np.arctan2(np.sin(rpy[:,2]-initial),np.cos(rpy[:,2]-initial))
    result=dict(samples=int(mask.sum()),seconds=float(mask.sum()*np.median(np.diff(z['time']))),
        first_seconds=float(z['time'][mask][0]),last_seconds=float(z['time'][mask][-1]),
        body_velocity_mean=velocity.mean(0,dtype=float).tolist(),
        body_velocity_rmse=np.sqrt(np.mean((velocity[:,:2]-z['command'][mask,0,:2]).astype(float)**2,axis=0)).tolist(),
        body_z_angular_speed_rms=float(np.sqrt(np.mean(angular[:,2].astype(float)**2))),
        roll_pitch_mean_rad=rpy[:,:2].mean(0,dtype=float).tolist(),
        roll_pitch_std_rad=rpy[:,:2].std(0,dtype=float).tolist(),
        roll_pitch_rms_rad=np.sqrt(np.mean(rpy[:,:2].astype(float)**2,axis=0)).tolist(),
        roll_pitch_rate_rms=np.sqrt(np.mean(rate.astype(float)**2,axis=0)).tolist() if len(rate) else None,
        yaw_target_rmse_rad=float(np.sqrt(np.mean(error.astype(float)**2))),
        yaw_drift_from_first_sample_rmse_rad=float(np.sqrt(np.mean(drift.astype(float)**2))),
        initial_yaw_rad=float(initial),action_clip_fraction=float(z['action_clip'][mask,0].mean()),
        root_displacement_xyz=(z['position'][mask,0][-1]-z['position'][mask,0][0]).tolist())
    if 'posture_width' not in z:return result
    offset=z['position'][mask,0,:2]-z['position'][0,0,:2]
    target=settings.get('target_yaw',0.)
    cross_track=-np.sin(target)*offset[:,0]+np.cos(target)*offset[:,1]
    result.update(quiet_score=float(z['posture_quiet_score'][eligible,0].mean()) if eligible.any() else None,
        pure_rate_quiet_score=float(np.exp(-np.mean((rate/settings.get('roll_pitch_rate_scale',1.))**2,axis=1)).mean()) if len(rate) else None,
        forward_score=float(z['posture_forward_score'][mask,0].mean()),
        along_speed_mean=float(z['posture_along_speed'][mask,0].mean()),
        world_cross_speed_rms=float(np.sqrt(np.mean(z['posture_cross_speed'][mask,0].astype(float)**2))),
        movement_direction_rms_rad=float(np.sqrt(np.mean(z['posture_direction_error'][mask,0].astype(float)**2))),
        cross_track_max_abs_from_start_m=float(np.abs(cross_track).max()),
        cross_track_first_last_from_start_m=[float(cross_track[0]),float(cross_track[-1])])
    for key in ('pose_quality','yaw_damping_cost'):
        if 'posture_'+key in z:result[key+'_mean']=float(z['posture_'+key][mask,0].mean())
    pair_rows={}
    for i,name in enumerate(('front','hind')):
        width=z['posture_width'][mask,0,i];count=z['contact'][mask,0,2*i:2*i+2].sum(1)
        target=settings[name+'_stance_width']
        lower=settings.get('stabilization_width_min',target-.03);upper=settings.get('stabilization_width_max',target+.03)
        row=dict(target_m=target,mean_m=float(width.mean()),std_m=float(width.std()),
            q0_q10_q50_q90_q100_m=np.quantile(width,[0,.1,.5,.9,1]).tolist(),
            rmse_m=float(np.sqrt(np.mean((width-target)**2))),
            fraction_within_1cm=float((np.abs(width-target)<=.01).mean()),
            score_mean=float(z['posture_'+name+'_width_score'][mask,0].mean()),by_contacts={})
        row['success_width_band_m']=[lower,upper]
        for n,label in enumerate(('both_air','one_support','both_support')):
            selected=width[count==n]
            if len(selected):row['by_contacts'][label]=dict(samples=len(selected),mean_m=float(selected.mean()),
                q10_q50_q90_m=np.quantile(selected,[.1,.5,.9]).tolist(),
                fraction_within_1cm=float((np.abs(selected-target)<=.01).mean()))
            if len(selected):row['by_contacts'][label]['fraction_in_success_width_band']=float(((selected>=lower)&(selected<=upper)).mean())
        pair_rows[name]=row
    result['foot_widths']=pair_rows
    return result


def training_group(rows):
    if not rows:return dict(episodes=0)
    weights=np.array([r['posture_seconds'] for r in rows]);rate_weights=np.array([r['posture_rate_samples'] for r in rows])
    def mean(key,w=weights):return float(np.average([r[key] for r in rows],weights=w)) if w.sum() else None
    def rms(key,w=weights):return float(np.sqrt(np.average([r[key]**2 for r in rows],weights=w))) if w.sum() else None
    formal=[r for r in rows if r.get('gallop_metric_seconds',0)>0]
    events=sum(r['leap_events'] for r in rows)
    settled=sum(r['flight_stable_landings']+r['flight_aborted_landings'] for r in rows)
    return dict(episodes=len(rows),startup_completed_fraction=float(np.mean([r['startup_completed'] for r in rows])),
        episodes_with_gallop_metrics=len(formal),reported_success_rate=float(np.mean([r['success'] for r in rows])),
        reported_tracking_pass_fraction=float(np.mean([r['tracking'] for r in formal])) if formal else None,
        natural_timeout_fraction=float(np.mean([r['timed_out'] for r in rows])),
        all_stage_weighted_yaw_rmse_rad=rms('posture_yaw_rmse'),
        all_stage_weighted_roll_pitch_rate_rms=[rms('posture_roll_rate_rms',rate_weights),rms('posture_pitch_rate_rms',rate_weights)],
        all_stage_weighted_quiet_score=mean('posture_quiet_score',rate_weights),
        all_stage_weighted_front_hind_width_m=[mean('posture_front_width'),mean('posture_hind_width')],
        all_stage_weighted_front_hind_width_scores=[mean('posture_front_width_score'),mean('posture_hind_width_score')],
        all_stage_weighted_forward_score=mean('posture_forward_score'),
        measured_flight_events=events,qualified_events=sum(r['leap_qualified_events'] for r in rows),
        event_weighted_distance_m=sum(r['leap_distance_mean']*r['leap_events'] for r in rows)/events if events else None,
        event_weighted_air_seconds=sum(r['leap_air_seconds_mean']*r['leap_events'] for r in rows)/events if events else None,
        stable_landing_fraction_of_finished_recoveries=sum(r['flight_stable_landings'] for r in rows)/settled if settled else None,
        note='Training aggregates are stochastic, not independent evaluation; posture uses all stages and event means are event-weighted.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--parent-play',type=Path)
    args=p.parse_args();out=args.report.resolve();summary=read(out/'summary.json');settings=summary['provenance']['gallop_settings']
    if not settings.get('heading_motion_enabled'):raise ValueError('Expected heading-motion data')
    z=archive(Path(summary['play'])/'gallop_trace.npz')
    valid=z['valid'][:,0].astype(bool);formal=rollout_metric_mask(z,settings,0)
    groups=dict(all_first_episode=segment(z,valid,settings),formal_gallop=segment(z,formal,settings),stages=[])
    for stage in range(4):
        mask=valid&(z['startup_stage'][:,0]==stage)
        if mask.any():groups['stages'].append(dict(stage=stage,**segment(z,mask,settings)))
    if settings.get('relearn_enabled'):
        groups['metric_context']='formal_gallop is the historical key for the full-student control window; it includes walking when relearn_gait is zero.'
        groups['saved_relearn_gaits']=np.unique(z['relearn_gait'][formal,0]).tolist()
    write(out/'heading_play_diagnostics.json',groups)
    comparison=dict(comparison_available=False,note='No matched parent replay supplied; current physical measurements are reported without a parent comparison.')
    if args.parent_play:
        old=archive(args.parent_play/'gallop_trace.npz');old_manifest=read(args.parent_play/'manifest.json')
        parent_mask=rollout_metric_mask(old,old_manifest['gallop_settings'],0)
        prior=segment(old,parent_mask,old_manifest['gallop_settings'])
        current=groups['formal_gallop']
        a=np.array(prior['roll_pitch_rate_rms']) if prior else None;b=np.array(current['roll_pitch_rate_rms']) if current else None
        common_heading=bool(settings.get('heading_motion_enabled') and old_manifest['gallop_settings'].get('heading_motion_enabled') and
            np.isclose(settings.get('target_yaw',0.),old_manifest['gallop_settings'].get('target_yaw',0.)))
        comparison=dict(comparison_available=True,parent_play=str(args.parent_play.resolve()),parent_formal=prior,current_formal=current,
            roll_pitch_rate_reduction_fraction=(1-b/a).tolist() if a is not None and b is not None else None,
            absolute_yaw_directly_comparable=common_heading,
            caveats=['Both use the same heading reference.' if common_heading else 'Heading references/reset distributions differ; absolute yaw errors are not a matched-start comparison.',
                     'Different checkpoints, commanded speeds, reward definitions and random states; these observations do not identify the effect of a single reward.',
                     'Formal control windows differ in duration because handover times differ.',
                     'Quiet and width reward formulas/weights can differ; compare physical pose, rates and widths separately.'])
        if 'posture_width' not in old:comparison['caveats'].append('Parent width was not logged; no inferred width is substituted.')
    write(out/'heading_parent_comparison.json',comparison)
    rows=read_rows(Path(summary['train'])/'episodes.jsonl');eligible=[r for r in rows if r['eligible']]
    # The exact rollout length is available in the saved configuration.
    from scripts.analysis.collect_unitree_baseline import load_yaml
    rollout=load_yaml(Path(summary['train'])/'params/agent.yaml')['num_steps_per_env'];total_steps=summary['iterations']*rollout
    cohorts=dict(all_eligible=eligible,first_200=[r for r in eligible if r['step']<=200*rollout],
        last_200=[r for r in eligible if r['step']>=total_steps-200*rollout],
        last_1000=[r for r in eligible if r['step']>=total_steps-1000*rollout])
    write(out/'heading_training_diagnostics.json',{name:training_group(group) for name,group in cohorts.items()})
    scalars={}
    with (out/'tensorboard_scalars.csv').open() as stream:
        for row in csv.DictReader(stream):scalars.setdefault(row['tag'],[]).append(row)
    completion=training_completion(summary['training_status'],summary['iterations'],[int(r['iteration_zero_based']) for r in scalars['Loss/value_function']])
    summary.update(training_complete=completion['complete'],completion_evidence=completion)
    summary['reward_scalar_coverage']=dict(count=len(scalars['Train/mean_reward']),first_iteration_zero_based=int(scalars['Train/mean_reward'][0]['iteration_zero_based']),
        last_iteration_zero_based=int(scalars['Train/mean_reward'][-1]['iteration_zero_based']),
        note='Completed-episode means may be absent before the first finished episode; absent early reward entries do not imply missing PPO updates.')
    write(out/'summary.json',summary)
    keys=[key for key in z if key.startswith(('posture_','startup_','stabilization_','bounded_','guarded_'))]
    with (out/'heading_play_trace.csv').open('w') as stream:
        records=[]
        for i,t in enumerate(z['time']):
            row=dict(time=float(t),first_episode_valid=int(valid[i]),formal_gallop=int(formal[i]))
            for key in keys:
                value=np.asarray(z[key][i,0])
                if value.ndim==0:row[key]=value.item()
                else:row.update({f'{key}_{j}':v.item() for j,v in enumerate(value)})
            records.append(row)
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    render(out,z,formal,settings,scalars)
    render_gallop_preview(out,Path(summary['play'])/'videos/gallop-play-step-0.mp4',z,formal)
    print(json.dumps(dict(report=str(out),training_complete=summary['training_complete'],formal=groups['formal_gallop']),ensure_ascii=False))


def render(out,z,formal,settings,scalars):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=z['time'];fig,axes=plt.subplots(5,1,figsize=(13,12),sharex=True,layout='constrained')
    axes[0].plot(t,np.rad2deg(z['posture_yaw_error'][:,0]));axes[0].axhline(0,color='k',ls='--');axes[0].set_ylabel('Yaw error (deg)')
    for i,name in enumerate(('roll','pitch')):
        axes[1].plot(t,np.rad2deg(z['rpy'][:,0,i]),label=name)
        axes[2].plot(t,z['posture_roll_pitch_rate'][:,0,i],label=name)
    axes[1].set_ylabel('Pose (deg)');axes[2].set_ylabel('Angle rate (rad/s)')
    for i,name in enumerate(('front','hind')):
        axes[3].plot(t,z['posture_width'][:,0,i]*100,label=name)
        axes[3].axhline(settings[name+'_stance_width']*100,color='k',ls='--')
    axes[3].set_ylabel('Foot width (cm)')
    axes[4].plot(t,z['posture_along_speed'][:,0],label='actual world forward');axes[4].plot(t,z['command'][:,0,0],'k--',label='command')
    axes[4].plot(t,z['posture_forward_score'][:,0],label='forward score');axes[4].set_ylabel('m/s or score');axes[4].set_xlabel('Time (s)')
    for ax in axes:
        if formal.any():ax.axvspan(t[formal][0],t[formal][-1],color='tab:green',alpha=.07)
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:ax.legend(loc='upper right')
    fig.savefig(out/'heading_motion_play.png',dpi=150);plt.close(fig)
    tags=['yaw_heading','roll_pitch_quiet','front_width','hind_width','forward_progress']
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    for ax,key in zip(axes.flat,tags):
        rows=scalars['Episode_Reward/gallop_'+key]
        ax.plot([int(r['iteration_zero_based'])+1 for r in rows],[float(r['value']) for r in rows],lw=.7)
        ax.set_title('Episode_Reward/gallop_'+key);ax.set_xlabel('Stage iteration');ax.grid(alpha=.2)
    axes.flat[-1].axis('off');fig.savefig(out/'heading_reward_curves.png',dpi=150);plt.close(fig)


def render_gallop_preview(out,path,z,formal):
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    wraps=np.flatnonzero(np.diff(z['phase'][:,0])<0)+1
    candidates=[(a,b) for a,b in zip(wraps[:-1],wraps[1:]) if z['time'][a]>=12 and formal[a:b+1].all()]
    if not candidates or not path.is_file():return
    first,last=candidates[0];times=np.linspace(z['time'][first],z['time'][last-1],6)
    with imageio.get_reader(path) as reader:
        fps=reader.get_meta_data()['fps'];fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
        for ax,t in zip(axes.flat,times):
            frame=round(t*fps);ax.imshow(reader.get_data(frame));ax.set_title(f'Video {frame/fps:.2f}s');ax.axis('off')
        walking='relearn_gait' in z and np.all(z['relearn_gait'][formal,0]==0)
        fig.suptitle('Student walking after handover (video timestamps)' if walking else 'Gallop after completed handover (video timestamps)')
        fig.savefig(out/'gallop_keyframes.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
