"""Collect Gallop and heading-Gallop evidence; no simulation/learning."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from gpd.paths import ROOT
from scripts.analysis.collect_unitree_baseline import load_yaml

CRITERIA=('tracking','pattern','flight','motion','front_load','front_motion','front_cycle','stable','mature')
STABILITY_CRITERIA=('stabilization_pose','stabilization_width','stabilization_landing','stabilization_distance','stabilization_measured')
BOUNDED_CRITERIA=('bounded_limits','guarded_limits','guarded_feet')
MEASURES=('vx_rmse','contact_accuracy','order_accuracy','flight_fraction','rom_fraction','fore_load_share',
          'fore_min_load_share','fore_rom_fraction','fore_min_rom_fraction','fore_participation','attitude_rms')
LEGS=('FL','FR','RL','RR')
EXTRA_MEASURES=('action_clip_fraction','terminated','timed_out','pair_joint_cost','pair_rom_cost','pair_load_cost','pair_stance_cost','pair_hip_cost','pair_valid_cycles',
                'leap_events','leap_qualified_events','leap_qualified_fraction','leap_distance_mean','leap_distance_max','leap_air_seconds_mean','leap_reward_points','leap_reward_steps','leap_physics_samples',
                'flight_takeoffs','flight_landings','flight_stable_landings','flight_aborted_landings','flight_takeoff_vz_mean',
                'flight_landing_peak_force_bw','flight_landing_peak_down_vz','flight_balance_cost','flight_support_cost',
                'bounded_near_limit_fraction','bounded_near_limit_cost','bounded_target_rate_cost','bounded_tracking_rmse','bounded_torque_saturation',
                'guarded_intervention','guarded_actual','guarded_front_high','guarded_hind_high','guarded_front_cost','guarded_hind_cost','guarded_tracking_rmse')


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def training_completion(status,checkpoint_iterations,loss_steps):
    """Reward scalars can be absent until the first episode has finished."""
    planned=int(status['planned_iterations'])
    checks=dict(status_completed=status['status']=='completed',checkpoint_count=checkpoint_iterations==planned,
        status_update_count=status.get('policy_updates')==planned,
        loss_iterations_complete=list(loss_steps)==list(range(planned)))
    return dict(complete=all(checks.values()),checks=checks,
                note='Mean reward/episode length may omit early iterations with no completed episodes; loss steps and checkpoint counters establish completion.')


def episode_stats(rows):
    measures=MEASURES+tuple(key for key in EXTRA_MEASURES if rows and all(key in row for row in rows))
    criteria=CRITERIA+tuple(key for key in STABILITY_CRITERIA+BOUNDED_CRITERIA if rows and all(key in row for row in rows))
    return dict(count=len(rows),success_rate=float(np.mean([r['success'] for r in rows])) if rows else None,
        criterion_pass_rates={key:float(np.mean([r[key] for r in rows])) for key in criteria} if rows else {},
        measures={key:dict(mean=float(np.mean([r[key] for r in rows])),
                          q10_q50_q90=np.quantile([r[key] for r in rows],[.1,.5,.9]).tolist()) for key in measures} if rows else {},
        failure_combinations=dict(Counter(','.join(k for k in criteria if not r[k]) for r in rows if not r['success'])))


def longest_run(mask):
    best=run=0
    for value in mask:
        run=run+1 if value else 0;best=max(best,run)
    return best


def rollout_metric_mask(d,settings,i):
    valid=d['valid'][:,i].astype(bool)
    if settings.get('startup_enabled',False):
        if 'gallop_metric_active' not in d:raise ValueError('Startup rollout is missing its Gallop metric mask')
        return valid&d['gallop_metric_active'][:,i].astype(bool)
    return valid&(d['time']>=settings['metric_warmup_seconds'])


def analyze_rollout(d,settings,reported):
    time=d['time'];dt=float(np.median(np.diff(time)));results=[]
    for i in range(d['valid'].shape[1]):
        valid=d['valid'][:,i].astype(bool);mask=rollout_metric_mask(d,settings,i)
        if not mask.any():
            results.append(dict(env=i,evaluated_samples=0,metric_segment_available=False,
                note='No full-Gallop measurement segment; inspect startup stage and resets.',
                reported_episode=next((r for r in reported if int(r['env'])==i),None)))
            continue
        resets=np.flatnonzero(d['episode_reset'][:,i]);contacts=d['contact'][mask,i]
        whole=d['foot_force'][mask if settings.get('startup_enabled',False) else valid,i].clip(0,None).sum(axis=0)
        mean_v=d['body_velocity'][mask,i].mean(axis=0,dtype=np.float64)
        errors=d['body_velocity'][mask,i,:2]-d['command'][mask,i,:2]
        phase=d['phase'][:,i];wraps=np.flatnonzero(np.diff(phase)<0)+1
        spans=[]
        for start,end in zip(wraps[:-1],wraps[1:]):
            if time[start]>=settings['metric_warmup_seconds'] and valid[start-1:end+1].all():
                if settings.get('startup_enabled',False) and not mask[start-1:end].all():continue
                spans.append(np.ptp(d['joint_pos'][start-1:end,i],axis=0))
        row=dict(env=i,warmup_seconds=settings['metric_warmup_seconds'],evaluated_samples=int(mask.sum()),
            metric_segment_start_seconds=float(time[mask][0]),metric_segment_available=True,
            evaluated_seconds=float(mask.sum()*dt),first_segment_end_seconds=float(time[valid][-1]),
            reset_times_seconds=time[resets].tolist(),mean_command=d['command'][mask,i].mean(axis=0,dtype=np.float64).tolist(),
            mean_body_velocity=mean_v.tolist(),planar_velocity_rmse=np.sqrt(np.mean(errors.astype(float)**2,axis=0)).tolist(),
            yaw_rmse=float(np.sqrt(np.mean(d['angular_velocity'][mask,i,2].astype(float)**2))),
            roll_pitch_rms=np.sqrt(np.mean(d['rpy'][mask,i,:2].astype(float)**2,axis=0)).tolist(),
            height_min_max=[float(d['position'][mask,i,2].min()),float(d['position'][mask,i,2].max())],
            net_planar_displacement=float(np.linalg.norm(d['position'][mask,i,:2][-1]-d['position'][mask,i,:2][0])),
            all_feet_contact_fraction=float(contacts.all(axis=1).mean()),airborne_time_fraction=float((~contacts).all(axis=1).mean()),
            longest_steady_air_seconds=longest_run((~contacts).all(axis=1))*dt,
            longest_full_first_segment_air_seconds=longest_run((~d['contact'][valid,i]).all(axis=1))*dt,
            foot_contact_duty=contacts.mean(axis=0).tolist(),vertical_impulse_share=whole/whole.sum(),
            mean_action_clip=float(d['action_clip'][mask,i].mean()),
            action_clip_quantiles=np.quantile(d['action_clip'][mask,i],[0,.5,.9,1]).tolist(),
            complete_phase_cycles_used_for_rom=len(spans),
            mean_cycle_joint_excursion_degrees=np.rad2deg(np.mean(spans,axis=0)).tolist() if spans else None,
            reported_episode=next((r for r in reported if int(r['env'])==i),None))
        row['vertical_impulse_share']=row['vertical_impulse_share'].tolist()
        results.append(row)
    return dict(duration_seconds=float(time[-1]),dt=dt,environments=results,
        note='Only first contiguous segment is analyzed; reset rows are pre-reset snapshots. Steady metrics exclude warmup. Runtime episode velocity sums precede interval events, whereas saved snapshots follow them.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True);p.add_argument('--play',type=Path,help='Optional matching saved play; omit to collect training only')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    train,play,out=a.train.resolve(),a.play.resolve() if a.play else None,a.output.resolve()
    folders=[train]+([play] if play else [])
    if any(out==x or x in out.parents for x in folders):raise ValueError('Report must be outside original runs')
    ts=read(train/'status.json');m=read(train/'manifest.json')
    ps=read(play/'status.json') if play else None;pm=read(play/'manifest.json') if play else None
    if ts['status']!='completed' or (ps and ps['status']!='completed'):raise ValueError('Wait for completed train/play runs')
    if m['interface'] not in ('unitree_go2_gallop_47_62_12_v1','unitree_go2_gallop_heading_49_64_12_v2','unitree_go2_gallop_bounded_49_64_12_v3','unitree_go2_gallop_guarded_49_64_12_v4') or (pm and pm['interface']!=m['interface']):raise ValueError('Wrong policy family')
    checkpoint=train/'model_final.pt';before=sha(checkpoint)
    if pm and pm['source_checkpoint']['sha256']!=before:raise ValueError('Play checkpoint differs from current training checkpoint')
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    iterations=int(weights['iter'])+1;agent=load_yaml(train/'params/agent.yaml');env=load_yaml(train/'params/env.yaml')
    rollout=int(agent['num_steps_per_env']);control_steps=iterations*rollout
    tb=EventAccumulator(str(train),size_guidance={'scalars':0});tb.Reload()
    scalars={tag:tb.Scalars(tag) for tag in tb.Tags()['scalars']}
    scalar_stats={tag:dict(count=len(rows),first=rows[0].value,last=rows[-1].value,
                         last_100_mean=float(np.mean([r.value for r in rows[-100:]]))) for tag,rows in scalars.items() if rows}
    episodes=read_rows(train/'episodes.jsonl');pe=read_rows(play/'episodes.jsonl') if play else [];c=read(train/'curriculum_summary.json')
    eligible=[r for r in episodes if r['eligible']]
    recent=[r for r in eligible if r['step']>=control_steps-1000*rollout]
    d=report=None
    if play:
        with np.load(play/'gallop_trace.npz') as archive:d={key:archive[key] for key in archive.files}
        if not all(np.isfinite(v).all() for v in d.values()):raise ValueError('Nonfinite trace')
        command=pm.get('evaluation_command')
        if command is not None:np.testing.assert_allclose(d['command'],np.broadcast_to(command,d['command'].shape),atol=1e-6,rtol=0)
        report=analyze_rollout(d,m['gallop_settings'],pe)
    out.mkdir(parents=True,exist_ok=False);shutil.copytree(train/'params',out/'params')
    shutil.copy2(train/'gallop_config.json',out/'gallop_config.json')
    for name,rows in [('train_episodes.csv',episodes),('play_episodes.csv',pe),('curriculum_decisions.csv',c['events'])]:
        if not rows:continue
        with (out/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'tensorboard_scalars.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['tag','iteration_zero_based','wall_time','value'])
        for tag,rows in scalars.items():writer.writerows([tag,r.step,r.wall_time,r.value] for r in rows)
    if d is not None:
        with (out/'play_state_trace.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['time','env','first_segment_valid','episode_reset','vx_cmd','vx','vy','yaw_rate','height','roll','pitch','action_clip','FL_force','FR_force','RL_force','RR_force'])
            for j,t in enumerate(d['time']):
                for i in range(d['valid'].shape[1]):writer.writerow([t,i,int(d['valid'][j,i]),int(d['episode_reset'][j,i]),d['command'][j,i,0],*d['body_velocity'][j,i,:2],d['angular_velocity'][j,i,2],d['position'][j,i,2],*d['rpy'][j,i,:2],d['action_clip'][j,i],*d['foot_force'][j,i]])
    import imageio.v2 as imageio
    videos=[]
    for folder,kind in [(train,'train')]+([(play,'play')] if play else []):
        paths=sorted((folder/'videos').glob('*.mp4'),key=lambda x:int(x.stem.split('-')[-1]))
        for video in paths:
            with imageio.get_reader(video) as reader:
                meta=reader.get_meta_data();frames=reader.count_frames();reader.get_data(0)
            videos.append(dict(path=str(video),kind=kind,frames=frames,duration_seconds=meta['duration'],fps=meta['fps'],resolution=list(meta['size'])))
    with (out/'videos.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['path','kind','frames','duration_seconds','fps','resolution']);writer.writeheader();writer.writerows(videos)
    summary=dict(collected_at=datetime.now().astimezone().isoformat(),train=str(train),play=str(play) if play else None,checkpoint=str(checkpoint),
        checkpoint_sha256=before,training_status=ts,play_status=ps,provenance=m,iterations=iterations,
        training_complete=training_completion(ts,iterations,[r.step for r in scalars['Loss/value_function']])['complete'],
        inherited_iterations=m.get('fine_tuning',m.get('migration',{})).get('parent_cumulative_iterations',0),
        cumulative_iterations=m.get('fine_tuning',m.get('migration',{})).get('parent_cumulative_iterations',0)+iterations,
        new_transitions=iterations*env['scene']['num_envs']*rollout,
        first_scalar_time=datetime.fromtimestamp(scalars['Train/mean_reward'][0].wall_time).astimezone().isoformat(),
        last_scalar_time=datetime.fromtimestamp(scalars['Train/mean_reward'][-1].wall_time).astimezone().isoformat(),
        policy_dt=env['sim']['dt']*env['decimation'],tensorboard=scalar_stats,
        episodes=dict(total=len(episodes),eligible=len(eligible),eligible_statistics=episode_stats(eligible),
                      recent_window_start_control_step=control_steps-1000*rollout,recent_statistics=episode_stats(recent)),
        curriculum=dict(decisions=len(c['events']),speed_caps=sorted({r['speed_cap'] for r in c['events']}),last_decision=c['events'][-1],
                        promotions=sum(r['reason']=='promote' for r in c['events']),reason_counts=dict(Counter(r['reason'] for r in c['events']))),
        rollout=report,rollout_status='collected' if report else 'not_available',videos=videos,policy_updates_during_collection=0,
        caveats=['Training criterion rates are not independent evaluation success rates.',
                 'stable combines timeout/survival, attitude and target-clipping gates; it does not mean falls alone.',
                 'Runtime flight_fraction counts cycles with sufficient air, not fraction of time airborne.',
                 'Target clipping is target-angle saturation, not actual joint-limit violation.',
                 'Reported joint excursions are per imposed phase cycle, not independently classified gait strides.',
                 'Original physics/noise/pushes remain active in play. No additional simulation or training was performed.'])
    summary['completion_evidence']=training_completion(ts,iterations,[r.step for r in scalars['Loss/value_function']])
    if m['gallop_settings'].get('stabilization_enabled'):
        summary['caveats'].append('Leap target is latched per episode; final leap_min_distance is not the target for every training episode.')
    summary['source_matches_training']={name:sha(train/'source'/name)==sha(ROOT/'isaac_task'/name) for name in ('gallop_core.py','gallop_env.py')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    if report:(out/'play_analysis.json').write_text(json.dumps(report,indent=2))
    render_plots(out,d,scalars,c['events'],rollout,m['gallop_settings'])
    if play and (play/'videos/gallop-play-step-0.mp4').is_file():render_video_preview(out,play/'videos/gallop-play-step-0.mp4')
    evidence={}
    for folder in folders:
        for path in folder.rglob('*'):
            if path.is_file():evidence[str(path.relative_to(ROOT))]=dict(sha256=sha(path),bytes=path.stat().st_size)
    if sha(checkpoint)!=before:raise ValueError('Model changed during collection')
    (out/'evidence_manifest.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(dict(report=str(out),iterations=iterations,transitions=summary['new_transitions'],
                         speed_cap=m['trained_speed_cap'],play=report['environments'] if report else None),indent=2))


def render_plots(out,d,scalars,events,rollout,settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    tags=['Train/mean_reward','Train/mean_episode_length','Gallop/speed_cap','Gallop/success_rate','Episode_Reward/gallop_flight','Gallop/last_rom_fraction']
    for ax,tag in zip(axes.flat,tags):
        rows=scalars[tag];x=np.array([r.step+1 for r in rows]);y=np.array([r.value for r in rows])
        if tag=='Train/mean_episode_length':y*=.02
        ax.plot(x,y,lw=.7,alpha=.7);ax.set_title(tag);ax.grid(alpha=.2);ax.set_xlabel('Stage PPO iteration')
    fig.savefig(out/'training_curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,5),layout='constrained')
    criteria=CRITERIA+tuple(k for k in STABILITY_CRITERIA+BOUNDED_CRITERIA if any(k in e['criterion_rates'] for e in events))
    data=np.array([[e['criterion_rates'].get(k,np.nan) for e in events] for k in criteria])
    im=ax.imshow(data,origin='upper',aspect='auto',vmin=0,vmax=1,cmap='RdYlGn',extent=[events[0]['step']/rollout,events[-1]['step']/rollout,len(criteria)-.5,-.5])
    ax.set_yticks(range(len(criteria)),criteria);ax.set_xlabel('Stage PPO iteration');ax.set_title('Curriculum gates: stable includes target clipping; mature is reward-ramp completion')
    fig.colorbar(im,ax=ax,label='Window pass rate');fig.savefig(out/'criterion_pass_rates.png',dpi=150);plt.close(fig)
    pair_tags=['Episode_Reward/gallop_pair_'+key for key in ('joint','rom','load','stance')]
    if all(tag in scalars for tag in pair_tags):
        fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
        for ax,tag in zip(axes.flat,pair_tags):
            rows=scalars[tag];ax.plot([r.step+1 for r in rows],[r.value for r in rows],lw=.8)
            ax.set_title(tag);ax.set_xlabel('Stage PPO iteration');ax.set_ylabel('Reward contribution');ax.grid(alpha=.2)
        fig.savefig(out/'pair_reward_curves.png',dpi=150);plt.close(fig)
    leap_tags=['Episode_Reward/gallop_leap_distance','Gallop/last_leap_distance_max','Gallop/last_leap_events','Gallop/last_leap_qualified_fraction']
    if all(tag in scalars for tag in leap_tags):
        fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
        for ax,tag in zip(axes.flat,leap_tags):
            rows=scalars[tag];ax.plot([r.step+1 for r in rows],[r.value for r in rows],lw=.8)
            if tag.endswith('distance_max'):ax.axhline(settings['leap_min_distance'],color='r',ls='--')
            ax.set_title(tag);ax.set_xlabel('Stage PPO iteration');ax.grid(alpha=.2)
        fig.savefig(out/'leap_curves.png',dpi=150);plt.close(fig)
    if d is None:return
    t=d['time'];mask=rollout_metric_mask(d,settings,0)
    if not mask.any():return
    fig,axes=plt.subplots(4,1,figsize=(13,9),layout='constrained',sharex=True)
    axes[0].plot(t,d['command'][:,0,0],'k--',label='command');axes[0].plot(t,d['body_velocity'][:,0,0],label='measured vx');axes[0].set_ylabel('m/s');axes[0].legend()
    axes[1].plot(t,np.rad2deg(d['rpy'][:,0,0]),label='roll');axes[1].plot(t,np.rad2deg(d['rpy'][:,0,1]),label='pitch');axes[1].set_ylabel('deg');axes[1].legend()
    axes[2].plot(t,d['action_clip'][:,0]);axes[2].axhline(settings['max_action_clipping'],color='r',ls='--');axes[2].set_ylabel('Target clip fraction')
    axes[3].imshow(d['contact'][:,0].T,origin='lower',aspect='auto',interpolation='nearest',extent=[t[0],t[-1],-.5,3.5],cmap='Blues');axes[3].set_yticks(range(4),LEGS);axes[3].set_ylabel('Actual contact');axes[3].set_xlabel('Time (s)')
    for ax in axes:
        ax.axvspan(0,settings['metric_warmup_seconds'],color='grey',alpha=.2)
        if (~d['valid'][:,0]).any():ax.axvspan(t[np.flatnonzero(~d['valid'][:,0])[0]],t[-1],color='grey',alpha=.2)
        ax.grid(alpha=.2)
    fig.savefig(out/'play_tracking.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(13,7),sharex=True,layout='constrained')
    x=t[mask]
    for ax,key in zip(axes[:2],['contact','target_contact']):
        ax.imshow(d[key][mask,0].T,origin='lower',aspect='auto',interpolation='nearest',extent=[x[0],x[-1],-.5,3.5],cmap='Blues');ax.set_yticks(range(4),LEGS);ax.set_ylabel(key)
    for j,leg in enumerate(LEGS):axes[2].plot(x,d['foot_height'][mask,0,j],lw=.8,label=leg)
    axes[2].axhline(settings['min_fore_swing_height'],color='k',ls='--');axes[2].set_ylabel('Foot clearance (m)');axes[2].legend(ncol=4);axes[2].set_xlabel('Time (s)')
    fig.savefig(out/'play_contacts.png',dpi=150);plt.close(fig)
    view_start=float(t[mask][0])+1 if settings.get('startup_enabled',False) else 2
    view=mask&(t>=view_start)&(t<=view_start+2)
    fig,axes=plt.subplots(4,2,figsize=(13,10),layout='constrained',sharex=True)
    for i,leg in enumerate(LEGS):
        for j,joint in enumerate(['thigh','calf']):
            ax=axes[i,j];ax.plot(t[view],np.rad2deg(d['joint_pos'][view,0,i,j+1]),label='measured')
            ax.plot(t[view],np.rad2deg(d['joint_reference'][view,0,i,j+1]),'k--',label='reference');ax.set_title(leg+' '+joint);ax.set_ylabel('deg');ax.grid(alpha=.2)
    axes[0,0].legend();fig.savefig(out/'joint_motion.png',dpi=150);plt.close(fig)


def render_video_preview(out,video):
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times=np.linspace(2,2.42,6)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    with imageio.get_reader(video) as reader:
        fps=reader.get_meta_data()['fps']
        for ax,t in zip(axes.flat,times):
            frame=round(t*fps)
            ax.imshow(reader.get_data(frame));ax.set_title(f'Video frame {frame}, {frame/fps:.2f} s');ax.axis('off')
    fig.suptitle('Saved policy playback: six frames across 0.42 s')
    fig.savefig(out/'play_keyframes.png',dpi=130);plt.close(fig)


if __name__=='__main__':main()
