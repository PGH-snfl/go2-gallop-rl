"""Diagnose saved startup/handover stalls without new simulation or training."""
import argparse
import json
from pathlib import Path

import numpy as np
from scripts.analysis.collect_unitree_baseline import load_yaml


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);a=parser.parse_args()
    out=a.report.resolve();summary=read(out/'summary.json');cfg=summary['provenance']['gallop_settings']
    with np.load(Path(summary['play'])/'gallop_trace.npz') as archive:z={k:archive[k] for k in archive.files}
    valid=z['valid'][:,0].astype(bool);stage=z['startup_stage'][:,0];mix=z['startup_gallop_mix'][:,0]
    formal=valid&z['gallop_metric_active'][:,0].astype(bool);dt=summary['policy_dt']
    segments={}
    for name,mask in [('all',valid),('blend',valid&(stage==2)),('last10s',valid&(z['time']>=20)),('last2s',valid&(z['time']>=28))]:
        if not mask.any():continue
        v=z['body_velocity'][mask,0];rpy=z['rpy'][mask,0];ang=z['angular_velocity'][mask,0];h=z['position'][mask,0,2];cmd=z['command'][mask,0,0]
        pose=(np.abs(rpy[:,:2])<=cfg['startup_max_tilt']).all(1)&(np.linalg.norm(ang,axis=1)<=cfg['startup_max_angular_speed'])
        pose&=(h>=cfg['min_height'])&(h<=cfg['max_height'])&(np.abs(v[:,1])<=.15)
        track=np.abs(v[:,0]-cmd)<=cfg['startup_tracking_error']
        segments[name]=dict(samples=int(mask.sum()),seconds=float(mask.sum()*dt),mean_vx=float(v[:,0].astype(float).mean()),
            command_min_max=[float(cmd.min()),float(cmd.max())],mean_command=float(cmd.astype(float).mean()),
            mean_absolute_speed_error=float(np.abs(v[:,0]-cmd).mean()),pose_pass_fraction=float(pose.mean()),speed_tracking_pass_fraction=float(track.mean()),
            pose_rms_degrees=np.degrees(np.sqrt(np.mean(rpy.astype(float)**2,axis=0))).tolist(),
            mix_first_last=mix[mask][[0,-1]].tolist(),all_off_contact_fraction=float((~z['contact'][mask,0]).all(1).mean()),
            raw_angle_clip_fraction=float(z['action_clip'][mask,0].mean()),filter_intervention_fraction=float(z['guarded_intervention'][mask,0].mean()),
            actual_joint_max_excess_rad=float(z['guarded_actual_excess'][mask,0].max()))
    train=Path(summary['train']);rows=[json.loads(l) for l in (train/'episodes.jsonl').read_text().splitlines() if l.strip()]
    rollout=load_yaml(train/'params/agent.yaml')['num_steps_per_env'];windows=[]
    for low in range(0,summary['iterations'],500):
        cohort=[r for r in rows if low*rollout<r['step']<=min(low+500,summary['iterations'])*rollout]
        if not cohort:continue
        windows.append(dict(iteration_start=low,iteration_end=min(low+500,summary['iterations']),episodes=len(cohort),
            handover_completions=sum(bool(r['startup_completed']) for r in cohort),measured_episodes=sum(r['gallop_metric_seconds']>=cfg['stabilization_min_metric_seconds'] for r in cohort),
            terminated=sum(bool(r['terminated']) for r in cohort),flight_events=sum(r['leap_events'] for r in cohort),
            max_leap_m=max(r['leap_distance_max'] for r in cohort),stage_counts={str(i):sum(r['startup_stage']==i for r in cohort) for i in range(4)}))
    completed=[r for r in rows if r['startup_completed']]
    result=dict(formal_samples=int(formal.sum()),formal_segment_available=bool(formal.any()),stage_sample_counts={str(i):int((valid&(stage==i)).sum()) for i in range(4)},
        final_stage=int(stage[valid][-1]),final_student_mix=float(mix[valid][-1]),final_teacher_mix=float(1-mix[valid][-1]),
        final_command_mps=float(z['command'][valid,0,0][-1]),segments=segments,training_windows=windows,
        training_episodes=len(rows),eligible_episodes=sum(bool(r['eligible']) for r in rows),handover_completions=len(completed),
        measured_episodes=sum(r['gallop_metric_seconds']>=cfg['stabilization_min_metric_seconds'] for r in rows),
        last_handover_iteration=max((r['step']/rollout for r in completed),default=None),
        full_successes=sum(bool(r['success']) for r in rows),maximum_recorded_training_leap_m=max(r['leap_distance_max'] for r in rows),
        stage_names=['hold','frozen_walk','blend','full_student'],
        caveats=['Saved snapshots follow physics; startup.advance checks before physics, so reconstructed gate fractions are approximate by one control step.',
        'Formal metric fields remain zero when handover never completes. They are not measured full-student performance.',
        'Contact air fraction during blend is separately checked from raw saved contacts.',
        'Several goal, timing and reward changes were combined; reward avoidance is a hypothesis, not a causally isolated result.'])
    (out/'startup_stall_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    axes[0,0].plot(z['time'],z['command'][:,0,0],label='command');axes[0,0].plot(z['time'],z['body_velocity'][:,0,0],label='actual vx',alpha=.8)
    axes[0,0].fill_between(z['time'],z['command'][:,0,0]-.25,z['command'][:,0,0]+.25,alpha=.15);axes[0,0].set_title('Command and actual speed (m/s)')
    axes[0,1].plot(z['time'],mix,label='student mix');axes[0,1].axhline(1,color='k',ls='--');axes[0,1].set_title('Handover stops below full student control')
    centers=[(w['iteration_start']+w['iteration_end'])/2 for w in windows]
    for key,label in [('handover_completions','handover'),('measured_episodes','measured >=5s')]:
        axes[1,0].plot(centers,[100*w[key]/w['episodes'] for w in windows],marker='o',label=label)
    axes[1,0].set_title('Training episode fractions (%)')
    axes[1,1].plot(centers,[w['max_leap_m']*100 if w['measured_episodes'] else np.nan for w in windows],marker='o')
    axes[1,1].axhline(cfg['leap_min_distance']*100,color='r',ls='--',label='80 cm goal');axes[1,1].set_title('Max formal training leap by window (cm)')
    axes[1,1].text(.03,.5,'Gaps = no measured student-gait episodes',transform=axes[1,1].transAxes)
    for ax in axes.flat:ax.grid(alpha=.2)
    for ax in axes.flat[:3]:ax.legend()
    axes[1,1].legend()
    for ax in axes[0]:ax.set_xlabel('Saved play seconds')
    for ax in axes[1]:ax.set_xlabel('PPO iteration')
    fig.savefig(out/'startup_stall_trends.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(report=str(out),formal_samples=int(formal.sum()),final_mix=result['final_student_mix'],training_handover_completions=len(completed),last_handover_iteration=result['last_handover_iteration'])))


if __name__=='__main__':main()
