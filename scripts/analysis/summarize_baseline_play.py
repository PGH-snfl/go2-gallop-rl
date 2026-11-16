"""Summarize saved play telemetry before the first reset, without running simulation."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def analyze_play_run(run, output, warmup_seconds=3.):
    run, output = Path(run), Path(output)
    trace = run / 'play_trace.npz'
    if not trace.exists():
        return dict(available=False, reason='No saved numeric play_trace.npz; video alone cannot establish velocity tracking.')
    status = json.loads((run / 'status.json').read_text())
    if status['status'] != 'completed':
        return dict(available=False, reason='Play has no completed status; do not treat partial recordings as a completed test.')
    if output.resolve() == run.resolve() or run.resolve() in output.resolve().parents:
        raise ValueError('Write analysis outside the original play run')
    with np.load(trace) as archive:
        d = {key: archive[key] for key in archive.files}
    time = d['time']
    assert len(time) > 1 and (np.diff(time) > 0).all()
    assert all(np.isfinite(value).all() for value in d.values())
    dt = float(np.median(np.diff(time)))
    n = d['command'].shape[1]
    manifest = json.loads((run / 'manifest.json').read_text())
    requested = manifest.get('evaluation_command')
    if requested is not None:
        np.testing.assert_allclose(d['command'], np.broadcast_to(requested, d['command'].shape), atol=1e-6, rtol=0)
    valid = np.zeros((len(time), n), dtype=bool)
    environments = []
    for i in range(n):
        resets = np.flatnonzero(d['post_reset_state'][:, i])
        stop = int(resets[0]) if len(resets) else len(time)
        mask = (time >= warmup_seconds) & (np.arange(len(time)) < stop)
        valid[:, i] = mask
        command, velocity = d['command'][mask, i], d['body_velocity'][mask, i]
        contacts = d['foot_force'][mask, i] > 5
        row = dict(env=i, reset_count=len(resets), reset_times_seconds=time[resets].tolist(),
                   first_episode_end_seconds=float(time[stop]) if stop < len(time) else None,
                   evaluated_samples=int(mask.sum()), evaluated_seconds=float(mask.sum()*dt))
        if mask.any():
            row.update(mean_command=command.mean(axis=0,dtype=np.float64).tolist(), mean_body_velocity=velocity.mean(axis=0,dtype=np.float64).tolist(),
                       planar_velocity_rmse=np.sqrt(((velocity[:, :2]-command[:, :2])**2).mean(axis=0)).tolist(),
                       yaw_rate_rmse=float(np.sqrt(((d['angular_velocity'][mask,i,2]-command[:,2])**2).mean())),
                       roll_pitch_rms=np.sqrt((d['rpy'][mask,i,:2]**2).mean(axis=0)).tolist(),
                       all_feet_contact_fraction=float(contacts.all(axis=1).mean()),
                       per_foot_contact_fraction=contacts.mean(axis=0).tolist(),
                       planar_displacement=float(np.linalg.norm(d['position'][mask,i,:2][-1]-d['position'][mask,i,:2][0])))
        environments.append(row)
    report = dict(available=True, play_run=str(run.resolve()), trace=str(trace.resolve()),
                  completed=True, total_steps=len(time), duration_seconds=float(time[-1]),
                  policy_updates=status['policy_updates'], warmup_seconds=warmup_seconds,
                  fixed_command=requested, fixed_command_trace_matches_manifest=requested is not None,
                  environments=environments,
                  caveats=['Summary excludes the warmup and all rows at/after the first reset.',
                           'Reset rows contain post-reset states. Reset reasons and pre-terminal states were not logged.',
                           'This play retains original noise/randomization/push events; it is not the nominal acceptance protocol.'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'play_analysis.json').write_text(json.dumps(report, indent=2))
    with (output / 'play_state_trace.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time','env','included_before_first_reset','post_reset_state','vx_cmd','vy_cmd','yaw_cmd','vx','vy','yaw_rate','height','roll','pitch','FL_force','FR_force','RL_force','RR_force'])
        for j, t in enumerate(time):
            for i in range(n):
                writer.writerow([t,i,bool(valid[j,i]),bool(d['post_reset_state'][j,i]),*d['command'][j,i],
                                 *d['body_velocity'][j,i,:2],d['angular_velocity'][j,i,2],d['position'][j,i,2],
                                 *d['rpy'][j,i,:2],*d['foot_force'][j,i]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
    axes[0].plot(time,d['command'][:,0,0],'k--',label='command vx')
    axes[0].plot(time,d['body_velocity'][:,0,0],label='measured vx');axes[0].set_ylabel('vx (m/s)');axes[0].legend()
    axes[1].plot(time,d['angular_velocity'][:,0,2],label='measured yaw rate');axes[1].set_ylabel('rad/s');axes[1].legend()
    axes[2].imshow((d['foot_force'][:,0]>5).T,origin='lower',aspect='auto',interpolation='nearest',
                   extent=[time[0],time[-1],-.5,3.5],cmap='Blues',vmin=0,vmax=1)
    axes[2].set_yticks(range(4),['FL','FR','RL','RR']);axes[2].set_ylabel('Contact >5N');axes[2].set_xlabel('Time (s)')
    for axis in axes:
        axis.axvspan(0,warmup_seconds,color='grey',alpha=.2)
        if environments[0]['first_episode_end_seconds'] is not None:
            axis.axvspan(environments[0]['first_episode_end_seconds'],time[-1],color='grey',alpha=.2)
        axis.grid(alpha=.2)
    fig.suptitle('Recorded play: grey intervals excluded from first-episode metrics')
    fig.savefig(output / 'play_tracking.png',dpi=150);plt.close(fig)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--play-run',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a = p.parse_args()
    report = analyze_play_run(a.play_run,a.report)
    summary_path = a.report / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary['deterministic_play_evaluation'] = report
        summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
