"""Collect existing logs and TensorBoard evidence only; never starts simulation/training."""
import argparse
from collections import Counter
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from gpd.paths import ROOT

def read(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--play',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    train,play,out=args.train.resolve(),args.play.resolve(),args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    ts,ps=read(train/'status.json'),read(play/'status.json')
    cfg=read(train/'config.json')
    events=[json.loads(line) for line in (play/'progress.jsonl').read_text().splitlines()]
    ends=[row for row in events if row['event']=='EPISODE_END']
    reports=read(play/'evaluation.json')
    completed=[r for r in reports if not r.get('incomplete_observation_window')]
    training_events=[json.loads(line) for line in (train/'progress.jsonl').read_text().splitlines()]
    updates=[row for row in training_events if row['event']=='PPO_UPDATE']
    tb=EventAccumulator(str(train),size_guidance={'scalars':0});tb.Reload()
    scalars={name:tb.Scalars(name) for name in tb.Tags()['scalars']}
    with (out/'tensorboard_scalars.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['tag','step_zero_based','wall_time','value'])
        for name,rows in scalars.items():
            w.writerows((name,x.step,x.wall_time,x.value) for x in rows)
    with (out/'episodes.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['episode','seconds_including_settle','gallop_seconds','reason','vx_at_reset','height_at_reset','accepted'])
        for i,e in enumerate(ends):
            w.writerow([i,e['episode_seconds'],max(0,e['episode_seconds']-cfg['settle_seconds']),e['reason'],e['vx'],e['height'],completed[i]['accepted']])
    selected=['model_final.pt','model_999.pt','actor.pt','config.json','agent.json','manifest.json','export_validation.json','status.json','progress.jsonl','console.log']
    evidence={str(f.relative_to(ROOT)):dict(sha256=digest(f),bytes=f.stat().st_size)
              for f in [*[train/x for x in selected],*[play/x for x in ['status.json','config.json','manifest.json','evaluation.json','progress.jsonl','console.log']],*train.glob('events.out.tfevents.*'),*play.glob('episode_*.npz')] if f.is_file()}
    (out/'evidence_manifest.json').write_text(json.dumps(evidence,indent=2))
    source_diff=[]
    for f in (train/'source').rglob('*.py'):
        relative=f.relative_to(train/'source')
        current=ROOT/relative
        if current.exists() and digest(current)!=digest(f):source_diff.append(str(relative))
    duration=[e['episode_seconds'] for e in ends]
    summary=dict(collected_at=datetime.now().astimezone().isoformat(),training_path=str(train),play_path=str(play),
        training_status=ts,play_status=ps,config=cfg,agent=read(train/'agent.json'),
        last_update=updates[-1],export_validation=read(train/'export_validation.json'),
        tensorboard={k:dict(count=len(v),first=v[0].value,last=v[-1].value,maximum=max(x.value for x in v),
            max_step_zero_based=max(v,key=lambda x:x.value).step) for k,v in scalars.items() if v},
        episode_summary=dict(completed=len(completed),partial=len(reports)-len(completed),accepted=sum(bool(x['accepted']) for x in completed),
            reset_reasons=dict(Counter(x['reason'] for x in ends)),duration_min=min(duration),duration_mean=float(np.mean(duration)),duration_max=max(duration),
            note='Durations include standing initialization. Reports after warmup cover only a short pre-failure window; do not treat those averages as sustained gait performance.'),
        current_source_differs_from_training=source_diff,collection_performed_no_simulation_or_training=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    # Export static figures from recorded data only.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    for ax,tag in zip(axes,['Train/mean_reward','Train/mean_episode_length','Loss/value_function']):
        rows=scalars[tag];ax.plot([x.step+1 for x in rows],[x.value for x in rows],linewidth=1)
        ax.set_ylabel(tag);ax.grid(alpha=.2)
    axes[-1].set_xlabel('PPO iteration (one-based)')
    fig.suptitle('Isaac Lab training: logged metrics, not gait acceptance')
    fig.savefig(out/'training_curves.png',dpi=150);plt.close(fig)
    data=np.load(play/'episode_0.npz',allow_pickle=False)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    t=data['time'];x=data['truth']
    axes[0].plot(t,np.rad2deg(x[:,3]),label='roll');axes[0].plot(t,np.rad2deg(x[:,4]),label='pitch')
    limit=np.rad2deg(cfg['roll_pitch_limit'])
    for sign in [-1,1]:axes[0].axhline(sign*limit,color='r',linestyle='--',alpha=.5)
    axes[0].set_ylabel('Attitude (deg)');axes[0].legend()
    axes[1].plot(t,x[:,6],label='vx');axes[1].axhline(cfg['speed'],linestyle='--',color='k');axes[1].set_ylabel('Speed (m/s)')
    axes[2].imshow(data['measured'].T,origin='lower',aspect='auto',interpolation='nearest',
        extent=[t[0],t[-1],-.5,3.5],vmin=0,vmax=1,cmap='Blues')
    axes[2].set_yticks(range(4),['FL','FR','RL','RR']);axes[2].set_ylabel('Detected contact')
    axes[2].set_xlabel('Episode time including initial stand (s)')
    fig.suptitle('First recorded play episode: trace ends before reset, not a successful cycle')
    fig.savefig(out/'play_episode_0.png',dpi=150);plt.close(fig)
    print(json.dumps({'output':str(out),'episodes':summary['episode_summary'],'source_differences':source_diff},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
