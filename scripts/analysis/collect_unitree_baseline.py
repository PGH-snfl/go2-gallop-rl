"""Archive metrics and provenance of a stock Unitree RL Lab run, without simulation or learning."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from gpd.paths import ROOT

class DataOnlyLoader(yaml.SafeLoader):
    pass

def python_tag_as_data(loader, suffix, node):
    # Saved Isaac configs use tuple/slice tags. Read data without constructing Python objects.
    if isinstance(node,yaml.ScalarNode):return loader.construct_scalar(node)
    if isinstance(node,yaml.SequenceNode):return loader.construct_sequence(node,deep=True)
    return loader.construct_mapping(node,deep=True)
DataOnlyLoader.add_multi_constructor('tag:yaml.org,2002:python/',python_tag_as_data)

def load_yaml(path):
    return yaml.load(path.read_text(),Loader=DataOnlyLoader)

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--play-run',type=Path,help='Optional separate official play run to collect alongside training')
    args=p.parse_args()
    run,out=args.run.resolve(),args.output.resolve()
    if out==run or run in out.parents:raise ValueError('Reports must be outside the original run')
    agent=load_yaml(run/'params/agent.yaml');env=load_yaml(run/'params/env.yaml')
    deploy=load_yaml(run/'params/deploy.yaml') if (run/'params/deploy.yaml').exists() else {'step_dt':env['sim']['dt']*env['decimation']}
    accumulator=EventAccumulator(str(run),size_guidance={'scalars':0});accumulator.Reload()
    series={tag:accumulator.Scalars(tag) for tag in accumulator.Tags()['scalars']}
    reward=series['Train/mean_reward']
    checkpoint_paths=sorted(run.glob('model_*.pt'),key=lambda f:int(f.stem.split('_')[-1]) if f.stem.split('_')[-1].isdigit() else float('inf'))
    final=checkpoint_paths[-1]
    checkpoint=torch.load(final,map_location='cpu',weights_only=True)
    iterations=int(checkpoint['iter'])+1
    shapes={k:list(v.shape) for k,v in checkpoint['model_state_dict'].items() if k.endswith('weight')}
    out.mkdir(parents=True,exist_ok=False)
    (out/'params').mkdir()
    for file in (run/'params').iterdir():
        if file.is_file():shutil.copy2(file,out/'params'/file.name)
    with (out/'tensorboard_scalars.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['tag','step','wall_time','value'])
        for tag,rows in series.items():writer.writerows([tag,e.step,e.wall_time,e.value] for e in rows)
    stats={tag:dict(count=len(rows),first=rows[0].value,last=rows[-1].value,
        last_100_mean=float(np.mean([e.value for e in rows[-100:]])),maximum=max(e.value for e in rows),
        maximum_step=max(rows,key=lambda e:e.value).step) for tag,rows in series.items() if rows}
    import imageio.v2 as imageio
    video_rows=[]
    sources=[(run/'videos','train')]
    if args.play_run:sources.append((args.play_run.resolve()/'videos','play'))
    for video,kind in [(video,kind) for folder,kind in sources for video in sorted(folder.rglob('*.mp4'))]:
        reader=imageio.get_reader(video)
        try:
            m=reader.get_meta_data()
            video_rows.append(dict(path=str(video),kind='play' if video.parent.name=='play' else kind,bytes=video.stat().st_size,
                fps=m['fps'],duration_seconds=m['duration'],resolution=list(m['size']),codec=m.get('codec')))
        finally:reader.close()
    with (out/'videos.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['path','kind','bytes','fps','duration_seconds','resolution','codec'])
        writer.writeheader();writer.writerows(video_rows)
    evidence={str(file.relative_to(ROOT)):dict(sha256=sha256(file),bytes=file.stat().st_size)
        for file in run.rglob('*') if file.is_file()}
    if args.play_run:
        evidence.update({str(file.relative_to(ROOT)):dict(sha256=sha256(file),bytes=file.stat().st_size)
            for file in args.play_run.resolve().rglob('*') if file.is_file()})
    (out/'evidence_manifest.json').write_text(json.dumps(evidence,indent=2))
    policy_dt=env['sim']['dt']*env['decimation']
    episode_stats=stats['Train/mean_episode_length']
    summary=dict(collected_at=datetime.now().astimezone().isoformat(),run=str(run),
        provenance=json.loads((run/'manifest.json').read_text()) if (run/'manifest.json').exists() else None,
        play_run=str(args.play_run.resolve()) if args.play_run else None,
        play_status=json.loads((args.play_run/'status.json').read_text()) if args.play_run and (args.play_run/'status.json').exists() else None,
        final_checkpoint=str(final),iteration_zero_based=int(checkpoint['iter']),iterations=iterations,
        expected_iterations=agent['max_iterations'],scalar_updates_observed=len(reward),
        training_completion_supported=iterations==agent['max_iterations'] and max(e.step for e in reward)==iterations-1,
        num_envs=env['scene']['num_envs'],rollout_steps_per_env=agent['num_steps_per_env'],
        total_transitions=iterations*env['scene']['num_envs']*agent['num_steps_per_env'],
        policy_dt=policy_dt,episode_target_seconds=env['episode_length_s'],
        final_mean_episode_seconds=episode_stats['last']*policy_dt,
        last_100_mean_episode_seconds=episode_stats['last_100_mean']*policy_dt,
        first_scalar_time=datetime.fromtimestamp(reward[0].wall_time).astimezone().isoformat(),
        last_scalar_time=datetime.fromtimestamp(reward[-1].wall_time).astimezone().isoformat(),
        scalar_recording_seconds=reward[-1].wall_time-reward[0].wall_time,
        checkpoint_count=len(checkpoint_paths),checkpoints=[str(f) for f in checkpoint_paths],
        model_weight_shapes=shapes,agent=agent,terrain_type=env['scene']['terrain']['terrain_type'],
        deployment_step_dt=deploy['step_dt'],tensorboard=stats,videos=video_rows,
        deterministic_play_evaluation=dict(available=False,reason='No separate play run supplied.'),
        interpretation_notes=[
            'Training reward and episode length are rolling training statistics, not held-out evaluation results.',
            'Episode_Termination statistics summarize recent episode endings across training environments; they are not independent test success rates.',
            'error_vel_xy and error_vel_yaw are command-resampling-time-normalized accumulated errors; they are not RMSE.',
            'Recorded terrain curriculum ended at 0; broad difficult-terrain competence is not established.',
            'Highest reward iteration is not necessarily the best checkpoint.',
            'This is a 45D actor/60D critic/12 joint-action baseline. It is incompatible with direct loading into the 57D/5-action GPD residual model.',
            'No new simulation, training, or policy updates were performed during collection.'])
    if args.play_run:
        from scripts.analysis.summarize_baseline_play import analyze_play_run
        summary['deterministic_play_evaluation']=analyze_play_run(args.play_run,out)
    if (run/'curriculum_summary.json').exists():
        summary['curriculum']=json.loads((run/'curriculum_summary.json').read_text())
    fine=(summary.get('provenance') or {}).get('fine_tuning')
    if fine:
        summary['fine_tuning']=fine
        summary['inherited_iterations']=fine['parent_cumulative_iterations']
        summary['cumulative_iterations']=fine['parent_cumulative_iterations']+iterations
        summary['total_transitions_scope']='This fine-tuning stage only; parent training is recorded separately.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    tags=['Train/mean_reward','Train/mean_episode_length','Episode_Termination/time_out',
          'Episode_Termination/bad_orientation','Metrics/base_velocity/error_vel_xy','Curriculum/lin_vel_cmd_levels']
    for axis,tag in zip(axes.flat,tags):
        rows=series.get(tag,[])
        if not rows:
            axis.set_title(tag);axis.text(.5,.5,'Not logged (command curriculum may be disabled)',ha='center',transform=axis.transAxes)
            continue
        x=np.array([e.step+1 for e in rows]);y=np.array([e.value for e in rows])
        if tag=='Train/mean_episode_length':y*=policy_dt
        axis.plot(x,y,alpha=.3,lw=.7,label='logged')
        if len(y)>=100:axis.plot(x[99:],np.convolve(y,np.ones(100)/100,mode='valid'),lw=1.5,label='100-update mean')
        axis.set_title(tag+(' (seconds)' if tag=='Train/mean_episode_length' else ''))
        axis.set_xlabel('PPO iteration (one-based)');axis.grid(alpha=.2)
    axes.flat[0].legend();fig.suptitle('Unitree Go2 baseline: training statistics, not deterministic evaluation')
    fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
    play=next((row for row in video_rows if row['kind']=='play'),None)
    if play:
        reader=imageio.get_reader(play['path'])
        try:
            fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
            for axis,at in zip(axes.flat,[0,2,5,8]):
                at=min(at,max(0,play['duration_seconds']-.1))
                axis.imshow(reader.get_data(round(at*play['fps'])));axis.set_title(f'{at:.1f} s');axis.axis('off')
            fig.suptitle('Existing final-policy play recording; frames do not establish quantitative stability')
            fig.savefig(out/'play_frames.png',dpi=130);plt.close(fig)
        finally:reader.close()
    print(json.dumps({k:summary[k] for k in ['run','final_checkpoint','iterations','total_transitions','final_mean_episode_seconds','training_completion_supported']},indent=2))
    print('Saved report:',out)

if __name__=='__main__':main()
