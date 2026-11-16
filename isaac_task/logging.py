"""Human-readable console events plus machine-readable progress, without simulator imports."""
from datetime import datetime
from contextlib import redirect_stdout
from io import StringIO
import json
import sys
import time
import math
import statistics

def duration(seconds):
    seconds = max(0,round(seconds))
    hours,remainder = divmod(seconds,3600)
    minutes,seconds = divmod(remainder,60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def console_detail(event, fields, elapsed, mode):
    if event == "PPO_UPDATE":
        return None  # RSL-RL already prints the requested iteration table.
    if event == 'GALLOP_CURRICULUM':
        rate=fields['success_rate'];rate_text='N/A' if rate is None else f'{rate:.0%}'
        if 'staged_index' in fields:
            goal=fields['staged_goal']
            before=fields['staged_index_before'];after=fields['staged_index']
            label=str(after) if before==after else f'{before}→{after}'
            return (f"stage={label}/8 | vx={goal['speed']:.2f}m/s | leap={goal['leap']*100:.0f}cm | "
                    f"성공률={rate_text} | cohort={fields['cohort_completed']}/{fields['cohort_size']} | {fields['reason']}")
        return f"step={fields['step']} | vx 상한={fields['speed_cap']:.2f} m/s | 성공률={rate_text} | episode={fields['episodes']} | {fields['reason']}"
    if event == 'RELEARN_EXPLORATION':
        return f"stage={fields['relearn_level']} | noise std={fields['noise_std']:.3f} | entropy=0"
    if event == 'STAGED_ADAPTATION':
        return (f"noise std={fields['parent_noise_mean']:.3f}→{fields['target_noise_mean']:.3f} | "
                f"ramp={fields['ramp_iterations']} PPO | entropy=0") if 'parent_noise_mean' in fields else f"noise std={fields['noise_mean']:.3f} | target reached"
    if event == "VELOCITY_CURRICULUM":
        score = fields['normalized_track_reward']
        score_text = 'N/A' if score is None else f'{score:.3f}'
        return (f"step={fields['step']} | vx 상한={fields['after']['lin_vel_x'][1]:.2f} m/s | "
                f"episode={fields['eligible_episodes']} | 추종={score_text}/{fields['threshold']:.2f} | {fields['reason']}")
    if event == "EPISODE_END" or (event == "PROGRESS" and mode == "train"):
        return None
    if event == "PROGRESS":
        speed = f"{fields['mean_vx']:.2f}" if fields['mean_vx'] is not None else "N/A"
        return (f"Step {fields['step']}/{fields['total_steps']} | "
                f"속도 {speed} m/s | reset {fields['resets']} | 누적 {duration(elapsed)}")
    if event in ("ENV_CREATING","SIM_READY","CHECKPOINT_LOADING","EXPORTING"):
        return None
    return " ".join(f"{k}={json.dumps(v,ensure_ascii=False)}" for k,v in fields.items())

def finite_fields(value):
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if isinstance(value,dict):
        return {k:finite_fields(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [finite_fields(v) for v in value]
    return value

class RunLog:
    def __init__(self, output, status):
        self.output, self.status = output, status
        self.started = time.monotonic()

    def emit(self, event, **fields):
        fields = finite_fields(fields)
        row = dict(time=datetime.now().astimezone().isoformat(timespec="seconds"),
                   elapsed_s=round(time.monotonic()-self.started,2), event=event, **fields)
        with (self.output/"progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
        detail = console_detail(event,fields,row["elapsed_s"],self.status.get("mode"))
        if detail is not None:
            print(f"[{row['time']}] [{event}] {detail}",flush=True)
        self.status["last_event"] = row
        temporary = self.output/"status.json.tmp"
        temporary.write_text(json.dumps(self.status,indent=2,ensure_ascii=False))
        temporary.replace(self.output/"status.json")

def logged_runner(base_class, emit, metrics=None, compact_episode_log=False):
    class LoggedRunner(base_class):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            # RSL-RL initializes this only in learn(), but adaptation saves a
            # valid checkpoint before any PPO call or writer initialization.
            if not hasattr(self,'logger_type'):
                self.logger_type=self.cfg.get('logger','tensorboard').lower()

        def log(self, locs, *args, **kwargs):
            # Preserve the native RSL-RL iteration table and TensorBoard writes.
            if compact_episode_log:
                prefixes=tuple({(key if '/' in key else 'Mean episode '+key)+':' for info in locs['ep_infos'] for key in info})
                buffer=StringIO()
                with redirect_stdout(buffer):super().log(locs,*args,**kwargs)
                sys.stdout.write(''.join(line for line in buffer.getvalue().splitlines(keepends=True) if not line.lstrip().startswith(prefixes)))
                sys.stdout.flush()
            else:super().log(locs,*args,**kwargs)
            extra_metrics=metrics() if metrics is not None else {}
            if self.writer is not None:
                for key, value in extra_metrics.items():
                    self.writer.add_scalar(key,value,locs['it'])
            completed = locs["it"]-locs["start_iter"]+1
            total = locs["tot_iter"]-locs["start_iter"]
            elapsed = locs["collection_time"]+locs["learn_time"]
            emit("PPO_UPDATE", iteration=locs["it"]+1, completed=completed,total=total,
                 iteration_limit=locs["tot_iter"],
                 mean_reward=float(statistics.mean(locs["rewbuffer"])) if len(locs["rewbuffer"]) else None,
                 progress_pct=round(100*completed/total,1),
                 transitions=self.tot_timesteps,iteration_s=round(elapsed,3),
                 collection_s=round(locs["collection_time"],3),learning_s=round(locs["learn_time"],3),
                 eta_s=round(self.tot_time/completed*(total-completed),1),
                 losses={k:float(v) for k,v in locs["loss_dict"].items()},
                 **({'diagnostics':{k:v for k,v in extra_metrics.items() if k.startswith('StagedAdapt/')}}
                    if any(k.startswith('StagedAdapt/') for k in extra_metrics) else {}))

        def save(self, path, *args, **kwargs):
            if self.logger_type in ('neptune','wandb') and self.writer is None and not self.disable_logs:
                self._prepare_logging_writer()
            super().save(path,*args,**kwargs)
            emit("CHECKPOINT_SAVED",path=str(path))
    return LoggedRunner
