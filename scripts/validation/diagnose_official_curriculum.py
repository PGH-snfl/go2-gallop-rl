"""Observe the original curriculum with a frozen checkpoint; never runs PPO updates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

from gpd.paths import new_run
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--num-envs', type=int, default=256)
p.add_argument('--steps', type=int, default=2200)
p.add_argument('--initial-phase', choices=['random', 'zero'], default='random')
p.add_argument('--action-mode', choices=['sampled', 'mean'], default='sampled')
p.add_argument('--curriculum', choices=['original', 'fixed'], default='original')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--output', type=Path)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if min(a.num_envs, a.steps) < 1:
    p.error('positive environment and step counts required')
out = a.output or new_run('isaac', 'check')
out.mkdir(parents=True, exist_ok=True)
if (out / 'status.json').exists():
    p.error('use a new output folder')
status = dict(status='starting', policy_updates=0, arguments=vars(a).copy())
status['arguments'] = {k: str(v) if isinstance(v, Path) else v for k, v in status['arguments'].items()}


def save():
    (out / 'status.json').write_text(json.dumps(status, indent=2))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


app = env = None
save()


def main():
    global env
    import csv
    import inspect
    import shutil
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.algorithms import PPO
    from rsl_rl.runners import OnPolicyRunner
    from scripts.analysis.collect_unitree_baseline import load_yaml

    manifest = json.loads((a.checkpoint.parent / 'manifest.json').read_text())
    if manifest['interface'] != 'unitree_go2_official_45_60_12_v1':
        raise RuntimeError('this diagnostic requires the clean official checkpoint')
    repo = Path(manifest['official_repo'])

    def verify():
        def git(*args):
            return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
        if git('rev-parse', 'HEAD') != manifest['official_commit'] or git('status', '--porcelain', '--untracked-files=no'):
            raise RuntimeError('official source is not the pinned clean checkout')
        if sha(Path(manifest['asset_path'])) != manifest['asset_sha256']:
            raise RuntimeError('asset differs from the training asset')

    verify()
    model_sha = sha(a.checkpoint)
    sys.path.insert(0, str(repo / 'source/unitree_rl_lab'))
    from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg
    if not Path(inspect.getfile(RobotEnvCfg)).is_relative_to(repo):
        raise RuntimeError('wrong task package imported')
    cfg = RobotEnvCfg()
    cfg.scene.robot.spawn.usd_path = manifest['asset_path']
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    cfg.sim.device = a.device
    original = cfg.curriculum.lin_vel_cmd_levels.func
    calls = []
    boundaries = []

    def observe_curriculum(e, env_ids, reward_term_name='track_lin_vel_xy'):
        term = e.command_manager.get_term('base_velocity')
        before = float(term.cfg.ranges.lin_vel_x[1])
        reward = float(e.reward_manager._episode_sums[reward_term_name][env_ids].mean() / e.max_episode_length_s)
        threshold = e.reward_manager.get_term_cfg(reward_term_name).weight * .8
        result = original(e, env_ids, reward_term_name)
        ids = torch.as_tensor(env_ids, device=e.device)
        calls.append(dict(step=int(e.common_step_counter), reset_count=len(ids),
            global_step_gate=bool(e.common_step_counter % e.max_episode_length == 0),
            normalized_track_reward=reward, threshold=threshold, reward_gate=reward > threshold,
            vx_limit_before=before, vx_limit_after=float(term.cfg.ranges.lin_vel_x[1]),
            mean_episode_length_buffer=float(e.episode_length_buf[ids].float().mean()),
            env_ids=ids.tolist()))
        return result

    # The wrapper records arguments/results and calls the original unchanged.
    if a.curriculum == 'fixed':
        from isaac_task.velocity_curriculum import curriculum_env_class
        cfg.curriculum.lin_vel_cmd_levels = None
        def curriculum_event(event, **fields):
            print('[' + event + '] ' + json.dumps(fields), flush=True)
        env = curriculum_env_class(ManagerBasedRLEnv)(cfg, curriculum_emit=curriculum_event)
    else:
        cfg.curriculum.lin_vel_cmd_levels.func = observe_curriculum
        env = ManagerBasedRLEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    agent = load_yaml(a.checkpoint.parent / 'params/agent.yaml')
    agent['device'] = a.device
    runner = OnPolicyRunner(wrapped, agent, log_dir=None, device=a.device)

    def no_updates(*args, **kwargs):
        raise RuntimeError('PPO updates forbidden in diagnostic')

    PPO.update = no_updates
    runner.load(str(a.checkpoint.resolve()), load_optimizer=False, map_location=a.device)
    policy = runner.alg.policy
    policy.requires_grad_(False)
    runner.train_mode()
    initial_weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    wrapped.episode_length_buf = (torch.randint_like(wrapped.episode_length_buf, high=int(wrapped.max_episode_length))
                                 if a.initial_phase == 'random' else torch.zeros_like(wrapped.episode_length_buf))
    offsets = wrapped.episode_length_buf.detach().cpu().numpy().copy()
    np.save(out / 'initial_episode_offsets.npy', offsets)
    obs = wrapped.get_observations().to(a.device)
    contact = env.scene['contact_forces']
    feet = [contact.body_names.index(f'{leg}_foot') for leg in ['FL', 'FR', 'RL', 'RR']]
    samples = []
    status.update(status='running', checkpoint_sha256=model_sha, official_commit=manifest['official_commit'],
        diagnostic_changes=[('periodic fixed curriculum' if a.curriculum == 'fixed' else 'observe native curriculum unchanged'),
                            f'initial episode phase: {a.initial_phase}', f'frozen actions: {a.action_mode}'],
        max_episode_length=int(env.max_episode_length), initial_zero_offset_count=int((offsets == 0).sum()))
    save()
    print('[DIAG_START] ' + json.dumps(status), flush=True)
    with torch.inference_mode():
        for step in range(a.steps):
            action = policy.act(obs) if a.action_mode == 'sampled' else policy.act_inference(obs)
            obs, reward, done, _ = wrapped.step(action)
            if not torch.isfinite(reward).all():
                raise RuntimeError('nonfinite rewards')
            if env.common_step_counter % env.max_episode_length == 0:
                recent = [x for x in calls if x['step'] == env.common_step_counter]
                boundaries.append(dict(step=int(env.common_step_counter), reset_count=int(done.sum()),
                    callback_ran=bool(recent), callback=recent,
                    mean_partial_track_sum_per_max_duration=float(env.reward_manager._episode_sums['track_lin_vel_xy'].mean() / env.max_episode_length_s),
                    vx_limit=float(env.command_manager.get_term('base_velocity').cfg.ranges.lin_vel_x[1])))
                print('[BOUNDARY] ' + json.dumps(boundaries[-1]), flush=True)
            if (step + 1) % 50 == 0:
                data = env.scene['robot'].data
                command = env.command_manager.get_command('base_velocity')
                samples.append(dict(step=step+1, mean_xy_command_norm=float(command[:, :2].norm(dim=1).mean()),
                    air_time_command_gate_fraction=float((command[:, :2].norm(dim=1) > .1).float().mean()),
                    mean_xy_speed=float(data.root_lin_vel_b[:, :2].norm(dim=1).mean()),
                    all_feet_contact_fraction=float((contact.data.net_forces_w[:, feet, 2] > 5).all(dim=1).float().mean()),
                    vx_limit=float(env.command_manager.get_term('base_velocity').cfg.ranges.lin_vel_x[1])))
            if (step + 1) % 250 == 0:
                status.update(step=step+1, last_sample=samples[-1])
                save()
                print('[PROGRESS] ' + json.dumps(status['last_sample']), flush=True)
    if not all(torch.equal(value, initial_weights[key]) for key, value in policy.state_dict().items()):
        raise RuntimeError('policy state changed during diagnostic')
    verify()
    if sha(a.checkpoint) != model_sha:
        raise RuntimeError('checkpoint file changed')
    active = [c for c in calls if c['step'] > 0]
    summary = dict(initial_phase=a.initial_phase, action_mode=a.action_mode, steps=a.steps, num_envs=a.num_envs,
        curriculum=a.curriculum,
        initial_zero_offset_count=int((offsets == 0).sum()), curriculum_callback_count=len(active),
        callbacks_passing_reward_gate=sum(c['reward_gate'] for c in active),
        callbacks_passing_global_step_gate=sum(c['global_step_gate'] for c in active),
        expansions=sum(c['vx_limit_after'] > c['vx_limit_before'] for c in active),
        final_vx_limit=float(env.command_manager.get_term('base_velocity').cfg.ranges.lin_vel_x[1]),
        boundaries=boundaries, samples=samples, policy_updates=0, unchanged_policy_state=True,
        checkpoint_sha256=model_sha, checkpoint_file_unchanged=True, official_source_clean=True,
        interpretation='Frozen-policy reproduction, not a reconstruction of historical PPO random states.')
    if a.curriculum == 'fixed':
        controller = env.velocity_curriculum
        summary['fixed_curriculum'] = controller.summary()
        summary['expansions'] = controller.expansions
        import isaac_task.velocity_curriculum as fixed_source
        shutil.copy2(fixed_source.__file__, out / 'velocity_curriculum.py')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'curriculum_calls.json').write_text(json.dumps(calls, indent=2))
    with (out / 'curriculum_calls.csv').open('w') as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=list(calls[0]) if calls else ['step'])
        csv_writer.writeheader()
        csv_writer.writerows(calls)
    shutil.copy2(__file__, out / 'diagnostic_source.py')
    status.update(status='completed', step=a.steps, summary=str(out / 'summary.json'),
                  unchanged_policy_state=True, checkpoint_file_unchanged=True, official_source_clean=True)
    save()
    print('[DIAG_COMPLETE] ' + json.dumps({k: v for k, v in summary.items() if k != 'samples'}), flush=True)


try:
    app = AppLauncher(a).app
    main()
except BaseException as error:
    status.update(status='failed', error=str(error), traceback=traceback.format_exc())
    save()
    raise
finally:
    if env is not None:
        env.close()
    if app is not None:
        app.close()
