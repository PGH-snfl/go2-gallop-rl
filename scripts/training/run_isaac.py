"""Isaac Lab is the training backend. MuJoCo is not imported here."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import traceback
from isaaclab.app import AppLauncher
from gpd.config import load_config
from gpd.paths import ROOT, new_run
from isaac_task.logging import RunLog, logged_runner

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("mode",choices=["train","check","play","export"])
p.add_argument("--config")
p.add_argument("--num-envs",type=int,default=4)
p.add_argument("--iterations",type=int,default=1000)
p.add_argument("--steps",type=int,default=440)
p.add_argument("--seed",type=int,default=42)
p.add_argument("--checkpoint",type=Path)
p.add_argument("--fresh",action="store_true",help="Train from random initialization; rejects --checkpoint")
p.add_argument("--output",type=Path)
p.add_argument("--log-every",type=int,default=50,help="Progress every N policy steps, or every 5 wall-clock seconds")
p.add_argument("--video",action="store_true",help="Record environment 0 to MP4 in the run's videos/ directory")
p.add_argument("--video-length",type=int,default=400,help="Policy steps per clip (400 steps = 10 seconds at 40 Hz)")
p.add_argument("--video-interval",type=int,default=5000,help="Training clip start interval in policy steps")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if min(args.num_envs,args.iterations,args.steps,args.log_every)<1:
    p.error("Counts must be positive")
if args.fresh and (args.mode!="train" or args.checkpoint):
    p.error("--fresh requires train without --checkpoint")
if args.video_length<1 or args.video_interval<args.video_length:
    p.error("Require 1 <= video-length <= video-interval")
if args.video and args.mode=="export":
    p.error("--video is for train/check/play, not export")
if args.video:
    args.enable_cameras=True
if args.mode in ("play","export") and not args.checkpoint:
    p.error("play/export requires --checkpoint")
if args.checkpoint:
    previous = args.checkpoint.resolve().parent
    manifest = json.loads((previous/"manifest.json").read_text())
    if manifest.get("interface") != "gpd_residual_57x5_isaac_v1":
        p.error("Checkpoint is not an Isaac Gallop policy; start fresh")
    cfg = load_config(previous/"config.json")
    if args.config and load_config(args.config) != cfg:
        p.error("Checkpoint controller configuration cannot be changed")
else:
    cfg = load_config(args.config)
out = args.output or new_run("isaac",args.mode)
out.mkdir(parents=True,exist_ok=True)
if (out/"status.json").exists():
    p.error("Output already contains a run; select a fresh directory")
status = dict(status="starting",mode=args.mode,backend="Isaac Lab / PhysX / RSL-RL",hardware_io=False,
    num_envs=args.num_envs,iterations=args.iterations,checkpoint=str(args.checkpoint) if args.checkpoint else None,
    training_initialization=("checkpoint" if args.checkpoint else "scratch") if args.mode=="train" else None,
    video_enabled=args.video)
def save_status():
    (out/"status.json").write_text(json.dumps(status,indent=2))
save_status()
log = RunLog(out,status)
log.emit("START",mode=args.mode,num_envs=args.num_envs,device=args.device,output=str(out.resolve()),
         policy_updates=args.mode=="train")
(out/"config.json").write_text(json.dumps(cfg,indent=2))
app = env = video_wrapper = None

def main():
    global env,video_wrapper
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from isaac_task.env import GallopIsaacCfg, GallopIsaacEnv, JOINTS, USD
    from isaac_task.export import export_actor
    ec = GallopIsaacCfg()
    ec.scene.num_envs = args.num_envs
    ec.seed = args.seed
    ec.sim.device = args.device
    ec.sim.dt = cfg["simulation_dt"]
    ec.decimation = round(cfg["policy_dt"]/cfg["simulation_dt"])
    ec.sim.render_interval = ec.decimation
    ec.episode_length_s = cfg["episode_seconds"]+cfg["settle_seconds"]
    if args.video:
        ec.viewer.resolution=(640,480)
    ec.robot.init_state.pos = (0.,0.,cfg["height"]+.018)
    ec.sim.physics_material.static_friction = cfg["friction"]
    ec.sim.physics_material.dynamic_friction = cfg["friction"]
    # Explicit torque model matches the deployment controller's limits.
    for j,name in enumerate(("hip","thigh","calf")):
        ec.robot.actuators["legs"].effort_limit[f".*_{name}_joint"] = cfg["torque_limits"][j]
        ec.robot.actuators["legs"].velocity_limit[f".*_{name}_joint"] = cfg["velocity_limits"][j]
    log.emit("ENV_CREATING",physics_dt=ec.sim.dt,control_dt=cfg["control_dt"],policy_dt=cfg["policy_dt"])
    env = GallopIsaacEnv(ec,cfg,record=args.mode in ("check","play"),
        log_callback=log.emit,log_every=args.log_every,
        total_steps=args.iterations*32 if args.mode=="train" else args.steps,
        follow_camera=args.video,render_mode="rgb_array" if args.video else None)
    env_api=env
    if args.video:
        import gymnasium as gym
        # Start periodically during training; record one continuous clip for play/check.
        # Internal per-env resets remain visible in the recording.
        video_wrapper=gym.wrappers.RecordVideo(env,video_folder=str(out/"videos"),
            step_trigger=(lambda step:step%args.video_interval==0) if args.mode=="train" else (lambda step:step==0),
            video_length=args.video_length,name_prefix=f"gallop-{args.mode}",
            fps=round(1/env.step_dt),disable_logger=True)
        env_api=video_wrapper
        status["video"]=dict(directory=str((out/"videos").resolve()),
            length_steps=args.video_length,interval_steps=args.video_interval if args.mode=="train" else None,
            fps=round(1/env.step_dt),environment=0)
        log.emit("VIDEO_RECORDING",**status["video"])
    wrapped = RslRlVecEnvWrapper(env_api,clip_actions=1.)
    log.emit("ENV_READY",observation_dim=57,action_dim=5,speed_target=cfg["speed"],
             lead=cfg["lead"],period=cfg["period"],contact_template=cfg.get("contact_template","paper"))
    manifest = dict(interface="gpd_residual_57x5_isaac_v1",trainer="Isaac Lab / RSL-RL PPO",
        observation_dim=57,action_dim=5,joint_names=JOINTS,usd_path=str(USD),
        usd_sha256=hashlib.sha256(USD.read_bytes()).hexdigest(),hardware_validated=False,
        control_implementation="shared gpd.controller.GallopController; CPU QP")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    for name in ("gpd","isaac_task","scripts"):
        shutil.copytree(ROOT/name,out/"source"/name,ignore=shutil.ignore_patterns("__pycache__"))
    agent = dict(seed=args.seed,device=args.device,num_steps_per_env=32,save_interval=50,
        obs_groups={"policy":["policy"],"critic":["policy"]},logger="tensorboard",
        policy=dict(class_name="ActorCritic",init_noise_std=.15,noise_std_type="log",
            actor_obs_normalization=False,critic_obs_normalization=False,
            actor_hidden_dims=[128,128],critic_hidden_dims=[128,128],activation="elu"),
        algorithm=dict(class_name="PPO",value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,
            entropy_coef=.005,num_learning_epochs=5,num_mini_batches=4,learning_rate=3e-4,
            schedule="adaptive",gamma=.99,lam=.95,desired_kl=.01,max_grad_norm=1.))
    (out/"agent.json").write_text(json.dumps(agent,indent=2))
    runner = logged_runner(OnPolicyRunner,log.emit)(wrapped,agent,log_dir=str(out),device=args.device)
    if args.checkpoint:
        log.emit("CHECKPOINT_LOADING",path=str(args.checkpoint))
        runner.load(str(args.checkpoint))
        log.emit("CHECKPOINT_LOADED",path=str(args.checkpoint))
    status["status"] = "running"
    save_status()
    if args.mode == "train":
        log.emit("TRAINING",iterations=args.iterations,rollout_steps=32)
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
        runner.save(str(out/"model_final.pt"))
    elif args.mode in ("check","play"):
        log.emit("EVALUATING",steps=args.steps,policy="checkpoint" if args.mode=="play" else "zero_residual",policy_updates=False)
        obs = wrapped.get_observations()
        actor = runner.get_inference_policy(device=args.device) if args.mode == "play" else None
        resets = 0
        with torch.inference_mode():
            for step in range(args.steps):
                action = actor(obs) if actor else torch.zeros((args.num_envs,5),device=args.device)
                obs,reward,done,info = wrapped.step(action)
                if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
                    raise RuntimeError("Nonfinite training transition")
                resets += int(done.sum().item())
        status.update(steps=args.steps,reset_count=resets)
        from gpd.metrics import summarize
        reports = []
        for i,episode in enumerate(env.episodes):
            reports.append(episode["report"])
            rows = episode["records"]
            np.savez_compressed(out/f"episode_{i}.npz",**{k:np.array([r[k] for r in rows]) for k in rows[0] if k!="fault"})
        complete_reports = list(reports)
        for rows in env.records:
            if rows:
                partial = summarize(rows,cfg,terminated=True,transport="isaaclab_physx")
                partial["incomplete_observation_window"] = True
                reports.append(partial)
        (out/"evaluation.json").write_text(json.dumps(reports,indent=2))
        status["gait_accepted"] = bool(complete_reports and all(r["accepted"] for r in complete_reports))
        log.emit("EVALUATION_RESULT",gait_accepted=status["gait_accepted"],resets=resets,
            completed_episodes=len(complete_reports),report=str(out/"evaluation.json"))
    if args.mode in ("train","export"):
        log.emit("EXPORTING",path=str(out/"actor.pt"))
        status["export"] = export_actor(runner.alg.policy,out)
        log.emit("EXPORT_VERIFIED",**status["export"])
        manifest["actor_sha256"] = hashlib.sha256((out/"actor.pt").read_bytes()).hexdigest()
        manifest["config_sha256"] = hashlib.sha256((out/"config.json").read_bytes()).hexdigest()
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    if video_wrapper is not None:
        # Flush a short final clip before reporting completion.
        if video_wrapper.recording:
            video_wrapper.stop_recording()
        status["videos"]=[str(path.resolve()) for path in sorted((out/"videos").glob("*.mp4"))]
        if not status["videos"]:
            raise RuntimeError("Video requested but no MP4 was saved")
        log.emit("VIDEO_SAVED",paths=status["videos"])
    status["status"] = "completed"
    log.emit("COMPLETED",mode=args.mode,output=str(out.resolve()))
    save_status()
    print(f"[RESULT] {out.resolve()}",flush=True)

try:
    log.emit("SIM_STARTING",backend="Isaac Lab / PhysX")
    app = AppLauncher(args).app
    log.emit("SIM_READY")
    main()
except BaseException as exc:
    status.update(status="failed",error=str(exc),traceback=traceback.format_exc())
    log.emit("FAILED",error=str(exc))
    save_status()
    raise
finally:
    if video_wrapper is not None:
        video_wrapper.close()
    elif env is not None:
        env.close()
    if app is not None:
        app.close()
