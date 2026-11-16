"""Frozen-policy MuJoCo diagnostics against official Go2 URDF limits; no DDS/hardware."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch

from gpd.paths import ROOT, new_run


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def official_limits(path, names):
    joints = {j.attrib['name']: j for j in ET.parse(path).getroot().findall('joint')}
    rows = []
    for name in names:
        limit = joints[name].find('limit')
        rows.append({key: float(limit.attrib[key]) for key in ('lower', 'upper', 'effort', 'velocity')})
    return rows


def restore_official_model(plant, source, mj):
    """Restore published MJCF physical parameters, keeping the saved target filter."""
    native = mj.MjModel.from_xml_path(str(source.resolve()))
    if (native.njnt, native.nv, native.nu) != (plant.model.njnt, plant.model.nv, plant.model.nu):
        raise ValueError('Native model topology differs')
    # Added plane/light do not change the source's leading geometry/joint ordering.
    for kind, count in ((mj.mjtObj.mjOBJ_JOINT, native.njnt),
                        (mj.mjtObj.mjOBJ_ACTUATOR, native.nu)):
        for i in range(count):
            if mj.mj_id2name(native, kind, i) != mj.mj_id2name(plant.model, kind, i):
                raise ValueError('Native model ordering differs')
    for name in ('jnt_range', 'dof_damping', 'dof_armature', 'dof_frictionloss',
                 'actuator_ctrlrange', 'actuator_ctrllimited'):
        getattr(plant.model, name)[:] = getattr(native, name)
    # MuJoCo compiles world geoms before body geoms: the added floor shifts
    # all source geom IDs. Match body-local order, including unnamed geoms.
    for body in range(native.nbody):
        body_name=mj.mj_id2name(native,mj.mjtObj.mjOBJ_BODY,body)
        dst_body=mj.mj_name2id(plant.model,mj.mjtObj.mjOBJ_BODY,body_name) if body else 0
        src=np.flatnonzero(native.geom_bodyid==body)
        dst=np.flatnonzero(plant.model.geom_bodyid==dst_body)
        dst=np.array([i for i in dst if mj.mj_id2name(plant.model,mj.mjtObj.mjOBJ_GEOM,int(i))!='floor'],dtype=int)
        if len(src)!=len(dst):
            raise ValueError('Native body geometry differs')
        for name in ('geom_friction', 'geom_condim'):
            getattr(plant.model,name)[dst]=getattr(native,name)[src]
    plane = mj.mj_name2id(plant.model, mj.mjtObj.mjOBJ_GEOM, 'floor')
    plant.model.geom_friction[plane] = [.4, .02, .01]
    plant.model.geom_condim[plane] = 3
    mj.mj_forward(plant.model, plant.data)
    plant.sync()
    plant.contract['validation_adjustments'] = [
        'Official MJCF joint stops restored: no artificial central-90-percent physical stops.',
        'Official MJCF passive damping/armature/frictionloss and robot geom friction/condim restored.',
        'Flat ground friction .4; source foot geom priority/contact settings retained.',
        'Official MJCF static actuator ranges restored (hip/thigh23.7, calf45.43Nm).',
        'Saved target position/rate filter, walking teacher and transition logic retained.',
        'LowCmd-equivalent PD formula evaluated locally, not through DDS or identified firmware.',
    ]
    plant.contract['model_adjustments'] = ['Native parameter restoration supersedes training-matching adjustments; see validation_adjustments.']
    plant.contract['runtime_native_joint_limits'] = plant.model.jnt_range[
        [mj.mj_name2id(plant.model,mj.mjtObj.mjOBJ_JOINT,n) for n in plant.native]
    ].tolist() if hasattr(plant,'native') else plant.model.jnt_range.tolist()
    plant.contract['runtime_actuator_ctrlrange'] = plant.model.actuator_ctrlrange.tolist()


def audit_physics(physics, names, rows, mass, dt):
    q, dq = physics['joint_pos'], physics['joint_velocity']
    pre_dq, torque = physics['pre_joint_velocity'], physics['applied_torque']
    targets, raw = physics['joint_target'], physics['computed_torque']
    lo = np.array([r['lower'] for r in rows]); hi = np.array([r['upper'] for r in rows])
    effort = np.array([r['effort'] for r in rows]); velocity = np.array([r['velocity'] for r in rows])
    excess = lambda x: np.maximum(lo-x, 0) + np.maximum(x-hi, 0)
    per_joint = []
    for i, name in enumerate(names):
        per_joint.append(dict(
            joint=name, official_lower_rad=float(lo[i]), official_upper_rad=float(hi[i]),
            official_effort_nm=float(effort[i]), official_velocity_rad_s=float(velocity[i]),
            actual_min_rad=float(q[:, i].min()), actual_max_rad=float(q[:, i].max()),
            max_position_excess_rad=float(excess(q)[:, i].max()),
            position_exceed_seconds=float((excess(q)[:, i]>1e-4).sum()*dt),
            target_max_excess_rad=float(excess(targets)[:, i].max()),
            max_abs_velocity_rad_s=float(np.abs(dq[:, i]).max()),
            velocity_exceed_seconds=float((np.abs(dq[:, i])>velocity[i]+1e-4).sum()*dt),
            max_abs_applied_torque_nm=float(np.abs(torque[:, i]).max()),
            applied_torque_exceed_seconds=float((np.abs(torque[:, i])>effort[i]+1e-4).sum()*dt),
            max_abs_unclipped_pd_nm=float(np.abs(raw[:, i]).max()),
            unclipped_pd_exceed_fraction=float((np.abs(raw[:, i])>effort[i]+1e-4).mean()),
            motor_saturation_fraction=float((np.abs(raw[:, i]-torque[:, i])>1e-5).mean()),
            torque_rms_nm=float(np.sqrt(np.mean(torque[:, i]**2))),
        ))
    forces = physics['foot_force']; norm = np.linalg.norm(forces, axis=-1)
    vertical = np.maximum(forces[:, :, 2], 0); total_vertical = vertical.sum(-1)
    # These are resultant simulated contact forces, not an official structural load allowance.
    power = torque*pre_dq
    return dict(
        per_joint=per_joint, samples=len(q), physics_dt=dt,
        all_positions_within_urdf=bool((excess(q)<=1e-4).all()),
        all_targets_within_urdf=bool((excess(targets)<=1e-4).all()),
        all_velocities_within_urdf=bool((np.abs(dq)<=velocity+1e-4).all()),
        all_applied_torques_within_urdf=bool((np.abs(torque)<=effort+1e-4).all()),
        max_abs_unclipped_pd_nm=float(np.abs(raw).max()),
        peak_foot_resultant_n={leg:float(norm[:, i].max()) for i, leg in enumerate(('FL','FR','RL','RR'))},
        peak_sum_vertical_force_n=float(total_vertical.max()),
        peak_sum_vertical_force_bw=float(total_vertical.max()/(mass*9.81)),
        max_base_contact_force_n=float(physics['base_contact_force'].max()),
        peak_positive_mechanical_power_w=float(np.maximum(power, 0).sum(-1).max()),
        mean_positive_mechanical_power_w=float(np.maximum(power, 0).sum(-1).mean()),
        contact_force_official_allowance='UNKNOWN: no manufacturer continuous/impact foot-load rating verified.',
        electrical_thermal_allowance='UNKNOWN: mechanical power/peak torque cannot establish current, temperature or continuous-duty safety.',
        thresholds='1e-4rad numerical position tolerance; 1e-4rad/s and1e-4Nm numeric tolerance.',
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--official-urdf', type=Path, required=True)
    p.add_argument('--model', type=Path, default=ROOT.parents[1]/'unitree/unitree_mujoco/unitree_robots/go2/go2.xml')
    p.add_argument('--profile', choices=('trained', 'sdk_proxy'), default='trained')
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--physics-dt', type=float)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--observation-noise', action='store_true')
    p.add_argument('--video', action='store_true')
    p.add_argument('--viewer', action='store_true', help='Show live MuJoCo window; follows robot')
    p.add_argument('--replay-seconds', type=float, default=0., help='After saving, replay recorded qpos in window for this wall duration')
    a = p.parse_args()
    if not 0<a.seconds<=30:
        p.error('Use one episode of0<seconds<=30')
    if a.replay_seconds<0 or a.replay_seconds>120:
        p.error('Replay duration must be between0 and120seconds')
    os.environ.setdefault('MUJOCO_GL', 'glfw' if a.viewer else 'egl')
    import mujoco as mj
    from gpd.rl_mujoco import GallopMuJoCo, sha
    from scripts.analysis.collect_gallop import analyze_rollout
    from scripts.validation.sim2sim_gallop import render_trace
    torch.set_num_threads(1)
    run = new_run('mujoco', 'evaluate')
    out = run.parent/'unitree_go2_gallop'/run.name
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status='starting', profile=a.profile, output=str(out), policy_updates=0, physical_hardware=False)
    write(out/'status.json', status)
    print(json.dumps(dict(event='START', **status)), flush=True)
    started = time.perf_counter()
    viewer=None
    try:
        plant = GallopMuJoCo(a.checkpoint, a.model, out, a.observation_noise, a.seed,
                            torque_mode='go2hv' if a.profile=='trained' else 'sdk_pd_static')
        if a.profile=='sdk_proxy':
            restore_official_model(plant, a.model, mj)
        if a.physics_dt is not None:
            ratio = round(plant.control_dt/a.physics_dt)
            if ratio<1 or not np.isclose(ratio*a.physics_dt, plant.control_dt):
                raise ValueError('Physics dt must divide the saved control dt')
            plant.physics_dt=a.physics_dt;plant.decimation=ratio;plant.model.opt.timestep=a.physics_dt
            plant.env.cfg.decimation=ratio
            plant.contract['physics_dt']=a.physics_dt
        rows = official_limits(a.official_urdf, plant.native)
        write(out/'model_contract.json', plant.contract)
        write(out/'official_limits.json', dict(source=str(a.official_urdf.resolve()),sha256=sha(a.official_urdf),joint_order=plant.native,limits=rows))
        controller_ms, inference_ms = [], []
        actor = plant.actor
        def timed_actor(obs):
            start=time.perf_counter_ns();value=actor(obs)
            inference_ms.append((time.perf_counter_ns()-start)/1e6)
            return value
        plant.actor=timed_actor
        samples={};steps=round(a.seconds/plant.control_dt)
        status.update(status='running', physics_dt=plant.physics_dt, control_dt=plant.control_dt)
        write(out/'status.json', status)
        if a.viewer:
            import mujoco.viewer
            viewer=mujoco.viewer.launch_passive(plant.model,plant.data)
            viewer.cam.distance=2.7;viewer.cam.azimuth=135;viewer.cam.elevation=-18
            print(json.dumps(dict(event='VIEWER_OPEN',display=os.environ.get('DISPLAY'))),flush=True)
        for i in range(steps):
            wall_start=time.perf_counter()
            start=time.perf_counter_ns();sample=plant.step(last=i==steps-1)
            controller_ms.append((time.perf_counter_ns()-start)/1e6)
            for key,value in sample.items():
                samples.setdefault(key,[]).append(value)
            if viewer and viewer.is_running():
                viewer.cam.lookat[:]=plant.data.qpos[:3];viewer.sync()
                time.sleep(max(0.,plant.control_dt-(time.perf_counter()-wall_start)))
            if (i+1)%250==0 or plant.fault:
                print(json.dumps(dict(event='PROGRESS',seconds=float(plant.data.time),startup_stage=int(plant.state.used_stage[0]),
                                      vx=float(plant.tensor.root_lin_vel_b[0,0]),fault=plant.fault)),flush=True)
            if plant.fault:
                break
        n=len(samples['phase']);trace={key:np.stack(value) for key,value in samples.items()}
        trace.update(time=np.arange(1,n+1)*plant.control_dt, valid=np.ones((n,1),bool),episode_reset=np.zeros((n,1),bool))
        physics={key:np.array([row[key] for row in plant.physics]) for key in plant.physics[0]}
        np.savez_compressed(out/'gallop_trace.npz',**trace)
        np.savez_compressed(out/'physics_trace.npz',**physics)
        if not plant.unchanged():
            raise RuntimeError('Policy changed during evaluation')
        hardware=audit_physics(physics,plant.native,rows,float(plant.model.body_mass.sum()),plant.physics_dt)
        with (out/'joint_limits.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(hardware['per_joint'][0]));writer.writeheader();writer.writerows(hardware['per_joint'])
        report=dict(profile=a.profile,seed=a.seed,checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha(a.checkpoint),
            model_source=str(a.model.resolve()),model_source_sha256=sha(a.model),
            completed_seconds=n*plant.control_dt, terminated=bool(plant.fault),termination_reason=plant.fault,
            policy_updates=0,policy_state_unchanged=True,physical_hardware=False,
            episode=plant.state.last_episode,analysis=analyze_rollout(trace,plant.manifest['gallop_settings'],[plant.state.last_episode]),
            hardware=hardware,timing=dict(inference_ms_quantiles=np.quantile(inference_ms,[.5,.95,.99,1]).tolist(),
                synchronous_sim_step_ms_quantiles=np.quantile(controller_ms,[.5,.95,.99,1]).tolist(),
                inference_calls_over20ms=int((np.array(inference_ms)>20).sum()),
                note='CPU inference timing only; full step includes synchronous physics and is not deploy-loop WCET/DDS latency.'),
            deployment_accepted=False,
            limitations=['No DDS, physical motors, deployable velocity/height estimator, transport delay or thermal model.',
                'Official generic Go2 URDF/MJCF are software limits; serial-specific EDU firmware/continuous-duty limits remain unverified.',
                'URDF effort and velocity maxima are independent bounds, not a simultaneous torque-speed capability guarantee.',
                'Static PD SDK proxy is not a validated real Go2 torque-speed curve.',
                'Saved supervisor still sees simulator-perfect velocity/height; actor and supervisor input availability differ.'])
        write(out/'report.json',report)
        if a.video:
            try:
                render_trace(plant,trace,out/'mujoco_gallop.mp4');report['video']=str(out/'mujoco_gallop.mp4')
            except Exception as error:
                report['video_error']=str(error)
            write(out/'report.json',report)
        status.update(status='completed',completed_seconds=n*plant.control_dt,terminated=bool(plant.fault),
                      termination_reason=plant.fault,policy_state_unchanged=True,elapsed_seconds=time.perf_counter()-started)
        write(out/'status.json',status)
        print(json.dumps(dict(event='COMPLETED',**status,urdf_position_pass=hardware['all_positions_within_urdf'],
              urdf_velocity_pass=hardware['all_velocities_within_urdf'],urdf_torque_pass=hardware['all_applied_torques_within_urdf'])),flush=True)
        if viewer and a.replay_seconds:
            print(json.dumps(dict(event='RECORDED_REPLAY',wall_seconds=a.replay_seconds,new_physics_steps=0)),flush=True)
            deadline=time.perf_counter()+a.replay_seconds;frame=0
            while viewer.is_running() and time.perf_counter()<deadline:
                plant.data.qpos[:]=trace['mujoco_qpos'][frame%n,0]
                mj.mj_forward(plant.model,plant.data)
                viewer.cam.lookat[:]=plant.data.qpos[:3];viewer.sync()
                frame+=1;time.sleep(plant.control_dt)
    except Exception as error:
        status.update(status='error',error=str(error));write(out/'status.json',status)
        raise
    finally:
        if viewer:
            viewer.close()


if __name__=='__main__':
    main()
