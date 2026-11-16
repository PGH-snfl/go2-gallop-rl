"""Collect frozen Go2 EDU validation runs and verify original artifacts unchanged."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    a=p.parse_args();out=a.report.resolve()
    summary=read(out/'summary.json');train=Path(summary['train']);play=Path(summary['play'])
    original=read(out/'evidence_manifest.json')
    root=Path(__file__).resolve().parents[2]
    changed=[name for name,record in original.items() if sha(root/name)!=record['sha256']]
    if changed:
        raise ValueError('Original artifacts changed: '+str(changed))
    cfg=summary['provenance']['gallop_settings'];contract=summary['provenance']['guarded_runtime_joint_contract']
    names=contract['joint_names'];physical=np.array(contract['physics_limits'])
    rows=[];reports=[];traces=[];videos=[]
    import imageio.v2 as imageio
    for folder in a.runs:
        status=read(folder/'status.json');r=read(folder/'report.json')
        if status['status']!='completed' or not r['policy_state_unchanged']:
            raise ValueError('Incomplete evaluation '+str(folder))
        if r['checkpoint_sha256']!=summary['checkpoint_sha256']:
            raise ValueError('Checkpoint differs from collected Isaac play')
        with np.load(folder/'physics_trace.npz') as z:
            data={key:z[key] for key in z.files}
        with np.load(folder/'gallop_trace.npz') as z:
            trace={key:z[key] for key in z.files}
        excess=np.maximum(physical[:,0]-data['joint_pos'],0)+np.maximum(data['joint_pos']-physical[:,1],0)
        masked=excess>cfg['guarded_actual_tolerance']
        exceed=[]
        for i,name in enumerate(names):
            indices=np.flatnonzero(masked[:,i])
            if len(indices):
                exceed.append(dict(joint=name,max_project_guard_excess_rad=float(excess[:,i].max()),
                    first_exceed_seconds=float(data['time'][indices[0]]),
                    total_exceed_seconds=float(len(indices)*r['hardware']['physics_dt'])))
        hip=[i for i,n in enumerate(names) if '_hip_' in n]
        brochure_hip_max=float(np.max(np.abs(data['joint_pos'][:,hip])))
        row=dict(run=str(folder.resolve()),profile=r['profile'],physics_dt=r['hardware']['physics_dt'],
            seconds=r['completed_seconds'],termination_reason=r['termination_reason'],
            startup_completed=bool(r['episode']['startup_completed']),
            formal_gallop_metric_seconds=r['episode']['gallop_metric_seconds'],
            urdf_position_pass=r['hardware']['all_positions_within_urdf'],
            urdf_target_position_pass=r['hardware']['all_targets_within_urdf'],
            urdf_velocity_pass=r['hardware']['all_velocities_within_urdf'],
            urdf_torque_pass=r['hardware']['all_applied_torques_within_urdf'],
            max_abs_calf_velocity_rad_s=max(j['max_abs_velocity_rad_s'] for j in r['hardware']['per_joint'] if '_calf_' in j['joint']),
            maximum_actual_hip_degrees=float(np.rad2deg(brochure_hip_max)),
            brochure_hip48degree_screen_pass=bool(brochure_hip_max<=np.deg2rad(48)),
            peak_foot_resultant_n=r['hardware']['peak_foot_resultant_n'],
            peak_vertical_force_bw=r['hardware']['peak_sum_vertical_force_bw'],
            guarded_physics_excess_diagnostics=exceed,
            end_mix=float(trace['startup_gallop_mix'][-1,0]) if 'startup_gallop_mix' in trace else None,
            end_command_vx=float(trace['command'][-1,0,0]),end_actual_vx=float(trace['body_velocity'][-1,0,0]),
            inference_ms_quantiles=r['timing']['inference_ms_quantiles'])
        if 'video' not in r or 'video_error' in r:
            raise ValueError('Required recording missing: '+str(folder))
        video=Path(r['video'])
        with imageio.get_reader(video) as reader:
            meta=reader.get_meta_data();frames=reader.count_frames()
            reader.get_data(0);reader.get_data(frames-1)
        videos.append(dict(path=str(video.resolve()),duration_seconds=meta['duration'],fps=meta['fps'],frames=frames,
                           size=meta['size'],sha256=sha(video)))
        rows.append(row);reports.append(r);traces.append(data)
    identical=[]
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            if rows[i]['profile']==rows[j]['profile'] and rows[i]['physics_dt']==rows[j]['physics_dt']:
                identical.append(dict(first=rows[i]['run'],second=rows[j]['run'],
                    same_positions=np.array_equal(traces[i]['joint_pos'],traces[j]['joint_pos']),
                    same_velocities=np.array_equal(traces[i]['joint_velocity'],traces[j]['joint_velocity']),
                    maximum_position_difference_rad=float(np.abs(traces[i]['joint_pos']-traces[j]['joint_pos']).max()),
                    maximum_velocity_difference_rad_s=float(np.abs(traces[i]['joint_velocity']-traces[j]['joint_velocity']).max()),
                    equivalent_within_1e8_tolerance=bool(np.allclose(traces[i]['joint_pos'],traces[j]['joint_pos'],atol=1e-8,rtol=0)
                        and np.allclose(traces[i]['joint_velocity'],traces[j]['joint_velocity'],atol=1e-8,rtol=0))))
    # Generic Go2 software model is not an EDU firmware calibration. Also screen
    # the manufacturer's narrower public hip range separately from the URDF.
    with np.load(play/'gallop_trace.npz') as z:
        isaac={k:z[k] for k in z.files}
    canonical=[f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
    q=isaac['joint_pos'][:,0].reshape(-1,12)[:,[canonical.index(n) for n in names]]
    isaac_hip_max=float(np.max(np.abs(q[:,[i for i,n in enumerate(names) if '_hip_' in n]])))
    action=isaac['action'][:,0]
    export_audit=dict(generic_export_file='isaac_task/export.py',generic_export_clamps_raw_actor_to_minus1_plus1=True,
        gallop_cli_supports_export=False,
        actual_saved_action_abs_max=float(np.abs(action).max()),
        actual_saved_action_elements_outside_minus1_plus1_fraction=float((np.abs(action)>1).mean()),
        conclusion='Generic clamped GPD exporter is incompatible with this unbounded49-to12 Gallop action contract; do not use it.')
    verdict=dict(sim2sim_pass=False,physical_deployment_accepted=False,policy_updates=0,physical_hardware_commands=0,
        runs=rows,videos=videos,repeated_nominal_gui_determinism=identical,
        isaac_saved_hip_abs_max_degrees=float(np.rad2deg(isaac_hip_max)),
        isaac_saved_hip48degree_screen_pass=bool(isaac_hip_max<=np.deg2rad(48)),
        isaac_actual_joint_velocity_validation='UNKNOWN: saved Isaac trace has no joint velocity/torque samples; target rate is not actual velocity.',
        export_audit=export_audit,
        unresolved_manufacturer_limits=['EDU serial/firmware operating limits','continuous/RMS motor torque','thermal/current/battery limits',
            'admissible foot impact or structural load','joint-specific identified torque-speed envelope'],
        deployment_blockers=[
            'No successful full Gallop handover in either MuJoCo validation profile.',
            'Actual calf velocity exceeds15.7rad/s official URDF software bound in trained-profile MuJoCo.',
            'Saved startup supervisor uses simulator-perfect body linear velocity and height; no verified deployed estimator connected.',
            'Existing five-action GPD DDS controller is not the49-observation/12-joint Gallop supervisor/filter/teacher runtime.',
            'No parity-verified Gallop export and official C++/DDS deployment integration.',
            'No measured end-to-end state age, DDS jitter/deadline or hardware receiver-timeout tests for this policy.',
            'Official peak torque/URDF bounds do not certify continuous duty or contact impacts.',
        ])
    write(out/'edu_sim2sim_verdict.json',verdict)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(rows),2,figsize=(14,3*len(rows)),squeeze=False,layout='constrained')
    for row,r,data,axs in zip(rows,reports,traces,axes):
        t=data['time'];calf=[i for i,n in enumerate(names) if '_calf_' in n]
        for i in calf:
            axs[0].plot(t,data['joint_velocity'][:,i],label=names[i].split('_')[0],lw=.7)
        for limit in (-15.7,15.7):axs[0].axhline(limit,color='r',ls='--',lw=1)
        axs[0].set_ylabel('Actual calf velocity (rad/s)');axs[0].legend(ncol=4)
        axs[1].plot(t,data['velocity'][:,0],label='body vx',lw=.8)
        axs[1].set_ylabel('Body vx (m/s)')
        axs[0].set_title(f"{row['profile']} dt={row['physics_dt']}s; stopped:{row['termination_reason']}")
        for ax in axs:ax.grid(alpha=.2);ax.set_xlabel('Simulation seconds')
    fig.savefig(out/'mujoco_velocity_limits.png',dpi=150);plt.close(fig)
    # Show one actual student-phase cycle from the Isaac recording, not startup.
    video=play/'videos/gallop-play-step-0.mp4'
    with imageio.get_reader(video) as reader:
        fps=reader.get_meta_data()['fps'];fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
        for ax,seconds in zip(axes.flat,np.linspace(12.,12.4,6)):
            ax.imshow(reader.get_data(round(seconds*fps)));ax.set_title(f'Isaac {seconds:.2f}s');ax.axis('off')
        fig.savefig(out/'isaac_student_keyframes.png',dpi=130);plt.close(fig)
    checks=dict(original_train_play_files_verified=len(original),original_hashes_unchanged=not changed,
        checkpoint_sha256=summary['checkpoint_sha256'],completed_validation_runs=len(rows),recordings_readable=len(videos),
        no_policy_updates=all(r['policy_updates']==0 for r in reports),
        policies_unchanged=all(r['policy_state_unchanged'] for r in reports),
        every_run_same_checkpoint=all(r['checkpoint_sha256']==summary['checkpoint_sha256'] for r in reports))
    write(out/'verification.json',checks)
    print(json.dumps(dict(checks=checks,runs=rows),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
