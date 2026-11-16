"""Check posture fine-tuning compatibility and kinematic reachability without PPO/physics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from gpd.paths import ROOT,new_run
from gpd.portable import external_path, resolve_go2_asset


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/gallop_stabilization.json')
    args=parser.parse_args()
    import mujoco as mj
    import torch
    from tensordict import TensorDict
    from rsl_rl.runners import OnPolicyRunner
    from isaac_task.gallop_core import GallopSettings,restore_gallop_policy
    from isaac_task.gallop_runtime import checkpoint_source_directory
    from scripts.analysis.collect_unitree_baseline import load_yaml

    torch.set_num_threads(1)
    run=new_run('isaac','check');out=run.parent/'unitree_go2_gallop'/run.name
    out.mkdir(parents=True,exist_ok=False)
    checkpoint=args.checkpoint.resolve();before=sha(checkpoint)
    manifest=json.loads((checkpoint.parent/'manifest.json').read_text())
    source=checkpoint_source_directory(checkpoint,manifest)
    settings=GallopSettings.load(args.config)
    if not settings.posture_enabled:raise ValueError('Select the new posture objective')
    actor_dim,critic_dim=(49,64) if settings.heading_motion_enabled else (47,62)
    class Probe:
        num_envs=2;num_actions=12
        def get_observations(self):return TensorDict(dict(policy=torch.zeros(2,actor_dim),critic=torch.zeros(2,critic_dim)),batch_size=[2])
    runner=OnPolicyRunner(Probe(),load_yaml(checkpoint.parent/'params/agent.yaml'),log_dir=None,device='cpu')
    def forbid(*a,**kw):raise RuntimeError('PPO/learning is forbidden in offline validation')
    runner.learn=forbid;runner.alg.update=forbid
    restored=restore_gallop_policy(runner,checkpoint)
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state_dict']
    restored_weights=runner.alg.policy.state_dict()
    exact=all(torch.equal(v if v.shape==weights[k].shape else v[:,:weights[k].shape[1]],weights[k]) for k,v in restored_weights.items())
    with torch.no_grad():
        obs=Probe().get_observations();obs['policy'].normal_(std=.1)
        action=runner.get_inference_policy(device='cpu')(obs)
        expected=obs['policy'][:,:weights['actor.0.weight'].shape[1]]
        layers=sorted(int(k.split('.')[1]) for k in weights if k.startswith('actor.') and k.endswith('.weight'))
        for i,index in enumerate(layers):
            expected=torch.nn.functional.linear(expected,weights[f'actor.{index}.weight'],weights[f'actor.{index}.bias'])
            if i<len(layers)-1:expected=torch.nn.functional.elu(expected)
    checks=dict(actor_critic_noise_exact=exact,fresh_optimizer=not runner.alg.optimizer.state,
        iteration_starts_at_zero=runner.current_learning_iteration==0,finite_12_joint_action=action.shape==(2,12) and bool(torch.isfinite(action).all()),
        unchanged_action_scale=manifest['action_scale']==settings.action_scale==.5,
        frozen_core_env_teacher_verified=source.is_dir(),zero_pitch_reference=settings.pitch_amplitude==0.)
    checks['initial_actor_output_matches_parent']=bool(torch.allclose(action,expected,atol=1e-6,rtol=1e-6))
    if settings.heading_motion_enabled and weights['actor.0.weight'].shape[1]==47:
        checks['new_actor_critic_heading_columns_zero']=bool((restored_weights['actor.0.weight'][:,-2:]==0).all() and (restored_weights['critic.0.weight'][:,-2:]==0).all())
    # Forward kinematics only: no mj_step and no physics rollout. The target
    # width is an engineering setting, not a measured chassis exterior width.
    model_path=external_path('unitree_mujoco')/'unitree_robots/go2/go2.xml'
    model=mj.MjModel.from_xml_path(str(model_path));data=mj.MjData(model)
    data.qpos[:3]=[0,0,.4];data.qpos[3:7]=[1,0,0,0]
    for leg in ('FL','FR','RL','RR'):
        for joint,value in [('hip',.1 if leg.endswith('L') else -.1),('thigh',.8 if leg.startswith('F') else 1.),('calf',-1.5)]:
            data.qpos[model.jnt_qposadr[model.joint(f'{leg}_{joint}_joint').id]]=value
    mj.mj_forward(model,data)
    nominal=[data.xpos[model.body(f'{l}_foot').id,1]-data.xpos[model.body(f'{r}_foot').id,1] for l,r in [('FL','FR'),('RL','RR')]]
    solved={}
    for leg in ('FL','FR','RL','RR'):
        target=(settings.front_stance_width if leg.startswith('F') else settings.hind_stance_width)/2
        side=1 if leg.endswith('L') else -1
        joint=model.joint(f'{leg}_hip_joint').id;address=model.jnt_qposadr[joint]
        lower,upper=-.25,.25
        for _ in range(35):
            angle=(lower+upper)/2;data.qpos[address]=side*angle;mj.mj_forward(model,data)
            y=side*data.xpos[model.body(f'{leg}_foot').id,1]
            if y<target:lower=angle
            else:upper=angle
        value=float(data.qpos[address]);lo,hi=model.jnt_range[joint]
        midpoint=(lo+hi)/2;radius=(hi-lo)*.9/2
        solved[leg]=dict(hip_rad=value,foot_y_m=float(data.xpos[model.body(f'{leg}_foot').id,1]),
            inside_soft_limits=bool(midpoint-radius<=value<=midpoint+radius),lateral_error_m=float(abs(y-target)))
    checks['target_foot_width_reachable_in_nominal_kinematics']=all(v['inside_soft_limits'] and v['lateral_error_m']<1e-6 for v in solved.values())
    checks['checkpoint_and_teacher_unchanged']=sha(checkpoint)==before and sha(source/'walking_teacher.pt')==manifest['walking_teacher']['sha256']
    checks['official_checkout_clean']=subprocess.run(['git','-C',manifest['official_repo'],'status','--porcelain'],capture_output=True,text=True,check=True).stdout==''
    original=ET.parse(model_path).getroot()
    hip=original.find('.//body[@name="FL_hip"]');thigh=original.find('.//body[@name="FL_thigh"]')
    lateral_axis_width=2*(float(hip.get('pos').split()[1])+float(thigh.get('pos').split()[1]))
    if settings.heading_motion_enabled:
        shoulder=json.loads((ROOT/'assets/go2_shoulder_widths.json').read_text())
        checks['training_usd_shoulder_widths_verified']=shoulder['asset_sha256']==manifest['asset_sha256'] and all(
            abs(shoulder[which+'_shoulder_width_m']-getattr(settings,which+'_shoulder_width'))<1e-6 for which in ('front','hind'))
        checks['independent_targets_below_shoulders']=all(getattr(settings,which+'_stance_width')<getattr(settings,which+'_shoulder_width') for which in ('front','hind'))
    result=dict(all_checks_passed=all(checks.values()),checks=checks,checkpoint=str(checkpoint),checkpoint_sha256=before,
        settings=settings.to_dict(),source_restore=restored,policy_updates=0,physics_rollouts=0,
        cuda_available=torch.cuda.is_available(),physics_limitation='Offline checks only; new Isaac dynamics not validated',
        geometry=dict(model=str(model_path),model_sha256=sha(model_path),leg_axis_width_m=lateral_axis_width,
            default_pose_front_hind_width_m=nominal,kinematic_target_width_pose=solved,
            note='MuJoCo model forward kinematics only, not a PhysX or locomotion stability test'),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'isaac_task/gallop_core.py',ROOT/'isaac_task/gallop_env.py',
            ROOT/'scripts/training/gallop_baseline.py',args.config.resolve()]})
    (out/'posture_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),all_checks_passed=result['all_checks_passed'],checks=len(checks))))
    if not result['all_checks_passed']:raise SystemExit(1)


if __name__=='__main__':main()
