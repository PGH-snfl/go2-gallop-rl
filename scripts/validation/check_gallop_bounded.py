"""Verify bounded migration with saved Go2 data; no optimization or physics."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import torch

from gpd.paths import ROOT,new_run
from isaac_task.gallop_core import GallopSettings,bounded_joint_targets,bounded_startup_targets,bounded_limit_cost
from isaac_task.gallop_adaptation import inverse_bounded_targets,legacy_student_targets,legacy_distillation_targets
from isaac_task.gallop_runtime import checkpoint_source_directory
from scripts.analysis.collect_unitree_baseline import load_yaml


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/gallop_bounded.json')
    parser.add_argument('--trace',type=Path,default=ROOT/'runs/isaac/play/unitree_go2_gallop/20261103_091200_02fe98/gallop_trace.npz')
    args=parser.parse_args();torch.set_num_threads(1)
    checkpoint=args.checkpoint.resolve();before=sha(checkpoint)
    manifest=json.loads((checkpoint.parent/'manifest.json').read_text());cfg=GallopSettings.load(args.config)
    source=checkpoint_source_directory(checkpoint,manifest)
    asset=json.loads((ROOT/'assets/go2_isaac_contract.json').read_text())
    robot=load_yaml(checkpoint.parent/'params/env.yaml')['scene']['robot']
    native=asset['native_joint_order'];hard={j['name']:np.deg2rad([j['lower'],j['upper']]) for j in asset['joints']}
    ranges=torch.tensor(np.array([hard[n] for n in native]),dtype=torch.float32)
    mid=ranges.mean(-1);radius=(ranges[:,1]-ranges[:,0])*.5*robot['soft_joint_pos_limit_factor']
    limits=torch.stack((mid-radius,mid+radius),dim=-1)
    default=torch.tensor([next(v for pattern,v in robot['init_state']['joint_pos'].items() if re.fullmatch(pattern,n)) for n in native])
    calves=torch.tensor([i for i,n in enumerate(native) if '_calf_' in n])
    allowed={k for k in cfg.to_dict() if k.startswith('bounded_')}
    checks=dict(frozen_parent_source_and_teacher_verified=source.is_dir(),bounded_profile=cfg.bounded_actions_enabled,
        stabilization_parent=manifest['interface']=='unitree_go2_gallop_heading_49_64_12_v2' and manifest['gallop_settings']['stabilization_enabled'],
        same_robot_asset=asset['isaac_asset_sha256']==manifest['asset_sha256']==sha(manifest['asset_path']),
        same_gait_conditions=all(k in allowed or cfg.to_dict()[k]==v for k,v in manifest['gallop_settings'].items()),
        default_pose_inside_soft_limits=bool(((default>limits[:,0])&(default<limits[:,1])).all()))
    with np.load(args.trace) as z:
        mask=z['gallop_metric_active'][:,0]&z['valid'][:,0]
        old_action=torch.tensor(z['action'][mask,0]);executed=torch.tensor(z['executed_action'][mask,0])
        desired=legacy_student_targets(old_action,default,limits,calves,torch.ones(len(old_action)),cfg)
        old_physical=(default+.5*executed).clamp(min=limits[:,0],max=limits[:,1])
        checks['teacher_target_matches_saved_executed_command']=bool(torch.allclose(desired,old_physical,atol=1e-6))
    latent=inverse_bounded_targets(desired,limits);mapped=bounded_joint_targets(latent,limits)
    difference=(mapped-old_physical).abs()
    near_cost,near=bounded_limit_cost(mapped,limits,calves,cfg.bounded_limit_margin)
    checks['finite_inverse_labels_at_clipped_boundaries']=bool(torch.isfinite(latent).all())
    checks['inverse_mapping_reconstructs_parent_within_0p003_rad']=bool(difference.max()<.003)
    checks['mapping_does_not_hide_parent_boundary_dwell']=bool(near.mean()>.2 and near_cost.mean()>0)
    huge=torch.linspace(-1e6,1e6,12000).reshape(1000,12)
    q=bounded_joint_targets(huge,limits)
    checks['extreme_outputs_inside_all_joint_ranges']=bool(torch.isfinite(q).all() and (q>=limits[:,0]-1e-6).all() and (q<=limits[:,1]+1e-6).all())
    mix=torch.linspace(0,1,len(latent));stages=torch.full((len(latent),),2)
    walking=torch.ones_like(latent)*100
    target=bounded_startup_targets(latent,walking,default,limits,stages,mix,cfg)
    checks['entire_walk_to_gallop_blend_bounded']=bool((target>=limits[:,0]-1e-6).all() and (target<=limits[:,1]+1e-6).all())
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state_dict']
    checks['parent_dimensions_49_64_12']=state['actor.0.weight'].shape[1]==49 and state['critic.0.weight'].shape[1]==64 and state['actor.6.weight'].shape[0]==12
    checks['parent_and_walking_teacher_unchanged']=sha(checkpoint)==before and sha(source/'walking_teacher.pt')==manifest['walking_teacher']['sha256']
    checks['official_checkout_clean']=not subprocess.run(['git','-C',manifest['official_repo'],'status','--porcelain'],check=True,capture_output=True,text=True).stdout
    run=new_run('isaac','check');out=run.parent/'unitree_go2_gallop'/run.name;out.mkdir(parents=True)
    result=dict(all_checks_passed=all(checks.values()),checks=checks,checkpoint=str(checkpoint),checkpoint_sha256=before,
        trace=str(args.trace.resolve()),trace_sha256=sha(args.trace),settings=cfg.to_dict(),
        saved_target_reconstruction=dict(samples=len(latent),max_error_rad=float(difference.max()),rmse_rad=float(difference.square().mean().sqrt()),
            near_limit_fraction=float(near.mean()),note='Formula-only reconstruction; no new actor has been trained'),
        policy_updates=0,supervised_updates=0,physics_rollouts=0,cuda_available=torch.cuda.is_available(),
        limitation='Offline validation only. User-run imitation and Isaac student preservation check still required.',
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'isaac_task/gallop_core.py',ROOT/'isaac_task/gallop_env.py',
            ROOT/'isaac_task/gallop_adaptation.py',ROOT/'scripts/training/gallop_baseline.py',args.config.resolve()]})
    (out/'bounded_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),checks=len(checks),all_checks_passed=result['all_checks_passed'],
        failed=[k for k,v in checks.items() if not v])))
    if not result['all_checks_passed']:raise SystemExit(1)


if __name__=='__main__':main()
