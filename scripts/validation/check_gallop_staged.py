"""Inspect a frozen Isaac check and test stage-specific replay without PPO."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from gpd.paths import ROOT,new_run
from gpd.portable import external_path, resolve_go2_asset
from isaac_task.gallop_runtime import staged_checkpoint_progress


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--isaac-check',type=Path,required=True)
    a=p.parse_args();source=a.isaac_check.resolve();manifest=json.loads((source/'manifest.json').read_text())
    status=json.loads((source/'status.json').read_text());assert status['status']=='completed' and status['policy_updates']==0 and status['policy_state_unchanged']
    run=new_run('isaac','check');out=run.parent/'unitree_go2_gallop'/run.name;out.mkdir(parents=True)
    parent=Path(manifest['source_checkpoint']['path']);parent_hash=sha(parent)
    checks={'isaac_completed_without_policy_updates':True}
    with np.load(source/'gallop_trace.npz') as z:
        limits=np.array(manifest['guarded_runtime_joint_contract']['target_limits'])
        checks['isaac_first_stage_only']=bool((z['staged_index']==0).all())
        checks['isaac_active_distance18cm']=bool(np.allclose(z['stabilization_leap_target'],.18))
        checks['isaac_active_air60ms']=bool(np.allclose(z['staged_flight_min_seconds'],.06))
        checks['isaac_target_angles_within_limits']=bool(((z['guarded_target']>=limits[:,0]-1e-6)&(z['guarded_target']<=limits[:,1]+1e-6)).all())
        rates=np.array([6. if '_hip_' in n else 16. for n in manifest['guarded_runtime_joint_contract']['joint_names']])
        checks['isaac_target_rates_within_limits']=bool((np.abs(z['guarded_target_rate'])<=rates+1e-4).all())
    episodes=[json.loads(l) for l in (source/'episodes.jsonl').read_text().splitlines() if l.strip()]
    first=[next(r for r in episodes if r['env']==i) for i in range(status['num_envs'])]
    actual=dict(episodes=len(first),startup_completed=sum(bool(r['startup_completed']) for r in first),
        terminated=sum(bool(r['terminated']) for r in first),full_stage_successes=sum(bool(r['success']) for r in first),
        final80cm_successes=sum(bool(r['staged_final_success']) for r in first))
    from gpd.rl_mujoco import GallopMuJoCo
    model=external_path('unitree_mujoco')/'unitree_robots/go2/go2.xml'
    fixtures=[]
    for index in (0,5,8):
        fixture=out/'fixtures'/f'stage{index}';fixture.mkdir(parents=True)
        shutil.copytree(source/'source',fixture/'source');shutil.copytree(source/'params',fixture/'params')
        checkpoint=fixture/'frozen_fixture.pt';shutil.copy2(parent,checkpoint)
        m=json.loads(json.dumps(manifest));goal=m['staged_design']['plan'][index]
        m['validation_fixture']=dict(trained=False,policy_updates=0,restored_checkpoint=str(parent),purpose='Replay contract only; not a trained stage policy')
        m['trained_staged_index']=index;m['trained_speed_cap']=goal['speed']
        m['staged_checkpoints']={checkpoint.name:dict(index=index,goal=goal,highest_confirmed_stage=None,
            sha256=sha(checkpoint),stage_iterations=0,cumulative_iterations=30500)}
        (fixture/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
        progress=staged_checkpoint_progress(checkpoint,m);assert progress['index']==index
        sim=fixture/'mujoco_adapter';sim.mkdir()
        plant=GallopMuJoCo(checkpoint,model,sim)
        checks[f'stage{index}_saved_goal_restored']=plant.state.curriculum.index==index and abs(float(plant.state.leap.distance_target[0])-goal['leap'])<1e-6
        checks[f'stage{index}_air_threshold_restored']=abs(float(plant.state.profile.flight_min_seconds[0])-goal['flight_min_seconds'])<1e-6
        checks[f'stage{index}_replay_frozen']=not plant.state.curriculum_enabled
        for step in range(10):
            sample=plant.step(last=step==9)
            if plant.fault:break
        checks[f'stage{index}_adapter_finite']=bool(np.isfinite(sample['position']).all() and np.isfinite(sample['guarded_target']).all())
        checks[f'stage{index}_policy_unchanged']=plant.unchanged()
        fixtures.append(str(checkpoint.relative_to(ROOT)))
    checks['parent_checkpoint_unchanged']=sha(parent)==parent_hash
    assert all(checks.values()),checks
    result=dict(status='completed',output=str(out),isaac_check=str(source),policy_updates=0,
        checks=checks,isaac_first_episodes=actual,validation_fixtures=fixtures,
        note='Fixtures restore existing500-update weights without training. Ten MuJoCo adapter steps verify wiring, not Sim2Sim performance or80cm feasibility.')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
