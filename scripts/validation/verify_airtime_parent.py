"""CPU tensor-restore check for the66k parent; no RSL runner, physics or learning."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import GallopSettings, restore_gallop_policy


class ShapeProbe(torch.nn.Module):
    """Fixed49/64 ->512/256/128 ELU networks; not a simulated environment."""
    def __init__(self,log_std=False):
        super().__init__()
        def net(dim,output):
            return torch.nn.Sequential(torch.nn.Linear(dim,512),torch.nn.ELU(),
                torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),
                torch.nn.ELU(),torch.nn.Linear(128,output))
        self.actor=net(49,12);self.critic=net(64,1)
        self.register_parameter('log_std' if log_std else 'std',torch.nn.Parameter(torch.ones(12)))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/gallop_2mps_50cm_airtime.json')
    parser.add_argument('--expected-sha',default='ed2ac0758317a294574d1d5a3d6c25e6e076472a64372ad4f5d384eb573a0afe')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();before=sha(args.checkpoint)
    if before!=args.expected_sha:raise ValueError('Unexpected parent checkpoint')
    parent=json.loads((args.checkpoint.parent/'manifest.json').read_text())
    old=GallopSettings(**parent['gallop_settings']);new=GallopSettings.load(args.config)
    protected=[key for key in old.to_dict() if key.startswith(('guarded_','edu_','velocity_brake_','startup_','flight_max_'))]
    protected+=['start_speed','max_speed','speed_band','leap_min_distance','flight_min_seconds',
                'flight_recovery_seconds','action_scale','min_height','max_height','fixed_goal_enabled']
    for key in protected:
        if getattr(old,key)!=getattr(new,key):raise ValueError('Protected field changed: '+key)
    data=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    saved=data['model_state_dict'];probe=ShapeProbe(log_std='log_std' in saved)
    runner=NS(device='cpu',alg=NS(policy=probe,optimizer=torch.optim.Adam(probe.parameters(),lr=1e-4)),
              current_learning_iteration=-1)
    restored=restore_gallop_policy(runner,args.checkpoint)
    for key,value in probe.state_dict().items():torch.testing.assert_close(value,saved[key],rtol=0,atol=0)
    assert not runner.alg.optimizer.state and runner.current_learning_iteration==0
    with torch.inference_mode():
        action=probe.actor(torch.zeros(4,49));value=probe.critic(torch.zeros(4,64))
    assert action.shape==(4,12) and value.shape==(4,1)
    assert torch.isfinite(action).all() and torch.isfinite(value).all()
    assert sha(args.checkpoint)==before
    result={'checkpoint_sha256':before,'config_sha256':sha(args.config),'parent_cumulative_updates':parent['cumulative_iterations'],
            'production_restore_function':restored,'all_restored_tensors_equal':True,'protected_settings_checked':len(protected),
            'actor_probe_shape':list(action.shape),'critic_probe_shape':list(value.shape),'probe_outputs_finite':True,
            'optimizer_updates':0,'physics_steps':0,'source_checkpoint_unchanged':True,
            'scope':'Production tensor restore function against a CPU shape-compatible probe, not an actual RSL/Isaac runtime check. weights_only=True used for the user-provided checkpoint.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
