"""Exercise production GallopState with synthetic tensors, without Isaac physics."""
import ast
from dataclasses import replace
import json
import math
from types import SimpleNamespace
import unittest

import torch

from gpd.paths import ROOT
from isaac_task.gallop_core import GallopSettings,PairBalanceTracker,SuccessSpeedCurriculum,forelimb_success,gait_pattern_success,pair_joint_penalty,phase_targets,LeapDistanceTracker,FlightStabilityTracker,flight_phase_targets
from isaac_task.gallop_core import WalkingToGallop,posture_terms,HeadingMotionTracker
from isaac_task.gallop_core import stabilization_leap_target
from isaac_task.gallop_core import GuardedJointController,guarded_foot_height_cost
from isaac_task.gallop_core import RelearnCurriculum,RequestLimitMonitor
from isaac_task.gallop_core import StagedGallopCurriculum,StagedEpisodeProfile


class Scene(dict):pass


def state_class():
    tree=ast.parse((ROOT/'isaac_task/gallop_env.py').read_text())
    selected=ast.Module(body=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='GallopState'],type_ignores=[])
    namespace=dict(torch=torch,math=math,json=json,LEGS=('FL','FR','RL','RR'),
                   PairBalanceTracker=PairBalanceTracker,SuccessSpeedCurriculum=SuccessSpeedCurriculum,
                   RelearnCurriculum=RelearnCurriculum,RequestLimitMonitor=RequestLimitMonitor,
                   StagedGallopCurriculum=StagedGallopCurriculum,StagedEpisodeProfile=StagedEpisodeProfile,
                   LeapDistanceTracker=LeapDistanceTracker,
                   FlightStabilityTracker=FlightStabilityTracker,flight_phase_targets=flight_phase_targets,
                   WalkingToGallop=WalkingToGallop,
                   posture_terms=posture_terms,
                   HeadingMotionTracker=HeadingMotionTracker,
                   stabilization_leap_target=stabilization_leap_target,GuardedJointController=GuardedJointController,guarded_foot_height_cost=guarded_foot_height_cost,
                   forelimb_success=forelimb_success,gait_pattern_success=gait_pattern_success,
                   pair_joint_penalty=pair_joint_penalty,phase_targets=phase_targets,
                   euler_xyz_from_quat=lambda q:(torch.zeros(len(q)),)*3)
    exec(compile(selected,'gallop_env.GallopState','exec'),namespace)
    return namespace['GallopState']


class GallopPairEnvironmentTests(unittest.TestCase):
    def test_reward_wiring_complete_episode_and_reset_with_synthetic_motion(self):
        n=3
        data=SimpleNamespace(joint_pos=torch.zeros(n,12),default_joint_pos=torch.zeros(n,12),
            soft_joint_pos_limits=torch.tensor([-10.,10.]).expand(n,12,2),
            root_lin_vel_b=torch.tensor([[2.,0.,0.]]).expand(n,3),root_ang_vel_b=torch.zeros(n,3),
            root_quat_w=torch.tensor([[1.,0.,0.,0.]]).expand(n,4),root_pos_w=torch.tensor([[0.,0.,.35]]).expand(n,3),
            body_pos_w=torch.zeros(n,4,3))
        robot=SimpleNamespace(data=data,find_bodies=lambda *a,**k:(list(range(4)),None),find_joints=lambda *a,**k:(list(range(12)),None))
        sensor=SimpleNamespace(body_names=[leg+'_foot' for leg in ('FL','FR','RL','RR')],data=SimpleNamespace(net_forces_w=torch.zeros(n,4,3)))
        scene=Scene(robot=robot,contact_forces=sensor);scene.env_origins=torch.zeros(n,3)
        env=SimpleNamespace(scene=scene,num_envs=n,device='cpu',step_dt=.02,max_episode_length=1000,common_step_counter=0,
            reset_terminated=torch.zeros(n,dtype=torch.bool),reset_time_outs=torch.ones(n,dtype=torch.bool),
            command_manager=SimpleNamespace(get_command=lambda name:torch.tensor([[2.,0.,0.]]).expand(n,3)))
        settings=replace(GallopSettings(),start_speed=2.,initial_gait_weight=1.)
        state=state_class()(env,settings,None,None,training_targets=False,curriculum_enabled=False)
        for step in range(1,1001):
            env.common_step_counter=step;state.begin_step()
            target=phase_targets(state.phase,state.command[:,0],settings)
            q=torch.stack((torch.zeros_like(target['thigh']),target['thigh'],target['calf']),dim=2)
            q[1,1]=0.;q[2,3]=0.  # Frozen right fore / right hind leg.
            data.joint_pos=q.reshape(n,12)
            data.body_pos_w[:,:,2]=target['foot_height']+settings.foot_radius
            sensor.data.net_forces_w[:,:,2]=target['stance'].float()*100.
            sensor.data.net_forces_w[1,1,2]=0.;sensor.data.net_forces_w[2,3,2]=0.
            state.update_measurements()
            for key in ('joint','rom','load','stance'):
                value=state.components['pair_'+key]
                self.assertTrue(torch.isfinite(value).all());self.assertTrue((value<=0).all())
        rows=[]
        for i in range(n):state.complete_episodes([i]);rows.append(state.last_episode.copy())
        for key in ('joint','rom','load','stance'):
            self.assertAlmostEqual(rows[0]['pair_'+key+'_cost'],0.,places=6)
            self.assertGreater(rows[1]['pair_'+key+'_cost'],0.)
            self.assertGreater(rows[2]['pair_'+key+'_cost'],0.)
        self.assertGreater(rows[0]['pair_valid_cycles'],40)
        self.assertEqual(rows[0]['action_clip_fraction'],0.)
        self.assertEqual(rows[0]['timed_out'],1.)
        self.assertIn('pair_load_asymmetry',state.snapshot())
        before=state.pair.valid_cycles[0].clone();state.reset(torch.tensor([1]))
        self.assertEqual(state.pair.valid_cycles[1].item(),0.)
        self.assertEqual(state.pair.valid_cycles[0].item(),before.item())


if __name__=='__main__':unittest.main()
