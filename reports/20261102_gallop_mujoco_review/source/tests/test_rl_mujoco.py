"""Named transfer contract, COM frames, raw actions and motor torque bounds."""
import tempfile
from pathlib import Path
import unittest

import mujoco as mj
import numpy as np
import torch

from gpd.paths import ROOT
from gpd.rl_mujoco import GallopMuJoCo,unitree_torque


class MuJoCoPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.plant=GallopMuJoCo(ROOT/'runs/isaac/train/unitree_go2_gallop/20261102_133218_92ee54/model_final.pt',
            ROOT.parents[1]/'unitree/unitree_mujoco/unitree_robots/go2/go2.xml',Path(cls.temp.name))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_native_joint_and_motor_maps_are_named_not_sdk_order(self):
        p=self.plant
        self.assertEqual(p.native[:4],['FL_hip_joint','FR_hip_joint','RL_hip_joint','RR_hip_joint'])
        self.assertEqual(p.act[:4],[3,0,9,6])
        for j,a in enumerate(p.act):
            joint=p.model.actuator_trnid[a,0]
            self.assertEqual(p.model.joint(joint).name,p.native[j])

    def test_root_com_velocity_and_angular_frame_match_mujoco_jacobians(self):
        p=self.plant
        oldq=p.data.qpos.copy();oldv=p.data.qvel.copy()
        p.data.qpos[3:7]=[.8,.2,.3,.4];p.data.qpos[3:7]/=np.linalg.norm(p.data.qpos[3:7])
        p.data.qvel[:]=np.linspace(-.3,.5,p.model.nv)
        mj.mj_forward(p.model,p.data);p.sync()
        linear=np.zeros((3,p.model.nv));angular=linear.copy()
        mj.mj_jacBodyCom(p.model,p.data,linear,angular,p.base)
        R=p.data.xmat[p.base].reshape(3,3)
        np.testing.assert_allclose(p.tensor.root_lin_vel_w[0],linear@p.data.qvel,atol=1e-7)
        np.testing.assert_allclose(p.tensor.root_ang_vel_b[0],R.T@angular@p.data.qvel,atol=1e-7)
        p.data.qpos[:]=oldq;p.data.qvel[:]=oldv;mj.mj_forward(p.model,p.data);p.sync()

    def test_observation_layout_preserves_previous_actions_above_one(self):
        p=self.plant
        p.last_action=np.linspace(-3,3,12).astype(np.float32)
        obs=p.observation().numpy()[0]
        self.assertEqual(obs.shape,(47,))
        np.testing.assert_allclose(obs[33:45],p.last_action)
        np.testing.assert_allclose(obs[9:21],p.data.qpos[p.qadr]-p.default,atol=1e-7)
        np.testing.assert_allclose(obs[3:6],[0,0,-1],atol=1e-7)
        np.testing.assert_allclose(obs[45:],[0,0],atol=1e-7)
        p.last_action[:]=0

    def test_torque_speed_envelope_uses_saved_forward_and_braking_limits(self):
        motor=self.plant.motor
        dq=np.array([0,13.5,21.75,30,40.]);q=np.zeros(5)
        forward=(100+.5*dq)/25
        braking=(-100+.5*dq)/25
        np.testing.assert_allclose(unitree_torque(forward,q,dq,motor),[23.4,20.2,10.1,0,0])
        np.testing.assert_allclose(unitree_torque(braking,q,dq,motor),[-23.4,-23.4,-11.7,0,0])

    def test_initial_hold_executes_default_pose_without_policy_updates(self):
        p=self.plant
        sample=p.step()
        np.testing.assert_allclose(sample['executed_action'],0,atol=1e-7)
        np.testing.assert_allclose(p.target,p.default,atol=1e-7)
        self.assertEqual(p.state.leap.physics_samples.item(),4.)
        self.assertTrue(p.unchanged())
        with self.assertRaises(RuntimeError):p.runner.alg.update()


if __name__=='__main__':unittest.main()
