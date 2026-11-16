"""Named transfer contract, COM frames, raw actions and motor torque bounds."""
import tempfile
from pathlib import Path
import unittest
import json
import shutil

import mujoco as mj
import numpy as np
import torch

from gpd.paths import ROOT
from gpd.rl_mujoco import GallopMuJoCo,unitree_torque,sha


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

    def test_heading_bundle_loads_49_observations_and_preserves_the_walking_teacher(self):
        # Migrate a temporary bundle for interface checks, without optimization
        # or physics integration. It is not a newly trained policy artifact.
        from scripts.analysis.collect_unitree_baseline import load_yaml
        from isaac_task.gallop_core import GallopSettings
        parent=self.plant.checkpoint.parent
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bundle=root/'bundle';bundle.mkdir();(bundle/'params').mkdir();(bundle/'source').mkdir()
            manifest=json.loads((parent/'manifest.json').read_text())
            cfg=GallopSettings.load(ROOT/'configs/gallop_2p8mps_heading_motion.json')
            manifest.update(interface='unitree_go2_gallop_heading_49_64_12_v2',observation_dim=49,critic_observation_dim=64,gallop_settings=cfg.to_dict())
            for filename,key in [('gallop_core.py','gallop_core_sha256'),('gallop_env.py','gallop_env_sha256')]:
                shutil.copy2(ROOT/'isaac_task'/filename,bundle/'source'/filename);manifest[key]=sha(bundle/'source'/filename)
            shutil.copy2(parent/'source/walking_teacher.pt',bundle/'source/walking_teacher.pt')
            (bundle/'manifest.json').write_text(json.dumps(manifest))
            shutil.copy2(parent/'params/agent.yaml',bundle/'params/agent.yaml')
            env=load_yaml(parent/'params/env.yaml')
            env['observations']['policy']['gallop_heading']=dict(clip=None,scale=None,noise=None)
            (bundle/'params/env.yaml').write_text(json.dumps(env))
            checkpoint=torch.load(parent/'model_final.pt',map_location='cpu',weights_only=True)
            for key in ('actor.0.weight','critic.0.weight'):
                old=checkpoint['model_state_dict'][key]
                checkpoint['model_state_dict'][key]=torch.cat((old,torch.zeros(len(old),2)),dim=1)
            torch.save(checkpoint,bundle/'model_final.pt');scene=root/'scene';scene.mkdir()
            plant=GallopMuJoCo(bundle/'model_final.pt',ROOT.parents[1]/'unitree/unitree_mujoco/unitree_robots/go2/go2.xml',scene)
            plant.data.qpos[3:7]=[np.cos(.1),0,0,np.sin(.1)];mj.mj_forward(plant.model,plant.data);plant.sync()
            obs=plant.observation()
            self.assertEqual(obs.shape,(1,49));np.testing.assert_allclose(obs[0,-2:],[np.sin(-.2),np.cos(-.2)-1],atol=1e-7)
            torch.testing.assert_close(plant.state.walking_policy(obs),plant.state.walking_policy(obs[:,:47]))
            self.assertTrue(plant.unchanged());self.assertTrue(plant.settings.heading_motion_enabled)
            # The stabilization profile must also restore the achieved speed
            # and distance instead of silently replaying the initial 2.2 stage.
            stable=GallopSettings.load(ROOT/'configs/gallop_stabilization.json')
            manifest.update(gallop_settings=stable.to_dict(),trained_speed_cap=2.6,highest_confirmed_speed=2.4)
            (bundle/'manifest.json').write_text(json.dumps(manifest))
            stable_scene=root/'stable_scene';stable_scene.mkdir()
            p=GallopMuJoCo(bundle/'model_final.pt',ROOT.parents[1]/'unitree/unitree_mujoco/unitree_robots/go2/go2.xml',stable_scene)
            self.assertAlmostEqual(p.state.startup.goal_speed.item(),2.6,places=5)
            self.assertAlmostEqual(p.state.leap.distance_target.item(),.16666667,places=6)
            self.assertAlmostEqual(p.state.curriculum.highest_confirmed_speed,2.4)
            p.state.startup.stage[:]=3;p.command[:,0]=2.6
            sample=p.step()  # A single software integration step, not a gait evaluation.
            np.testing.assert_allclose(sample['executed_action'][0,p.state.calf_ids],sample['action'][0,p.state.calf_ids]*.85,atol=1e-6)
            self.assertTrue(np.isfinite(sample['stabilization_excess_cost']).all());self.assertTrue(p.unchanged())

    def test_bounded_bundle_executes_same_transform_without_policy_optimization(self):
        from isaac_task.gallop_core import GallopSettings
        parent=ROOT/'runs/isaac/train/unitree_go2_gallop/20261102_161522_fe6276'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bundle=root/'bundle';bundle.mkdir();(bundle/'params').mkdir();(bundle/'source').mkdir()
            manifest=json.loads((parent/'manifest.json').read_text());cfg=GallopSettings.load(ROOT/'configs/gallop_bounded.json')
            manifest.update(interface='unitree_go2_gallop_bounded_49_64_12_v3',gallop_settings=cfg.to_dict())
            for filename,key in [('gallop_core.py','gallop_core_sha256'),('gallop_env.py','gallop_env_sha256')]:
                shutil.copy2(ROOT/'isaac_task'/filename,bundle/'source'/filename);manifest[key]=sha(bundle/'source'/filename)
            shutil.copy2(parent/'source/walking_teacher.pt',bundle/'source/walking_teacher.pt')
            (bundle/'manifest.json').write_text(json.dumps(manifest))
            for name in ['env.yaml','agent.yaml']:shutil.copy2(parent/'params'/name,bundle/'params'/name)
            shutil.copy2(parent/'model_final.pt',bundle/'model_final.pt');scene=root/'scene';scene.mkdir()
            p=GallopMuJoCo(bundle/'model_final.pt',ROOT.parents[1]/'unitree/unitree_mujoco/unitree_robots/go2/go2.xml',scene)
            p.state.startup.stage[:]=3;p.state.startup.mix[:]=1.;p.command[:,0]=2.2
            latent=torch.linspace(-100,100,12)[None];p.actor=lambda obs:latent.clone()
            expected=p.core.bounded_joint_targets(latent,p.tensor.soft_joint_pos_limits)
            sample=p.step()  # One temporary software step, not a gait test.
            np.testing.assert_allclose(p.target,expected[0],atol=2e-7)
            np.testing.assert_allclose(sample['executed_action'][0],(p.target-p.default)/.5,atol=3e-7)
            self.assertEqual(float(sample['action_clip'][0]),0.);self.assertGreater(float(sample['bounded_near_limit_fraction'][0]),.9)
            self.assertGreaterEqual(float(sample['bounded_torque_saturation'][0]),0.)
            self.assertTrue(p.unchanged())

    def test_guarded_bundle_uses_shrunk_physics_and_rate_limited_legacy_actions(self):
        from isaac_task.gallop_core import GallopSettings
        from scripts.analysis.collect_unitree_baseline import load_yaml
        parent=ROOT/'runs/isaac/train/unitree_go2_gallop/20261102_161522_fe6276'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bundle=root/'bundle';bundle.mkdir();(bundle/'params').mkdir();(bundle/'source').mkdir()
            manifest=json.loads((parent/'manifest.json').read_text());cfg=GallopSettings.load(ROOT/'configs/gallop_guarded_2p5.json')
            manifest.update(interface='unitree_go2_gallop_guarded_49_64_12_v4',gallop_settings=cfg.to_dict(),trained_speed_cap=2.5)
            for filename,key in [('gallop_core.py','gallop_core_sha256'),('gallop_env.py','gallop_env_sha256')]:
                shutil.copy2(ROOT/'isaac_task'/filename,bundle/'source'/filename);manifest[key]=sha(bundle/'source'/filename)
            shutil.copy2(parent/'source/walking_teacher.pt',bundle/'source/walking_teacher.pt')
            (bundle/'manifest.json').write_text(json.dumps(manifest));shutil.copy2(parent/'params/agent.yaml',bundle/'params/agent.yaml')
            env=load_yaml(parent/'params/env.yaml');env['scene']['robot']['soft_joint_pos_limit_factor']=1.
            (bundle/'params/env.yaml').write_text(json.dumps(env));shutil.copy2(parent/'model_final.pt',bundle/'model_final.pt')
            scene=root/'scene';scene.mkdir()
            p=GallopMuJoCo(bundle/'model_final.pt',ROOT.parents[1]/'unitree/unitree_mujoco/unitree_robots/go2/go2.xml',scene)
            self.assertAlmostEqual(p.state.startup.goal_speed.item(),2.5);self.assertAlmostEqual(p.state.leap.distance_target.item(),.14,places=6)
            midpoint=p.original_limits.mean(-1);radius=np.ptp(p.original_limits,axis=1)*.45
            np.testing.assert_allclose(p.soft,np.column_stack((midpoint-radius,midpoint+radius)))
            joint_ids=[p.model.joint(n).id for n in p.native]
            np.testing.assert_allclose(p.model.jnt_range[joint_ids],p.soft)
            p.state.startup.stage[:]=3;p.state.startup.mix[:]=1.;p.command[:,0]=2.5
            sample=torch.ones(1,12)*100;p.actor=lambda obs:sample.clone();previous=p.default.copy()
            result=p.step()  # Temporary one-step connection check only.
            self.assertTrue(np.all(np.abs(p.target-previous)<=p.state.guard.rate.numpy()*.02+1e-6))
            np.testing.assert_allclose(result['executed_action'][0],(p.target-p.default)/.5,atol=1e-6)
            self.assertGreater(float(result['guarded_intervention'][0]),0.)
            self.assertTrue(p.unchanged())

            # A promoted guarded bundle must replay its saved speed and leap
            # goal using the same physical range and target-rate controller.
            adaptive=GallopSettings.load(ROOT/'configs/gallop_guarded_3mps_leap.json')
            manifest.update(gallop_settings=adaptive.to_dict(),trained_speed_cap=3.,highest_confirmed_speed=2.9)
            (bundle/'manifest.json').write_text(json.dumps(manifest))
            promoted_scene=root/'promoted_scene';promoted_scene.mkdir()
            promoted=GallopMuJoCo(bundle/'model_final.pt',ROOT.parents[1]/'unitree/unitree_mujoco/unitree_robots/go2/go2.xml',promoted_scene)
            self.assertAlmostEqual(promoted.state.startup.goal_speed.item(),3.)
            self.assertAlmostEqual(promoted.state.leap.distance_target.item(),.22,places=6)
            self.assertAlmostEqual(promoted.state.curriculum.highest_confirmed_speed,2.9)
            self.assertFalse(promoted.state.curriculum_enabled)
            np.testing.assert_allclose(promoted.soft,p.soft)
            promoted.state.startup.stage[:]=3;promoted.state.startup.mix[:]=1.;promoted.command[:,0]=3.
            promoted.actor=lambda obs:sample.clone();previous=promoted.default.copy()
            promoted.step()  # Software connection only; no gait-performance claim.
            self.assertTrue(np.all(np.abs(promoted.target-previous)<=promoted.state.guard.rate.numpy()*.02+1e-6))
            self.assertTrue(promoted.unchanged())

            from isaac_task.gallop_core import RelearnCurriculum
            relearn=GallopSettings.load(ROOT/'configs/gallop_relearn.json')
            plan=RelearnCurriculum(relearn).plan
            manifest.update(interface='unitree_go2_gallop_relearn_49_64_12_v5',gallop_settings=relearn.to_dict(),
                trained_relearn_level=0,trained_speed_cap=1.,highest_confirmed_speed=None,
                relearn_checkpoints={'model_final.pt':dict(level=0,speed_cap=1.,highest_confirmed_speed=None,
                    sha256=sha(bundle/'model_final.pt'),cumulative_iterations=4000)})
            (bundle/'manifest.json').write_text(json.dumps(manifest))
            walking_scene=root/'walking_scene';walking_scene.mkdir()
            walking=GallopMuJoCo(bundle/'model_final.pt',ROOT.parents[1]/'unitree/unitree_mujoco/unitree_robots/go2/go2.xml',walking_scene)
            self.assertEqual(walking.state.curriculum.index,0);self.assertFalse(walking.state.curriculum_enabled)
            self.assertEqual(float(walking.state.relearn_gait[0]),0.)
            walking.state.startup.stage[:]=3
            self.assertTrue(torch.equal(walking.observation()[0,45:47],torch.zeros(2)))
            walking.actor=lambda obs:torch.ones(1,12)*.6;walking.command[:,0]=1.
            walking.step()
            np.testing.assert_allclose((walking.state.guard.requested[0]-walking.tensor.default_joint_pos[0]).numpy(),.3,atol=1e-6)
            self.assertTrue(walking.unchanged())
            manifest['relearn_checkpoints']['model_final.pt'].update(level=len(plan)-1,speed_cap=3.,highest_confirmed_speed=2.9)
            (bundle/'manifest.json').write_text(json.dumps(manifest))
            final_scene=root/'final_scene';final_scene.mkdir()
            final=GallopMuJoCo(bundle/'model_final.pt',ROOT.parents[1]/'unitree/unitree_mujoco/unitree_robots/go2/go2.xml',final_scene)
            self.assertAlmostEqual(final.state.leap.distance_target.item(),.22,places=6)
            final.state.startup.stage[:]=3;final.command[:,0]=3.;final.actor=lambda obs:torch.ones(1,12)*100
            final.step()
            self.assertIn('requested_joint_limit',final.fault)
            self.assertTrue(np.all(np.abs(final.target-final.default)<=final.state.guard.rate.numpy()*.02+1e-6))
            self.assertTrue(final.unchanged())


if __name__=='__main__':unittest.main()
