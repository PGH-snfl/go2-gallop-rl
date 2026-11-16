"""Reject false hardware passes and restore native geoms despite an added floor."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco as mj
import numpy as np

from gpd.paths import ROOT
from gpd.portable import external_path, resolve_go2_asset
from gpd.rl_mujoco import make_scene
from scripts.validation.validate_gallop_edu import audit_physics, restore_official_model
from scripts.analysis.collect_edu_validation import motion_blockers, audit_isaac_joint_samples


class EduValidationTests(unittest.TestCase):
    def test_safe_velocity_is_not_reported_as_a_hardware_exceedance_after_handover_failure(self):
        row=dict(startup_completed=False,formal_gallop_metric_seconds=0,urdf_position_pass=True,urdf_velocity_pass=True,urdf_torque_pass=True)
        blockers=motion_blockers([row])
        self.assertEqual(len(blockers),1);self.assertIn('full-student',blockers[0])
        row['urdf_velocity_pass']=False
        self.assertTrue(any('velocity' in b for b in motion_blockers([row])))

    def test_missing_isaac_actual_velocity_cannot_receive_a_pass(self):
        self.assertFalse(audit_isaac_joint_samples({},[],[])['available'])

    def test_velocity_excess_fails_even_with_safe_angles_and_torque(self):
        physics=dict(joint_pos=np.full((2,1),-1.5),joint_velocity=np.array([[0.],[16.2]]),
            pre_joint_velocity=np.zeros((2,1)),applied_torque=np.ones((2,1)),
            joint_target=np.full((2,1),-1.5),computed_torque=np.ones((2,1)),
            foot_force=np.zeros((2,4,3)),base_contact_force=np.zeros(2))
        result=audit_physics(physics,['FR_calf_joint'],[dict(lower=-2.7227,upper=-.83776,effort=45.43,velocity=15.7)],15.,.005)
        self.assertTrue(result['all_positions_within_urdf'])
        self.assertTrue(result['all_applied_torques_within_urdf'])
        self.assertFalse(result['all_velocities_within_urdf'])
        self.assertEqual(result['per_joint'][0]['velocity_exceed_seconds'],.005)
        self.assertIn('UNKNOWN',result['contact_force_official_allowance'])

    def test_native_geometry_restore_matches_body_local_geoms_not_shifted_global_ids(self):
        source=external_path('unitree_mujoco')/'unitree_robots/go2/go2.xml'
        with tempfile.TemporaryDirectory() as directory:
            model=make_scene(source,Path(directory),.005,1.,dict(Y2=23.4,friction=.01))
            plant=SimpleNamespace(model=model,data=mj.MjData(model),contract={},sync=lambda:None)
            restore_official_model(plant,source,mj)
            native=mj.MjModel.from_xml_path(str(source))
            np.testing.assert_equal(model.jnt_range,native.jnt_range)
            np.testing.assert_equal(model.actuator_ctrlrange,native.actuator_ctrlrange)
            np.testing.assert_equal(model.dof_armature,native.dof_armature)
            for name in ('FL','FR','RL','RR'):
                dst=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,name)
                src=mj.mj_name2id(native,mj.mjtObj.mjOBJ_GEOM,name)
                np.testing.assert_equal(model.geom_friction[dst],native.geom_friction[src])
                self.assertEqual(model.geom_condim[dst],native.geom_condim[src])
            floor=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'floor')
            np.testing.assert_allclose(model.geom_friction[floor],[.4,.02,.01])


if __name__=='__main__':
    unittest.main()
