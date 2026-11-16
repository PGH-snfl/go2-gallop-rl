import unittest
import numpy as np
from gpd.config import load_config
from gpd.schedule import GallopSchedule
from gpd.qp import project_forces, rpy_rate_world, GpdAttitudeQP
from gpd.controller import GallopController
from gpd.sim import FlatPlant
from gpd.env import GallopEnv

class Contracts(unittest.TestCase):
    def test_float32_imu_contract(self):
        cfg=load_config(); plant=FlatPlant(cfg); controller=GallopController(cfg)
        state=plant.low_state()
        for name in ("quaternion","gyro","acceleration","q","dq","foot_force"):
            setattr(state,name,getattr(state,name).astype(np.float32))
        torque,info=controller.step(state)
        self.assertFalse(info["fault"])
        self.assertTrue(np.isfinite(torque).all())

    def test_incomplete_run_cannot_pass(self):
        from gpd.metrics import summarize
        cfg=load_config(); cfg["acceptance_min_cycles"]=0; cfg["acceptance_valid_fraction"]=0
        truth=np.zeros(12); truth[6]=cfg["speed"]
        row={"time":3., "truth":truth, "estimated":truth, "desired":np.ones(4,bool),
             "measured":np.ones(4,bool), "clip_fraction":0., "cycle_ms":1., "fault":None}
        result=summarize([row],cfg)
        self.assertFalse(result["gates"]["full_duration"])
        self.assertFalse(result["accepted"])

    def test_paper_eight_contact_phases(self):
        from gpd import ROOT
        s = GallopSchedule(load_config(ROOT / "configs/paper_gallop.json"))
        # Independent fixtures transcribed from paper Eq.3, in FL FR RL RR.
        expected = [[1,0,1,1],[1,0,0,1],[1,1,0,1],[1,1,0,0],[0,1,0,0],[0,0,0,0],[0,0,1,0],[0,0,1,1]]
        for k, mask in enumerate(expected):
            s.phase = (0.25+(k+0.5)/8)%1
            np.testing.assert_array_equal(s.mask(),mask)

    def test_topology_and_phase_preserved_at_action_extremes(self):
        import itertools
        for lead in ("left","right"):
            s = GallopSchedule(load_config(lead=lead))
            phase = s.phase
            for action in itertools.product([-1,1],repeat=5):
                s.apply(action,10)
                self.assertEqual(s.phase,phase)
                ordered = np.argsort(s.touchdown).tolist()
                self.assertEqual(ordered,[2,3,0,1] if lead=="right" else [3,2,1,0])

    def test_force_projection(self):
        f=project_forces(np.array([500,-500,-10,500,-500,900,1,2,3,4,5,6]),np.array([1,1,0,0],bool),.8,200).reshape(4,3)
        self.assertTrue((f[:,2]>=0).all())
        self.assertTrue((f[:,2]<=200).all())
        self.assertTrue((np.abs(f[:,:2])<=.8*f[:,2,None]+1e-9).all())
        self.assertTrue((np.linalg.norm(f[:,:2],axis=1)<=.8*f[:,2]+1e-9).all())
        np.testing.assert_array_equal(f[2:],0)

    def test_euler_rate_at_zero_yaw(self):
        np.testing.assert_allclose(rpy_rate_world([0,0,0]),np.eye(3))

    def test_nan_action_rejected(self):
        with self.assertRaises(ValueError):
            GallopSchedule(load_config()).apply([0,0,np.nan,0,0],.025)

    def test_nan_sensor_has_finite_bounded_fallback(self):
        cfg=load_config();plant=FlatPlant(cfg);controller=GallopController(cfg)
        state=plant.low_state();state.dq[0]=np.nan
        torque,info=controller.step(state)
        self.assertTrue(info["fault"])
        self.assertTrue(np.isfinite(torque).all())
        self.assertTrue((np.abs(torque)<=np.tile(cfg["torque_limits"],4)).all())

    def test_stale_timestamp_latches_fault(self):
        cfg=load_config();plant=FlatPlant(cfg);controller=GallopController(cfg)
        state=plant.low_state();controller.step(state)
        torque,info=controller.step(state)
        self.assertTrue(info["fault"])

    def test_zero_residual_parity(self):
        cfg=load_config();a=GallopEnv(cfg);b=GallopEnv(cfg)
        a.reset();b.reset()
        for _ in range(6):
            a.controller.set_action(np.zeros(5))
            ta,_=a.controller.step(a.plant.low_state())
            tb,_=b.controller.step(b.plant.low_state())
            np.testing.assert_allclose(ta,tb,atol=1e-9,rtol=0)
            a.plant.step(ta);b.plant.step(tb)

    def test_gpd_flight_force_zero(self):
        from gpd import ROOT
        cfg=load_config(ROOT / "configs/paper_gallop.json");c=GallopController(cfg);c.schedule.phase=.94
        np.testing.assert_array_equal(c.gpd.solve(c.estimator.robot,c.schedule),np.zeros(12))

if __name__=="__main__":unittest.main()
