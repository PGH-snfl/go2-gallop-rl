"""Offline collector helpers, loaded without simulator or TensorBoard imports."""
import ast
import json
import unittest

import numpy as np

from gpd.paths import ROOT


def helpers():
    path=ROOT/'scripts/analysis/collect_flight_diagnostics.py'
    tree=ast.parse(path.read_text())
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef)
              and node.name in ('flight_event_rows','flight_trace_rows')]
    namespace={'np':np,'json':json}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def trace():
    counters={'leap_events':[0,1,1,2,2],'leap_qualified_events':[0]*5,
              'flight_landings':[0,1,1,2,2],'flight_stable_landings':[0,0,1,1,1],
              'flight_aborted_landings':[0,0,0,0,1]}
    z={key:np.array(values)[:,None] for key,values in counters.items()}
    z.update(time=np.arange(5)*.02,valid=np.ones((5,1),bool),episode_reset=np.zeros((5,1),bool),
             leap_last_distance=np.full((5,1),.15),leap_last_seconds=np.full((5,1),.08),
             leap_reward_score=np.array([0,.01,.03,.02,0])[:,None],
             flight_step_touchdown_distance_score=np.array([0,.01,0,.02,0])[:,None],
             flight_step_recovery_distance_score=np.array([0,0,.03,0,0])[:,None],
             flight_abort_slide=np.array([0,0,0,0,1])[:,None],
             flight_abort_sample_foot_slide=np.array([0,0,0,0,.8])[:,None],
             flight_abort_sample_foot_velocity_w=np.full((5,1,4,3),.8))
    return z


class FlightCollectionTests(unittest.TestCase):
    def test_event_totals_include_touchdown_credit_even_if_later_aborted(self):
        rows=helpers()['flight_event_rows'](trace(),{'leap_min_distance':.5,'leap_reward_weight':24})
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['status'],'stable')
        self.assertAlmostEqual(rows[0]['distance_reward_points'],.04*24)
        self.assertAlmostEqual(rows[0]['touchdown_distance_reward_points'],.01*24)
        self.assertAlmostEqual(rows[0]['recovery_distance_reward_points'],.03*24)
        self.assertEqual(rows[1]['status'],'aborted')
        self.assertAlmostEqual(rows[1]['distance_reward_points'],.02*24)
        self.assertEqual(json.loads(rows[1]['abort_reason_counts']),{'flight_abort_slide':1.})

    def test_multidimensional_foot_velocities_are_flattened_as_numbers(self):
        rows=helpers()['flight_trace_rows'](trace())
        self.assertEqual(rows[-1]['flight_abort_sample_foot_velocity_w_3_2'],.8)
        self.assertEqual(len([key for key in rows[-1] if key.startswith('flight_abort_sample_foot_velocity_w_')]),12)
        self.assertIsInstance(rows[-1]['flight_abort_sample_foot_velocity_w_0_0'],float)

    def test_legacy_trace_without_split_fields_keeps_its_total(self):
        z={key:value for key,value in trace().items() if not key.startswith(('flight_step_','flight_abort_'))}
        rows=helpers()['flight_event_rows'](z,{'leap_min_distance':.5,'leap_reward_weight':24})
        self.assertAlmostEqual(sum(row['distance_reward_points'] for row in rows),.06*24)
        self.assertEqual(rows[-1]['abort_reason_counts'],'{}')

    def test_multiple_unattributable_events_remain_rejected(self):
        z=trace();z['leap_events'][1,0]=3
        with self.assertRaises(ValueError):helpers()['flight_event_rows'](z,{'leap_min_distance':.5,'leap_reward_weight':24})


if __name__=='__main__':unittest.main()
