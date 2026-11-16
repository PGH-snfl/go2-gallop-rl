"""No simulator or learning required: check normalized RSL export and bundle rejection."""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from isaac_task.export import export_actor
from scripts.validation.validate_mujoco import validate_bundle

class IsaacExportContracts(unittest.TestCase):
    def test_normalized_export_after_action_distribution(self):
        obs=TensorDict({"policy":torch.randn(128,57)},batch_size=[128])
        policy=ActorCritic(obs,{"policy":["policy"],"critic":["policy"]},5,
            actor_hidden_dims=[128,128],critic_hidden_dims=[128,128],actor_obs_normalization=True)
        policy.update_normalization(obs)
        policy.act(obs)  # Populate live distribution: full-policy deepcopy must not be needed.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=export_actor(policy,root)
            self.assertTrue(result["accepted"])
            (root/"config.json").write_text('{}')
            manifest={"interface":"gpd_residual_57x5_isaac_v1",
                "actor_sha256":hashlib.sha256((root/"actor.pt").read_bytes()).hexdigest(),
                "config_sha256":hashlib.sha256((root/"config.json").read_bytes()).hexdigest()}
            (root/"manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(validate_bundle(root),root)
            (root/"config.json").write_text('{"speed":99}')
            with self.assertRaisesRegex(ValueError,"config checksum"):
                validate_bundle(root)

if __name__=="__main__":
    unittest.main()
