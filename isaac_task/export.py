"""Export RSL-RL actor and normalizer together; compare against actual policy output."""
import copy
import json
import torch
from tensordict import TensorDict

class DeploymentActor(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.normalizer = copy.deepcopy(policy.actor_obs_normalizer).cpu()
        self.actor = copy.deepcopy(policy.actor).cpu()

    def forward(self, observation):
        return self.actor(self.normalizer(observation)).clamp(-1,1)

def export_actor(policy, output):
    output.mkdir(parents=True,exist_ok=True)
    # RSL-RL stores a live action distribution with non-leaf tensors after PPO;
    # copying the whole policy fails. Export occurs only after learning ends.
    policy = policy.cpu().eval()
    module = torch.jit.script(DeploymentActor(policy).eval())
    module.save(str(output/"actor.pt"))
    module = torch.jit.load(str(output/"actor.pt"))
    obs = torch.randn(128,57,generator=torch.Generator().manual_seed(17))
    with torch.no_grad():
        reference = policy.act_inference(TensorDict({"policy":obs},batch_size=[128])).clamp(-1,1)
        error = (reference-module(obs)).abs().max().item()
    report = {"accepted":error<1e-5,"max_abs_error":error,"observation_dim":57,"action_dim":5,
        "trainer":"Isaac Lab / RSL-RL PPO","normalization":"included in actor.pt"}
    (output/"export_validation.json").write_text(json.dumps(report,indent=2))
    if not report["accepted"]:
        raise RuntimeError("Export numerical parity failed")
    return report
