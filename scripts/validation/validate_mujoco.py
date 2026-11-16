"""Validate an Isaac-trained bundle in MuJoCo. No optimization/training occurs."""
import argparse
from gpd.paths import new_run
import hashlib
import json
from pathlib import Path
import subprocess
import sys

def validate_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle/"manifest.json").read_text())
    validation = json.loads((bundle/"export_validation.json").read_text())
    if manifest.get("interface") != "gpd_residual_57x5_isaac_v1" or not validation.get("accepted"):
        raise ValueError("Requires an Isaac Gallop bundle with a passing numerical export check")
    if manifest.get("actor_sha256") != hashlib.sha256((bundle/"actor.pt").read_bytes()).hexdigest():
        raise ValueError("Exported actor checksum mismatch")
    if manifest.get("config_sha256") != hashlib.sha256((bundle/"config.json").read_bytes()).hexdigest():
        raise ValueError("Controller config checksum mismatch")
    return bundle

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle",type=Path)
    p.add_argument("--seconds",type=float,default=30)
    p.add_argument("--output",type=Path)
    p.add_argument("--dds",action="store_true")
    p.add_argument("--binary",default="deploy/build_316/gallop_dds")
    p.add_argument("--require-pass",action="store_true")
    a = p.parse_args()
    a.output = a.output or new_run("mujoco","dds" if a.dds else "evaluate")
    bundle = validate_bundle(a.bundle)
    from gpd.paths import ROOT as root
    command = [sys.executable,"-m",("scripts.validation.sim2sim" if a.dds else "scripts.validation.evaluate"),
        "--config",str(bundle/"config.json"),"--actor",str(bundle/"actor.pt"),
        "--seconds",str(a.seconds),"--output",str(a.output.resolve())]
    if a.dds:
        command += ["--binary",a.binary]
    if a.require_pass:
        command += ["--require-pass"]
    raise SystemExit(subprocess.call(command,cwd=root))

if __name__ == "__main__":
    main()
