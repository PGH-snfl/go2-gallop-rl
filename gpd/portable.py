"""Resolve pinned dependencies and hash-verified assets after moving machines."""
import hashlib
import os
from pathlib import Path

from gpd.paths import ROOT


def external_path(name):
    aliases={'unitree_rl_lab_official':'unitree_rl_lab','unitree_rl_lab':'unitree_rl_lab',
             'unitree_mujoco':'unitree_mujoco','IsaacLab':'IsaacLab'}
    name=aliases[name]
    variable={'unitree_rl_lab':'GALLOP_UNITREE_RL_LAB','unitree_mujoco':'GALLOP_UNITREE_MUJOCO','IsaacLab':'GALLOP_ISAACLAB'}[name]
    if os.environ.get(variable):return Path(os.environ[variable]).expanduser().resolve()
    base=Path(os.environ.get('GALLOP_EXTERNAL_ROOT',ROOT/'third_party')).expanduser()
    return (base/name).resolve()


def resolve_go2_asset(manifest=None):
    manifest=manifest or {};expected=manifest.get('asset_sha256')
    candidates=[]
    if os.environ.get('GALLOP_GO2_USD'):candidates.append(Path(os.environ['GALLOP_GO2_USD']).expanduser())
    candidates.append(ROOT/'assets/vendor/unitree_go2/usd/go2.usd')
    if manifest.get('asset_path'):candidates.append(Path(manifest['asset_path']))
    for path in candidates:
        if path.is_file() and (expected is None or hashlib.sha256(path.read_bytes()).hexdigest()==expected):return path.resolve()
    raise FileNotFoundError('Matching Go2 USD not found; restore assets or set GALLOP_GO2_USD. Saved checkpoint hashes are never rewritten.')
