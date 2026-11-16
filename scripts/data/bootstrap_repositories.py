"""Fetch exact upstream revisions without modifying an existing dirty checkout."""
import subprocess
from pathlib import Path

from gpd.paths import ROOT

REPOS={
    'unitree_rl_lab':('https://github.com/unitreerobotics/unitree_rl_lab.git','4960b84732b0c2ec593dccbfe963fda1bcd7b1e3'),
    'unitree_mujoco':('https://github.com/unitreerobotics/unitree_mujoco.git','1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d'),
    'IsaacLab':('https://github.com/isaac-sim/IsaacLab.git','37ddf626871758333d6ed89cf64ad702aef127d0'),
}


def main():
    base=ROOT/'third_party';base.mkdir(exist_ok=True)
    for name,(url,commit) in REPOS.items():
        folder=base/name
        if folder.exists():
            head=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
            dirty=subprocess.check_output(['git','-C',str(folder),'status','--porcelain','--untracked-files=no'],text=True).strip()
            if head!=commit or dirty:raise RuntimeError(f'Existing {name} is not the required clean revision; preserve it and choose another location.')
        else:
            folder.mkdir();subprocess.run(['git','init',str(folder)],check=True)
            subprocess.run(['git','-C',str(folder),'remote','add','origin',url],check=True)
            subprocess.run(['git','-C',str(folder),'fetch','--depth','1','origin',commit],check=True)
            subprocess.run(['git','-C',str(folder),'checkout','--detach','FETCH_HEAD'],check=True)
        print(f'{name}: {commit}',flush=True)


if __name__=='__main__':main()
