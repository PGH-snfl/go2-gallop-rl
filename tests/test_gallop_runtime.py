"""Saved source and iteration lineage remain usable after changing rewards."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from gpd.paths import ROOT
from isaac_task.gallop_runtime import checkpoint_source_directory,parent_cumulative_iterations


class GallopRuntimeTests(unittest.TestCase):
    def test_legacy_gallop_source_is_selected_in_a_clean_process(self):
        parent=ROOT/'runs/isaac/train/unitree_go2_gallop/20261028_181029_379d1c'
        if not parent.exists():self.skipTest('Local trained model is not present')
        code='''
import json
from pathlib import Path
from isaac_task.gallop_runtime import activate_checkpoint_source
p=Path(%r)
m=json.loads((p/'manifest.json').read_text())
activate_checkpoint_source(p/'model_final.pt',m)
from isaac_task import gallop_core
assert Path(gallop_core.__file__).resolve()==p/'source/gallop_core.py'
assert not hasattr(gallop_core.GallopSettings(),'pair_load_weight')
print('legacy source selected')
'''%str(parent)
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_changed_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory);source=parent/'source';source.mkdir()
            manifest={}
            for name,key in (('gallop_core.py','gallop_core_sha256'),('gallop_env.py','gallop_env_sha256')):
                p=source/name;p.write_text('# saved version\n');manifest[key]=hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(checkpoint_source_directory(parent/'model_final.pt',manifest),source)
            (source/'gallop_core.py').write_text('# changed\n')
            with self.assertRaises(ValueError):checkpoint_source_directory(parent/'model_final.pt',manifest)

    def test_cumulative_iterations_do_not_double_count_parent_stages(self):
        self.assertEqual(parent_cumulative_iterations({'fine_tuning':{'parent_cumulative_iterations':2000}},2000),4000)
        self.assertEqual(parent_cumulative_iterations({'migration':{'parent_cumulative_iterations':4000}},6000),10000)
        self.assertEqual(parent_cumulative_iterations({'cumulative_iterations':12000,'fine_tuning':{'parent_cumulative_iterations':10000}},2000),12000)


if __name__=='__main__':unittest.main()
