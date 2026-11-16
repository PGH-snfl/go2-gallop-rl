"""Checkpoint bookkeeping and replay source selection without starting Isaac."""
import hashlib
import importlib
from pathlib import Path
import sys


def checkpoint_source_directory(checkpoint,manifest):
    directory=Path(checkpoint).resolve().parent/'source'
    for name,key in (('gallop_core.py','gallop_core_sha256'),('gallop_env.py','gallop_env_sha256')):
        path=directory/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest[key]:
            raise ValueError(f'Training source snapshot is missing or changed: {path}')
    teacher=manifest.get('walking_teacher')
    if teacher:
        path=directory/'walking_teacher.pt'
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=teacher['sha256']:
            raise ValueError(f'Walking teacher snapshot is missing or changed: {path}')
    if manifest.get('gallop_teacher'):
        path=directory/'gallop_teacher.pt'
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['gallop_teacher']['sha256']:
            raise ValueError(f'Gallop teacher snapshot is missing or changed: {path}')
    if manifest.get('adaptation_source_sha256'):
        path=directory/'gallop_adaptation.py'
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['adaptation_source_sha256']:
            raise ValueError(f'Adaptation source snapshot is missing or changed: {path}')
    return directory


def activate_checkpoint_source(checkpoint,manifest):
    """Play/check use their own verified task, including older reward versions."""
    if any(name in sys.modules for name in ('isaac_task.gallop_core','isaac_task.gallop_env')):
        raise RuntimeError('Select checkpoint source before importing the Gallop task')
    directory=checkpoint_source_directory(checkpoint,manifest)
    import isaac_task
    isaac_task.__path__.insert(0,str(directory))
    importlib.invalidate_caches()
    return str(directory)


def parent_cumulative_iterations(manifest,stage_iterations):
    if 'cumulative_iterations' in manifest:return int(manifest['cumulative_iterations'])
    lineage=manifest.get('fine_tuning',manifest.get('migration',{}))
    return int(lineage.get('parent_cumulative_iterations',0))+int(stage_iterations)
