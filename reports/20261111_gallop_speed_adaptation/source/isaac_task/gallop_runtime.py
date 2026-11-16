"""Checkpoint bookkeeping and replay source selection without starting Isaac."""
import hashlib
import importlib
import math
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
    if manifest.get('staged_adaptation_source_sha256'):
        path=directory/'staged_adaptation.py'
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['staged_adaptation_source_sha256']:
            raise ValueError(f'Staged adaptation source snapshot is missing or changed: {path}')
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


def relearn_checkpoint_progress(checkpoint,manifest):
    """Controller progress belongs to this checkpoint, including interim saves."""
    checkpoint=Path(checkpoint)
    record=manifest.get('relearn_checkpoints',{}).get(checkpoint.name)
    if record is None:raise ValueError('Missing checkpoint-specific relearning progress: '+checkpoint.name)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=record['sha256']:
        raise ValueError('Relearning checkpoint/progress hash mismatch')
    return record.copy()


def staged_checkpoint_progress(checkpoint,manifest):
    """Speed alone cannot identify a saved 3m/s distance/flight stage."""
    checkpoint=Path(checkpoint)
    record=manifest.get('staged_checkpoints',{}).get(checkpoint.name)
    plan=manifest.get('staged_design',{}).get('plan',[])
    if record is None:raise ValueError('Missing checkpoint-specific staged Gallop progress: '+checkpoint.name)
    index=record.get('index')
    if type(index) is not int or not 0<=index<len(plan) or record.get('goal')!=plan[index]:
        raise ValueError('Saved staged goal and plan do not match')
    confirmed=record.get('highest_confirmed_stage')
    if confirmed is not None and (type(confirmed) is not int or not 0<=confirmed<len(plan)):
        raise ValueError('Invalid saved confirmed Gallop stage')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=record['sha256']:
        raise ValueError('Staged checkpoint/progress hash mismatch')
    return record.copy()


def map_relearn_progress(progress,manifest,plan):
    """Insert curriculum stages without reinterpreting a saved numeric index."""
    saved=manifest.get('relearn_design',{}).get('plan')
    if not saved or not 0<=progress['level']<len(saved):
        raise ValueError('Missing saved relearning goal for stage migration')
    goal=saved[progress['level']]
    keys=('speed','gait','request_tolerance','request_seconds','request_fraction')
    matches=[i for i,item in enumerate(plan) if item['name']==goal['name'] and
             all(math.isclose(float(item[k]),float(goal[k]),rel_tol=1e-8,abs_tol=1e-9) for k in keys)]
    if len(matches)!=1:raise ValueError('Saved relearning goal has no unique match in the new plan')
    return dict(progress,level=matches[0],parent_level=progress['level'],goal=plan[matches[0]].copy())
