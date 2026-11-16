"""Package every regular project file, with separate quick-resume and full archives."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

EXCLUDED={'.venv','.isaacvenv','__pycache__','.git','build','build_316'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def pack(paths,source,dest,zstd=False):
    process=subprocess.Popen(['zstd','-T4','-3','-q','-o',str(dest)],stdin=subprocess.PIPE) if zstd else None
    stream=tarfile.open(fileobj=process.stdin,mode='w|') if process else tarfile.open(dest,'w:gz',compresslevel=3)
    try:
        for rel in paths:
            p=source/rel;info=stream.gettarinfo(str(p),arcname=rel)
            info.uid=info.gid=0;info.uname=info.gname=''
            with p.open('rb') as f:stream.addfile(info,f)
    finally:
        stream.close()
        if process:
            process.stdin.close()
            if process.wait():raise RuntimeError('zstd failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest-dir',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);a.manifest_dir.mkdir(parents=True,exist_ok=True)
    files={};symlinks=[]
    for parent,dirs,names in os.walk(a.source):
        dirs[:]=sorted(d for d in dirs if d not in EXCLUDED and not (Path(parent)/d).is_symlink())
        for name in sorted(names):
            path=Path(parent)/name;rel=path.relative_to(a.source).as_posix()
            if path.is_symlink():symlinks.append(rel);continue
            if name.endswith(('.pyc','.pyo')):continue
            if name in ('.env','.netrc','.git-credentials') or name.endswith(('.pem','.key')):
                raise RuntimeError('Unexpected credential-like file; review before publishing: '+rel)
            files[rel]=dict(bytes=path.stat().st_size,sha256=sha(path))
    manifest=dict(files=files,total_bytes=sum(v['bytes'] for v in files.values()),
        exclusions=sorted(EXCLUDED),excluded_symlinks=symlinks,
        note='All regular Gallop project source/data/reports/runs/archive files retained; machine-specific virtualenvs/builds/caches excluded.')
    (a.manifest_dir/'artifact-files.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(event='INVENTORY',files=len(files),bytes=manifest['total_bytes'])),flush=True)
    resume_prefixes=('runs/isaac/train/unitree_go2_gallop/20261112_124757_47b3b7/',
        'runs/isaac/train/unitree_go2_gallop/20261111_182833_2552ed/',
        'runs/isaac/train/unitree_go2_gallop/20261106_000029_077a6f/',
        'runs/isaac/train/unitree_go2_forward_only/20261028_160258_54620c/')
    pack([n for n in files if n.startswith(resume_prefixes)],a.source,a.output/'resume-pack.tar.gz')
    print(json.dumps(dict(event='RESUME_READY',bytes=(a.output/'resume-pack.tar.gz').stat().st_size)),flush=True)
    full=a.output/'full-snapshot.tar.zst';pack(list(files),a.source,full,True)
    parts=[]
    with full.open('rb') as f:
        i=1
        while block:=f.read(1024**3):
            name=f'full-snapshot.tar.zst.part{i:02d}';(a.output/name).write_bytes(block);parts.append(name);i+=1
    full.unlink()
    assets={n:dict(bytes=(a.output/n).stat().st_size,sha256=sha(a.output/n)) for n in ['resume-pack.tar.gz']+parts}
    result=dict(tag='handoff-2026-11-16',assets=assets,full_parts=parts,
        repository='PGH-snfl/go2-gallop-rl',inventory='handoff/artifact-files.json')
    (a.manifest_dir/'release-assets.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.output/'SHA256SUMS').write_text(''.join(f"{v['sha256']}  {n}\n" for n,v in assets.items()))
    print(json.dumps(dict(event='COMPLETE',assets=assets)),flush=True)


if __name__=='__main__':main()
