"""Download/check release assets and safely restore resume or full project data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import urllib.request

from gpd.paths import ROOT


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def restore(stream,destination,inventory,full):
    count=0
    for member in stream:
        name=member.name;path=Path(name)
        if path.is_absolute() or '..' in path.parts or not member.isfile():raise ValueError('Unsafe archive member: '+name)
        if name not in inventory:raise ValueError('Unlisted archive member: '+name)
        expected=inventory[name]
        relative=path if path.parts[0] in ('runs','reports','archive','data') else Path('archive/original_source_20261116')/path
        target=destination/relative
        if not target.resolve().is_relative_to(destination.resolve()):raise ValueError('Extraction escapes destination')
        if target.exists():
            if target.is_symlink() or sha(target)!=expected['sha256']:raise FileExistsError('Conflicting local file: '+str(target))
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        with stream.extractfile(member) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
        if target.stat().st_size!=expected['bytes'] or sha(target)!=expected['sha256']:
            target.unlink();raise ValueError('Extracted checksum mismatch: '+name)
        target.chmod(member.mode&0o777);count+=1
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true',help='All runs, logs, videos, archives and original source; default is quick resume')
    parser.add_argument('--assets-dir',type=Path,default=ROOT/'.release_assets')
    parser.add_argument('--destination',type=Path,default=ROOT)
    parser.add_argument('--repository',help='Override the release repository, e.g. for a fork')
    a=parser.parse_args();meta=json.loads((ROOT/'handoff/release-assets.json').read_text())
    inventory=json.loads((ROOT/'handoff/artifact-files.json').read_text())['files']
    names=meta['full_parts'] if a.full else ['resume-pack.tar.gz'];a.assets_dir.mkdir(parents=True,exist_ok=True)
    repository=a.repository or meta['repository']
    for name in names:
        dest=a.assets_dir/name;record=meta['assets'][name]
        if not dest.exists():
            partial=dest.with_suffix(dest.suffix+'.download')
            url=f"https://github.com/{repository}/releases/download/{meta['tag']}/{name}"
            print('Downloading '+name,flush=True)
            with urllib.request.urlopen(url) as src,partial.open('wb') as dst:shutil.copyfileobj(src,dst)
            if partial.stat().st_size!=record['bytes'] or sha(partial)!=record['sha256']:
                partial.unlink();raise ValueError('Download checksum mismatch: '+name)
            partial.rename(dest)
        if dest.stat().st_size!=record['bytes'] or sha(dest)!=record['sha256']:raise ValueError('Asset checksum mismatch: '+name)
    a.destination.mkdir(parents=True,exist_ok=True)
    if a.full:
        if not shutil.which('zstd'):raise RuntimeError('Install zstd first: sudo apt install zstd')
        process=subprocess.Popen(['zstd','-dq'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        errors=[]
        def feed():
            try:
                for name in names:
                    with (a.assets_dir/name).open('rb') as src:shutil.copyfileobj(src,process.stdin)
            except Exception as error:errors.append(error)
            finally:process.stdin.close()
        thread=threading.Thread(target=feed);thread.start()
        try:
            with tarfile.open(fileobj=process.stdout,mode='r|') as stream:count=restore(stream,a.destination,inventory,True)
        finally:
            process.stdout.close();thread.join()
            code=process.wait()
        if errors or code:raise RuntimeError('Decompression failed: '+str(errors))
    else:
        with tarfile.open(a.assets_dir/names[0],'r:gz') as stream:count=restore(stream,a.destination,inventory,False)
    print(json.dumps(dict(restored_files=count,destination=str(a.destination.resolve()),full=a.full,checksums_verified=True)))


if __name__=='__main__':main()
