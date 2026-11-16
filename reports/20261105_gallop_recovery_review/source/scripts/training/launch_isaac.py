"""Preserve failure exit status even when Kit shuts down the child with exit code 0."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from gpd.paths import ROOT as root, new_run
arguments=sys.argv[1:]
gallop=bool(arguments and arguments[0]=='gallop')
baseline=bool(arguments and arguments[0]=='baseline')
forward=bool(arguments and arguments[0]=='forward')
official=bool(arguments and arguments[0] in ('official','baseline','forward'))
if official or gallop:arguments=arguments[1:]
module='scripts.training.gallop_baseline' if gallop else 'scripts.training.official_baseline' if official else 'scripts.training.run_isaac'
if not arguments or "--help" in arguments or "-h" in arguments:
    if not arguments or arguments[0] in ("--help","-h"):
        arguments=["check","--help"]
    raise SystemExit(subprocess.call([sys.executable,"-m",module,*arguments]))
if baseline:arguments += ['--curriculum-fixed']
if forward:arguments += ['--forward-only']
out=None
for i,arg in enumerate(arguments):
    if arg=="--output" and i+1<len(arguments):
        out=Path(arguments[i+1])
    elif arg.startswith("--output="):
        out=Path(arg.split("=",1)[1])
if out is None:
    out=new_run("isaac",arguments[0])
    if gallop:out=out.parent/'unitree_go2_gallop'/out.name
    if official:
        tag=('unitree_go2_forward_only' if '--forward-only' in arguments else
             'unitree_go2_curriculum_fixed' if '--curriculum-fixed' in arguments else 'unitree_go2_official')
        out=out.parent/tag/out.name
    arguments += ["--output",str(out)]
if (out/"status.json").exists():
    raise SystemExit("Output already contains a run; select a fresh directory")
out.mkdir(parents=True,exist_ok=True)
print(f"[LOG] {out.resolve()/'console.log'}",flush=True)
with (out/"console.log").open("w",buffering=1) as log:
    child=subprocess.Popen([sys.executable,"-u","-m",module,*arguments],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        for line in child.stdout:
            log.write(line)
            print(line,end="",flush=True)
        code=child.wait()
    except KeyboardInterrupt:
        import signal
        child.send_signal(signal.SIGINT)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.terminate()
            child.wait()
        raise SystemExit(130)
if code:
    raise SystemExit(code)
status=json.loads((out/"status.json").read_text()) if (out/"status.json").exists() else {}
if status.get("status")!="completed":
    print(f"Isaac run failed: {status.get('error','missing completion status')}; see {out}",file=sys.stderr)
    raise SystemExit(1)
