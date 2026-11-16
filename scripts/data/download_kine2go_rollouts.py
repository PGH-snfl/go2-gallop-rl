"""Download one public simulation rollout per clip, plus repeats for the run candidates."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from scripts.data.download_kine2go_references import ROOT, REPO, get

def main():
    reference_manifest = json.loads((ROOT / "data/download_manifest.json").read_text())
    revision = reference_manifest["revision"]
    jobs = [(c["name"], 0) for c in reference_manifest["clips"]]
    candidates = ["ai4_dog_canter", "ai4_dog_run_00", "ai4_dog_run_02", "ai4_dog_run_04"]
    jobs += [(name, index) for name in candidates for index in (1, 2)]
    def download(job):
        name, index = job
        remote = f"data/{name}/traj_{index:04d}/traj.pkl"
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{remote}"
        destination = ROOT / "data/rollouts" / name / f"traj_{index:04d}.pkl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = destination.read_bytes() if destination.exists() else get(url)
        if not destination.exists():
            destination.write_bytes(raw)
        return {"name": name, "index": index, "url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(download,job) for job in jobs]):
            result = future.result()
            results.append(result)
            print(f"Downloaded rollout {len(results)}/{len(jobs)}: {result['name']}/{result['index']}", flush=True)
    (ROOT / "data/rollout_manifest.json").write_text(json.dumps({"revision":revision,"rollouts":sorted(results,key=lambda x:(x['name'],x['index']))},indent=2)+"\n")
    for filename in ("metadata.json", "scripts/load_clip.py"):
        raw = get(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{filename}")
        (ROOT/"data"/filename.split('/')[-1]).write_bytes(raw)

if __name__ == "__main__":
    main()
