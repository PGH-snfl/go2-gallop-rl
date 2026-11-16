"""Download the 40 public Kine2Go references and metadata, without policies or rollouts."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import time
import urllib.request

from gpd.paths import ROOT
REPO = "MIMUW-Robotics/kine2go"

def get(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)

def main():
    info = json.loads(get(f"https://huggingface.co/api/datasets/{REPO}"))
    revision = info["sha"]
    entries = json.loads(get(f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/data"))
    clips = sorted(e["path"].split("/")[-1] for e in entries if e["type"] == "directory")
    def download(name):
        destination = ROOT / "data" / "references" / name
        destination.mkdir(parents=True, exist_ok=True)
        files = []
        for filename in ("motion.npy", "clip.json", "config.json"):
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/data/{name}/{filename}"
            raw = get(url)
            (destination / filename).write_bytes(raw)
            files.append({"filename": filename, "url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        return {"name": name, "files": files}
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(download, name): name for name in clips}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Downloaded {len(results)}/{len(clips)}: {result['name']}", flush=True)
    manifest = {"repository": REPO, "revision": revision, "clips": sorted(results, key=lambda x: x["name"])}
    (ROOT / "data" / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed: {len(results)} references at {revision}", flush=True)

if __name__ == "__main__":
    main()
