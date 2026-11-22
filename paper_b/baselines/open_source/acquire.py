"""Pin public upstream trees and preserve byte-exact author source files.

Large datasets, checkpoints and images stay upstream; the full tree manifest
records those exclusions. Re-run with --verify to check the local snapshots.
No credentials, author code rewrites or Git history rewriting are involved.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
REPOS = {
    "aidgn": "TCCofWANG/AIDGN-RUL",
    "two_phase": "mjliu-advertising/RUL",
    "dual_mixer": "fuen1590/PhmDeepLearningProjects",
    "ffve": "LLerDagu/RUL-FFVE",
    "fcstgnn": "Frank-Wang-oss/FCSTGNN",
    "cpt": "ZeyU1999/RUL-Prediction",
}
EXTENSIONS = {".py", ".md", ".rst", ".yaml", ".yml", ".json", ".toml",
              ".ini", ".cfg", ".sh", ".bat", ".ipynb"}


def get(url):
    urls = [url]
    if url.startswith("https://raw.githubusercontent.com/"):
        pieces = url.removeprefix("https://raw.githubusercontent.com/").split("/", 3)
        urls.append(f"https://github.com/{pieces[0]}/{pieces[1]}/raw/{pieces[2]}/{pieces[3]}")
    for index, endpoint in enumerate(urls):
        try:
            req = urllib.request.Request(endpoint, headers={"User-Agent": "PaperB-source-reproduction"})
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read()
        except Exception:
            if index == len(urls) - 1:
                raise


def acquire(key, repo):
    out = ROOT / "upstream" / key
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "SOURCE_MANIFEST.json"
    preserve_manifest = manifest_path.exists()
    if preserve_manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sha = manifest["commit"]
        tree = manifest["tree"]
    else:
        cache = ROOT / ".cache" / key
        if not (cache / ".git").exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth=1", "--filter=blob:none", "--no-checkout",
                            f"https://github.com/{repo}.git", str(cache)], check=True,
                           capture_output=True, timeout=180)
        def git(*args):
            return subprocess.check_output(["git", "-C", str(cache), *args], text=True, encoding="utf-8").strip()
        sha = git("rev-parse", "HEAD")
        tree = []
        # -l would lazily fetch EVERY promised blob merely to get its size,
        # including multi-GB datasets/checkpoints. Only inspect tree objects.
        for record in git("ls-tree", "-rz", "HEAD").split("\0"):
            if not record:
                continue
            meta, path = record.split("\t", 1)
            mode, typ, blob = meta.split()
            if typ == "blob":
                tree.append({"path": path, "sha": blob})
        manifest = {"repository": f"https://github.com/{repo}", "commit": sha,
                    "commit_date": git("show", "-s", "--format=%cI", "HEAD"),
                    "tree": tree, "files": [],
                    "policy": "All code/config/notebook/README/license files <=2MB; no datasets/checkpoints/images. Full upstream tree is recorded."}
    candidates = [x for x in tree if x.get("size", 0) <= 2_000_000 and (
        Path(x["path"]).suffix.lower() in EXTENSIONS or
        Path(x["path"]).name.lower().startswith(("license", "requirements", "environment")))]

    def fetch(entry):
        path = entry["path"]
        dst = out / path
        if dst.exists():
            data = dst.read_bytes()
        else:
            data = get(f"https://raw.githubusercontent.com/{repo}/{sha}/{urllib.parse.quote(path)}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != entry["sha"]:
            raise RuntimeError(f"upstream blob mismatch: {repo}/{path}")
        return {"path": path, "git_blob_sha": blob, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}

    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        verified_files = list(pool.map(fetch, candidates))
    if preserve_manifest:
        # Preserve the registered manifest bytes, including original newline
        # style; Linux/Windows restoration must retain its SHA identity.
        assert verified_files == manifest["files"], f"source manifest changed: {key}"
    else:
        manifest["files"] = verified_files
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{key} {sha} code_files={len(candidates)} total_blobs={len(tree)}", flush=True)
    return {"model": key, "repository": manifest["repository"], "commit": sha,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}


def verify():
    lock = json.loads((ROOT / "sources.lock.json").read_text())
    n = 0
    for source in lock:
        folder = ROOT / "upstream" / source["model"]
        manifest_path = folder / "SOURCE_MANIFEST.json"
        assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == source["manifest_sha256"]
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["files"]:
            assert hashlib.sha256((folder / entry["path"]).read_bytes()).hexdigest() == entry["sha256"], entry["path"]
            n += 1
    print(f"PASS: {len(lock)} pinned source trees; {n} original files byte-exact")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify()
    else:
        with cf.ThreadPoolExecutor(max_workers=3) as pool:
            records = list(pool.map(lambda pair: acquire(*pair), REPOS.items()))
        lock_path = ROOT / "sources.lock.json"
        if lock_path.exists():
            assert json.loads(lock_path.read_text()) == records, "registered source lock changed"
        else:
            lock_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
