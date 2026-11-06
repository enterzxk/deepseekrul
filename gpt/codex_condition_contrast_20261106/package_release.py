"""Build a source-only handoff ZIP and verify every archived byte."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    upstream = json.loads((root/"UPSTREAM.json").read_text(encoding="utf-8"))
    for source in upstream["sources"]:
        path = root/source["local_path"]
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        variants = (raw, raw.rstrip(b"\n")+b"\n", raw.rstrip(b"\n"))
        matched = None
        for data in variants:
            blob = hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
            if blob == source["git_blob_sha1"]:
                matched = data
                break
        if matched is None:
            raise ValueError(f"Upstream source is not the pinned Git blob: {path}")
        path.write_bytes(matched)
        source["sha256"] = digest(matched)
    (root/"UPSTREAM.json").write_text(json.dumps(upstream, indent=2)+"\n", encoding="utf-8")
    log = (root/"validation_output.txt").read_text(encoding="utf-8-sig")
    count = re.search(r"Ran (\d+) tests", log)
    if count is None or not log.rstrip().endswith("OK"):
        raise ValueError("A successful actual validation transcript is required")
    from tim_contrast.run import code_fingerprint
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "status": "CORE_IMPLEMENTED_AWAITING_DEEPSEEK_EXPERIMENTS",
               "upstream_commit": upstream["commit"], "code_sha256": code_fingerprint(),
               "tests_passed": int(count.group(1)), "validation_command": "python -m unittest discover -s tests -v",
               "validation_log_sha256": digest((root/"validation_output.txt").read_bytes()),
               "real_cmapss_training_executed": False, "linux_gpu_pipeline_executed": False,
               "novelty_proven": False, "performance_gain_proven": False,
               "original_project_files_overwritten": False, "external_handoff_sent": False}
    (root/"DELIVERY.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    files = sorted(p for p in root.rglob("*") if p.is_file()
                   and not any(x in {"__pycache__", "outputs", "cache", ".pytest_cache"} for x in p.relative_to(root).parts)
                   and p.name != "MANIFEST.sha256" and p.suffix not in {".pyc", ".pyo", ".zip"})
    lines = [f"{digest(p.read_bytes())}  {p.relative_to(root).as_posix()}" for p in files]
    (root/"MANIFEST.sha256").write_text("\n".join(lines)+"\n", encoding="utf-8")
    files.append(root/"MANIFEST.sha256")
    destination = root.parent/(root.name+".zip")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, (Path(root.name)/path.relative_to(root)).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
        for path in files:
            name = (Path(root.name)/path.relative_to(root)).as_posix()
            if digest(archive.read(name)) != digest(path.read_bytes()):
                raise ValueError(f"Archive mismatch: {name}")
    print(json.dumps({"archive": str(destination), "files": len(files),
                      "bytes": destination.stat().st_size,
                      "sha256": digest(destination.read_bytes()),
                      "tests_passed": receipt["tests_passed"], "code_sha256": receipt["code_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
