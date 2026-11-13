"""Preserve pinned upstream bytes; do not import or execute upstream modules."""
import concurrent.futures, datetime, hashlib, json, os, pathlib, stat, sys, time, urllib.parse, urllib.request
ROOT = pathlib.Path(__file__).resolve().parent
SPEC = json.loads((ROOT / "PINNED_TREES.json").read_text(encoding="utf-8"))
TEXT_EXT = {".py",".yaml",".yml",".json",".toml",".sh",".cfg",".ini",".txt",".md",".rst",".sql",".lock",".c",".h",".cpp",".hpp",".cc",".ipynb",".ps1",".bat"}
SPECIAL = {".gitattributes",".gitignore",".env.example",".editorconfig","makefile","dockerfile","license","license.txt","license.md","copying","notice","authors"}
def reason(path, size, mode):
    parts = pathlib.PurePosixPath(path).parts
    low = path.lower()
    name = parts[-1].lower()
    if mode == "120000": return "symlink excluded; no target materialization"
    if "__pycache__" in parts or low.endswith((".pyc",".pyo")): return "generated Python bytecode, not author source"
    if parts[0] == ".idea": return "personal IDE/deployment metadata, not runtime source"
    if parts[0].lower() in {"logs","ea_log"} or name == "corr_log.csv": return "generated experiment results/checkpoints/logs; excluded including stored predictions"
    if parts[0] in {"CMAPSS","N-CMAPSS"} and not name.startswith(("readme","license","notice")): return "dataset or dataset-placeholder/document asset; no data downloaded"
    if low.endswith((".pt",".pth",".ckpt",".pkl",".pickle",".npz",".npy",".h5",".hdf5",".zip",".7z",".parquet")): return "data/checkpoint/model binary excluded"
    if low.endswith(".ipynb") and size > 5000000: return "large notebook with embedded outputs; not required for retained source entry points; original not rewritten"
    if size > 10000000: return "large nonessential blob excluded"
    if name in SPECIAL or pathlib.PurePosixPath(path).suffix.lower() in TEXT_EXT: return None
    if low.endswith(".csv") and "File_DevUnits_TestUnits" in path: return None
    return "non-source asset/figure or generated artifact; no runtime source omitted"
def checked_dest(relative):
    rel = pathlib.PurePosixPath(relative)
    assert not rel.is_absolute() and '..' not in rel.parts and ':' not in relative and '\\' not in relative, relative
    dest = pathlib.Path(os.path.abspath(ROOT / relative))
    assert os.path.normcase(os.path.commonpath([str(dest),str(ROOT)])) == os.path.normcase(str(ROOT)), relative
    return dest
def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
def get_source(job):
    spec, item = job
    raw_url = "https://raw.githubusercontent.com/{}/{}/{}".format(spec["repo"],spec["commit"],urllib.parse.quote(item["path"],safe="/"))
    dest = checked_dest(spec["name"] + "/" + item["path"])
    if dest.exists():
        data = dest.read_bytes()
    else:
        failure = None
        for attempt in range(2):
            try:
                request = urllib.request.Request(raw_url,headers={"User-Agent":"pinned-research-source-preservation/1"})
                with urllib.request.urlopen(request,timeout=60) as response:
                    data = response.read()
                failure = None
                break
            except Exception as exc:
                failure = exc
                if attempt == 0: time.sleep(0.4)
        if failure is not None: raise RuntimeError(f"{spec['name']}/{item['path']}: {failure}")
    assert len(data) == item["size"], (item["path"],len(data),item["size"])
    assert blob_sha(data) == item["sha"], (item["path"],"git blob SHA mismatch")
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        with dest.open("xb") as handle: handle.write(data)
    os.chmod(dest,stat.S_IREAD)
    return spec["name"],{
      "path":item["path"],"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),
      "git_blob_sha1":item["sha"],"git_mode":item["mode"],
      "source_url":f"https://github.com/{spec['repo']}/blob/{spec['commit']}/{urllib.parse.quote(item['path'],safe='/')}",
      "raw_url":raw_url,"unmodified_original_bytes":True,"local_read_only":not bool(dest.stat().st_mode & stat.S_IWRITE)
    }
def main():
    jobs=[]; manifests={}
    for spec in SPEC["repos"]:
        assert len(spec["commit"]) == 40 and not spec["truncated"]
        included=[]; excluded=[]
        for item in spec["entries"]:
            why=reason(item["path"],item["size"],item["mode"])
            if why is None: jobs.append((spec,item))
            else: excluded.append({**item,"reason":why,"source_url":f"https://github.com/{spec['repo']}/blob/{spec['commit']}/{urllib.parse.quote(item['path'],safe='/')}"})
        manifests[spec["name"]] = {
          "schema_version":1,"repository":spec["repo"],"repository_url":"https://github.com/"+spec["repo"],
          "full_commit_sha":spec["commit"],"commit_url":f"https://github.com/{spec['repo']}/commit/{spec['commit']}",
          "tree_api_url":f"https://api.github.com/repos/{spec['repo']}/git/trees/{spec['commit']}?recursive=1",
          "git_tree_response_sha":spec["tree_sha"],"tree_truncated":False,
          "acquisition":"public raw.githubusercontent.com fixed-commit bytes, verified by Git blob SHA-1 and SHA-256",
          "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "scope":"author source/configuration/build dependencies/documentation; no upstream imports, training, data or prediction evaluation",
          "included":included,"excluded":excluded,"upstream_blob_count":len(spec["entries"]),
          "author_source_modified":False,"notebook_policy":"retained notebooks preserved verbatim including outputs; large output notebooks excluded, never stripped",
          "license": {"status":"explicit","identifier":"Apache-2.0","path":"LICENSE","redistribution":"subject to upstream Apache-2.0 terms; retain full LICENSE and source notices"} if spec["name"] == "moo_elm" else {"status":"no_explicit_repository_license_found_in_pinned_tree","identifier":None,"redistribution":"not authorized by this audit; local read-only snapshot only, source ignored by Git; inspect third-party file-specific notices independently","git_submission":"SOURCE_MANIFEST.json metadata only; author source must remain untracked"},
        }
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for name, item in pool.map(get_source,jobs):
            manifests[name]["included"].append(item)
    for name, manifest in manifests.items():
        manifest["included"].sort(key=lambda x:x["path"])
        manifest["acquisition_script_sha256"] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
        notices = []
        for item in manifest["included"]:
            if pathlib.PurePosixPath(item["path"]).suffix.lower() not in {'.py','.c','.h','.cpp','.hpp','.toml'}: continue
            lines = checked_dest(name + '/' + item["path"]).read_text(encoding='utf-8',errors='replace').splitlines()
            for idx, line in enumerate(lines[:80]):
                if any(phrase in line.lower() for phrase in ('general public license','mit license','apache license','bsd license','licensed under')):
                    notices.append({'path':item['path'],'line':idx+1,'text':line.strip()})
        manifest['license']['file_specific_notices'] = notices
        manifest['license']['scope_note'] = 'Root license metadata does not override retained third-party/file/package-specific notices. No license compatibility or redistribution approval is inferred.'
        manifest["included_file_count"]=len(manifest["included"])
        manifest["excluded_file_count"]=len(manifest["excluded"])
        manifest["included_total_bytes"]=sum(x["bytes"] for x in manifest["included"])
        assert manifest["included_file_count"]+manifest["excluded_file_count"] == manifest["upstream_blob_count"]
        assert all(x["local_read_only"] for x in manifest["included"])
        root=checked_dest(name);root.mkdir(exist_ok=True)
        destination=root/"SOURCE_MANIFEST.json"
        destination.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
        print(json.dumps({"name":name,"commit":manifest["full_commit_sha"],"included":manifest["included_file_count"],"excluded":manifest["excluded_file_count"],"bytes":manifest["included_total_bytes"],"license":manifest["license"]["status"]}),flush=True)
if __name__ == "__main__": main()

