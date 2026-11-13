"""Verify preserved bytes and Git exclusion without importing author code."""
import hashlib,json,os,pathlib,stat,subprocess
ROOT=pathlib.Path(__file__).resolve().parent
SPEC=json.loads((ROOT/"PINNED_TREES.json").read_text(encoding="utf-8"))
def main():
    total=0;results=[];unlicensed_paths=[];manifests=[]
    for spec in SPEC["repos"]:
        folder=ROOT/spec["name"]
        manifest_path=folder/"SOURCE_MANIFEST.json"
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["full_commit_sha"]==spec["commit"]
        assert manifest["included_file_count"]+manifest["excluded_file_count"]==len(spec["entries"])
        assert {x["path"] for x in manifest["included"]}.isdisjoint(x["path"] for x in manifest["excluded"])
        assert {x["path"] for x in manifest["included"]+manifest["excluded"]}=={x["path"] for x in spec["entries"]}
        for item in manifest["included"]:
            path=folder/item["path"];data=path.read_bytes()
            assert hashlib.sha256(data).hexdigest()==item["sha256"],path
            assert hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()==item["git_blob_sha1"],path
            assert not(path.stat().st_mode & stat.S_IWRITE),path
            if manifest["license"]["status"]!="explicit":unlicensed_paths.append(path)
            total+=1
        manifests.append(manifest_path)
        results.append({"name":spec["name"],"files":len(manifest["included"]),"excluded":len(manifest["excluded"]),"manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),"license_notices":manifest["license"]["file_specific_notices"]})
    worktree=ROOT.parents[2]
    relative=[p.relative_to(worktree).as_posix() for p in unlicensed_paths]
    process=subprocess.run(["git","check-ignore","--no-index","-z","--stdin"],input=("\0".join(relative)+"\0").encode('utf-8'),cwd=worktree,capture_output=True,check=False)
    assert process.returncode==0,process.stderr
    ignored={p.decode('utf-8') for p in process.stdout.split(b'\0') if p}
    assert ignored==set(relative),(len(ignored),len(relative),sorted(set(relative)-ignored)[:3])
    for p in manifests:
        proc=subprocess.run(["git","check-ignore","--no-index",str(p.relative_to(worktree))],cwd=worktree,capture_output=True,text=True,check=False)
        assert proc.returncode==1,(p,proc.stdout,proc.stderr)
    report={"status":"PASS","source_files_verified":total,"unlicensed_source_files_confirmed_git_ignored":len(relative),"source_imports_or_execution":0,"repos":results}
    (ROOT/"PRESERVATION_ACCEPTANCE.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()

