"""Adversarial metadata probes using real freeze documents, never model inference."""
import argparse,copy,json,tempfile
from pathlib import Path
from . import joint
from .common import Guard,write_json,digest

def main(original,author):
 original,author=Path(original).resolve(),Path(author).resolve()
 before={str(p):digest(p) for p in [original/'JOINT_COMPARISON_FREEZE.json',original/'BATCH_FREEZE.json',author/'BATCH_FREEZE.json']}
 checks=[]
 with Guard('pre_TEST_metadata_checks') as guard:
  master=joint.validate_ready(original,author);checks.append({'case':'real_READY_94physical_168logical','result':'PASS','physical':len(master['physical_runs']),'logical':len(master['logical_conditions'])})
  joint.freeze(original,author)
  assert before=={p:digest(p) for p in before};checks.append({'case':'freeze_idempotence','result':'PASS'})
  for case in ['missing_author_batch','duplicate_batch_key','prepared_not_ready','wrong_component_hash']:
   with tempfile.TemporaryDirectory(prefix='joint_metadata_probe_') as tmp:
    left,right=Path(tmp)/'fixed',Path(tmp)/'author';left.mkdir();right.mkdir()
    for name in ['JOINT_COMPARISON_FREEZE.json','JOINT_COMPARISON_PREPARED.json','BATCH_FREEZE.json']:(left/name).write_bytes((original/name).read_bytes())
    (right/'BATCH_FREEZE.json').write_bytes((author/'BATCH_FREEZE.json').read_bytes())
    if case=='missing_author_batch':(right/'BATCH_FREEZE.json').unlink()
    elif case=='duplicate_batch_key':
     x=json.loads((right/'BATCH_FREEZE.json').read_text());x['runs'][-1]=x['runs'][0];(right/'BATCH_FREEZE.json').write_text(json.dumps(x))
    elif case=='prepared_not_ready':
     x=json.loads((left/'JOINT_COMPARISON_FREEZE.json').read_text());x['state']='PREPARED';x['frozen']=False;(left/'JOINT_COMPARISON_FREEZE.json').write_text(json.dumps(x))
    else:
     x=json.loads((left/'JOINT_COMPARISON_FREEZE.json').read_text());x['component_batch_sha256']['author_limits']='0'*64;(left/'JOINT_COMPARISON_FREEZE.json').write_text(json.dumps(x))
    try:joint.validate_ready(left,right)
    except (AssertionError,FileNotFoundError):checks.append({'case':case,'result':'REJECTED_BEFORE_TEST'})
    else:raise AssertionError('invalid READY accepted: '+case)
  with tempfile.TemporaryDirectory(prefix='joint_quota_probe_') as tmp:
   quota=Path(tmp)/'TEST_ATTEMPT.json';joint.write_once(quota,{'mode':'reuse_without_forward'})
   try:joint.write_once(quota,{'mode':'actual_forward'})
   except FileExistsError:checks.append({'case':'shared_atomic_quota_reuse_vs_forward','result':'SECOND_OWNER_REJECTED'})
   else:raise AssertionError('quota overwritten')
 assert not guard.record['official_test_reads'] and not guard.record['violations']
 assert before=={p:digest(p) for p in before}
 result={'new_training':False,'model_TEST_forward':False,'official_file_reads':0,'real_weights_or_configs_modified':False,'checks':checks}
 write_json(original/'PRE_TEST_ACCEPTANCE.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--author-root',required=True);a=p.parse_args();main(a.root,a.author_root)
