"""Negative acceptance probes on copied real histories; never model inference."""
from pathlib import Path
import argparse,copy,json
from .common import Guard,write_json
from .collect import collect
from .verify import replay,read

def main(root,data_dir):
 root=Path(root).resolve();source=root/'matched/aidgn/FD001/s101'
 history=read(source/'history.json');metric=read(source/'training_metrics.json')
 replay(history,metric);checks=[{'case':'real_trained_history','result':'PASS'}]
 variants=[]
 x=copy.deepcopy(history);x[0]['val_engine_mean_rmse']+=.5;variants.append(('stale_engine_score',x))
 x=copy.deepcopy(history);x[metric['best_epoch']-1]['state_content_sha256']='0'*64;variants.append(('wrong_accepted_state',x))
 variants.append(('premature_stop',copy.deepcopy(history[:-1])))
 for label,x in variants:
  try:replay(x,metric)
  except AssertionError:checks.append({'case':label,'result':'REJECTED'})
  else:raise AssertionError('bad evidence accepted: '+label)
 with Guard('collector_negative_acceptance') as guard:
  try:collect(root,data_dir,root/'metadata_probe_target',root/'metadata_probe_docs')
  except (AssertionError,FileNotFoundError):checks.append({'case':'collector_before_all_tests_complete','result':'REJECTED_BEFORE_LABELS'})
  else:raise AssertionError('premature collector accepted')
 assert not guard.record['official_test_reads'] and not guard.record['violations']
 result={'new_training':False,'new_test_forward':False,'official_file_reads':0,'real_training_files_modified':False,'checks':checks}
 write_json(root/'metadata_acceptance_checks.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--data-dir',required=True);a=p.parse_args();main(a.root,a.data_dir)
