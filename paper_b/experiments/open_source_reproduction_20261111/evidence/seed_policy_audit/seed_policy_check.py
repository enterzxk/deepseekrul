# Exact Python body executed read-only by audit_training_integrity.
# Invocation: ssh a40 /home/yndx/miniconda3/envs/a40_env/bin/python -B -
# stdin was the body below; recorded exit code: 0.
import sys, json, hashlib
from pathlib import Path
root=Path('/home/yndx/paper_b_public_baselines_20261111')
sys.path.insert(0,str(root))
import torch
from paper_b.baselines.open_source.adapters import UpstreamAdapter,module
from paper_b.baselines.open_source.common import seed_everything
pre_mod=module('two_phase','Transformer_pre_model.py','audit_pre_seed')
ft_mod=module('two_phase','Transformer_ft_serialatten_model.py','audit_ft_seed')
def digest_state(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def rng_hash():
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
print(json.dumps({'audit':'initialization_only_no_data_no_forward_no_training','torch':torch.__version__}))
for subset,layers,dropout in [('FD001',1,.1),('FD002',2,0.)]:
    for seed in [101,202,303]:
        seed_everything(seed)
        fresh_rng=rng_hash()
        src_pre=pre_mod.Transformer_pre(17,16,45,32,32,layers,2,dropout)
        src_pre_rng=rng_hash()
        seed_everything(seed)
        src_ft=ft_mod.Transformer_ft(17,16,45,32,32,layers,2,dropout)
        src_ft_rng=rng_hash()
        adapter=UpstreamAdapter({'model':'two_phase','subset':subset,'seed':seed,'window':45})
        current_before_pre_rng=rng_hash()
        curr_pre=adapter.pre_module.Transformer_pre(17,16,45,32,32,layers,2,dropout)
        current_after_pre_rng=rng_hash()
        pdiffs=[(name,(value-curr_pre.state_dict()[name]).abs().max().item()) for name,value in src_pre.state_dict().items()]
        ftdiffs=[(name,(value-adapter.net.state_dict()[name]).abs().max().item()) for name,value in src_ft.state_dict().items()]
        print(json.dumps({'subset_group':subset,'seed':seed,'source_pre_weights':digest_state(src_pre),'current_pre_weights':digest_state(curr_pre),'source_ft_weights':digest_state(src_ft),'current_ft_weights':digest_state(adapter.net),'pre_differing_tensors':sum(d>0 for _,d in pdiffs),'pre_total_tensors':len(pdiffs),'pre_max_abs_delta':max(d for _,d in pdiffs),'ft_differing_tensors':sum(d>0 for _,d in ftdiffs),'fresh_rng':fresh_rng,'source_pre_after_rng':src_pre_rng,'source_ft_after_rng':src_ft_rng,'current_before_pre_rng':current_before_pre_rng,'current_after_pre_rng':current_after_pre_rng}))
