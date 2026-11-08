import time,os,sys
t0=time.time()
import torch
print("torch import %.1fs"%(time.time()-t0), torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), flush=True)
t=time.time(); x=torch.zeros(1,device="cuda"); print("cuda alloc %.1fs"%(time.time()-t), x.device, flush=True)
sys.path[:0]=["/home/user/rulb/tim_20261105","/home/user/rulb/scripts","/home/user/rulb/src","/home/user/rulb"]
t=time.time(); import run_fold as base; print("run_fold import %.1fs"%(time.time()-t), flush=True)
import run_ida as ida
t=time.time(); folds,df=ida.outer_folds("FD001"); print("load FD001 %.1fs"%(time.time()-t), len(df), flush=True)
df=df.sort_values(["unit","cycle"]).reset_index(drop=True)
train,valid=folds[0]
t=time.time(); tables=base.prior_tables(df,"FD001",train,valid,0); print("prior_tables FD001 %.1fs"%(time.time()-t), flush=True)
t=time.time(); tr,va=ida.prep_fold(df,"FD001",0,train,valid); print("prep_fold FD001 %.1fs"%(time.time()-t), tr[0].shape, va[0].shape, flush=True)
