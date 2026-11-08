import sys
sys.path[:0]=['/home/user/rulb/tim_20261105','/home/user/rulb/scripts','/home/user/rulb/src','/home/user/rulb']
import run_ida as ida
for w in (50,100,150):
    m=ida.CandidateModel(num_sensors=14, window_size=w, d_model=64, num_layers=1, dropout=0.15,
                         use_baseline=True, use_ema_features=True, use_cycle=True, use_pe=True,
                         use_graph=False, use_dual_head=False)
    print(w, sum(p.numel() for p in m.parameters()))