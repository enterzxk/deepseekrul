#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
sed -i 's/unit=int(u), T=T, band=band,/unit=int(u), TT=T, band=band,/' scripts/dsh_r3_Lstrat.py
sed -i 's/np.nanmedian(g.g\/g.T)/np.nanmedian(g["g"]\/g["TT"])/' scripts/dsh_r3_Lstrat.py
sed -i 's/np.nanmedian(g.cl\/g.T)/np.nanmedian(g["cl"]\/g["TT"])/' scripts/dsh_r3_Lstrat.py
sed -i 's/np.nanmedian(g.rt\/g.T)/np.nanmedian(g["rt"]\/g["TT"])/' scripts/dsh_r3_Lstrat.py
sed -i 's/np.nanmedian(r.g\/r.T), np.nanmedian(r.cl\/r.T), np.nanmedian(r.rt\/r.T)/np.nanmedian(r["g"]\/r["TT"]), np.nanmedian(r["cl"]\/r["TT"]), np.nanmedian(r["rt"]\/r["TT"])/' scripts/dsh_r3_Lstrat.py
sed -i 's/np.corrcoef(r\[k\], r.T)/np.corrcoef(r[k], r["TT"])/' scripts/dsh_r3_Lstrat.py
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python scripts/dsh_r3_Lstrat.py 2>&1 | grep -v Warning | tail -10
