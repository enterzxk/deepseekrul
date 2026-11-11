#!/usr/bin/env bash
# DSPT-HC / Paper B 环境。跑任何东西前先 source 这个文件：
#   source /media/sangfor/vdb/user/zxk/env.sh
# 约定：代码、数据、产物全部在 2TB 盘（/dev/vdb）的 zxk 目录下，不占系统盘。
export HC_ROOT=/media/sangfor/vdb/user/zxk/deepseek-rul
export HC_OUT_ROOT=$HC_ROOT/experiments/hc
export HC_DATA=$HC_ROOT/data/CMAPSS
export TORCH_HOME=/media/sangfor/vdb/user/zxk/cache/torch
# 所有可能未定义的变量一律用 ${VAR:-} 展开，保证 `set -u` 下 source 不会因未绑定变量中断
export PYTHONPATH=$HC_ROOT:$HC_ROOT/src:${PYTHONPATH:-}

# 2026-11-11：tmh 在 a40 上遇到 sklearn 报
#   ImportError: libstdc++.so.6: version GLIBCXX_3.4.29 not found
# env 自带的 libstdc++ 是新的，系统那份太旧，所以把 env 的 lib 放到最前。
# 注意：主控在 zxk 新目录实测「不加也能 import sklearn / KMeans / prepare_train」，
# 因此这一行是防御性的；若日后出现相反的副作用，优先怀疑这里。
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH:-}

export PY=/home/yndx/miniconda3/envs/a40_env/bin/python
export PATH=/home/yndx/miniconda3/envs/a40_env/bin:$PATH
cd $HC_ROOT 2>/dev/null || true
