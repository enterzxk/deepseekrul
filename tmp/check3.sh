#!/bin/bash
echo "=== femto feat2 ==="
tail -3 ~/deepseek-rul/logs/femto_feat2.log
echo "files: $(ls ~/deepseek-rul/data/femto_features/*.csv 2>/dev/null | wc -l)"
echo "=== seeds progress ==="
echo "cqr:  $(ls ~/deepseek-rul/experiments/punet_seeds/*/f*_s*/e2e_cqr_none.json 2>/dev/null | wc -l)/60"
echo "point:$(ls ~/deepseek-rul/experiments/punet_seeds/*/f*_s*/e2e_point_none.json 2>/dev/null | wc -l)/60"
echo "qr:   $(ls ~/deepseek-rul/experiments/punet_seeds/*/f*_s*/e2e_qr_raw_none.json 2>/dev/null | wc -l)/60"
tail -2 ~/deepseek-rul/logs/seeds_all.log
