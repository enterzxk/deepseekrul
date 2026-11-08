#!/bin/bash
echo "--- git version ---"; git --version
echo "--- ls-remote test ---"
timeout 25 git ls-remote --heads https://github.com/Eng-Ismail7/ieee-phm-2012-data-challenge-dataset.git 2>&1 | head -3
echo "--- curl github ---"
timeout 20 curl -sS -o /dev/null -w "%{http_code}\n" https://github.com 2>&1
echo "--- curl codeload ---"
timeout 20 curl -sS -o /dev/null -w "%{http_code}\n" https://codeload.github.com/Eng-Ismail7/ieee-phm-2012-data-challenge-dataset/tar.gz/refs/heads/master 2>&1
