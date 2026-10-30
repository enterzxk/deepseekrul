#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
cd /home/yndx/zxk/bearing-rul-pi
codex exec -m 6.1sol -c model_reasoning_effort=xhigh --sandbox read-only --skip-git-repo-check "$(cat /home/yndx/p8/advice.md)" > /home/yndx/p8/advice.out 2>&1
echo "EXIT=$?" >> /home/yndx/p8/advice.out
