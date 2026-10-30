#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
echo '--- FD001 runs:'; ls experiments/order03/*/FD001 2>/dev/null | head
echo '--- FD004 runs:'; ls experiments/order03/*/FD004 2>/dev/null | head
