#!/bin/bash
echo "cpus: $(nproc)"
echo "load: $(cat /proc/loadavg)"
echo "--- top cpu ---"
ps -eo pcpu,etime,cmd --sort=-pcpu | head -6 | cut -c1-100
echo "--- mem ---"
free -g | head -2
