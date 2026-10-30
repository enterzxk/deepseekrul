$b2 = 'Y2QgL2hvbWUveW5keC96eGsvYmVhcmluZy1ydWwtcGkKL2hvbWUveW5keC9taW5pY29uZGEzL2VudnMvYTQwX2Vudi9iaW4vcHl0aG9uIHJlc3VsdHMvZHNoL2FuYWx5c2lzL2FnZ3JlZ2F0ZV9sZXZlbGIucHk='
$n = '0'
for ($i=0; $i -lt 34; $i++) {
  $c = ssh -o BatchMode=yes -o ConnectTimeout=10 -o UpdateHostKeys=no lab-server "find /home/yndx/zxk/bearing-rul-pi/experiments/levelb -name metrics.json | wc -l" 2>$null
  $n = (($c -join '') -replace '\s','')
  Write-Output ("check {0}: {1}/12 at {2}" -f $i, $n, (Get-Date -Format HH:mm:ss))
  if ($n -eq '12') { break }
  Start-Sleep -Seconds 120
}
if ($n -eq '12') {
  ssh -o BatchMode=yes -o ConnectTimeout=10 -o UpdateHostKeys=no lab-server "echo $b2 | base64 -d | bash" 2>&1
} else { Write-Output 'TIMEOUT_WAITING_FOR_LEVELB' }
