$exe = "C:\Users\20562\AppData\Local\OpenAI\Codex\bin\f71a721e0e6193a9\codex.exe"
$out = & $exe exec --skip-git-repo-check --sandbox read-only "Reply with exactly: CODEX_CLI_OK" 2>&1
$out | Select-Object -Last 12
