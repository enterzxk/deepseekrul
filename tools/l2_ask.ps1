# L2 helper: ask Codex via `codex exec` (bypasses the flaky app-server plugin path).
param(
  [Parameter(Mandatory=$true)][string]$PromptFile,
  [string]$WorkDir = (Get-Location).Path,
  [int]$TimeoutSec = 600
)
$ErrorActionPreference = 'Stop'
$codex = "$env:LOCALAPPDATA\OpenAI\Codex\bin\f71a721e0e6193a9\codex.exe"
if (-not (Test-Path $codex)) { throw "codex.exe not found at $codex" }
$prompt = Get-Content -Path $PromptFile -Raw -Encoding utf8
if ([string]::IsNullOrWhiteSpace($prompt)) { throw "empty prompt file: $PromptFile" }
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $codex
$psi.Arguments = 'exec --skip-git-repo-check -C "' + $WorkDir + '" -'
$psi.UseShellExecute = $false
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.StandardOutputEncoding = (New-Object System.Text.UTF8Encoding $false)
$psi.StandardErrorEncoding  = (New-Object System.Text.UTF8Encoding $false)
$p = [System.Diagnostics.Process]::Start($psi)
$bytes = (New-Object System.Text.UTF8Encoding $false).GetBytes($prompt)
$p.StandardInput.BaseStream.Write($bytes, 0, $bytes.Length)
$p.StandardInput.BaseStream.Flush()
$p.StandardInput.Close()
$stdoutTask = $p.StandardOutput.ReadToEndAsync()
$stderrTask = $p.StandardError.ReadToEndAsync()
$finished = $p.WaitForExit($TimeoutSec * 1000)
if (-not $finished) { try { $p.Kill() } catch {}; throw "codex exec timed out after $TimeoutSec s" }
"=== EXIT: $($p.ExitCode) ==="
$stdoutTask.Result
if ($stderrTask.Result) { "=== STDERR ==="; $stderrTask.Result }