# Build the IEEEtran manuscript from the Markdown draft.
# Fails loudly on missing sections or LaTeX errors (both happened during development).

$ErrorActionPreference = "Stop"
$pandoc   = "C:\Users\20562\AppData\Local\Pandoc\pandoc.exe"
$tectonic = "C:\Users\20562\AppData\Local\Temp\tectonic\tectonic.exe"
$md  = "docs\manuscripts\v1.0-draft.md"
$tex = "docs\manuscripts\manuscript.tex"

$text = Get-Content $md -Raw -Encoding UTF8
$required = @("## I. Introduction","## II. Related Work","## III. Method",
              "## IV. Experimental Protocol","## V. Results",
              "## VI. Discussion","## VII. Conclusion")
$missing = @($required | Where-Object { -not $text.Contains($_) })
if ($missing.Count -gt 0) { throw "STRUCTURE CHECK FAILED - missing: $($missing -join ', ')" }
Write-Host "structure check: 7/7 sections present"

# markdown -> latex fragment with natbib citations
$body = $text.Substring($text.IndexOf("## I. Introduction"))
$body = $body -replace '(?m)^### ', '@@SUB@@'
$body = $body -replace '(?m)^## ', '# '
$body = $body -replace '(?m)^@@SUB@@', '## '
Set-Content -Path "tmp\body.md" -Value $body -Encoding UTF8
& $pandoc "tmp\body.md" -f markdown -t latex -o "tmp\body.tex" --wrap=preserve --natbib --bibliography "docs\manuscripts\references.bib"

$t = Get-Content "tmp\body.tex" -Raw -Encoding UTF8

# longtable -> table/tabular
$parts = $t -split [regex]::Escape("\begin{longtable}")
$out = $parts[0]
for ($i = 1; $i -lt $parts.Length; $i++) {
    $b = $parts[$i]
    $e = $b.IndexOf("\end{longtable}")
    $inner = $b.Substring(0, $e)
    $rest  = $b.Substring($e + "\end{longtable}".Length)
    $nl = $inner.IndexOf("`n")
    $spec = $inner.Substring(0, $nl).Trim() -replace '^\[\]', ''
    $inner = $inner.Substring($nl + 1) -replace '\\endhead', '' -replace '\\endlastfoot', ''
    $inner = $inner -replace '\\bottomrule\\noalign\{\}', '' -replace '\\noalign\{\}', ''
    $out += "\begin{table}[!t]`n\centering\scriptsize`n\begin{tabular}$spec`n$($inner.TrimEnd())`n\bottomrule`n\end{tabular}`n\end{table}`n$rest"
}


# reuse preamble from the existing tex (title/abstract live there)
$old = Get-Content $tex -Raw -Encoding UTF8
$docStart = $old.IndexOf("\begin{document}") + "\begin{document}".Length
$hdr = $old.Substring(0, $docStart)
$pre = $old.Substring($docStart, $old.IndexOf("\section{I. Introduction}") - $docStart)
Set-Content -Path $tex -Value ($hdr + "`n" + $pre + $out + "`n\bibliography{references}`n\end{document}`n") -Encoding UTF8

Push-Location "docs\manuscripts"
$prev = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$log = & $tectonic manuscript.tex 2>&1 | Out-String
$ErrorActionPreference = $prev
Pop-Location
$errs = ($log -split "`n") | Where-Object { $_ -match "^error" }
if ($errs) { throw ("LATEX COMPILE FAILED:" + "`n" + ($errs -join "`n")) }
Write-Host "latex: no errors"
if (Test-Path "docs\manuscripts\manuscript.pdf") {
    Write-Host ("PDF rebuilt: {0:N0} KB" -f ((Get-Item "docs\manuscripts\manuscript.pdf").Length/1KB))
} else { throw "PDF not produced" }
