#!/usr/bin/env node
// Build the IEEEtran manuscript: markdown -> LaTeX (pandoc) -> PDF (tectonic).
// Line-based conversion of pandoc longtables; hoists captions out of tabular.
import { readFileSync, writeFileSync, existsSync, statSync } from "node:fs";
import { execFileSync } from "node:child_process";

const PANDOC = "C:\\Users\\20562\\AppData\\Local\\Pandoc\\pandoc.exe";
const TECTONIC = "C:\\Users\\20562\\AppData\\Local\\Temp\\tectonic\\tectonic.exe";
const MD = "docs/manuscripts/v1.0-draft.md";
const TEX = "docs/manuscripts/manuscript.tex";
const ABS = "docs/manuscripts/abstract.tex";

const text = readFileSync(MD, "utf8");
const required = ["## I. Introduction", "## II. Related Work", "## III. Method",
  "## IV. Experimental Protocol", "## V. Results", "## VI. Discussion", "## VII. Conclusion"];
const missing = required.filter(h => !text.includes(h));
if (missing.length) { console.error("STRUCTURE CHECK FAILED: missing " + missing.join(", ")); process.exit(1); }
console.log("structure check: 7/7 sections present");

let body = text.slice(text.indexOf("## I. Introduction"));
body = body.replace(/^### /gm, "@@SUB@@").replace(/^## /gm, "# ").replace(/^@@SUB@@/gm, "## ");
writeFileSync("tmp/body.md", body, "utf8");

execFileSync(PANDOC, ["tmp/body.md", "-f", "markdown", "-t", "latex", "-o", "tmp/body.tex",
  "--wrap=preserve", "--natbib", "--bibliography", "docs/manuscripts/references.bib"], { stdio: "inherit" });

const lines = readFileSync("tmp/body.tex", "utf8").split(/\r?\n/);
const out = [];
let i = 0;
while (i < lines.length) {
  const line = lines[i];
  if (!line.startsWith("\\begin{longtable}")) { out.push(line); i++; continue; }

  // spec lives on the same line: \begin{longtable}[]{@{}llll@{}}
  const m = line.match(/\[\]\{(.*)\}\s*$/);
  const spec = m ? m[1] : "@{}l@{}";
  i++;
  const inner = [];
  while (i < lines.length && !lines[i].startsWith("\\end{longtable}")) { inner.push(lines[i]); i++; }
  i++; // skip \end{longtable}

  let caption = null;
  const kept = [];
  let skipping = false;
  for (const l of inner) {
    const t = l.trim();
    if (t.startsWith("\\caption{")) { caption = t.replace(/\\tabularnewline\s*$/, ""); continue; }
    if (t === "\\endfirsthead") { skipping = true; continue; }
    if (skipping) { if (t.includes("\\midrule")) skipping = false; continue; }
    if (t === "\\endhead" || t === "\\endlastfoot") continue;
    if (t.startsWith("\\bottomrule")) continue;
    if (t === "\\noalign{}") continue;
    kept.push(l);
  }
  out.push("\\begin{table}[!t]");
  out.push("\\centering\\scriptsize");
  if (caption) out.push(caption);
  out.push("\\begin{tabular}{" + spec + "}");
  for (const l of kept) out.push(l);
  out.push("\\bottomrule");
  out.push("\\end{tabular}");
  out.push("\\end{table}");
}

let tex = out.join("\n");

const old = readFileSync(TEX, "utf8");
const docStart = old.indexOf("\\begin{document}") + "\\begin{document}".length;
const hdr = old.slice(0, docStart);
let pre = old.slice(docStart, old.indexOf("\\section{I. Introduction}"));

if (existsSync(ABS)) {
  const abs = readFileSync(ABS, "utf8").trim();
  pre = pre.replace(/\\begin\{abstract\}[\s\S]*?\\end\{abstract\}/, "\\begin{abstract}\n" + abs + "\n\\end{abstract}");
}

writeFileSync(TEX, hdr + "\n" + pre + tex + "\n\\bibliography{references}\n\\end{document}\n", "utf8");

let log = "";
try {
  log = execFileSync(TECTONIC, ["manuscript.tex"], { cwd: "docs/manuscripts", encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
} catch (e) {
  log = String(e.stdout ?? "") + String(e.stderr ?? "") + String(e.message ?? "");
}
const errs = log.split("\n").filter(l => l.startsWith("error"));
if (errs.length) { console.error("LATEX COMPILE FAILED:\n" + errs.join("\n")); process.exit(1); }
console.log("latex: no errors");
const pdf = "docs/manuscripts/manuscript.pdf";
console.log("PDF rebuilt: " + Math.round(statSync(pdf).size / 1024) + " KB");
