// Build the submission-format manuscript with the official IJPHM LaTeX template.
// Input : docs/manuscripts/ijphm_v2/{01,03,04,05}_*.md + references_v2.bib
// Output: tmp/ijphm_submission/manuscript.tex -> manuscript.pdf (tectonic)
//
// The template (PHMSociety.cls) is two-column, uses apacite, and requires
// \PHMbibliography.  Tables produced by pandoc are longtables, which do not work
// in two-column mode; they are converted to plain tabular inside table* floats.

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, mkdirSync, copyFileSync, existsSync } from "node:fs";

const PANDOC = "C:\\Users\\20562\\AppData\\Local\\Pandoc\\pandoc.exe";
const TECTONIC = "C:\\Users\\20562\\AppData\\Local\\Temp\\tectonic\\tectonic.exe";
const DIR = "docs/manuscripts/ijphm_v2";
const TPL = "tmp/lit/templates/latex";
const OUT = "tmp/ijphm_submission";

mkdirSync(OUT, { recursive: true });
copyFileSync(`${TPL}/PHMSociety.cls`, `${OUT}/PHMSociety.cls`);
copyFileSync(`${DIR}/references_v2.bib`, `${OUT}/references_v2.bib`);

const body = ["01_intro_related.md", "03_protocol.md", "04_results.md", "05_discussion.md"]
  .map((f) => {
    const lines = readFileSync(`${DIR}/${f}`, "utf8").split("\n");
    // drop the leading draft-note blockquote (lines starting with '>' until the first non-quote)
    let i = 0;
    while (i < lines.length && (lines[i].trim() === "" || lines[i].startsWith(">"))) i++;
    return lines.slice(i).join("\n");
  })
  .join("\n\n");
// The IJPHM template order is: sections -> acknowledgment -> bibliography -> biographies -> appendix.
// The appendix is therefore converted separately and appended after \PHMbibliography.
const appendixMd = readFileSync(`${DIR}/06_appendix.md`, "utf8");
writeFileSync(`${OUT}/appendix.md`, appendixMd, "utf8");
execFileSync(PANDOC, [`${OUT}/appendix.md`, "-f", "markdown", "-t", "latex", "--natbib",
                      "-o", `${OUT}/appendix.tex`], { stdio: "inherit" });
writeFileSync(`${OUT}/body.md`, body, "utf8");

execFileSync(PANDOC, [`${OUT}/body.md`, "-f", "markdown", "-t", "latex", "--natbib",
                      "-o", `${OUT}/body.tex`], { stdio: "inherit" });

let tex = readFileSync(`${OUT}/body.tex`, "utf8");
let app = readFileSync(`${OUT}/appendix.tex`, "utf8");
const ROOT = "C:/Users/20562/Desktop/deepseek-rul";
const absFig = (s) => s.replace(/\\includegraphics(\[[^\]]*\])?\{([^}]+)\}/g,
                                (m, opt, p) => `\\includegraphics${opt || ""}{${ROOT}/${p}}`);
tex = absFig(tex);
app = absFig(app);
// unnumbered sections required by the template
tex = tex.replace(/\\section\{(Acknowledgment|Biographies)\}/g, "\\section*{$1}");
app = app.replace(/\\section\{(Appendix[^}]*)\}/g, "\\section*{$1}");
// strip the hand-written section numbers ("1. Introduction" -> "Introduction"):
// LaTeX numbers sections itself, otherwise the PDF shows "1. 1. Introduction".
tex = tex.replace(/\\(section|subsection)\{(\d+(?:\.\d+)*)\.?\s+/g, "\\$1{");
// longtable -> table* + tabular (two-column safe)
tex = tex.replace(/\\begin\{longtable\}\[\]\{@\{\}([^@]*)@\{\}\}/g, "\\begin{table*}[t]\\centering\\begin{tabular}{$1}");
tex = tex.replace(/\\end\{longtable\}/g, "\\end{tabular}\\end{table*}");
tex = tex.replace(/\\toprule/g, "\\hline").replace(/\\bottomrule/g, "\\hline");
// remaining longtable-only commands
tex = tex.replace(/\\midrule/g, "\\hline")
         .replace(/\\endhead/g, "")
         .replace(/\\endlastfoot/g, "");

const abstract = readFileSync(`${DIR}/00_abstract.md`, "utf8")
  .split("abstract: |")[1].split("keywords:")[0]
  .split("\n").map((l) => l.replace(/^ {2}/, "")).join("\n").trim()
  // the abstract is injected as raw LaTeX, so escape the characters that TeX would eat
  .replace(/\\/g, "\\textbackslash{}")
  .replace(/([%&_#])/g, "\\$1")
  .replace(/\^/g, "\\^{}")
  .replace(/~/g, "\\textasciitilde{}");

const doc = String.raw`\documentclass[IJPHM, 2026, 00]{PHMSociety}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{array}
\usepackage{calc}   % pandoc's longtable column widths use \real{} from calc
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
% apacite provides \cite (parenthetical) and \citeA (textual); pandoc --natbib emits
% \citep/\citet, so alias them.
\newcommand{\citep}{\cite}
\newcommand{\citet}{\citeA}
\begin{document}
\title{What conformal RUL intervals deliver: an 80-cell audit of coverage, per-engine failure, and a conditional epistemic scale}
\author{First Author\authorNumber{1}}
\address{\affiliation{{1}}{Affiliation, City, Country}{{\email{author@example.edu}}}}
\maketitle
\pagestyle{fancy}
\thispagestyle{plain}
\phmLicenseFootnote{First Author}
\begin{abstract}
${abstract}
\end{abstract}
${tex}
\bibliographystyle{apacite}
\PHMbibliography{references_v2}
${app}
\end{document}
`;
writeFileSync(`${OUT}/manuscript.tex`, doc, "utf8");
console.log("wrote manuscript.tex");

try {
  execFileSync(TECTONIC, ["manuscript.tex"], { cwd: OUT, stdio: "inherit" });
  console.log("wrote manuscript.pdf");
} catch (e) {
  console.error(`tectonic failed: ${e.message}`);
  process.exit(1);
}
