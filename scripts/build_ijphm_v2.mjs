// Build the IJPHM v2 manuscript: markdown sections -> docx (APA citations) and -> PDF.
// Node is invoked with the bundled binary; pandoc and tectonic are absolute paths.
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync } from "node:fs";

const PANDOC = "C:\\Users\\20562\\AppData\\Local\\Pandoc\\pandoc.exe";
const TECTONIC = "C:\\Users\\20562\\AppData\\Local\\Temp\\tectonic\\tectonic.exe";
const DIR = "docs/manuscripts/ijphm_v2";
const SECTIONS = [
  "00_abstract.md",
  "01_intro_related.md",
  "03_protocol.md",
  "04_results.md",
  "05_discussion.md",
  "06_appendix.md",
];

const parts = [];
for (const f of SECTIONS) {
  const p = `${DIR}/${f}`;
  if (!existsSync(p)) { console.error(`missing ${p}`); process.exit(1); }
  parts.push(readFileSync(p, "utf8"));
}
const body = parts.join("\n\n");
writeFileSync("tmp/ijphm_v2_body.md", body, "utf8");
console.log(`body: ${body.length} chars from ${SECTIONS.length} files`);

const common = [
  "tmp/ijphm_v2_body.md",
  "--citeproc",
  `--bibliography=${DIR}/references_v2.bib`,
  `--csl=${DIR}/apa.csl`,
  "--metadata", "lang=en-US",
];

// 1) DOCX (no LaTeX needed; IJPHM accepts Word)
execFileSync(PANDOC, [...common, "-o", `${DIR}/draft.docx`], { stdio: "inherit" });
console.log("wrote draft.docx");

// 2) HTML for quick inspection of citation rendering
execFileSync(PANDOC, [...common, "-s", "--toc", "-o", `${DIR}/draft.html`], { stdio: "inherit" });
console.log("wrote draft.html");

// 3) PDF via LaTeX + tectonic (best effort; may fail if the TeX bundle is unavailable offline)
try {
  // -s is required: the standalone template defines the CSL macros (\citeproctext,
  // \CSLRightInline, \tightlist) that a bare fragment does not.
  execFileSync(PANDOC, [...common, "-s", "-V", "geometry:margin=1in",
                         "-t", "latex", "-o", "tmp/ijphm_v2_manuscript.tex"], { stdio: "inherit" });
  // figures are compiled from tmp/, so make the paths absolute
  const texPath = "tmp/ijphm_v2_manuscript.tex";
  const ROOT = "C:/Users/20562/Desktop/deepseek-rul";
  let t = readFileSync(texPath, "utf8").replace(
    /\\includegraphics(\[[^\]]*\])?\{([^}]+)\}/g,
    (m, opt, p) => `\\includegraphics${opt || ""}{${ROOT}/${p}}`);
  writeFileSync(texPath, t, "utf8");
  execFileSync(TECTONIC, ["ijphm_v2_manuscript.tex"], { cwd: "tmp", stdio: "inherit" });
  console.log("wrote tmp/ijphm_v2_manuscript.pdf");
} catch (e) {
  console.error(`PDF step skipped/failed: ${e.message}`);
}
