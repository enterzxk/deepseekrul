#!/usr/bin/env node
// Keep the LaTeX abstract in sync with abstract.tex (single source of truth).
import { readFileSync, writeFileSync } from "node:fs";

const TEX = "docs/manuscripts/manuscript.tex";
const ABS = "docs/manuscripts/abstract.tex";

const abs = readFileSync(ABS, "utf8").trim();
let tex = readFileSync(TEX, "utf8");
const re = /\\begin\{abstract\}[\s\S]*?\\end\{abstract\}/;
if (!re.test(tex)) {
  console.error("no abstract block found in " + TEX);
  process.exit(1);
}
const before = tex;
tex = tex.replace(re, "\\begin{abstract}\n" + abs + "\n\\end{abstract}");
writeFileSync(TEX, tex, "utf8");
console.log(before === tex ? "abstract already in sync" : "abstract synced from abstract.tex");
