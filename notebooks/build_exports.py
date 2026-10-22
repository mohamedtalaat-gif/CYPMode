# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Export the executed notebook to a single HTML file and a PDF: ``python -m notebooks.build_exports``.

The HTML is written here rather than by nbconvert so that it needs nothing from the network: the equations are typeset in the page by the vendored KaTeX (scripts, stylesheet and
fonts inlined from ``notebooks/_vendor/katex``), code is highlighted by Pygments, and figures are embedded. The PDF is the same page printed by headless Chrome, with the headings as its bookmarks and the entries of the contents as links to their sections.
"""

import base64
import html
import json
import re
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

from notebooks.notebook_builder import slug

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "cypmode_notebook.ipynb"
VENDOR = HERE / "_vendor" / "katex"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

TITLE = "CYPMode: predicting how drugs switch off liver enzymes, from first principles"

CSS = """
:root { --ink: #1c2230; --muted: #5a6477; --rule: #d9deea; --accent: #1f4e8c; --code-bg: #f4f6fb; --card: #ffffff; }
* { box-sizing: border-box; }
body { margin: 0; background: #eef1f7; color: var(--ink); font-family: Charter, 'Iowan Old Style', Georgia, serif; font-size: 17px; line-height: 1.62; }
main { max-width: 860px; margin: 0 auto; padding: 36px 40px 80px; background: var(--card); }
h1, h2, h3, h4 { font-family: 'Avenir Next', 'Helvetica Neue', Arial, sans-serif; line-height: 1.25; color: #14213d; }
h1 { font-size: 2.1rem; margin: 2.4em 0 0.6em; padding-bottom: 0.25em; border-bottom: 3px solid var(--accent); }
h1.title { font-size: 2.7rem; border: none; margin-top: 1.2em; }
h2 { font-size: 1.5rem; margin: 2em 0 0.5em; color: var(--accent); }
h3 { font-size: 1.2rem; margin: 1.6em 0 0.4em; }
h4 { font-size: 1.05rem; margin: 1.3em 0 0.3em; }
p { margin: 0.7em 0; hyphens: auto; }
a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
code { font-family: Menlo, 'SF Mono', Consolas, monospace; font-size: 0.84em; background: var(--code-bg); padding: 1px 4px; border-radius: 4px; }
table { border-collapse: collapse; margin: 1em 0; font-size: 0.88em; width: 100%; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--rule); padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #e9eef8; font-family: 'Avenir Next', 'Helvetica Neue', Arial, sans-serif; }
blockquote { margin: 1em 0; padding: 0.2em 1em; border-left: 4px solid var(--accent); background: #f1f5fc; color: #2a3550; }
hr { border: none; border-top: 1px solid var(--rule); margin: 2em 0; }
.cell.code { margin: 1.1em 0; border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; background: var(--code-bg); }
.cell.code pre { margin: 0; padding: 10px 14px; font-family: Menlo, 'SF Mono', Consolas, monospace; font-size: 0.78em; line-height: 1.45; white-space: pre-wrap; word-break: break-word; }
.cell.code .label { font: 700 0.62rem 'Avenir Next', 'Helvetica Neue', Arial, sans-serif; letter-spacing: 0.12em; color: var(--muted); padding: 5px 14px 0; text-transform: uppercase; }
.cell.code .outputs { background: #fff; border-top: 1px solid var(--rule); padding: 8px 14px; }
.cell.code .outputs pre { padding: 4px 0; font-size: 0.78em; color: #222; }
.cell.code .outputs pre.error { color: #a01d26; }
.cell.code .outputs img { display: block; max-width: 100%; margin: 8px auto; border-radius: 8px; }
.cell.code .outputs .caption { font-size: 0.9em; color: var(--muted); margin: 0 0 8px; }
.cell.code .outputs img + .caption { margin-top: -2px; }
.katex-display { overflow-x: auto; overflow-y: hidden; padding: 0.2em 0; }
.toc { margin-top: 2em; }
.toc ul { margin: 0.2em 0 0.9em; padding-left: 1.4em; }
.toc p { margin: 0.9em 0 0.1em; }
.toc a { color: var(--accent); }
.toc li { margin: 0.1em 0; }
.titlepage { min-height: 78vh; display: flex; flex-direction: column; justify-content: center; text-align: center; }
.titlepage > div { max-width: 36em; margin: 0 auto; }
.titlepage h1 { font-size: 3.4rem; border: none; margin: 0 0 0.3em; padding: 0; color: #14213d; letter-spacing: 0.01em; }
.titlepage h3 { font-size: 1.35rem; font-weight: 500; color: var(--accent); margin: 0 0 1.6em; }
.titlepage p { color: var(--muted); }
.cell.shown { margin: 1.4em 0; }
.cell.shown img { display: block; max-width: 100%; margin: 0 auto 6px; border-radius: 10px; box-shadow: 0 2px 10px rgba(20, 33, 61, 0.25); }
.cell.shown .caption { font-size: 0.88em; color: var(--muted); }
.cell.shown .caption p { margin: 0.2em 0; }
@media print {
  body { background: #fff; font-size: 10.4pt; line-height: 1.5; }
  main { max-width: none; padding: 0; box-shadow: none; }
  h1 { break-before: page; margin-top: 0; font-size: 20pt; }
  .titlepage { min-height: 0; height: 245mm; break-after: page; }
  .titlepage h1 { break-before: avoid; font-size: 40pt; }
  .toc { break-before: page; margin-top: 0; }
  .toc-part { break-inside: avoid; }
  .toc a { color: #1f4e8c; }
  h2, h3, h4 { break-after: avoid; }
  .cell.code { break-inside: auto; } .cell.code .outputs img { break-inside: avoid; max-height: 230mm; }
  table { display: table; break-inside: auto; } tr { break-inside: avoid; }
  a { color: inherit; }
}
@page { size: A4; margin: 18mm 17mm 20mm; @bottom-center { content: counter(page); font: 9pt 'Avenir Next', Arial, sans-serif; color: #777; } }
"""

AUTORENDER = """
renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}], throwOnError: false, strict: 'ignore'});
"""

CODE = re.compile(r"```.*?```|`[^`\n]+`", re.DOTALL)
DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
INLINE = re.compile(r"(?<![\\$])\$(?![\s$])((?:[^$\n\\]|\\.)+?)(?<![\s\\])\$(?![\w$])")


def katex_css() -> str:
    """katex.min.css with the fonts inlined (woff2 where the font exists in that format, else woff)."""
    css = (VENDOR / "katex.min.css").read_text()

    def face(match: re.Match) -> str:
        name = match.group(1)
        for ext, fmt in (("woff2", "woff2"), ("woff", "woff")):
            font = VENDOR / "fonts" / f"{name}.{ext}"
            if font.exists():
                return f"src:url(data:font/{ext};base64,{base64.b64encode(font.read_bytes()).decode()}) format(\"{fmt}\")"
        raise FileNotFoundError(f"KaTeX font {name} missing in {VENDOR / 'fonts'}")

    return re.sub(r"src:url\(fonts/(KaTeX_[A-Za-z0-9-]+)\.woff2\)[^}]*?(?=\})", face, css)


def markdown_engine() -> MarkdownIt:
    def highlighter(text: str, lang: str, _attrs: str) -> str:
        if lang in ("python", "py", ""):
            return f'<pre class="source">{highlight(text, PythonLexer(), HtmlFormatter(nowrap=True))}</pre>'
        return f"<pre>{html.escape(text)}</pre>"

    return MarkdownIt("commonmark", {"highlight": highlighter, "html": True}).enable("table")


def render_markdown(text: str, engine: MarkdownIt) -> str:
    """Markdown to HTML with the TeX spans protected from the Markdown parser and put back untouched for KaTeX (code spans and fenced code are left alone)."""
    code_parts: list[str] = []
    spans: list[str] = []

    def hold_code(match: re.Match) -> str:
        code_parts.append(match.group(0))
        return f"CODEPART{len(code_parts) - 1}END"

    def keep(match: re.Match) -> str:
        spans.append(match.group(0))
        return f"MATHSPAN{len(spans) - 1}END"

    protected = INLINE.sub(keep, DISPLAY.sub(keep, CODE.sub(hold_code, text)))
    protected = re.sub(r"CODEPART(\d+)END", lambda m: code_parts[int(m.group(1))], protected)
    tokens = engine.parse(protected)
    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            token.attrSet("id", slug(tokens[i + 1].content))
    out = engine.renderer.render(tokens, engine.options, {})
    return re.sub(r"MATHSPAN(\d+)END", lambda m: html.escape(spans[int(m.group(1))], quote=False), out)


def render_outputs(outputs: list[dict], engine: MarkdownIt) -> str:
    parts = []
    for o in outputs:
        kind = o["output_type"]
        if kind == "stream":
            parts.append(f"<pre>{html.escape(''.join(o['text']))}</pre>")
        elif kind == "error":
            parts.append(f"<pre class=\"error\">{html.escape(chr(10).join(o['traceback']))}</pre>")
        elif kind == "display_data":
            data = o["data"]
            if "image/png" in data:
                parts.append(f'<img alt="{html.escape(o.get("metadata", {}).get("title", "figure"))}" src="data:image/png;base64,{data["image/png"]}">')
            elif "text/markdown" in data:
                body = render_markdown("".join(data["text/markdown"]) if isinstance(data["text/markdown"], list) else data["text/markdown"], engine)
                parts.append(f'<div class="caption">{body}</div>' if o.get("metadata", {}).get("caption") else f"<div>{body}</div>")
    return "".join(parts)


def build_html(notebook: Path = NOTEBOOK, out: Path | None = None) -> Path:
    nb = json.loads(notebook.read_text())
    engine = markdown_engine()
    body = []
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            meta = cell.get("metadata", {})
            kind = " titlepage" if meta.get("titlepage") else " toc" if meta.get("generated") and source.startswith("## Table of contents") else ""
            body.append(f'<section class="cell md{kind}">{render_markdown(source, engine)}</section>')
        else:
            outputs = render_outputs(cell.get("outputs", []), engine)
            if cell.get("metadata", {}).get("jupyter", {}).get("source_hidden"):
                body.append(f'<div class="cell shown">{outputs}</div>')
                continue
            body.append(f'<div class="cell code"><div class="label">code</div><pre class="source">{highlight(source, PythonLexer(), HtmlFormatter(nowrap=True))}</pre>' + (f'<div class="outputs">{outputs}</div>' if outputs else "") + "</div>")
    page = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(TITLE)}</title>'
        f"<style>{katex_css()}{HtmlFormatter().get_style_defs('.source')}{CSS}</style></head><body><main>{''.join(body)}</main>"
        f"<script>{(VENDOR / 'katex.min.js').read_text()}</script><script>{(VENDOR / 'auto-render.min.js').read_text()}</script><script>{AUTORENDER}</script></body></html>"
    )
    out = out or notebook.with_suffix(".html")
    out.write_text(page)
    return out


def build_pdf(page: Path, out: Path | None = None) -> Path:
    out = out or page.with_suffix(".pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files", "--generate-pdf-document-outline", f"--print-to-pdf={out}", page.resolve().as_uri()], check=True, capture_output=True, timeout=1200)
    return out


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"{NOTEBOOK.name} not found; run python -m notebooks.build_notebook first", file=sys.stderr)
        return 1
    page = build_html()
    print(f"{page.name}: {page.stat().st_size / 1e6:.1f} MB")
    if Path(CHROME).exists():
        pdf = build_pdf(page)
        print(f"{pdf.name}: {pdf.stat().st_size / 1e6:.1f} MB")
    else:
        print("Chrome not found; the PDF was not written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
