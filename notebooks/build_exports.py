# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Export the executed notebook to a single HTML file, a PDF, and a web edition: ``python -m notebooks.build_exports``.

The HTML is written here rather than by nbconvert so that it needs nothing from the network: the equations are typeset in the page by the vendored KaTeX (scripts, stylesheet and
fonts inlined from ``notebooks/_vendor/katex``), code is highlighted by Pygments, and figures are embedded. The PDF is the same page printed by headless Chrome, with the headings as its bookmarks and the entries of the contents as links to their sections.

The web edition (``notebooks/web/index.html``) is the same text for reading on a screen: a dark and a light theme that follow the page that embeds it, a contents beside the text with a search of the whole
notebook, code and long outputs that open on request, and each catalogued figure in both themes as a file of its own that is fetched when it comes into view. ``python -m notebooks.build_exports --web`` writes only that edition.
"""

import base64
import html
import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

from notebooks.notebook_builder import INLINE_MATH, slug

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "cypmode_notebook.ipynb"
VENDOR = HERE / "_vendor" / "katex"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FIGURES = HERE.parent / "docs" / "figures"
WEB = HERE / "web"
REPOSITORY = "https://github.com/mohamedtalaat-gif/CYPMode"

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
INLINE = INLINE_MATH


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
            token.attrSet("id", slug(re.sub(r"MATHSPAN(\d+)END", lambda m: spans[int(m.group(1))], tokens[i + 1].content)))  # from the heading as written, so that it is the anchor of the contents
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


def dangling_anchors(page: str) -> list[str]:
    """The targets of the internal links of a page that no element id answers. A browser tries the fragment as written and then percent-decoded, so both are tried."""
    ids = {html.unescape(i) for i in re.findall(r'\sid="([^"]*)"', page)}
    targets = [html.unescape(t) for t in re.findall(r'<a [^>]*href="#([^"]*)"', page)]
    return sorted({t for t in targets if t not in ids and unquote(t) not in ids})


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
    dangling = dangling_anchors(page)
    if dangling:
        raise ValueError(f"links of the page that point to no heading, a click on them would do nothing: {dangling}")
    out = out or notebook.with_suffix(".html")
    out.write_text(page)
    return out


WEB_CSS = """
:root, [data-theme="dark"] { --bg: #0f1826; --bg-deep: #0a111c; --panel: #162234; --rule: #2b3d57; --text: #e1e8f2; --muted: #93a3ba; --fe: #e5773d; --n: #6c97ff; --o: #e8575d; --code: #0a111c; color-scheme: dark; }
[data-theme="light"] { --bg: #f2f5f9; --bg-deep: #e6ecf3; --panel: #ffffff; --rule: #cdd6e3; --text: #0e1726; --muted: #51627b; --fe: #b4511a; --n: #2c54c9; --o: #c43c42; --code: #f8fafc; color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: 'Source Serif 4 Variable', Charter, 'Iowan Old Style', Georgia, serif; font-size: 17px; line-height: 1.62; }
h1, h2, h3, h4, .bar, .side, summary, button, input { font-family: 'Barlow Semi Condensed', 'Avenir Next', 'Helvetica Neue', Arial, sans-serif; }
.bar { position: sticky; top: 0; z-index: 20; display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: center; padding: 0.55rem 1rem; background: color-mix(in srgb, var(--bg) 92%, transparent); backdrop-filter: blur(6px); border-bottom: 1px solid var(--rule); }
.bar .name { margin-right: auto; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 34rem; }
.bar button, .bar input { font-size: 0.95rem; color: var(--text); background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 0.3rem 0.7rem; }
.bar button { cursor: pointer; }
.bar button:hover, .bar input:focus { border-color: var(--n); outline: none; }
.bar input { width: min(20rem, 40vw); }
#contents-btn { display: none; }
.embedded .bar .name, .embedded #theme { display: none; }
.embedded .bar input { margin-right: auto; }
.layout { display: grid; grid-template-columns: 310px minmax(0, 1fr); max-width: 1400px; margin: 0 auto; }
.side { position: sticky; top: 49px; align-self: start; height: calc(100vh - 49px); overflow-y: auto; padding: 1rem 0.8rem 3rem 1rem; border-right: 1px solid var(--rule); font-size: 0.95rem; line-height: 1.3; }
.side ul { list-style: none; margin: 0.15rem 0 0.5rem; padding-left: 0.7rem; }
.side a { display: block; padding: 0.18rem 0.4rem; color: var(--muted); text-decoration: none; border-left: 2px solid transparent; border-radius: 0 4px 4px 0; }
.side a:hover { color: var(--text); }
.side a.active { color: var(--text); background: var(--bg-deep); border-left-color: var(--fe); }
.side details > summary { cursor: pointer; list-style: none; }
.side details > summary::-webkit-details-marker { display: none; }
.side .part > summary a { color: var(--text); font-weight: 600; }
.side .hits a { padding: 0.35rem 0.4rem; }
.side .hits b { display: block; color: var(--text); font-weight: 600; }
.side .hits small { color: var(--muted); }
.side .note { margin: 0 0 0.5rem; color: var(--muted); }
.side .files { margin-top: 1.6rem; padding-top: 0.8rem; border-top: 1px solid var(--rule); }
.side .files p { margin: 0 0 0.3rem; color: var(--muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; }
main { min-width: 0; max-width: 880px; padding: 1.5rem 2rem 6rem; }
h1, h2, h3, h4 { line-height: 1.25; color: var(--text); scroll-margin-top: 64px; }
h1 { font-size: 2rem; margin: 2.6em 0 0.6em; padding-bottom: 0.25em; border-bottom: 2px solid var(--fe); }
h1.title { font-size: 2.6rem; border: none; margin-top: 0.6em; }
h2 { font-size: 1.5rem; margin: 2em 0 0.5em; color: var(--n); }
h3 { font-size: 1.2rem; margin: 1.6em 0 0.4em; }
h4 { font-size: 1.05rem; margin: 1.3em 0 0.3em; }
.anc { margin-left: 0.4em; font-size: 0.8em; color: var(--muted); text-decoration: none; opacity: 0; }
.anc::before { content: '#'; }
h2:hover .anc, h3:hover .anc, h4:hover .anc, .anc:focus { opacity: 1; }
p { margin: 0.7em 0; hyphens: auto; }
a { color: var(--n); text-decoration: none; }
a:hover { text-decoration: underline; }
code { font-family: Menlo, 'SF Mono', Consolas, monospace; font-size: 0.84em; background: var(--bg-deep); padding: 1px 4px; border-radius: 4px; }
table { display: block; border-collapse: collapse; margin: 1em 0; font-size: 0.88em; overflow-x: auto; }
th, td { border: 1px solid var(--rule); padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: var(--bg-deep); }
blockquote { margin: 1em 0; padding: 0.2em 1em; background: var(--bg-deep); border-left: 4px solid var(--fe); }
hr { border: none; border-top: 1px solid var(--rule); margin: 2em 0; }
.katex-display { overflow-x: auto; overflow-y: hidden; padding: 0.2em 0; }
.cell.toc { display: none; }
.titlepage { display: flex; flex-direction: column; justify-content: center; min-height: 42vh; text-align: center; }
.titlepage h1 { margin: 0 0 0.3em; padding: 0; border: none; font-size: 3rem; }
.titlepage h3 { margin: 0 0 1.4em; color: var(--n); font-weight: 500; }
.titlepage p { color: var(--muted); }
.cell.code { margin: 1.2em 0; }
details.src { overflow: hidden; background: var(--code); border: 1px solid var(--rule); border-radius: 8px; }
details.src > summary { padding: 0.35rem 0.9rem; color: var(--muted); font-size: 0.78rem; letter-spacing: 0.1em; text-transform: uppercase; cursor: pointer; }
details.src > summary span { margin-left: 0.6em; letter-spacing: 0; text-transform: none; }
details.src > pre { margin: 0; padding: 0.7rem 0.9rem; border-top: 1px solid var(--rule); font: 0.78em/1.45 Menlo, 'SF Mono', Consolas, monospace; white-space: pre-wrap; word-break: break-word; }
.outputs { margin-top: 0.4rem; }
.outputs pre { margin: 0.3rem 0; padding: 0.5rem 0.9rem; background: var(--bg-deep); border-radius: 6px; font: 0.78em/1.45 Menlo, 'SF Mono', Consolas, monospace; white-space: pre-wrap; word-break: break-word; }
.outputs pre.error { color: var(--o); }
.outputs .caption { margin: 0 0 8px; color: var(--muted); font-size: 0.9em; }
details.out > summary { padding: 0.2rem 0; color: var(--muted); font-size: 0.82rem; cursor: pointer; }
img { max-width: 100%; height: auto; }
.figure, .cell.shown { margin: 1.6em 0; }
.figure img { display: block; margin: 0 auto 6px; border-radius: 10px; }
.figure figcaption { color: var(--muted); font-size: 0.88em; }
.figure figcaption p { margin: 0.2em 0; }
.figure .fig-light { display: none; }
[data-theme="light"] .figure .fig-light { display: block; }
[data-theme="light"] .figure .fig-dark { display: none; }
[data-theme="light"] .figure img { box-shadow: 0 2px 10px rgba(14, 23, 38, 0.15); }
img.plain { display: block; margin: 8px auto; background: #fff; border-radius: 8px; }
[data-theme="dark"] img.plain, :root:not([data-theme]) img.plain { background: transparent; filter: invert(0.9) hue-rotate(180deg); }
@media (max-width: 980px) {
  .layout { grid-template-columns: 1fr; }
  #contents-btn { display: inline-block; }
  .bar .name, #code, #outputs { display: none; }
  .bar input { flex: 1 1 8rem; width: auto; }
  .side { position: fixed; z-index: 30; top: 49px; left: 0; width: min(340px, 88vw); height: calc(100vh - 49px); background: var(--bg); transform: translateX(-102%); transition: transform 0.2s; }
  body.side-open .side { transform: none; box-shadow: 0 0 30px rgba(0, 0, 0, 0.45); }
  main { padding: 1rem 1rem 5rem; }
}
"""

WEB_HEAD_SCRIPT = """(function () { var t = null; try { t = new URLSearchParams(location.search).get('theme') || localStorage.getItem('cypmode-notebook-theme'); } catch (e) {}
if (t !== 'light' && t !== 'dark') t = window.matchMedia && matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
document.documentElement.setAttribute('data-theme', t); if (window.self !== window.top) document.documentElement.className = 'embedded'; })();"""

WEB_JS = """
(function () {
  'use strict';
  var root = document.documentElement;
  var main = document.getElementById('main');
  var side = document.getElementById('side');
  var tree = document.getElementById('tree');
  var hits = document.getElementById('hits');

  function setTheme(value) { if (value === 'dark' || value === 'light') root.setAttribute('data-theme', value); }
  window.addEventListener('message', function (event) {
    if (event.origin === location.origin && event.data && event.data.cypmodeTheme) setTheme(event.data.cypmodeTheme);
  });
  document.getElementById('theme').addEventListener('click', function () {
    var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    setTheme(next);
    try { localStorage.setItem('cypmode-notebook-theme', next); } catch (e) {}
  });
  document.getElementById('contents-btn').addEventListener('click', function () { document.body.classList.toggle('side-open'); });

  renderMathInElement(main, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }], throwOnError: false, strict: 'ignore' });

  var heads = Array.prototype.filter.call(main.querySelectorAll('h1[id], h2[id]'), function (h) {
    return !h.classList.contains('title') && !h.closest('.titlepage') && !h.closest('.toc');
  });
  var links = {};
  var list = null;
  heads.forEach(function (h) {
    var a = document.createElement('a');
    a.href = '#' + h.id;
    a.setAttribute('data-for', h.id);
    a.innerHTML = h.innerHTML;
    links[h.id] = a;
    if (h.tagName === 'H1') {
      var part = document.createElement('details');
      var summary = document.createElement('summary');
      part.className = 'part';
      summary.appendChild(a);
      part.appendChild(summary);
      list = document.createElement('ul');
      part.appendChild(list);
      tree.appendChild(part);
    } else if (list) {
      var item = document.createElement('li');
      item.appendChild(a);
      list.appendChild(item);
    }
  });
  tree.addEventListener('click', function (event) { if (event.target.closest('a')) document.body.classList.remove('side-open'); });

  Array.prototype.forEach.call(main.querySelectorAll('h2[id], h3[id], h4[id]'), function (h) {
    var anchor = document.createElement('a');
    anchor.className = 'anc';
    anchor.href = '#' + h.id;
    anchor.setAttribute('aria-label', 'Link to this section');
    h.appendChild(anchor);
  });

  var current = null;
  var openedPart = null;
  function activate(id) {
    if (id === current || !links[id]) return;
    if (current && links[current]) links[current].classList.remove('active');
    current = id;
    var link = links[id];
    link.classList.add('active');
    var part = link.closest('details.part');
    if (part && part !== openedPart) {
      if (openedPart) openedPart.open = false;
      part.open = true;
      openedPart = part;
    }
    if (!tree.hidden) {
      var box = side.getBoundingClientRect();
      var spot = link.getBoundingClientRect();
      if (spot.top < box.top + 40 || spot.bottom > box.bottom - 40) side.scrollTop += spot.top - box.top - side.clientHeight / 2;
    }
  }
  var waiting = false;
  function spy() {
    waiting = false;
    var found = null;
    for (var i = 0; i < heads.length; i += 1) {
      if (heads[i].getBoundingClientRect().top <= 90) found = heads[i]; else break;
    }
    if (found) activate(found.id);
  }
  window.addEventListener('scroll', function () { if (!waiting) { waiting = true; requestAnimationFrame(spy); } }, { passive: true });

  function reveal() {
    var id = '';
    try { id = decodeURIComponent(location.hash.slice(1)); } catch (e) { id = location.hash.slice(1); }
    var target = id ? document.getElementById(id) : null;
    if (!target) return;
    for (var d = target.closest('details'); d; d = d.parentElement ? d.parentElement.closest('details') : null) d.open = true;
    target.scrollIntoView();
  }
  window.addEventListener('hashchange', reveal);

  function toggleAll(button, selector, show, hide) {
    button.addEventListener('click', function () {
      var open = button.getAttribute('aria-pressed') !== 'true';
      Array.prototype.forEach.call(main.querySelectorAll(selector), function (d) { d.open = open; });
      button.setAttribute('aria-pressed', String(open));
      button.textContent = open ? hide : show;
    });
  }
  toggleAll(document.getElementById('code'), 'details.src', 'Show all code', 'Hide all code');
  toggleAll(document.getElementById('outputs'), 'details.out', 'Show all outputs', 'Hide all outputs');

  var index = null;
  function buildIndex() {
    index = [];
    var entry = null;
    Array.prototype.forEach.call(main.children, function (cell) {
      if (cell.classList.contains('toc')) return;
      var h = cell.classList.contains('titlepage') ? null : cell.querySelector('h1[id], h2[id]');
      if (h) { entry = { id: h.id, title: h.textContent, html: links[h.id] ? links[h.id].innerHTML : '', text: '' }; index.push(entry); }
      if (entry) entry.text += ' ' + cell.textContent;
    });
    index.forEach(function (e) { e.text = e.text.replace(/\\s+/g, ' '); e.low = e.text.toLowerCase(); e.lowTitle = e.title.toLowerCase(); });
  }
  function search(query) {
    var terms = query.toLowerCase().split(/\\s+/).filter(Boolean);
    hits.textContent = '';
    if (!terms.length || query.trim().length < 2) { hits.hidden = true; tree.hidden = false; return; }
    if (!index) buildIndex();
    var found = [];
    index.forEach(function (e) {
      var score = 0;
      for (var i = 0; i < terms.length; i += 1) {
        var n = e.low.split(terms[i]).length - 1;
        if (!n) return;
        score += n + (e.lowTitle.indexOf(terms[i]) >= 0 ? 20 : 0);
      }
      found.push({ e: e, score: score });
    });
    found.sort(function (a, b) { return b.score - a.score; });
    tree.hidden = true;
    hits.hidden = false;
    var note = document.createElement('p');
    note.className = 'note';
    note.textContent = found.length ? found.length + (found.length === 1 ? ' section' : ' sections') + (found.length > 30 ? ', the best 30 shown' : '') : 'No section contains all the words.';
    hits.appendChild(note);
    found.slice(0, 30).forEach(function (f) {
      var a = document.createElement('a');
      var at = f.e.low.indexOf(terms[0], f.e.lowTitle.length);
      if (at < 0) at = Math.max(0, f.e.low.indexOf(terms[0]));
      var snippet = f.e.text.slice(Math.max(0, at - 60), at + 90);
      a.href = '#' + f.e.id;
      a.innerHTML = '<b></b><small></small>';
      if (f.e.html) a.firstChild.innerHTML = f.e.html; else a.firstChild.textContent = f.e.title;
      a.lastChild.textContent = (at > 60 ? '\\u2026' : '') + snippet + '\\u2026';
      a.addEventListener('click', function () { document.body.classList.remove('side-open'); });
      hits.appendChild(a);
    });
  }
  var timer = null;
  document.getElementById('q').addEventListener('input', function (event) {
    var value = event.target.value;
    clearTimeout(timer);
    timer = setTimeout(function () { search(value); }, 160);
  });

  reveal();
  spy();
})();
"""


def png_size(head: bytes) -> tuple[int, int]:
    """Width and height from the first bytes of a PNG file."""
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    return struct.unpack(">II", head[16:24])


def figure_files(figure_id: str) -> dict[str, Path]:
    """The light and the dark file of a catalogued figure (docs/figures/<group>/<id>-<theme>.png)."""
    files = {}
    for theme in ("light", "dark"):
        found = sorted(FIGURES.glob(f"*/{figure_id}-{theme}.png"))
        if not found:
            raise FileNotFoundError(f"figure {figure_id}-{theme}.png not found under docs/figures; run docs/figures/build_all.py")
        files[theme] = found[0]
    return files


def web_figure(figure_id: str, title: str, caption: str) -> str:
    """A catalogued figure for the web edition: both themes as files of their own, each with its size so that the page does not move while they load."""
    images = []
    for theme, path in figure_files(figure_id).items():
        with path.open("rb") as handle:
            width, height = png_size(handle.read(32))
        images.append(f'<img class="fig-{theme}" loading="lazy" decoding="async" width="{width}" height="{height}" src="figures/{figure_id}-{theme}.png" alt="{html.escape(title)}">')
    return f'<figure class="figure" id="fig-{html.escape(figure_id)}">{"".join(images)}<figcaption>{caption}</figcaption></figure>'


def web_outputs(outputs: list[dict], engine: MarkdownIt) -> tuple[str, list[str]]:
    """The outputs of a cell for the web edition and the catalogued figures that they show. A long stream folds away; a figure takes the caption that follows it."""
    parts: list[str] = []
    shown: list[str] = []
    i = 0
    while i < len(outputs):
        o = outputs[i]
        kind = o["output_type"]
        data = o.get("data", {})
        meta = o.get("metadata", {})
        i += 1
        if kind == "stream":
            text = "".join(o["text"])
            lines = text.count("\n") + 1
            parts.append(f'<details class="out"><summary>output, {lines} lines</summary><pre>{html.escape(text)}</pre></details>' if lines > 16 else f"<pre>{html.escape(text)}</pre>")
        elif kind == "error":
            parts.append(f'<pre class="error">{html.escape(chr(10).join(o["traceback"]))}</pre>')
        elif kind == "display_data" and "image/png" in data and "figure" in meta:
            caption = ""
            nxt = outputs[i] if i < len(outputs) else None
            if nxt and nxt.get("output_type") == "display_data" and "text/markdown" in nxt.get("data", {}) and nxt.get("metadata", {}).get("caption") == meta["figure"]:
                markdown = nxt["data"]["text/markdown"]
                caption = render_markdown("".join(markdown) if isinstance(markdown, list) else markdown, engine)
                i += 1
            shown.append(meta["figure"])
            parts.append(web_figure(meta["figure"], meta.get("title", "figure"), caption))
        elif kind == "display_data" and "image/png" in data:
            width, height = png_size(base64.b64decode(data["image/png"][:48]))
            parts.append(f'<img class="plain" loading="lazy" width="{width}" height="{height}" alt="{html.escape(meta.get("title", "figure"))}" src="data:image/png;base64,{data["image/png"]}">')
        elif kind == "display_data" and "text/markdown" in data:
            markdown = data["text/markdown"]
            body = render_markdown("".join(markdown) if isinstance(markdown, list) else markdown, engine)
            parts.append(f'<div class="caption">{body}</div>' if meta.get("caption") else f"<div>{body}</div>")
    return "".join(parts), shown


def build_web(notebook: Path = NOTEBOOK, out: Path | None = None) -> Path:
    """Write the web edition: index.html, the list of the catalogued figures it shows (figures.txt), and a copy of those figures in a folder of their own (figures/, not committed; the Pages workflow copies them from docs/figures)."""
    out = out or WEB
    nb = json.loads(notebook.read_text())
    engine = markdown_engine()
    body: list[str] = []
    figures: list[str] = []
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            meta = cell.get("metadata", {})
            kind = " titlepage" if meta.get("titlepage") else " toc" if meta.get("generated") and source.startswith("## Table of contents") else ""
            body.append(f'<section class="cell md{kind}">{render_markdown(source, engine)}</section>')
            continue
        outputs, shown = web_outputs(cell.get("outputs", []), engine)
        figures.extend(shown)
        if cell.get("metadata", {}).get("jupyter", {}).get("source_hidden"):
            body.append(f'<div class="cell shown">{outputs}</div>')
            continue
        lines = source.count("\n") + 1
        code = f'<details class="src"><summary>code<span>{lines} lines</span></summary><pre class="source">{highlight(source, PythonLexer(), HtmlFormatter(nowrap=True))}</pre></details>'
        body.append(f'<div class="cell code">{code}{f"<div class=outputs>{outputs}</div>" if outputs else ""}</div>')
    files = (
        f'<div class="files"><p>Files</p><a href="standalone/">The complete page, one file</a><a href="{REPOSITORY}/blob/main/notebooks/cypmode_notebook.pdf">PDF</a>'
        f'<a href="{REPOSITORY}/blob/main/notebooks/cypmode_notebook.ipynb">Jupyter notebook</a><a href="{REPOSITORY}">Repository</a></div>'
    )
    dark_code = HtmlFormatter(style="github-dark").get_style_defs('[data-theme="dark"] .source')
    styles = f"{katex_css()}{HtmlFormatter().get_style_defs('.source')}{dark_code}{WEB_CSS}"
    page = (
        f'<!doctype html><html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(TITLE)}</title>'
        f"<script>{WEB_HEAD_SCRIPT}</script><style>{styles}</style></head><body>"
        '<header class="bar"><button id="contents-btn" type="button">Contents</button>'
        f'<span class="name">{html.escape(TITLE)}</span><input id="q" type="search" placeholder="Search the notebook" aria-label="Search the notebook">'
        '<button id="code" type="button" aria-pressed="false">Show all code</button><button id="outputs" type="button" aria-pressed="false">Show all outputs</button>'
        '<button id="theme" type="button">Theme</button></header>'
        f'<div class="layout"><nav id="side" class="side" aria-label="Contents"><div id="hits" class="hits" hidden></div><div id="tree"></div>{files}</nav><main id="main">{"".join(body)}</main></div>'
        f"<script>{(VENDOR / 'katex.min.js').read_text()}</script><script>{(VENDOR / 'auto-render.min.js').read_text()}</script><script>{WEB_JS}</script></body></html>"
    )
    dangling = dangling_anchors(page)
    if dangling:
        raise ValueError(f"links of the page that point to no heading, a click on them would do nothing: {dangling}")
    out.mkdir(parents=True, exist_ok=True)
    unique = sorted(set(figures))
    (out / "figures.txt").write_text("".join(f"{name}\n" for name in unique))
    folder = out / "figures"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir()
    for name in unique:
        for path in figure_files(name).values():
            shutil.copyfile(path, folder / path.name)
    target = out / "index.html"
    target.write_text(page)
    return target


def build_pdf(page: Path, out: Path | None = None) -> Path:
    out = out or page.with_suffix(".pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files", "--generate-pdf-document-outline", f"--print-to-pdf={out}", page.resolve().as_uri()], check=True, capture_output=True, timeout=1200)
    return out


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"{NOTEBOOK.name} not found; run python -m notebooks.build_notebook first", file=sys.stderr)
        return 1
    if "--web" in sys.argv[1:]:
        web = build_web()
        print(f"{web.parent.name}/{web.name}: {web.stat().st_size / 1e6:.1f} MB")
        return 0
    page = build_html()
    print(f"{page.name}: {page.stat().st_size / 1e6:.1f} MB")
    if Path(CHROME).exists():
        pdf = build_pdf(page)
        print(f"{pdf.name}: {pdf.stat().st_size / 1e6:.1f} MB")
    else:
        print("Chrome not found; the PDF was not written")
    web = build_web()
    print(f"{web.parent.name}/{web.name}: {web.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
