# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The animated preview of the interactive map: ``python docs/map/preview/build_preview.py`` writes ``docs/map/preview/explorer.gif`` from the page ``docs/map/explorer.html``.

GitHub shows a picture and not a page, so the README carries a short loop of the map: the search for a word, a module with what it imports and what imports it, the job that runs it, the flow of
the data and the kinds of idea. The frames are drawn by the headless Chrome that draws the film (``docs/video/anime/chrome.py``) from the page itself, so that nothing in the loop is a
drawing: it is the page, driven through its own functions (``go`` selects a module, a job, a stage or a kind; ``render`` redraws).
"""

import io
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "docs/video/anime"))
PAGE = ROOT / "docs/map/explorer.html"
OUT = ROOT / "docs/map/preview/explorer.gif"
WIDTH = 960  # pixels of the loop; the README shows it at about 800
MODULE = "cypmode.validation.final_recipe"
SEARCH = "boltz"


def shot(chrome) -> Image.Image:
    import base64

    data = chrome.call("Page.captureScreenshot", format="png")["data"]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def frames_of(chrome) -> list[tuple[Image.Image, int]]:
    """The frames of the loop with the time that each stays on the screen, in milliseconds."""
    chrome.open(PAGE)
    time.sleep(1.5)
    frames: list[tuple[Image.Image, int]] = []

    def keep(ms: int, settle: float = 0.35) -> None:
        time.sleep(settle)
        frames.append((shot(chrome), ms))

    keep(1800)  # the graph of the modules
    for typed in (SEARCH[:1], SEARCH[:3], SEARCH):
        chrome.evaluate(f"document.querySelector('#q').value = {typed!r}; state.query = {typed!r}; render()")
        keep(350 if typed != SEARCH else 1900)
    chrome.evaluate(f"document.querySelector('#q').value = ''; state.query = ''; go('module', {MODULE!r})")
    keep(2800)  # a module: what it imports, what imports it, the jobs that run it, the tests, the section of the notebook
    job = chrome.evaluate(f"modById.get({MODULE!r}).runBy[0] || ''")
    if job:
        chrome.evaluate(f"go('job', {job!r})")
        keep(2400)  # the job that runs it: what it waits for and which logs it writes
    chrome.evaluate("go('stage', 'logs')")
    keep(2000)
    chrome.evaluate("go('kind', DATA.evidence[DATA.evidence.length - 1].kind)")
    keep(2000)
    chrome.evaluate("state.view = 'code'; state.sel = null; render()")
    keep(1200)
    return frames


def main() -> int:
    from chrome import Chrome

    with Chrome() as chrome:
        frames = frames_of(chrome)
    height = round(frames[0][0].height * WIDTH / frames[0][0].width)
    small = [f.resize((WIDTH, height), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=160) for f, _ in frames]
    small[0].save(OUT, save_all=True, append_images=small[1:], duration=[ms for _, ms in frames], loop=0, optimize=True, disposal=2)
    print(f"{OUT.relative_to(ROOT)}: {len(frames)} frames, {OUT.stat().st_size / 1e6:.2f} MB, {sum(ms for _, ms in frames) / 1000:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
