# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The animated preview of the interactive card: ``python docs/card/build_preview.py`` writes ``docs/card/cypmode_interactive_preview.gif`` from the public page, and ``--full`` writes
``docs/card/cypmode_interactive_preview_full.gif`` from the full page, which exists only after the deadline (``scripts/release_after_deadline.py --rebuild-card`` builds both).

GitHub shows a picture and not a page, so the README carries a short loop of the public interactive page (``docs/card/interactive/index.html``): the enzyme turns once, then the second
complex is loaded and the view moves to its ligand and turns, with the iron-nitrogen distance measured on the deposited coordinates on screen. The frames are drawn by the headless Chrome
that draws the film (``docs/video/anime/chrome.py``) from the page itself, so that nothing in the loop is a drawing of a result: it is the page.
"""

import io
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/video/anime"))
PAGES = {"public": ROOT / "docs/card/interactive/index.html", "full": ROOT / "docs/card/interactive/full/index.html"}
OUTS = {"public": ROOT / "docs/card/cypmode_interactive_preview.gif", "full": ROOT / "docs/card/cypmode_interactive_preview_full.gif"}
WIDTH = 560  # pixels of the loop; the viewer is about 780 px wide in the page
TURN_FRAMES, LIGAND_FRAMES, STEP_MS = 24, 16, 90
COMPLEX = 2  # the tile that is loaded after the first turn


def capture(chrome, clip: dict) -> Image.Image:
    import base64

    data = chrome.call("Page.captureScreenshot", format="png", clip={**clip, "scale": 1})["data"]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def frames_of(chrome, page: Path) -> list[Image.Image]:
    """The frames of the loop, in order: one turn of the whole enzyme, then the second complex at its ligand."""
    chrome.open(page)
    time.sleep(1.5)
    rect = chrome.evaluate("(() => { const r = document.querySelector('section.view').getBoundingClientRect(); return {x: r.x, y: r.y, width: r.width, height: r.height}; })()")
    clip = {key: float(rect[key]) for key in ("x", "y", "width", "height")}
    frames = []
    chrome.evaluate("focus('whole'); viewer.render(); 1")
    for _ in range(TURN_FRAMES):
        frames.append(capture(chrome, clip))
        chrome.evaluate(f"viewer.rotate({360 / TURN_FRAMES}, 'y'); viewer.render(); 1")
    chrome.evaluate(f"load({COMPLEX}); document.getElementById('cart').click(); focus('ligand'); viewer.render(); 1")  # the cartoon is switched off so that the ligand is seen
    time.sleep(0.4)
    for _ in range(LIGAND_FRAMES):
        frames.append(capture(chrome, clip))
        chrome.evaluate(f"viewer.rotate({360 / LIGAND_FRAMES}, 'y'); viewer.render(); 1")
    return frames


def assemble(frames: list[Image.Image], path: Path, width: int = WIDTH, step_ms: int = STEP_MS) -> Path:
    """Write the frames as a looping GIF of ``width`` pixels with one adaptive palette of 64 colours per frame."""
    small = [f.resize((width, round(f.height * width / f.width)), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    small[0].save(path, save_all=True, append_images=small[1:], duration=step_ms, loop=0, optimize=True, disposal=2)
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    from chrome import Chrome

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="draw the loop from the full page (built after the deadline) into the file of the complete edition")
    mode = "full" if parser.parse_args(argv).full else "public"
    if not PAGES[mode].exists():
        print(f"{PAGES[mode].relative_to(ROOT)} does not exist: build the {mode} interactive card first (docs/card/build_interactive_card.py{' --full' if mode == 'full' else ''})")
        return 1
    with Chrome() as chrome:
        frames = chrome.retrying(lambda: frames_of(chrome, PAGES[mode]))
    path = assemble(frames, OUTS[mode])
    print(f"wrote {path.relative_to(ROOT)}: {len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
