"""Draw single frames of the animation (or any page function) to PNG files: ``python docs/video/anime/preview.py <out dir> <expression> [<expression> ...]``.

An expression is JavaScript evaluated in ``index.html`` before the screenshot, for example ``render(12.5)``.
"""

import sys
import time
from pathlib import Path

from chrome import Chrome

HERE = Path(__file__).resolve().parent


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    with Chrome() as chrome:
        chrome.open(HERE / "index.html")
        chrome.evaluate("ready()")
        for number, expression in enumerate(sys.argv[2:]):
            chrome.evaluate(expression)
            time.sleep(0.4)
            chrome.screenshot(out / f"preview_{number:02d}.png")
            print(f"preview_{number:02d}.png <- {expression}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
