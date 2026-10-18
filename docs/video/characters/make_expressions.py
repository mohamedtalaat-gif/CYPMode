"""Paint every expression variant that does not exist yet, one after another: ``python docs/video/characters/make_expressions.py [--members kai bits ...]``.

A variant is written to ``docs/video/anime/characters/<member>_<expression>.png`` by ``expressions.py``; a finished file is kept, so the script can be stopped and started again.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bible
import expressions
import generate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs="*", default=list(bible.STAFF), choices=list(bible.STAFF))
    args = parser.parse_args()
    for member in args.members:
        source = min(generate.OUT.glob(f"{member}_calm_s*.png"))
        for expression in (e for e in bible.EXPRESSIONS if e != "calm"):
            target = generate.OUT / f"{member}_{expression}.png"
            if target.exists():
                continue
            started = time.time()
            expressions.repaint(member, expression, source).save(target)
            print(f"{target.name} ({time.time() - started:.0f} s)", flush=True)
    print("EXPRESSIONS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
