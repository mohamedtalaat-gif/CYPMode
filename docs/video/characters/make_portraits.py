"""Paint the calm portrait of every member that has none yet, then every missing expression: ``python docs/video/characters/make_portraits.py [--members stacker dock ...]``.

A finished file is kept, so the script can be stopped and started again. The calm portrait comes from ``generate.py``, the other expressions from ``expressions.py``.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bible  # noqa: E402
import expressions  # noqa: E402
import generate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs="*", default=list(bible.STAFF), choices=list(bible.STAFF))
    args = parser.parse_args()
    generate.OUT.mkdir(parents=True, exist_ok=True)
    for member in args.members:
        seed = bible.STAFF[member]["seed"]
        calm = generate.OUT / f"{member}_calm_s{seed}.png"
        if not calm.exists():
            started = time.time()
            calm.write_bytes(generate.run(generate.workflow(bible.positive(member, "calm"), bible.NEGATIVE, seed, f"{member}_calm")))
            print(f"{calm.name} ({time.time() - started:.0f} s)", flush=True)
    print("PORTRAITS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
