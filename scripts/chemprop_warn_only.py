"""Run the Chemprop command line with PyTorch's deterministic algorithms in warn-only mode (used for the runs on the Apple GPU only).

Chemprop's training asks PyTorch for deterministic algorithms whenever it is given a seed. On the Apple GPU (MPS) the scatter reduction of the message passing has no deterministic implementation, so PyTorch stops
the run at once (the speed test of 2026-10-17 ended that way, before any GPU time was measured). With ``warn_only`` the operation runs and PyTorch warns; every other operation stays deterministic and the seed
stays fixed, so a run is repeatable except for the order of the additions inside that one operation. The runs on the processor do not use this launcher.
"""

import sys

import torch

_original = torch.use_deterministic_algorithms


def _warn_only(mode, *, warn_only=False):
    return _original(mode, warn_only=bool(mode) or warn_only)


torch.use_deterministic_algorithms = _warn_only

from chemprop.cli.main import main

if __name__ == "__main__":
    sys.argv[0] = "chemprop"
    sys.exit(main())
