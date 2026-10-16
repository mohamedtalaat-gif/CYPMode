"""Cut a painted portrait out of its plain white background: ``python docs/video/characters/cutout.py <portrait.png> [<out.png>]``.

The portraits are painted on flat white. The background is the white region connected to the border of the picture (a flood fill with a small tolerance); white that lies inside the figure, such as a
lab coat or a shirt, is not connected to the border through white and stays. The edge is softened over a few pixels and any white fringe is removed by pulling the colour of the edge pixels
towards the figure (un-premultiplying against white), so the figure sits on a coloured scene without a halo.
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

TOLERANCE = 18  # largest distance to pure white (per channel) that still counts as background


def background_mask(rgb: np.ndarray, tolerance: int = TOLERANCE) -> np.ndarray:
    """True where a pixel is background: white within ``tolerance`` and connected to the border through such pixels."""
    near_white = (255 - rgb).max(axis=2) <= tolerance
    h, w = near_white.shape
    seen = np.zeros_like(near_white)
    queue = deque()
    for x in range(w):
        for y in (0, h - 1):
            if near_white[y, x] and not seen[y, x]:
                seen[y, x] = True
                queue.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if near_white[y, x] and not seen[y, x]:
                seen[y, x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and near_white[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                queue.append((ny, nx))
    return seen


def cutout(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    background = background_mask(rgb.astype(np.uint8))
    alpha = Image.fromarray(np.where(background, 0, 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.1))
    a = np.asarray(alpha).astype(np.float32) / 255.0
    # remove the white that the soft edge still carries: colour = (observed - (1 - a) * white) / a
    safe = np.clip(a, 0.15, 1.0)[..., None]
    colour = np.clip((rgb - (1.0 - a[..., None]) * 255.0) / safe, 0, 255)
    colour = np.where(a[..., None] < 0.15, rgb, colour)
    out = np.dstack([colour, a * 255.0]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def main() -> int:
    source = Path(sys.argv[1])
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_name(source.stem + "_cut.png")
    cutout(Image.open(source)).save(target)
    print(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
