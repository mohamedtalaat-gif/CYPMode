"""Expression variants of a painted portrait by repainting only the face: ``python docs/video/characters/expressions.py --member kai --expression blink``.

The calm portrait of a member (``generate.py``) stays the body, the hair and the clothes of every variant. A square crop around the face is cut from it, soft elliptical masks mark only the parts
that move in the expression (both eyes for a blink, the mouth for speech, both for a happy face), and the Anima model repaints those regions in the same ComfyUI workflow family as the portrait (an
image-to-image pass with a noise mask: the latent outside the mask is kept, the inside is re-noised to ``DENOISE`` and denoised under a prompt that names the new expression). The repainted crop
is pasted back through the same soft masks, so the seams are invisible and the hair, the skin and the clothes do not change.
"""

import argparse
import io
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bible  # noqa: E402
import generate  # noqa: E402

CROP = 512
# positions in the 768 x 1024 calm portraits, read off a gridded view of each face: (left eye x, y), (right eye x, y), (mouth x, y)
LANDMARKS = {
    "kai": ((297, 366), (440, 366), (373, 469)),
    "bits": ((320, 265), (440, 265), (388, 342)),
    "stacker": ((343, 241), (423, 252), (380, 328)),
    "gatekeeper": ((347, 195), (427, 192), (384, 263)),
    "dock": ((328, 187), (428, 187), (377, 277)),
    "auditor": ((317, 280), (447, 278), (383, 392)),
    "scout": ((322, 305), (450, 306), (384, 399)),
    "foundry": ((347, 170), (424, 168), (378, 264)),
}
EYE_RX, EYE_RY, MOUTH_RX, MOUTH_RY = 0.34, 0.20, 0.30, 0.17  # half-sizes of the masks as fractions of the distance between the eyes
DENOISE = {"blink": 0.88, "talk": 0.80, "happy": 0.80}
# A mouth that sits under a mustache is a thin line that a small mask leaves unchanged: the mask is larger and the repaint stronger for that member, and the prompt names the teeth.
MOUTH_SCALE = {"foundry": (1.6, 2.0)}  # (width, height) multipliers of the mouth mask
MOUTH_DENOISE = {"foundry": 0.95}
MOUTH_TAGS = {"foundry": "wide open mouth, teeth, lower lip, chin"}
EXPRESSION_TAGS = {
    "blink": ("closed eyes, relaxed, serene, closed mouth, slight smile", "open eyes, looking at viewer, half-closed eyes"),
    "talk": ("open mouth, speaking, looking at viewer", "closed mouth"),
    "happy": ("closed eyes, smile, open mouth, happy, laughing", "open eyes, serious"),
}
STEPS = 24


def upload(image: Image.Image, name: str) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    reply = requests.post(f"{generate.SERVER}/upload/image", files={"image": (name, buffer.getvalue(), "image/png")}, data={"overwrite": "true"}, timeout=60)
    reply.raise_for_status()
    return reply.json()["name"]


def soft_mask(size: int, ellipses: list[tuple[float, float, float, float]]) -> Image.Image:
    """White where the model may repaint: the union of the ellipses (centre x, centre y, half-width, half-height) with a soft edge."""
    yy, xx = np.mgrid[0:size, 0:size]
    mask = np.zeros((size, size))
    for cx, cy, rx, ry in ellipses:
        mask = np.maximum(mask, np.clip((1.15 - (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)) * 4.0, 0, 1))
    return Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3))


def regions(member: str, expression: str, left: int, top: int) -> list[tuple[float, float, float, float]]:
    """The mask ellipses of an expression in the coordinates of the face crop."""
    (lx, ly), (rx_, ry_), (mx, my) = LANDMARKS[member]
    iod = float(np.hypot(rx_ - lx, ry_ - ly))
    eyes = [(x - left, y - top, EYE_RX * iod, EYE_RY * iod) for x, y in ((lx, ly), (rx_, ry_))]
    sx, sy = MOUTH_SCALE.get(member, (1.0, 1.0))
    mouth = [(mx - left, my - top, MOUTH_RX * iod * sx, MOUTH_RY * iod * sy)]
    return {"blink": eyes, "talk": mouth, "happy": eyes + mouth}[expression]


def inpaint_workflow(positive: str, negative: str, image_name: str, mask_name: str, seed: int, denoise: float, prefix: str) -> dict:
    graph = generate.workflow(positive, negative, seed, prefix, width=CROP, height=CROP, steps=STEPS)
    graph["50"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    graph["51"] = {"class_type": "LoadImage", "inputs": {"image": mask_name}}
    graph["52"] = {"class_type": "ImageToMask", "inputs": {"image": ["51", 0], "channel": "red"}}
    graph["53"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["50", 0], "vae": ["15", 0]}}
    graph["54"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["53", 0], "mask": ["52", 0]}}
    graph["19"]["inputs"]["latent_image"] = ["54", 0]
    graph["19"]["inputs"]["denoise"] = denoise
    return graph


def repaint(member: str, expression: str, source: Path, seed: int | None = None) -> Image.Image:
    (lx, ly), (rx_, ry_), (mx, my) = LANDMARKS[member]
    cx, cy = (lx + rx_ + mx) / 3, (ly + ry_ + my) / 3
    full = Image.open(source).convert("RGB")
    left, top = int(max(0, min(full.width - CROP, cx - CROP // 2))), int(max(0, min(full.height - CROP, cy - CROP // 2)))
    crop = full.crop((left, top, left + CROP, top + CROP))
    mask = soft_mask(CROP, regions(member, expression, left, top))
    tag = uuid.uuid4().hex[:8]
    image_name, mask_name = upload(crop, f"face_{tag}.png"), upload(mask.convert("RGB"), f"mask_{tag}.png")
    seed = bible.STAFF[member]["seed"] + 17 if seed is None else seed
    tags, avoid = EXPRESSION_TAGS[expression]
    if expression != "blink" and member in MOUTH_TAGS:
        tags = f"{tags}, {MOUTH_TAGS[member]}"
    positive = f"{bible.QUALITY}, {bible.STAFF[member]['traits']}, {tags}, upper body, simple background, white background, flat colors"
    graph = inpaint_workflow(positive, f"{bible.NEGATIVE}, {avoid}", image_name, mask_name, seed, MOUTH_DENOISE.get(member, DENOISE[expression]) if expression != "blink" else DENOISE[expression], f"{member}_{expression}_face")
    painted = Image.open(io.BytesIO(generate.run(graph))).convert("RGB")
    blended = Image.composite(painted, crop, mask)
    out = full.copy()
    out.paste(blended, (left, top))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", required=True, choices=list(bible.STAFF))
    parser.add_argument("--expression", required=True, choices=[e for e in bible.EXPRESSIONS if e != "calm"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    source = sorted(generate.OUT.glob(f"{args.member}_calm_s*.png"))[0]
    started = time.time()
    result = repaint(args.member, args.expression, source, args.seed)
    target = generate.OUT / f"{args.member}_{args.expression}.png"
    result.save(target)
    print(f"{target} ({time.time() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
