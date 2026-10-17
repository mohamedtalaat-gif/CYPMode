"""Cut out every painted portrait and tell the film where they are: ``python docs/video/characters/build_sprites.py``.

For each member and expression found in ``docs/video/anime/characters/`` (``generate.py`` makes ``<member>_calm_s<seed>.png``, ``expressions.py`` the others) this writes the cut-out
``cut/<member>_<expression>.png`` (``cutout.py``), then ``docs/video/anime/assets_characters.js`` (the index the film reads) and ``docs/video/characters/manifest.json`` (what made every
picture: the prompt, the seed, the sampler settings, the SHA-256 of the model files and of every sprite). The manifest is tracked; the pictures themselves are regenerated from it.
"""

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bible
import cutout
import generate

CUT = generate.OUT / "cut"
INDEX = HERE.parent / "anime" / "assets_characters.js"
MANIFEST = HERE / "manifest.json"
MODELS = Path("/Volumes/Data/ComfyUI/models")
MODEL_FILES = {"diffusion": MODELS / "diffusion_models" / generate.UNET, "text_encoder": MODELS / "text_encoders" / "qwen_3_06b_base.safetensors", "vae": MODELS / "vae" / "qwen_image_vae.safetensors"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def source_of(member: str, expression: str) -> Path | None:
    found = sorted(generate.OUT.glob(f"{member}_{expression}_s*.png" if expression == "calm" else f"{member}_{expression}.png"))
    return found[0] if found else None


def main() -> int:
    CUT.mkdir(parents=True, exist_ok=True)
    index, manifest = {}, {"model": {"name": "Anima Aesthetic v1.1 (CircleStone Labs)", "source": "https://huggingface.co/circlestone-labs/Anima", "files": {k: (sha256(p) if p.exists() else None) for k, p in MODEL_FILES.items()}},
                           "settings": {"width": generate.WIDTH, "height": generate.HEIGHT, "steps": generate.STEPS, "cfg": generate.CFG, "sampler": generate.SAMPLER, "scheduler": "simple"},
                           "negative_prompt": bible.NEGATIVE, "members": {}}
    for member in bible.STAFF:
        entry, record = {}, {"traits": bible.STAFF[member]["traits"], "seed": bible.STAFF[member]["seed"], "sprites": {}}
        for expression in bible.EXPRESSIONS:
            source = source_of(member, expression)
            if source is None:
                continue
            target = CUT / f"{member}_{expression}.png"
            if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
                cutout.cutout(Image.open(source)).save(target)
            entry[expression] = f"characters/cut/{target.name}"
            record["sprites"][expression] = {"source": source.name, "sha256": sha256(target)}
        if entry:
            index[member] = {"size": [generate.WIDTH, generate.HEIGHT], "expressions": entry}
            manifest["members"][member] = record
    INDEX.write_text("window.CHARACTERS = " + json.dumps(index, indent=1) + ";\n")
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{sum(len(v['expressions']) for v in index.values())} sprites of {len(index)} members; wrote {INDEX} and {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
