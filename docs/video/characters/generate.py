# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Paint the staff of the film with the Anima model through a local ComfyUI server: ``python docs/video/characters/generate.py --member kai [--expression calm] [--seed N]``.

ComfyUI (https://github.com/comfyanonymous/ComfyUI, GPL-3.0) runs outside the repository (in the folder that ``COMFYUI_HOME`` names) on the Apple GPU; start it with
``cd $COMFYUI_HOME && ./venv/bin/python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch``. The model is Anima (CircleStone Labs, non-commercial licence for the model; the pictures it
makes may be used commercially), which needs ``anima-aesthetic-v1.1.safetensors``, ``qwen_3_06b_base.safetensors`` and ``qwen_image_vae.safetensors`` in ComfyUI's ``models`` folders. The prompts come from
``bible.py``; the pictures go to ``docs/video/anime/characters/``. The workflow is the one the model card publishes (er_sde sampler, 30 steps, CFG 4) with the Aesthetic checkpoint.
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bible

SERVER = "http://127.0.0.1:8188"
OUT = Path(__file__).resolve().parents[1] / "anime" / "characters"
UNET = "anima-aesthetic-v1.1.safetensors"
WIDTH, HEIGHT, STEPS, CFG, SAMPLER = 768, 1024, 30, 4.0, "er_sde"


def workflow(positive: str, negative: str, seed: int, prefix: str, width: int = WIDTH, height: int = HEIGHT, steps: int = STEPS, cfg: float = CFG) -> dict:
    return {
        "44": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "45": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion", "device": "default"}},
        "15": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["45", 0]}},
        "12": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["45", 0]}},
        "28": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "19": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": SAMPLER, "scheduler": "simple", "denoise": 1.0, "model": ["44", 0], "positive": ["11", 0], "negative": ["12", 0], "latent_image": ["28", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["19", 0], "vae": ["15", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


def run(graph: dict, timeout: float = 3600.0) -> bytes:
    request = urllib.request.Request(f"{SERVER}/prompt", data=json.dumps({"prompt": graph}).encode(), headers={"Content-Type": "application/json"})
    prompt_id = json.load(urllib.request.urlopen(request))["prompt_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = json.load(urllib.request.urlopen(f"{SERVER}/history/{prompt_id}"))
        if prompt_id in history:
            entry = history[prompt_id]
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError(json.dumps(entry["status"])[:600])
            image = next(iter(entry["outputs"].values()))["images"][0]
            return urllib.request.urlopen(f"{SERVER}/view?filename={image['filename']}&subfolder={image['subfolder']}&type={image['type']}").read()
        time.sleep(3)
    raise TimeoutError("ComfyUI did not finish")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", required=True, choices=list(bible.STAFF))
    parser.add_argument("--expression", default="calm", choices=list(bible.EXPRESSIONS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else bible.STAFF[args.member]["seed"]
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data = run(workflow(bible.positive(args.member, args.expression), bible.NEGATIVE, seed, f"{args.member}_{args.expression}", steps=args.steps))
    target = OUT / f"{args.member}_{args.expression}_s{seed}.png"
    target.write_bytes(data)
    print(f"{target} ({time.time() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
