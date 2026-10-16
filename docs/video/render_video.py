"""Render the animation: ``python docs/video/render_video.py``.

The film is an SVG page (``anime/index.html``) that draws the whole scene as a function of the time t. This script writes the numbers it shows (``anime/facts.js``, read from the
project's final files), drives a headless Chrome frame by frame at 24 frames per second and 1920 x 1080, encodes the .mp4 with ``encode.swift`` (the system's AVFoundation, nothing to
install), and writes a short looping .gif preview and a poster frame next to it. The film says that the audit reproduced, so it is only rendered when the audit log says so.
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "anime"))
sys.path.insert(0, str(HERE.parents[1]))

import facts as facts_module  # noqa: E402
from chrome import Chrome  # noqa: E402

PAGE = HERE / "anime" / "index.html"
FACTS_JS = HERE / "anime" / "facts.js"
FRAMES = HERE / "_frames"
BUILD = HERE / "_build"
MP4 = HERE / "cypmode_overview.mp4"
GIF = HERE / "cypmode_overview_preview.gif"
WEB = HERE / "cypmode_overview_web.mp4"
POSTER = HERE / "cypmode_overview_poster.png"
VOICE_TRACK = HERE / "voices" / "voice_track.wav"
FPS = 24
GIF_SIZE, GIF_FPS, GIF_START, GIF_SECONDS = (640, 360), 12, 3.2, 8.0


def render_frames(chrome: Chrome, duration: float) -> int:
    if FRAMES.exists():
        shutil.rmtree(FRAMES)
    FRAMES.mkdir(parents=True)
    total = round(duration * FPS)
    for index in range(total):
        chrome.evaluate(f"renderAsync({index / FPS})")
        chrome.screenshot(FRAMES / f"frame_{index:05d}.png")
        if index % 240 == 0:
            print(f"frame {index} of {total}", flush=True)
    return total


def dom_digests(poster_only: bool = False) -> dict:
    """The digest of the DOM of every frame of the film and of the poster (deterministic, unlike the raster and the encoded video), recomputed by the audit to prove that the film still says what it said."""
    out = {}
    with Chrome() as chrome:
        chrome.open(PAGE)
        out["poster"] = chrome.evaluate("posterDigest()")
        if not poster_only:
            total, chain = round(float(chrome.evaluate("DURATION")) * FPS), []
            for start in range(0, total, 480):  # twenty seconds at a time
                chain.append(chrome.evaluate(f"domDigest({start}, {min(start + 480, total)}, {FPS})"))
            out["film"] = hashlib.sha256(":".join(chain).encode()).hexdigest()
    return out


def write_poster(chrome: Chrome) -> None:
    """The poster is its own picture (poster.js: the title, the question, four real numbers and the whole team), not a frame of the film."""
    chrome.evaluate("posterAsync()")
    chrome.screenshot(POSTER)


def mux_voice() -> None:
    """Add the spoken track (``voices/voice_track.wav``, made by voices.py) to the silent film when it exists: the video is copied, the voice is encoded as AAC and padded with silence to the end."""
    ffmpeg = shutil.which("ffmpeg")
    if not VOICE_TRACK.exists() or ffmpeg is None:
        return
    silent = MP4.with_name("_silent.mp4")
    MP4.replace(silent)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(silent), "-i", str(VOICE_TRACK), "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-af", "apad", "-shortest", str(MP4)], check=True)
    silent.unlink()


def write_web_copy() -> None:
    """The copy that the README links to: 1280 x 720, H.264 at a modest quality and 96 kbit/s AAC, small enough to live in the repository (GitHub refuses files above 100 MB)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(MP4), "-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", "slow", "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-c:a", "aac", "-b:a", "96k", str(WEB)], check=True)


def encode() -> None:
    BUILD.mkdir(exist_ok=True)
    binary = BUILD / "encode"
    if not binary.exists() or binary.stat().st_mtime < (HERE / "encode.swift").stat().st_mtime:
        subprocess.run(["swiftc", "-O", str(HERE / "encode.swift"), "-o", str(binary)], check=True, capture_output=True)
    subprocess.run([str(binary), str(FRAMES), str(MP4), str(FPS)], check=True)


def write_gif() -> None:
    step = FPS // GIF_FPS
    first = round(GIF_START * FPS)
    frames = [Image.open(FRAMES / f"frame_{i:05d}.png").convert("RGB").resize(GIF_SIZE, Image.LANCZOS) for i in range(first, first + round(GIF_SECONDS * FPS), step)]
    frames[0].save(GIF, save_all=True, append_images=frames[1:], duration=round(1000 / GIF_FPS), loop=0, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-audit-check", action="store_true", help="development only: render although the audit log does not say REPRODUCIBLE")
    parser.add_argument("--write-facts", action="store_true", help="only write anime/facts.js (to open the page by hand) and stop")
    args = parser.parse_args()
    facts = facts_module.load()
    if args.write_facts:
        FACTS_JS.write_text(facts_module.to_javascript(facts, facts_module.real_data()))
        return 0
    if not facts.audit_reproducible and not args.skip_audit_check:
        print("the audit log does not say REPRODUCIBLE; run queue job 900 first (the animation states that the results reproduce)")
        return 1
    FACTS_JS.write_text(facts_module.to_javascript(facts, facts_module.real_data()))
    with Chrome() as chrome:
        chrome.open(PAGE)
        total = render_frames(chrome, float(chrome.evaluate("DURATION")))
        write_poster(chrome)
    encode()
    mux_voice()
    write_web_copy()
    write_gif()
    shutil.rmtree(FRAMES)
    from cypmode.audit.artifacts import asset_digest, record

    digests = dom_digests()
    record("film", [p for p in (MP4, WEB, GIF) if p.exists()], {"dom_digest": digests["film"], "assets": asset_digest(), "frames": total, "voices": VOICE_TRACK.exists()})
    record("poster", [POSTER], {"dom_digest": digests["poster"]})
    print(f"wrote {MP4.name} ({MP4.stat().st_size / 1e6:.1f} MB), {GIF.name} ({GIF.stat().st_size / 1e6:.1f} MB) and {POSTER.name}; {total} frames at {FPS} per second")
    return 0


if __name__ == "__main__":
    sys.exit(main())
