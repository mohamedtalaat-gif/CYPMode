"""The voices of the film: ``python docs/video/voices.py harvest`` reads every spoken line and its time from the film itself, ``python docs/video/voices.py speak`` (in ``.venv-voice``) speaks them.

The lines are the ones the film already shows: the three lines of every introduction card and every subtitle. ``harvest`` draws the film once a second in headless Chrome and collects them (the page
records them when ``SPEECH_LOG`` is defined), assigns the speaker of each by the scene it sits in, and writes ``voices/lines.json``. ``speak`` speaks each line with Kokoro-82M (hexgrad, Apache-2.0,
https://huggingface.co/hexgrad/Kokoro-82M; trained on permissively licensed and synthetic audio, 24 kHz), one voice per member, fits it into the time the line is on screen (a faster reading, never beyond
``MAX_SPEED``), mixes the clips into ``voices/voice_track.wav`` and writes ``anime/assets_voice.js``: the intervals in which somebody is speaking, the time of every subtitle and, per frame of the film, whether the voice is loud enough for an open mouth, so that the
painted staff open their mouths when the voice does. ``voices/manifest.json`` records the text, voice, speed and duration of every clip and the SHA-256 of the model file. The audio is regenerated from it, so it is not tracked.
"""

import hashlib
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VOICES = HERE / "voices"
LINES = VOICES / "lines.json"
TRACK = VOICES / "voice_track.wav"
MANIFEST = VOICES / "manifest.json"
INTERVALS = HERE / "anime" / "assets_voice.js"
RATE = 24000
BASE_SPEED = 1.08  # the reading pace of a line
PACE = {"Narrator": 1.16, "Quanta": 1.14}  # the narrator and the mascot read a little faster than the staff
MAX_SPEED = 1.2  # a line that would run past the end of its scene is read faster, never beyond this
GAP = 0.3  # silence between two lines, whoever speaks
VOICE_OF = {"Narrator": "bm_fable", "Kai": "am_puck", "Scout": "am_liam", "Bits": "am_eric", "Stacker": "am_michael", "Gatekeeper": "bm_george", "Quanta": "am_echo", "Foundry": "am_fenrir",
            "Dock": "am_onyx", "Auditor": "bm_daniel"}  # one voice per member; Cap speaks no word
CARD_SPEECH_START = 1.5  # seconds after the card appears, when its first line fades in
CARD_GAP = 0.45  # silence between two lines of a card


def speaker_of(scene_id: str, local: float, card: float, gate_second_card: float) -> str:
    """The member who says a line of the scene at the local time; the opening and the finale are told by the narrator."""
    if scene_id == "gate":
        return "Gatekeeper" if local < gate_second_card else "Quanta"
    return {"open": "Narrator", "finale": "Narrator", "kai": "Kai", "scout": "Scout", "bits": "Bits", "stacker": "Stacker", "foundry": "Foundry", "dock": "Dock", "auditor": "Auditor"}[scene_id]


def harvest() -> int:
    """Draw the film once a second and write ``voices/lines.json``: [{"speaker", "kind", "start", "end", "text"}] in film seconds."""
    sys.path[:0] = [str(HERE), str(HERE / "anime")]
    import facts as facts_module
    from chrome import Chrome

    facts_path = HERE / "anime" / "facts.js"
    facts_path.write_text(facts_module.to_javascript(facts_module.load(), facts_module.real_data()))
    found: dict[tuple, dict] = {}
    with Chrome() as chrome:
        chrome.open(HERE / "anime" / "index.html")
        chrome.evaluate("window.SPEECH_LOG = []")
        scenes = chrome.evaluate("SCENES")
        card, duration = float(chrome.evaluate("CARD")), int(chrome.evaluate("DURATION"))
        for second in range(duration):
            chrome.evaluate(f"render({second + 0.5})")
        log = chrome.evaluate("SPEECH_LOG")
    gate = next(s for s in scenes if s["id"] == "gate")
    scene_at = lambda t: next(s for s in scenes if s["start"] <= t < s["end"])
    for item in log:
        if item["kind"] == "card":
            scene = scene_at(item["start"] + 0.1)
            speaker = item["name"] if item["name"] in VOICE_OF else "Narrator"
            at = item["start"] + CARD_SPEECH_START
            for line in item["lines"]:
                found[("card", round(item["start"], 2), line)] = {"speaker": speaker, "kind": "card", "start": at, "end": item["start"] + card - 0.3, "text": line, "scene": scene["id"]}
                at += 1.0  # nominal; the real start of the next line follows the end of the previous one in `speak`
        else:
            scene = scene_at(item["a"] + 0.1)
            speaker = speaker_of(scene["id"], item["a"] - scene["start"], card, card + 16)
            found[("sub", round(item["a"], 2), item["text"])] = {"speaker": speaker, "kind": "subtitle", "start": item["a"] + 0.15, "end": item["b"] - 0.25, "text": item["text"], "scene": scene["id"]}
    lines = sorted(found.values(), key=lambda x: (x["start"], x["kind"]))
    VOICES.mkdir(exist_ok=True)
    LINES.write_text(json.dumps(lines, indent=1))
    (VOICES / "scenes.json").write_text(json.dumps(scenes))
    print(f"{len(lines)} lines of {len({x['speaker'] for x in lines})} speakers over {duration} s -> {LINES}")
    return 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def speak() -> int:
    from huggingface_hub import hf_hub_download
    from kokoro import KPipeline

    lines = json.loads(LINES.read_text())
    pipelines: dict[str, KPipeline] = {}

    def synth(text: str, voice: str, speed: float) -> np.ndarray:
        pipe = pipelines.setdefault(voice[0], KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M"))
        parts = [audio.numpy() for _, _, audio in pipe(text, voice=voice, speed=speed, split_pattern=None)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    scenes = json.loads((VOICES / "scenes.json").read_text())
    scene_end = lambda t0: next(s["end"] for s in scenes if s["start"] <= t0 + 0.1 < s["end"])
    clips, report, cursor = [], [], 0.0
    for line in lines:  # one speaker at a time: a line starts when it is due or when the previous one has ended
        voice = VOICE_OF[line["speaker"]]
        start = max(line["start"], cursor)
        base = PACE.get(line["speaker"], BASE_SPEED)
        audio = synth(line["text"], voice, base)
        limit = scene_end(line["start"]) - 0.35
        speed = base
        if start + len(audio) / RATE > limit:
            speed = min(MAX_SPEED, base * (len(audio) / RATE) / max(0.5, limit - start) * 1.02)
            audio = synth(line["text"], voice, speed)
        seconds = len(audio) / RATE
        cursor = start + seconds + GAP
        clips.append((start, audio))
        report.append({"speaker": line["speaker"], "voice": voice, "kind": line["kind"], "scene": line["scene"], "due": round(line["start"], 2), "start": round(start, 2), "end": round(start + seconds, 2),
                       "scene_end": round(scene_end(line["start"]), 2), "speed": round(speed, 2), "text": line["text"], "fits": bool(start + seconds <= scene_end(line["start"]))})
        print(f"{line['speaker']:10s} due {line['start']:7.2f} -> {start:7.2f}-{start + seconds:7.2f} (scene ends {scene_end(line['start']):.2f}, speed {speed:.2f}) {line['text'][:56]}", flush=True)
    total = int((max(s + len(a) / RATE for s, a in clips) + 1.0) * RATE)
    track = np.zeros(total, dtype=np.float32)
    for start, audio in clips:
        i = int(start * RATE)
        fade = min(len(audio) // 4, int(0.012 * RATE))
        shaped = audio.copy()
        if fade:
            shaped[:fade] *= np.linspace(0, 1, fade)
            shaped[-fade:] *= np.linspace(1, 0, fade)
        track[i:i + len(shaped)] += shaped
    track *= 0.89 / max(1e-6, float(np.abs(track).max()))
    with wave.open(str(TRACK), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((track * 32767).astype("<i2").tobytes())
    envelope = np.convolve(np.abs(track), np.ones(int(0.04 * RATE)) / int(0.04 * RATE), mode="same")
    speaking = envelope > 0.012
    edges = np.flatnonzero(np.diff(np.concatenate([[0], speaking.astype(int), [0]])))
    spans = [[round(a / RATE, 3), round(b / RATE, 3)] for a, b in zip(edges[::2], edges[1::2], strict=True)]
    merged: list[list[float]] = []
    for a, b in spans:  # a pause shorter than a breath does not close the mouth
        if merged and a - merged[-1][1] < 0.16:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    frame = RATE // 24  # the mouth is opened or closed once per frame of the film, following the loudness of the voice (about one syllable)
    loud = np.array([np.abs(track[i:i + frame]).mean() for i in range(0, len(track) - frame, frame)])
    reference = np.percentile(loud[loud > 0.01], 80) if (loud > 0.01).any() else 1.0
    mouth, run = [], 0
    for x in loud:  # an open mouth closes for one frame every eighth frame: a vowel held by the synthesis is still a syllable
        run = run + 1 if x > 0.45 * reference else 0
        mouth.append("1" if run and run % 8 else "0")
    mouth = "".join(mouth)
    spoken: dict[str, list[list[float]]] = {}
    for r in report:
        if r["kind"] == "subtitle":
            spoken.setdefault(r["text"], []).append([r["start"], r["end"]])
    INTERVALS.write_text("window.SPEECH_ON = " + json.dumps(merged) + ";\nwindow.SPEECH_LINES = " + json.dumps(spoken) + ";\nwindow.SPEECH_ENV = " + json.dumps(mouth) + ";\n")
    model = Path(hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth"))
    MANIFEST.write_text(json.dumps({"model": {"name": "Kokoro-82M (hexgrad, Apache-2.0)", "source": "https://huggingface.co/hexgrad/Kokoro-82M", "sha256": sha256(model), "sample_rate": RATE},
                                    "voices": VOICE_OF, "base_speed": BASE_SPEED, "pace": PACE, "max_speed": MAX_SPEED, "clips": report}, indent=1))
    late = [r for r in report if not r["fits"]]
    print(f"{len(report)} clips, {len(merged)} speech intervals, track {total / RATE:.0f} s; {len(late)} clips run past the end of their scene")
    for r in late:
        print(f"  late: {r['speaker']} {r['start']}-{r['end']} (scene ends {r['scene_end']}, speed {r['speed']}): {r['text'][:70]}")
    return 0


if __name__ == "__main__":
    started = time.time()
    sys.exit({"harvest": harvest, "speak": speak}[sys.argv[1]]())
