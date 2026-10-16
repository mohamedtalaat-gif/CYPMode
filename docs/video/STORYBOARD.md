# The CYPMode animation: storyboard

Purpose: a viewer who watches the film and nothing else understands what the project is, how it works and what it delivers. The film is an anime-style short of about six and a half
minutes, 1920 x 1080 at 24 frames per second, silent, with subtitles. Every member of the project's team introduces themselves on a character card (name, role, what they do, what they
stand for in the project) and then does the job in a scene on real data; one molecule, Cap, is passed from station to station until the results come out.

## What is real

Every number on screen is read from the project's files when the film is rendered (`facts.py`), and every picture of a molecule or a pocket is drawn from coordinates:

- the compound Cap is a real compound of the challenge (OCNT-2395436): its measurements, its flat structure, its 2,048-bit fingerprint and its descriptors come from the files;
- the scatter of the label rule is a seeded sample of the real CYP3A4 training compounds, with the counts of the whole set;
- the models are the 23 candidates of the final stack, and the ROC curves are rebuilt from the cached out-of-fold predictions;
- the ledger of the Gatekeeper is the gate ledger of the logs, and Quanta's bars are the measured signal of the physics columns;
- the enzyme, the heme pockets of a Type II and a Type I crystal structure and the predicted poses of a donor ligand with and without the restraint are rendered with PyMOL from the
  coordinates (`pymol_clips.py`), and the PoseBusters list is the table of checks of the submitted poses;
- the two cinematic clips (the whole CYP3A4 enzyme turning with its heme and ligand glowing inside, and the slow approach to the pocket with the ligand nitrogen over the heme iron) are drawn
  by Blender with the Molecular Nodes add-on from the coordinates of the crystal structure 6MA7 (`blender_clips.py`, `blender/scene.py`); the leader-line callouts follow the projected position of
  the iron and of the coordinating nitrogen frame by frame;
- the audit hash, the counts of tests, modules and queue jobs and the hash of the final zip are computed from the repository.

## Style

Anime-style characters about seven and a half heads tall: a small face with a defined jaw, narrow eyes with layered irises and highlights, hair built from separate locks with a glossy
band, drawn hands, thin coloured line art, soft shading. Painted-looking laboratory backgrounds with out-of-focus depth, halftone and speed-line backdrops on the character cards, diagonal
wipes between scenes, a slow push-in on every scene and a final grade. It is drawn entirely by code as SVG (`anime/*.js`); it is a stylised vector film and makes no claim to be studio work.

## Cast

| Character | Role in the project |
| --- | --- |
| **Cap** (a smiling capsule) | the molecule whose journey the film follows: "will I switch off a liver enzyme?" |
| **The enzyme** (a purple body with the heme in its pocket) | CYP3A4 / CYP2D6, the target |
| **Kai** (goggles, teal hair) | the assay analyst: the two measurements per compound and the label rule |
| **Scout** (blond hair, freckles, binoculars) | the data scout: the files the organisers released, compared by name with the blinded test set, and the guard that takes a test compound out before anything is computed |
| **Bits** (glasses, ginger hair) | the chemoinformatician: fingerprints, descriptors, alerts, physics, as numbers |
| **Stacker** (headband, red scarf, stubble) | the modeller: the committee of models and the logistic stack |
| **Gatekeeper** (silver hair, goatee, clipboard) | the statistician: nested validation, corrected statistics, the pre-declared gates |
| **Quanta** (a glowing orbital) | the quantum chemist: the GFN2-xTB physics, tested and not used |
| **Foundry** (bald, mustache, safety goggles) | the foundation-model engineer: the pretrained models, each added to the stack and judged by the same rule |
| **Dock** (visor, beard) with the assistants **Boltz-2** and **OpenFold3** | the structural biologist: co-folding, the iron-nitrogen restraint, PoseBusters |
| **Auditor** (frames) | the reproducibility officer: everything runs twice and must match byte for byte |

The eight members of staff are male adults painted with the Anima text-to-image model (CircleStone Labs, https://huggingface.co/circlestone-labs/Anima; run locally in ComfyUI, GPL-3.0). Each is one portrait,
painted from a fixed prompt (`docs/video/characters/bible.py`: the traits that identify him, the seed, the sampler settings) and cut from its white background (`cutout.py`); the other expressions are made
by repainting only the eyes or the mouth of the same picture (`expressions.py`), so the hair, the clothes and the face stay identical. The film swaps between four pictures of a member, calm, blink, talk and
happy, and moves him with a slow breath and sway; `sprites.js` draws him, and `characters/manifest.json` records the prompt, seed, settings and SHA-256 of the model files and of every picture. When no pictures exist, the film falls back
to the vector people of `people.js`.

| Member | Look |
| --- | --- |
| Kai | young, teal messy hair, green eyes, blush, goggles on the head, lab coat with rolled sleeves over a green shirt, blue gloves, test tubes in the pocket |
| Scout | young, blond messy hair, freckles, brown eyes, a magenta turtleneck under the lab coat, binoculars round the neck, a satchel strap |
| Bits | young, orange side-parted hair, purple eyes, round glasses, purple hoodie under a lab coat, headphones round the neck, a pen in the pocket |
| Stacker | adult, brown spiked hair, stubble, red headband and scarf, charcoal work jacket with rolled sleeves, broad shoulders |
| Gatekeeper | mature, grey hair, goatee, blue eyes, lined face, navy blazer with a pocket square and a blue tie, a clipboard |
| Foundry | adult, bald, thick mustache, brown eyes, safety goggles on the forehead, a cyan shirt under the lab coat, rolled sleeves, stocky |
| Dock | adult, dark skin, short black hair, full beard, amber eyes, blue visor, olive utility jacket over an orange shirt |
| Auditor | adult, navy hair, rectangular glasses, green eyes, green blazer and tie, a lanyard with a card |

Cap, the enzyme, Quanta and the assistants are not gendered and stay vector drawings. The enzyme has stubby arms, feet and two alpha helices on its body; Cap wears laced sneakers over striped socks and mittens with a cuff; Quanta has twinkling stars; Boltz-2 waves and OpenFold3 sleeps. Every round face has eyebrows that follow the mood. All of them idle: breathing, a slow sway, blinking, a small sway of the arms and head.

## Scenes (seconds)

Each introduction card lasts 14 seconds so that the lines on it can be read aloud.

| Time | Scene | What happens |
| --- | --- | --- |
| 0 - 30 | Opening | dawn in the laboratory; Cap rolls in and lands; the title; the real enzyme; "Will I switch off a liver enzyme and keep it off?"; the two tasks, with their sizes |
| 30 - 68 | Kai | card; Cap's two dose-response curves and the shift; the real scatter of 2,334 compounds and the rule line |
| 68 - 106 | Scout | card; the six files with their compound counts and how many compounds of the blinded test set each holds; the test compounds taken out |
| 106 - 144 | Bits | card; Cap's flat structure scanned; the 2,048 yes/no questions with the real answers lit; the descriptors; the 18 feature blocks and the count of numbers |
| 144 - 182 | Stacker | card; the 23 models by kind; the real ROC curves of the two enzymes; the blend |
| 182 - 242 | Gatekeeper and Quanta | card; the rule written first; the ledger of every idea and its verdict; Quanta's card; the physics columns and the NOT MET stamp |
| 242 - 282 | Foundry | card; the pretrained models, the change in AUC each brought to the stack for both enzymes, and its verdict (running while a job is not finished) |
| 282 - 334 | Dock | card; the real enzyme; the real Type II pocket with the iron and the nitrogen marked; a donor ligand without and with the restraint; the PoseBusters checks |
| 334 - 366 | Auditor | card; two audit runs side by side, the same hash, the IDENTICAL stamp; the counts of tests, modules, jobs and the zip hash |
| 366 - 394 | Finale | the results board, the whole team, confetti, the credits |

## Voices

Every member speaks the lines of his own card and the subtitles of his scene, and the narrator speaks the opening and the finale; Cap and the other mascots speak no word (Quanta has a voice of its own).
The voices are Kokoro-82M (hexgrad, Apache-2.0, https://huggingface.co/hexgrad/Kokoro-82M; trained on permissively licensed and synthetic audio), one English male voice per member
(`docs/video/voices.py`: Kai am_puck, Scout am_liam, Bits am_eric, Stacker am_michael, Gatekeeper bm_george, Quanta am_echo, Foundry am_fenrir, Dock am_onyx, Auditor bm_daniel, narrator bm_fable).
`voices.py harvest` draws the film once a second in headless Chrome and collects every line with its time (the page records them), `voices.py speak` (in the separate environment `.venv-voice`)
speaks them one after the other, mixes `voices/voice_track.wav` and writes `anime/assets_voice.js`, which makes the subtitles stay on screen exactly as long as the line is spoken and opens the
mouths of the painted staff only while a voice is heard. `render_video.py` adds the track to the mp4 (ffmpeg, AAC). `voices/manifest.json` records the text, voice, pace and duration of every clip and the SHA-256 of the model file.
No recording of any person or of any anime is used.

## Build

- `docs/video/anime/index.html` loads `lib.js` (easing, cel shading, geometry, text measurement and fitting), `faces.js` (the mascots' eyes and the mouths), `people.js` (the staff), `mascots.js`
  (Cap, Quanta, the enzyme, the assistants), `props.js` (the laboratory, panels, pills, bubbles, stamps, subtitles, real pictures), `film1.js` to `film5.js` (the scenes) and `main.js` (the page entry points).
- Every text in a box is measured with the real font and fitted into it; `tests/test_video.py` draws the film every two seconds and fails when a text leaves its box or the frame.
- `docs/video/blender_clips.py` renders the two cinematic clips with Blender 5 and Molecular Nodes (queue job 943; `blender/scene.py` is GPL-3.0-or-later because it drives those two programs, the rest
  of the repository is MIT OR Apache-2.0); the frames are in `docs/video/anime/clips/` (not tracked) and the film falls back to the PyMOL pictures when a clip is absent;
- `docs/video/pymol_clips.py` renders the real-structure pictures with PyMOL (queue job 944); `docs/video/render_video.py` drives a headless Chrome over the DevTools protocol, one screenshot per frame,
  encodes the .mp4 with `encode.swift` (the system's AVFoundation, no extra software), and writes a 720p copy for the README (`cypmode_overview_web.mp4`, small enough for the repository; the 1080p original is over GitHub's limit of 100 MB and stays local), a short .gif preview and the poster (`poster.js`: the title, the question the film answers, four numbers read from the facts and the whole team; its own picture, not a frame of the film; queue job 945). About 6,800 frames take one to two hours.
- `python docs/video/anime/preview.py <dir> "render(20)"` draws single frames while the film is being edited; `main.js` also has `sheet()`, `faces()` and `portrait()` for the cast.
