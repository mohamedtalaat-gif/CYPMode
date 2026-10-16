"""The eight members of staff as prompts for the Anima text-to-image model (CircleStone Labs, https://huggingface.co/circlestone-labs/Anima).

Anima reads lowercase Danbooru-style tags: quality and safety tags first, then the subject (``1boy``), then the traits that identify the character, then the picture (pose, framing,
background). The traits below are the identity of each member and repeat in every picture of him, so that the expression variants of one member are recognisably the same person.
Each member is a male adult researcher of the laboratory and wears the white lab coat; what makes him recognisable is the hair, the face and what he wears or carries under and beside it (``docs/video/STORYBOARD.md``).
"""

QUALITY = "masterpiece, best quality, safe"  # the Aesthetic checkpoint takes score_ tags in the negative prompt only
NEGATIVE = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, artist name, signature, watermark, text, 1girl, multiple boys, extra fingers, bad hands"
FRAMING = "upper body, looking at viewer, facing viewer, front view, simple background, white background, flat colors"

# the identity of each member: hair, eyes, face, clothing and the things he carries
STAFF = {
    "kai": {
        "traits": "1boy, young man, teal hair, messy hair, short hair, green eyes, blush, goggles on head, white labcoat, rolled sleeves, green shirt, blue gloves, slim, pipette in breast pocket",
        "seed": 1101,
    },
    "bits": {
        "traits": "1boy, young man, orange hair, side-parted hair, purple eyes, round glasses, purple hoodie, white labcoat, headphones around neck, tall, slender",
        "seed": 1202,
    },
    "stacker": {
        "traits": "1boy, adult man, brown hair, spiked hair, brown eyes, red headband, red scarf, stubble, tan skin, white labcoat, grey shirt, rolled sleeves, broad shoulders, confident",
        "seed": 1303,
    },
    "gatekeeper": {
        "traits": "1boy, mature man, grey hair, short hair, goatee, blue eyes, wrinkles, white labcoat, white shirt, blue necktie, holding clipboard, tall, calm",
        "seed": 1404,
    },
    "dock": {
        "traits": "1boy, adult man, dark skin, black hair, full beard, amber eyes, blue visor, white labcoat, orange shirt, rolled sleeves, broad shoulders, friendly",
        "seed": 1505,
    },
    "auditor": {
        "traits": "1boy, adult man, navy blue hair, short hair, rectangular glasses, green eyes, tan skin, white labcoat, white shirt, green necktie, lanyard with id card, neat",
        "seed": 1606,
    },
    "scout": {
        "traits": "1boy, young man, blonde hair, messy short hair, freckles, brown eyes, binoculars around neck, white labcoat, magenta sweater, satchel strap, curious, smile",
        "seed": 1707,
    },
    "foundry": {
        "traits": "1boy, adult man, bald, thick mustache, brown eyes, safety goggles on forehead, white labcoat, cyan shirt, rolled sleeves, stocky, broad shoulders, confident",
        "seed": 1808,
    },
}

# what changes between the pictures of one member: the face only
EXPRESSIONS = {
    "calm": "closed mouth, slight smile",
    "blink": "closed eyes, closed mouth, slight smile",
    "talk": "open mouth, looking at viewer",
    "happy": "smile, open mouth, happy",
}


def positive(name: str, expression: str = "calm") -> str:
    """The positive prompt of one member in one expression."""
    return f"{QUALITY}, {STAFF[name]['traits']}, {EXPRESSIONS[expression]}, {FRAMING}"
