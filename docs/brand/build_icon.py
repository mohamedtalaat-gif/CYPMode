# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The icon of the project: ``python docs/brand/build_icon.py`` writes ``cypmode-icon.svg``, ``favicon-32.png``, ``apple-touch-icon.png`` and ``favicon.ico`` next to this file.

The picture is a structure, drawn from the deposited coordinates of CYP3A4 in the crystal structure 6MA7 (Protein Data Bank): the porphyrin ring of the heme (residue HEM 601, chain A), the iron as an
orange sphere, and the triazole ring of the ligand (residue TPF 602) whose nitrogen N5 is bound to the iron at 2.20 angstrom. That bond is the Type II interaction that the project looks for
(section 1.9 of the notebook). The atoms are projected from their real positions, the macrocycle seen at an angle so that the ligand stands above it; the coordinates are copied below (three decimals,
angstrom) so that the script needs no download.

The icon is placed on every page that the Pages site publishes by the workflow ``.github/workflows/pages.yml``, so that a tab shows it and not the icon of the domain.
"""

import base64
import io
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TILT_DEG, TURN_DEG = 52.0, 18.0
HEME = {
    "FE": (-16.010, -22.914, -9.665),
    "NA": (-15.774, -21.340, -10.952),
    "NB": (-16.425, -24.088, -11.298),
    "NC": (-16.265, -24.550, -8.382),
    "ND": (-15.565, -21.798, -8.124),
    "C1A": (-15.509, -20.056, -10.578),
    "C2A": (-15.518, -19.238, -11.778),
    "C3A": (-15.733, -20.026, -12.811),
    "C4A": (-15.903, -21.369, -12.316),
    "C1B": (-16.463, -23.722, -12.601),
    "C2B": (-16.835, -24.870, -13.408),
    "C3B": (-17.046, -25.901, -12.589),
    "C4B": (-16.776, -25.429, -11.250),
    "C1C": (-16.479, -25.841, -8.853),
    "C2C": (-16.271, -26.749, -7.760),
    "C3C": (-15.932, -26.025, -6.697),
    "C4C": (-15.914, -24.624, -7.071),
    "C1D": (-15.524, -22.211, -6.800),
    "C2D": (-15.288, -21.040, -5.975),
    "C3D": (-15.177, -19.978, -6.784),
    "C4D": (-15.325, -20.443, -8.156),
    "CHA": (-15.300, -19.651, -9.279),
    "CHB": (-16.180, -22.481, -13.088),
    "CHC": (-16.854, -26.224, -10.126),
    "CHD": (-15.649, -23.507, -6.308),
}
LIGAND = {
    "N5": (-18.133, -22.477, -9.298),
    "C7": (-19.235, -23.125, -9.717),
    "N6": (-20.318, -22.376, -9.721),
    "N4": (-19.879, -21.154, -9.256),
    "C6": (-18.510, -21.209, -9.009),
}

CORE_BONDS_MAX, LIGAND_BONDS_MAX = 1.75, 1.65  # angstrom: two atoms closer than this are bonded
PYRROLES = [[f"N{p}", f"C1{p}", f"C2{p}", f"C3{p}", f"C4{p}"] for p in "ABCD"]
RING_ORDER = ["N5", "C7", "N6", "N4", "C6"]


def projected() -> dict[str, np.ndarray]:
    """The screen position of every atom, in angstrom: the plane of the macrocycle is found by a singular value decomposition and the view is tilted towards the side of the ligand."""
    heme = {n: np.array(v) for n, v in HEME.items()}
    ligand = {"L" + n: np.array(v) for n, v in LIGAND.items()}
    centre = heme["FE"]
    vt = np.linalg.svd(np.array(list(heme.values())) - centre)[2]
    axis_u, axis_v, normal = vt[0], vt[1], vt[2]
    if np.dot(ligand["LN5"] - centre, normal) < 0:
        normal, axis_v = -normal, -axis_v
    tilt, turn = np.radians(TILT_DEG), np.radians(TURN_DEG)
    out = {}
    for name, position in {**heme, **ligand}.items():
        q = position - centre
        x, y, z = np.dot(q, axis_u), np.dot(q, axis_v), np.dot(q, normal)
        x, y = x * np.cos(turn) - y * np.sin(turn), x * np.sin(turn) + y * np.cos(turn)
        out[name] = np.array([x, y * np.cos(tilt) + z * np.sin(tilt)])
    return out


def bonds(table: dict[str, tuple], limit: float, prefix: str = "") -> list[tuple[str, str]]:
    names = [n for n in table if n != "FE"]
    return [(prefix + p, prefix + q) for i, p in enumerate(names) for q in names[i + 1:] if np.linalg.norm(np.array(table[p]) - np.array(table[q])) < limit]


def svg_text() -> str:
    pos = projected()
    pts = np.array(list(pos.values()))
    low, high = pts.min(axis=0), pts.max(axis=0)
    scale, mid = 46.0 / max(high[0] - low[0], high[1] - low[1]), (low + high) / 2

    def xy(name: str) -> tuple[float, float]:
        p = pos[name]
        return 32 + (p[0] - mid[0]) * scale, 32 + (mid[1] - p[1]) * scale  # the screen runs downward and the ligand is above the ring

    def line(a: str, b: str, colour: str, width: float) -> str:
        (x1, y1), (x2, y2) = xy(a), xy(b)
        return f'<path d="M{x1:.2f} {y1:.2f}L{x2:.2f} {y2:.2f}" stroke="{colour}" stroke-width="{width}" stroke-linecap="round"/>'

    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">',
           '<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#16539c"/><stop offset="1" stop-color="#0a2f63"/></linearGradient>',
           '<radialGradient id="fe" cx="0.35" cy="0.3" r="0.8"><stop offset="0" stop-color="#ffd08a"/><stop offset="1" stop-color="#f0883e"/></radialGradient></defs>',
           '<rect width="64" height="64" rx="14" fill="url(#bg)"/>']
    for ring in PYRROLES:  # the four pyrrole rings of the porphyrin, as light surfaces
        out.append('<polygon points="' + " ".join(f"{xy(n)[0]:.2f},{xy(n)[1]:.2f}" for n in ring) + '" fill="#7fb7ff" fill-opacity="0.38"/>')
    out += [line(p, q, "#eef3fa", 2.6) for p, q in bonds(HEME, CORE_BONDS_MAX)]
    out += [line("FE", n, "#c9d6ea", 1.5) for n in ("NA", "NB", "NC", "ND")]
    out.append('<polygon points="' + " ".join(f"{xy('L' + n)[0]:.2f},{xy('L' + n)[1]:.2f}" for n in RING_ORDER) + '" fill="#2ee6c8" fill-opacity="0.30"/>')
    out.append(line("FE", "LN5", "#2ee6c8", 2.4))
    out += [line(p, q, "#2ee6c8", 2.6) for p, q in bonds(LIGAND, LIGAND_BONDS_MAX, "L")]
    for n in ("NA", "NB", "NC", "ND"):
        x, y = xy(n)
        out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" fill="#9ccbff" stroke="#0a2f63" stroke-width="0.8"/>')
    for n in RING_ORDER:
        if n.startswith("N"):
            x, y = xy("L" + n)
            out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="#7ff5de" stroke="#0a2f63" stroke-width="0.8"/>')
    x, y = xy("FE")
    out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.2" fill="url(#fe)" stroke="#a24e12" stroke-width="1.1"/>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_png(svg: str, size: int) -> bytes:
    """The picture at a size, with a transparent background, drawn by the headless Chrome of the film."""
    sys.path.insert(0, str(HERE.parents[1] / "docs/video/anime"))
    from chrome import Chrome

    page = HERE / "_icon_page.html"
    page.write_text(f"<!doctype html><meta charset=utf-8><body style='margin:0;background:transparent'><img src='data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}' width={size} height={size}>")
    try:
        with Chrome() as chrome:
            chrome.open(page)
            time.sleep(0.6)
            chrome.call("Emulation.setDefaultBackgroundColorOverride", color={"r": 0, "g": 0, "b": 0, "a": 0})
            time.sleep(0.2)
            data = chrome.call("Page.captureScreenshot", format="png", clip={"x": 0, "y": 0, "width": size, "height": size, "scale": 1}, fromSurface=True)["data"]
    finally:
        page.unlink(missing_ok=True)
    return base64.b64decode(data)


def main() -> int:
    from PIL import Image

    svg = svg_text()
    (HERE / "cypmode-icon.svg").write_text(svg)
    (HERE / "favicon-32.png").write_bytes(render_png(svg, 32))
    (HERE / "apple-touch-icon.png").write_bytes(render_png(svg, 180))
    Image.open(io.BytesIO(render_png(svg, 48))).convert("RGBA").save(HERE / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    for name in ("cypmode-icon.svg", "favicon-32.png", "apple-touch-icon.png", "favicon.ico"):
        print(f"{name}: {(HERE / name).stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
