"""Real structures rendered with PyMOL (headless, ray traced) for the figures and the film.

``render(key, body, ...)`` writes a PyMOL script (the shared set-up, the caller's ``body`` that selects, colours and sets the view, then a footer that projects chosen atoms to image
coordinates and writes the picture), runs it, and returns the PNG and the pixel position of every anchor, so that crisp labels can be drawn on top of the render by the caller.
Renders are cached by a hash of the script and of the input files; a missing PyMOL is an error with an explanation, not a silent fallback to a sketch.
"""

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "docs/figures/_pymol_cache"
PDB_CACHE = ROOT / "data/structure_track/inputs/pdb_cache"

PALETTE = {
    "dark": {"bg": "0x0d1117", "protein": "0x5f78b0", "surface": "0xb9cdf5", "pocket_c": "0xa9b8dc", "n": "0x5b9bff", "o": "0xff6b6b", "s": "0xf2d24a", "ligand": "0x2ee6c8", "heme": "0xf08a3c",
             "iron": "0xffb347", "dash": "0xffe14d", "cartoon_t": 0.4, "surface_t": 0.5},
    "light": {"bg": "0xffffff", "protein": "0x8d9bbd", "surface": "0xb9c8e8", "pocket_c": "0x59688f", "n": "0x2457c5", "o": "0xc43a3a", "s": "0xb8860b", "ligand": "0x0e9f8a", "heme": "0xd9691f",
              "iron": "0xe0891c", "dash": "0xb8860b", "cartoon_t": 0.55, "surface_t": 0.62},
}

HEADER = '''
import json
import numpy as np
from pymol import cmd

PAL = {pal!r}
ROOT = {root!r}
cmd.reinitialize()
cmd.bg_color(PAL["bg"])
cmd.set("ray_shadow", "off")
cmd.set("antialias", 2)
cmd.set("ambient", 0.7)
cmd.set("specular", 0.02)
cmd.set("reflect", 0.25)
cmd.set("direct", 0.5)
cmd.set("depth_cue", 1)
cmd.set("fog_start", 0.4)
cmd.set("ray_opaque_background", "on")
cmd.set("cartoon_fancy_helices", 1)
cmd.set("cartoon_smooth_loops", 1)
cmd.set("stick_radius", 0.15)
cmd.set("dash_color", PAL["dash"])
cmd.set("dash_radius", 0.09)
cmd.set("dash_gap", 0.18)
cmd.set("dash_length", 0.2)
'''

FOOTER = '''
_W, _H = {width}, {height}
_view = cmd.get_view()
_R = np.array(_view[0:9]).reshape(3, 3)
_pos, _origin = np.array(_view[9:12]), np.array(_view[12:15])
_f = (_H / 2) / np.tan(np.radians(float(cmd.get("field_of_view")) / 2))
_anchors = {{}}
for _name, _sel in {anchors!r}.items():
    _xyz = cmd.get_coords(_sel)
    if _xyz is None or len(_xyz) == 0:
        continue
    _p = _R.T @ (_xyz.mean(axis=0) - _origin) + _pos
    _anchors[_name] = [float(_W / 2 + _p[0] / (-_p[2]) * _f), float(_H / 2 - _p[1] / (-_p[2]) * _f)]
json.dump(_anchors, open({json_path!r}, "w"))
cmd.png({png_path!r}, width=_W, height=_H, ray=1)
'''


ORIENT = '''
_fe = cmd.get_coords("{ref}elem FE").mean(axis=0)
_ring = cmd.get_coords("{ref}hem and not elem FE")
_n = np.linalg.svd(_ring - _ring.mean(axis=0))[2][2]
_lig = cmd.get_coords("lig")
if np.dot(_n, _lig.mean(axis=0) - _fe) < 0:
    _n = -_n
_plane = (_lig - _fe) - np.outer((_lig - _fe) @ _n, _n)
_x = np.linalg.svd(_plane - _plane.mean(axis=0), full_matrices=False)[2][0]
if np.dot(_x, _plane.mean(axis=0)) < 0:
    _x = -_x
_x = _x - np.dot(_x, _n) * _n
_x = _x / np.linalg.norm(_x)
_z = np.cross(_x, _n)
_t = np.radians({tilt})
_ey, _ez = np.cos(_t) * _n - np.sin(_t) * _z, np.cos(_t) * _z + np.sin(_t) * _n
_v = list(cmd.get_view())
_v[0:9] = [float(a) for a in np.column_stack([_x, _ey, _ez]).flatten()]
cmd.set_view(_v)
cmd.center("lig or hem")
cmd.zoom("lig or hem", {zoom})
'''


@dataclass
class Scene:
    png: Path
    anchors: dict


def pymol_executable() -> str:
    found = shutil.which("pymol") or ("/opt/homebrew/bin/pymol" if Path("/opt/homebrew/bin/pymol").exists() else None)
    if not found:
        raise RuntimeError("PyMOL is needed to render the structures of the figures (https://pymol.org); install it or restore docs/figures/_pymol_cache from a previous build")
    return found


def render(key: str, body: str, size=(1800, 1100), theme: str = "dark", anchors: dict | None = None, files=()) -> Scene:
    """Render ``body`` (PyMOL Python that loads, styles and orients; ``PAL`` holds the theme colours) to a PNG; ``anchors`` maps a name to a selection whose centre is projected to pixels."""
    anchors = anchors or {}
    digest = hashlib.sha256(json.dumps([HEADER, FOOTER, key, body, size, theme, anchors, [(str(f), Path(f).stat().st_size) for f in files]], sort_keys=True).encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    png, meta, script = CACHE / f"{key}-{theme}-{digest}.png", CACHE / f"{key}-{theme}-{digest}.json", CACHE / f"{key}-{theme}-{digest}.py"
    if not png.exists() or not meta.exists():
        script.write_text(HEADER.format(pal=PALETTE[theme], root=str(ROOT)) + body + FOOTER.format(width=size[0], height=size[1], anchors=anchors, json_path=str(meta), png_path=str(png)))
        result = subprocess.run([pymol_executable(), "-cq", str(script)], capture_output=True, text=True, timeout=1800)
        if not png.exists():
            raise RuntimeError(f"PyMOL did not write {png.name}:\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
    return Scene(png, json.loads(meta.read_text()))


def orient(zoom: float = 9, tilt: float = 14, ref: str = "") -> str:
    return ORIENT.format(ref=ref, tilt=tilt, zoom=zoom)


def cif(pdb_id: str) -> Path:
    return PDB_CACHE / f"{pdb_id}.cif"


# -- scenes -----------------------------------------------------------------------------------------------------------------------------

def pocket(pdb_id: str, ligand: str, theme: str = "dark", size=(1800, 1100), side: bool = True, shell: float = 4.0, slab: float = 15.0) -> Scene:
    """The heme pocket of a CYP3A4 crystal structure: translucent cartoon and pocket surface, the lining residues, the heme with its iron, the ligand, and the iron coordination as a dashed line."""
    body = f'''
cmd.load({str(cif(pdb_id))!r}, "s")
cmd.remove("solvent")
cmd.hide("everything")
cmd.select("lig", "resn {ligand}")
cmd.select("hem", "resn HEM")
cmd.select("pocket", "byres (polymer within {shell} of lig) or ((polymer and name SG and resn CYS) within 4 of elem FE)")
cmd.show("cartoon", "polymer"); cmd.color(PAL["protein"], "polymer"); cmd.set("cartoon_transparency", PAL["cartoon_t"])
cmd.create("pk", "(polymer within 10 of lig) and not (byres (polymer within 3.2 of lig))")
cmd.hide("everything", "pk"); cmd.show("surface", "pk"); cmd.color(PAL["surface"], "pk"); cmd.set("transparency", PAL["surface_t"], "pk"); cmd.set("surface_quality", 1)
cmd.show("sticks", "pocket")
cmd.color(PAL["pocket_c"], "pocket and elem C"); cmd.color(PAL["n"], "pocket and elem N"); cmd.color(PAL["o"], "pocket and elem O"); cmd.color(PAL["s"], "pocket and elem S")
cmd.show("sticks", "lig"); cmd.color(PAL["ligand"], "lig and elem C"); cmd.set("stick_radius", 0.2, "lig")
cmd.show("sticks", "hem"); cmd.color(PAL["heme"], "hem and elem C"); cmd.set("stick_radius", 0.17, "hem")
cmd.show("spheres", "elem FE"); cmd.set("sphere_scale", 0.5, "elem FE"); cmd.color(PAL["iron"], "elem FE")
cmd.distance("coord", "elem FE", "(lig and elem N) within 2.8 of elem FE"); cmd.hide("labels")
{orient(zoom=9, tilt=14 if side else 90)}
cmd.clip("slab", {slab})
'''
    anchors = {"fe": "elem FE", "ligand": "lig", "coordinating_n": "(lig and elem N) within 2.8 of elem FE", "cys": "(polymer and name SG and resn CYS) within 4 of elem FE", "heme": "hem and not elem FE"}
    return render(f"pocket_{pdb_id}_{ligand}", body, size, theme, anchors, files=(cif(pdb_id),))


def overview(pdb_id: str, ligand: str, theme: str = "dark", size=(1800, 1100)) -> Scene:
    """The whole enzyme as a cartoon with the heme and the ligand as spheres, to show where the pocket is."""
    body = f'''
cmd.load({str(cif(pdb_id))!r}, "s")
cmd.remove("solvent")
cmd.hide("everything")
cmd.select("lig", "resn {ligand}"); cmd.select("hem", "resn HEM")
cmd.show("cartoon", "polymer")
cmd.color(PAL["protein"], "polymer"); cmd.set("cartoon_transparency", 0.15); cmd.set("cartoon_fancy_helices", 1)
cmd.show("spheres", "hem"); cmd.color(PAL["heme"], "hem and elem C"); cmd.color(PAL["iron"], "elem FE"); cmd.set("sphere_scale", 0.6, "hem")
cmd.show("sticks", "lig"); cmd.color(PAL["ligand"], "lig and elem C"); cmd.set("stick_radius", 0.25, "lig")
cmd.orient("polymer"); cmd.turn("y", 30); cmd.zoom("polymer", 2)
'''
    return render(f"overview_{pdb_id}_{ligand}", body, size, theme, {"fe": "elem FE", "ligand": "lig"}, files=(cif(pdb_id),))


def predicted_pocket(cif_path: Path, theme: str = "dark", size=(1800, 1100), key: str = "predicted", slab: float = 15.0, shell: float = 4.0) -> Scene:
    """The pocket of a predicted complex (a Boltz-2 output: the ligand is residue LIG1, the heme HEM), drawn like ``pocket``."""
    body = f'''
cmd.load({str(cif_path)!r}, "s")
cmd.remove("solvent")
cmd.hide("everything")
cmd.select("lig", "resn LIG1 or resn LIG")
cmd.select("hem", "resn HEM")
cmd.select("pocket", "byres (polymer within {shell} of lig) or ((polymer and name SG and resn CYS) within 4 of elem FE)")
cmd.show("cartoon", "polymer"); cmd.color(PAL["protein"], "polymer"); cmd.set("cartoon_transparency", PAL["cartoon_t"])
cmd.create("pk", "(polymer within 10 of lig) and not (byres (polymer within 3.2 of lig))")
cmd.hide("everything", "pk"); cmd.show("surface", "pk"); cmd.color(PAL["surface"], "pk"); cmd.set("transparency", PAL["surface_t"], "pk"); cmd.set("surface_quality", 1)
cmd.show("sticks", "pocket")
cmd.color(PAL["pocket_c"], "pocket and elem C"); cmd.color(PAL["n"], "pocket and elem N"); cmd.color(PAL["o"], "pocket and elem O"); cmd.color(PAL["s"], "pocket and elem S")
cmd.show("sticks", "lig"); cmd.color(PAL["ligand"], "lig and elem C"); cmd.set("stick_radius", 0.2, "lig")
cmd.show("sticks", "hem"); cmd.color(PAL["heme"], "hem and elem C"); cmd.set("stick_radius", 0.17, "hem")
cmd.show("spheres", "elem FE"); cmd.set("sphere_scale", 0.5, "elem FE"); cmd.color(PAL["iron"], "elem FE")
cmd.distance("coord", "elem FE", "(lig and elem N) within 3.0 of elem FE"); cmd.hide("labels")
{orient(zoom=9)}
cmd.clip("slab", {slab})
cmd.set("depth_cue", 0)
'''
    anchors = {"fe": "elem FE", "ligand": "lig", "coordinating_n": "(lig and elem N) within 3.0 of elem FE", "cys": "(polymer and name SG and resn CYS) within 4 of elem FE"}
    return render(f"{key}", body, size, theme, anchors, files=(cif_path,))


def pose_comparison(reference: Path, others: list[tuple[Path, str]], theme: str = "dark", size=(1800, 1100), key: str = "poses", slab: float = 18.0, crystal: tuple[Path, str] | None = None, zoom: float = 11.0) -> Scene:
    """Several predicted poses of one ligand in the pocket of ``reference``: every other complex is aligned on the protein backbone and only its ligand is drawn, in the colour given with it.

    The reference ligand is teal; the heme and the iron come from the reference. ``crystal`` is an optional (CIF path, ligand code) of the experimental structure: the reference is aligned on it and its
    ligand is drawn in white."""
    loads = "\n".join(f'cmd.load({str(p)!r}, "o{i}"); cmd.align("o{i} and polymer and name CA", "s and polymer and name CA"); cmd.hide("everything", "o{i}"); cmd.select("l{i}", "o{i} and (resn LIG1 or resn LIG)"); cmd.show("sticks", "l{i}"); cmd.color({c!r}, "l{i} and elem C"); cmd.set("stick_radius", 0.17, "l{i}")'
                      for i, (p, c) in enumerate(others))
    crystal_block = "" if crystal is None else (f'cmd.load({str(crystal[0])!r}, "cryst"); cmd.remove("cryst and solvent"); cmd.hide("everything", "cryst"); '
                                                 f'cmd.align("s and polymer and name CA", "cryst and polymer and name CA"); cmd.select("cl", "cryst and resn {crystal[1]}"); cmd.show("sticks", "cl"); '
                                                 f'cmd.color("white" if PAL["bg"] != "0xffffff" else "0x444444", "cl and elem C"); cmd.set("stick_radius", 0.24, "cl")')
    body = f'''
cmd.load({str(reference)!r}, "s")
cmd.remove("solvent")
cmd.hide("everything")
cmd.select("lig", "s and (resn LIG1 or resn LIG)")
cmd.select("hem", "s and resn HEM")
cmd.show("cartoon", "s and polymer"); cmd.color(PAL["protein"], "s and polymer"); cmd.set("cartoon_transparency", PAL["cartoon_t"])
cmd.create("pk", "(s and polymer within 10 of lig) and not (byres (s and polymer within 3.2 of lig))")
cmd.hide("everything", "pk"); cmd.show("surface", "pk"); cmd.color(PAL["surface"], "pk"); cmd.set("transparency", PAL["surface_t"], "pk"); cmd.set("surface_quality", 1)
cmd.show("sticks", "lig"); cmd.color(PAL["ligand"], "lig and elem C"); cmd.set("stick_radius", 0.22, "lig")
cmd.show("sticks", "hem"); cmd.color(PAL["heme"], "hem and elem C"); cmd.set("stick_radius", 0.17, "hem")
cmd.show("spheres", "s and elem FE"); cmd.set("sphere_scale", 0.5, "s and elem FE"); cmd.color(PAL["iron"], "s and elem FE")
{crystal_block}
{loads}
{orient(zoom=zoom, ref="s and ")}
cmd.clip("slab", {slab})
cmd.set("depth_cue", 0)
'''
    anchors = {"fe": "s and elem FE", "ligand": "lig"} | {f"pose{i}": f"l{i}" for i in range(len(others))}
    return render(f"{key}", body, size, theme, anchors, files=(reference, *[p for p, _ in others], *([crystal[0]] if crystal else [])))


def heme_closeup(pdb_id: str, ligand: str, theme: str = "dark", size=(1500, 1000)) -> Scene:
    """The heme of a crystal structure seen edge-on, with its iron as a sphere and the axial cysteine below it; the ligand is not drawn (the iron's sixth position is empty or holds a water in the resting enzyme)."""
    body = f'''
cmd.load({str(cif(pdb_id))!r}, "s")
cmd.remove("solvent")
cmd.hide("everything")
cmd.select("lig", "resn {ligand}")
cmd.select("hem", "resn HEM")
cmd.select("cys", "((polymer and resn CYS) within 4.5 of elem FE)")
cmd.show("sticks", "hem"); cmd.color(PAL["heme"], "hem and elem C"); cmd.color(PAL["n"], "hem and elem N"); cmd.color(PAL["o"], "hem and elem O"); cmd.set("stick_radius", 0.22, "hem")
cmd.show("sticks", "byres cys"); cmd.color(PAL["pocket_c"], "(byres cys) and elem C"); cmd.color(PAL["s"], "(byres cys) and elem S"); cmd.color(PAL["n"], "(byres cys) and elem N"); cmd.color(PAL["o"], "(byres cys) and elem O"); cmd.set("stick_radius", 0.2, "byres cys")
cmd.show("spheres", "elem FE"); cmd.set("sphere_scale", 0.6, "elem FE"); cmd.color(PAL["iron"], "elem FE")
{orient(zoom=5, tilt=18)}
cmd.set("depth_cue", 0)
'''
    return render(f"heme_{pdb_id}", body, size, theme, {"fe": "elem FE", "cys": "(polymer and name SG and resn CYS) within 4 of elem FE"}, files=(cif(pdb_id),))


def autocrop(image, margin: int = 24):
    """The picture (an array read with matplotlib) cropped to the bounding box of everything that differs from its background, plus a margin."""
    import numpy as np

    rgb = image[..., :3]
    background = rgb[0, 0]
    changed = np.abs(rgb - background).max(axis=2) > 0.04
    rows, cols = np.nonzero(changed)
    if len(rows) == 0:
        return image
    top, bottom, left, right = max(rows.min() - margin, 0), min(rows.max() + margin, image.shape[0]), max(cols.min() - margin, 0), min(cols.max() + margin, image.shape[1])
    return image[top:bottom, left:right]
