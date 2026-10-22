# SPDX-License-Identifier: GPL-3.0-or-later
# This file drives Blender's Python interface (and, with the Molecular Nodes backend, the Molecular Nodes add-on, GPL-3.0-or-later); like Blender's own scripts it is offered under the
# GPL. The rest of the code of the repository is AGPL-3.0-or-later (compatible with the GPL through section 13 of the AGPL); nothing of Blender or Molecular Nodes is copied into it.
"""One cinematic molecular scene rendered with Blender from real coordinates: ``Blender -b -P scene.py -- --config scene.json``.

Two backends draw the enzyme. ``molecularnodes`` (the default of the film) loads the structure file itself with the Molecular Nodes add-on and styles it through geometry nodes: a
translucent surface for the protein and glowing ball-and-stick styles on selections (the heme, the ligand, the residues around it). The ``obj`` backend takes a protein surface mesh exported
from PyMOL and the atoms listed in the configuration. The picture follows the language of a cinematic explainer: a translucent blue enzyme with a bright rim, the heme and the ligand glowing,
a shallow depth of field that keeps the active site sharp, out-of-focus particles in the background and a slow orbit.

Everything that is drawn comes from coordinates: no generated imagery is used. The per-frame screen position of the atoms named in ``anchors`` (the iron, the coordinating nitrogen) is written
to ``anchors.json`` so that the film can attach leader-line callouts to the moving atoms.
"""

import json
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

ELEMENT_COLOUR = {"FE": (1.0, 0.85, 0.2), "C": (1.0, 0.55, 0.12), "N": (0.2, 0.55, 1.0), "O": (1.0, 0.25, 0.2), "S": (1.0, 0.9, 0.2), "CL": (0.4, 1.0, 0.4), "F": (0.5, 1.0, 0.9)}
RADIUS = {"FE": 0.9, "C": 0.55, "N": 0.5, "O": 0.5, "S": 0.7, "CL": 0.7, "F": 0.45}


def load_config() -> dict:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return json.loads(Path(argv[argv.index("--config") + 1]).read_text())


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(block):
            block.remove(item)


def protein_material(colour=(0.03, 0.14, 0.75), alpha: float = 0.5) -> bpy.types.Material:
    material = bpy.data.materials.new("enzyme")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    out, bsdf = nodes.new("ShaderNodeOutputMaterial"), nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.28
    bsdf.inputs["Alpha"].default_value = alpha
    try:
        material.surface_render_method = "BLENDED"  # translucent enzyme: the glowing heme and ligand show through
    except (AttributeError, TypeError):
        material.blend_method = "BLEND"
    material.use_backface_culling = True
    for name, value in (("Subsurface Weight", 0.6), ("Specular IOR Level", 0.7), ("Coat Weight", 0.4)):
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = value
    if "Subsurface Radius" in bsdf.inputs:
        bsdf.inputs["Subsurface Radius"].default_value = (0.8, 0.5, 0.3)
    rim, ramp, emission, mix = (nodes.new("ShaderNodeLayerWeight"), nodes.new("ShaderNodeValToRGB"), nodes.new("ShaderNodeEmission"), nodes.new("ShaderNodeAddShader"))
    rim.inputs["Blend"].default_value = 0.35
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.55, 0.95
    ramp.color_ramp.elements[0].color, ramp.color_ramp.elements[1].color = (0, 0, 0, 1), (0.25, 0.75, 1.0, 1)
    emission.inputs["Strength"].default_value = 1.6
    links.new(rim.outputs["Fresnel"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], emission.inputs["Color"])
    links.new(bsdf.outputs["BSDF"], mix.inputs[0])
    links.new(emission.outputs["Emission"], mix.inputs[1])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return material


def glow_material(name: str, colour, strength: float) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    out, emission = nodes.new("ShaderNodeOutputMaterial"), nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*colour, 1.0)
    emission.inputs["Strength"].default_value = strength
    links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return material


def import_surface(path: str, material) -> bpy.types.Object:
    bpy.ops.wm.obj_import(filepath=path)
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.01)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.shade_smooth()
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def add_atoms(atoms: list[dict], scale: float) -> dict[str, bpy.types.Object]:
    made = {}
    for atom in atoms:
        element = atom["element"].upper()
        bpy.ops.mesh.primitive_uv_sphere_add(radius=RADIUS.get(element, 0.5) * scale, location=atom["xyz"], segments=24, ring_count=12)
        sphere = bpy.context.active_object
        sphere.data.materials.append(glow_material(f"{atom['group']}_{element}", ELEMENT_COLOUR.get(element, (1, 1, 1)), 6.0 if atom["group"] == "ligand" else 4.0))
        bpy.ops.object.shade_smooth()
        made[atom["name"]] = sphere
    return made


def add_lights(centre: Vector, size: float, unit: float = 1.0) -> None:
    for name, location, colour, energy in (("key", (1.6, -1.2, 1.4), (0.8, 0.9, 1.0), 6e4), ("rim_cyan", (-1.8, 1.3, 0.8), (0.2, 0.8, 1.0), 2.5e5), ("rim_orange", (0.4, 1.8, -0.9), (1.0, 0.55, 0.2), 1.2e5)):
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.color, light.size = energy * unit**2, colour, size * 0.8
        obj = bpy.data.objects.new(name, light)
        obj.location = centre + Vector(location) * size
        bpy.context.collection.objects.link(obj)
        track = obj.constraints.new("TRACK_TO")
        track.target = bpy.data.objects["focus"]
        track.track_axis, track.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"


def add_particles(centre: Vector, size: float, count: int, seed: int) -> None:
    import random

    rng = random.Random(seed)
    for _ in range(count):
        direction = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))).normalized()
        distance = size * rng.uniform(1.4, 3.2)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size * rng.uniform(0.005, 0.018), location=centre + direction * distance, segments=12, ring_count=6)
        orb = bpy.context.active_object
        warm = rng.random() < 0.2
        orb.data.materials.append(glow_material("orb", (1.0, 0.6, 0.2) if warm else (0.3, 0.7, 1.0), rng.uniform(1.5, 5)))


def add_bloom(scene) -> None:
    """A glow around the bright atoms, through the compositor (its API differs between Blender 4 and 5)."""
    if hasattr(scene, "compositing_node_group"):
        group = bpy.data.node_groups.new("bloom", "CompositorNodeTree")
        group.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        layers, glare, out = group.nodes.new("CompositorNodeRLayers"), group.nodes.new("CompositorNodeGlare"), group.nodes.new("NodeGroupOutput")
        for name, value in (("Type", "Fog Glow"), ("Quality", "High"), ("Threshold", 0.9), ("Size", 0.7), ("Strength", 1.0)):
            try:
                glare.inputs[name].default_value = value
            except (TypeError, KeyError):
                pass
        group.links.new(layers.outputs["Image"], glare.inputs["Image"])
        group.links.new(glare.outputs["Image"], out.inputs["Image"])
        scene.compositing_node_group = group
        return
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers, glare, composite = tree.nodes.new("CompositorNodeRLayers"), tree.nodes.new("CompositorNodeGlare"), tree.nodes.new("CompositorNodeComposite")
    glare.glare_type, glare.quality, glare.threshold, glare.size = "FOG_GLOW", "HIGH", 0.9, 7
    tree.links.new(layers.outputs["Image"], glare.inputs["Image"])
    tree.links.new(glare.outputs["Image"], composite.inputs["Image"])


def uniform_glow_material(name: str, colour, strength: float) -> bpy.types.Material:
    """A lit, softly glowing material of one colour (the atom spheres of the styles are instances and do not carry the ``Color`` attribute that the bonds read, so the colour is set here).

    The emission keeps the atoms bright against the dark enzyme; the lit base colour gives them shape, so that balls and sticks read as volumes and not as a flat shape.
    """
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    out, bsdf = nodes.new("ShaderNodeOutputMaterial"), nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*colour[:3], 1.0)
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Emission Color"].default_value = (*colour[:3], 1.0)
    bsdf.inputs["Emission Strength"].default_value = strength
    for key, value in (("Specular IOR Level", 0.8), ("Coat Weight", 0.5)):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = value
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return material


MN_UNIT = 0.1  # Molecular Nodes draws one angstrom as 0.1 Blender units


def load_with_molecular_nodes(cfg: dict):
    """The complex drawn by Molecular Nodes (GPL-3.0-or-later, https://github.com/BradyAJohnston/MolecularNodes) from the structure file itself.

    The enzyme is a translucent surface style; every entry of ``highlight`` is a style on a selection (MDAnalysis selection language) with a glowing material. The atoms named in
    ``anchors`` become empties whose screen position is written for the film's callouts. Returns the molecule object, the anchor empties and the focus point in Blender units.
    """
    import importlib

    import numpy as np

    bpy.ops.preferences.addon_enable(module="bl_ext.user_default.molecularnodes")
    mn = importlib.import_module("bl_ext.user_default.molecularnodes")
    mol = mn.Molecule.load(cfg["structure"], style=None, name="complex")
    mol.add_style("surface", selection=cfg.get("protein_selection", "protein"), material=protein_material(tuple(cfg.get("protein_colour", (0.03, 0.14, 0.75))), cfg.get("protein_alpha", 0.5)),
                  color=(*cfg.get("protein_colour", (0.03, 0.14, 0.75)), 1.0))
    for k, group in enumerate(cfg.get("highlight", [])):
        mol.add_style(group.get("style", "ball_and_stick"), selection=group["selection"], material=uniform_glow_material(f"glow_{k}", group["colour"], group.get("strength", 5.0)), color=(*group["colour"][:3], 1.0))
    atoms = mol.atoms
    names, resnames, elements, positions = np.array(atoms.names), np.array(atoms.resnames), np.array(atoms.elements), np.array(atoms.positions)
    found: dict[str, Vector] = {}
    anchors = {}
    for label, spec in cfg.get("anchors", {}).items():
        mask = resnames == spec["resname"]
        if "name" in spec:
            mask &= names == spec["name"]
        if "element" in spec:
            mask &= np.char.upper(elements.astype(str)) == spec["element"].upper()
        candidates = np.flatnonzero(mask)
        if len(candidates) == 0:
            raise ValueError(f"anchor {label}: no atom matches {spec}")
        if "nearest_to" in spec:
            reference = found[spec["nearest_to"]] / MN_UNIT
            candidates = candidates[[int(np.argmin(np.linalg.norm(positions[candidates] - np.array(reference), axis=1)))]]
        point = Vector(positions[candidates[0]] * MN_UNIT)
        bpy.ops.object.empty_add(location=point)
        empty = bpy.context.active_object
        empty.name = label
        anchors[label] = empty
        found[label] = point
    focus = found[cfg["focus_anchor"]] if cfg.get("focus_anchor") else Vector(cfg["focus"]) * MN_UNIT
    return mol.object, anchors, focus


def main() -> None:
    cfg = load_config()
    clear_scene()
    scene = bpy.context.scene
    engines = set(bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys())
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = cfg.get("resolution", [1920, 1080])
    scene.render.fps = cfg.get("fps", 24)
    scene.frame_start, scene.frame_end = 1, cfg["frames"]
    scene.render.image_settings.file_format = "PNG"
    try:
        scene.eevee.taa_render_samples = int(cfg.get("samples", 48))
    except AttributeError:
        pass
    for transform in ("AgX", "Filmic", "Standard"):
        try:
            scene.view_settings.view_transform = transform
            break
        except TypeError:
            continue
    world = scene.world or bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.012, 0.035, 1.0)

    unit = 1.0
    if cfg.get("backend") == "molecularnodes":
        surface, atoms, focus_xyz = load_with_molecular_nodes(cfg)
        unit = MN_UNIT
    else:
        surface = import_surface(cfg["surface_obj"], protein_material(tuple(cfg.get("protein_colour", (0.03, 0.14, 0.75))), cfg.get("protein_alpha", 0.5)))
        atoms = add_atoms(cfg["atoms"], cfg.get("atom_scale", 1.0))
        focus_xyz = Vector(cfg["focus"])
    bpy.ops.object.empty_add(location=focus_xyz)
    focus = bpy.context.active_object
    focus.name = "focus"
    bbox = [surface.matrix_world @ Vector(c) for c in surface.bound_box]
    centre = sum(bbox, Vector()) / 8
    size = max((b - centre).length for b in bbox)
    add_lights(centre, size, unit)
    add_particles(centre, size, cfg.get("particles", 160), cfg.get("seed", 3))

    camera_data = bpy.data.cameras.new("camera")
    camera_data.lens, camera_data.dof.use_dof = cfg.get("lens", 55), True
    camera_data.dof.focus_object, camera_data.dof.aperture_fstop = focus, cfg.get("fstop", 1.6)
    camera = bpy.data.objects.new("camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    track = camera.constraints.new("TRACK_TO")
    track.target, track.track_axis, track.up_axis = focus, "TRACK_NEGATIVE_Z", "UP_Y"

    # camera path: an orbit around the focus whose radius, height and angle are interpolated between the start and the end of the clip
    start, end = cfg["camera"]["start"], cfg["camera"]["end"]
    anchors = {}
    for frame in range(1, cfg["frames"] + 1):
        k = (frame - 1) / max(cfg["frames"] - 1, 1)
        k = k * k * (3 - 2 * k)  # ease in and out
        azimuth = math.radians(start["azimuth"] + (end["azimuth"] - start["azimuth"]) * k)
        elevation = math.radians(start["elevation"] + (end["elevation"] - start["elevation"]) * k)
        radius = (start["radius"] + (end["radius"] - start["radius"]) * k) * unit
        camera.location = focus_xyz + Vector((radius * math.cos(elevation) * math.cos(azimuth), radius * math.cos(elevation) * math.sin(azimuth), radius * math.sin(elevation)))
        camera.keyframe_insert("location", frame=frame)
        bpy.context.view_layer.update()
        anchors[frame] = {name: [round(v, 5) for v in world_to_camera_view(scene, camera, obj.location)[:2]] for name, obj in atoms.items()}
    add_bloom(scene)
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "anchors.json").write_text(json.dumps(anchors))
    scene.render.filepath = str(out / "f_")
    if cfg.get("still"):
        scene.frame_set(cfg["still"])
        scene.render.filepath = str(out / "still.png")
        bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(animation=True)


main()
