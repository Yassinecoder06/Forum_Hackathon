"""Convert a MetaHuman v2 export (UE 5.7 "Optimized" FBX meshes + DCC textures, see docs/avatar/METAHUMAN.md) into one
web GLB: body + detailed face + outfit + ponytail on ONE skeleton, 52 ARKit morph targets, eyebrows painted into
the skin, a 4-bone ponytail chain for spring physics.

    blender -b --factory-startup --python scripts/metahuman/convert_metahuman_v2.py -- \
        --src D:/MetaHuman_Export_v2/F01 --out .runtime/avatar-candidates/MetaHuman.glb [--renders DIR]

Inputs (read only, never modified): <ID>_Body.fbx, <ID>_Face_LOD0.fbx, <ID>_Face_ARKit.fbx (LOD1 face + one ARKit pose
per frame, pose i at Blender frame i+2), arkit_frames.json, <ID>_Outfit.fbx, Hair/*Ponytail*.fbx, Hair/*Eyebrows*.fbx,
Textures/{Head,Body,Eyes,Teeth,Hair,Outfit}.

What it does, and why (sign language needs readable hands, eyebrows and mouth):
* ARKit morph targets on the DETAILED face: the ARKit animation drives joints of the face skeleton, which LOD0 shares,
  so the animation is played on the LOD0 face and each pose is captured as a shape key (delta from the neutral pose).
  287 corrective joints were recorded at rest by Unreal: wrinkle/corrective detail is missing, main shapes are not.
* Eyebrows: the brow cards have no hair-atlas UVs and the head texture has no painted brows, so the cards are baked
  as a soft mask into the head colour map (Cycles, selected-to-active). Painted brows deform with every brow shape.
* Face material slots are mislabelled by the export (shifted by one), so roles come from geometry: skin = largest,
  eyes = compact one-sided parts, teeth = largest part in the mouth, lashes = largest card part in the eye band;
  transparent Unreal layers (eye shell, lacrimal edge, saliva) are dropped.
* Ponytail: skinned to the head; a 4-bone chain (hair_ponytail_01..04) along the tail gets smooth weights for the
  Studio's spring physics. No cloth simulation: the garment is fitted.
* Outfit: rendered dark (navy top, charcoal shorts) with the garment's own normal and AO maps: interpreters wear dark
  plain tops so the hands read clearly; the exported garment is pale.
* Units metres, +Z forward in glTF, WebP textures, morph targets without normals (sparse when smaller).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_metahuman import (add_finger_tips, copy_facial_bones, import_fbx, load_image, log, of_type, pbr,  # noqa: E402
                               pil_rgba, image_from_array, prune, reskin, set_active, delete_slot_faces)

HAIR_SRGB = (0.125, 0.082, 0.098)          # manifest #201519 (estimated, "really darker brown-black")
BROW_SRGB = (0.1, 0.07, 0.065)       # a shade lighter than the near-black hair, as natural brows are
SHIRT, SHORTS = (0.012, 0.016, 0.034), (0.02, 0.02, 0.024)   # linear: navy, charcoal
srgb_to_lin = lambda c: tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c)  # noqa: E731


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--face-texture-size", type=int, default=2048)
    p.add_argument("--texture-size", type=int, default=1024)
    p.add_argument("--blend", default=None, help="also save the assembled scene as .blend (for checks)")
    return p.parse_args(argv)


# ------------------------------------------------------------------------------------------------ ARKit shapes
def bake_arkit_shapes(face, face_arm, arkit_path, frames):
    """Play the ARKit animation on the LOD0 face skeleton and store each pose as a shape key on the LOD0 mesh."""
    objs = import_fbx(arkit_path)
    src_arm = of_type(objs, "ARMATURE")[0]
    action = src_arm.animation_data.action
    # the rest poses must agree, else the local-space animation would not transfer
    worst = 0.0
    for b in face_arm.data.bones:
        o = src_arm.data.bones.get(b.name)
        if o is not None:
            worst = max(worst, (b.matrix_local.translation - o.matrix_local.translation).length)
    log(f"ARKit skeleton vs LOD0 face skeleton: max rest offset {worst:.4f} (armature units, cm)")
    if worst > 0.05:
        raise RuntimeError("ARKit and LOD0 face skeletons differ; refusing to bake")
    # the action also keys the armature OBJECT (location/rotation/scale 0.01): keep only joint channels, else the
    # face skeleton is rescaled during the bake and every shape is wrong by the object transform
    def fcurve_sets(act):
        if hasattr(act, "layers") and len(act.layers):
            for layer in act.layers:
                for strip in layer.strips:
                    for bag in strip.channelbags:
                        yield bag.fcurves
        elif hasattr(act, "fcurves"):
            yield act.fcurves
    dropped = 0
    for fcs in fcurve_sets(action):
        for fc in [fc for fc in fcs if not fc.data_path.startswith("pose.bones")]:
            fcs.remove(fc)
            dropped += 1
    log(f"ARKit action: removed {dropped} object-level channels")
    face_arm.animation_data_create()
    face_arm.animation_data.action = action
    if hasattr(face_arm.animation_data, "action_slot") and getattr(action, "slots", None):
        face_arm.animation_data.action_slot = action.slots[0]
    scene = bpy.context.scene

    def evaluated(frame):
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = face.evaluated_get(dg)
        me = ev.to_mesh()
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        ev.to_mesh_clear()
        return co.reshape(-1, 3)

    neutral = evaluated(2)
    basis = np.empty(len(face.data.vertices) * 3, np.float32)
    face.data.vertices.foreach_get("co", basis)
    basis = basis.reshape(-1, 3)
    log(f"neutral pose vs rest mesh: max {np.abs(neutral - basis).max():.4f} cm")
    poses = [(name, evaluated(i + 2) - neutral) for i, name in frames]     # all poses first: keys would feed back
    face.shape_key_add(name="Basis", from_mix=False)
    made = []
    for name, delta in poses:
        delta[np.abs(delta) < 2e-3] = 0.0            # below 0.02 mm: noise, keeps sparse storage small
        moved = int((np.abs(delta).max(1) > 0).sum())
        key = face.shape_key_add(name=name, from_mix=False)
        key.data.foreach_set("co", (basis + delta).ravel())
        key.value = 0.0
        key.slider_min, key.slider_max = 0.0, 1.0
        made.append((name, moved, float(np.abs(delta).max())))
    face_arm.animation_data.action = None
    for pb in face_arm.pose.bones:
        pb.matrix_basis.identity()
    scene.frame_set(1)
    for o in objs:
        bpy.data.objects.remove(o, do_unlink=True)
    log(f"ARKit shape keys: {len(made)}; e.g. " + ", ".join(f"{n} {m} verts {d:.2f} cm" for n, m, d in made[:6]))
    small = [n for n, m, d in made if d < 0.05]
    if small:
        log("shapes with almost no motion (check):", small)
    return made


# ------------------------------------------------------------------------------------------------ face slots
def classify_face_slots_v2(face):
    parts = {}
    for p in face.data.polygons:
        parts.setdefault(p.material_index, []).append(tuple(face.matrix_world @ p.center))
    st = {i: np.array(c) for i, c in parts.items()}
    roles = {}
    skin = max(st, key=lambda i: len(st[i]))
    roles[skin] = "skin"
    rest = [i for i in st if i != skin]
    for i in rest:
        xs = st[i][:, 0]
        if len(st[i]) > 200 and xs.min() > 0:
            roles[i] = "eye_l"
        elif len(st[i]) > 200 and xs.max() < 0:
            roles[i] = "eye_r"
    eye_z = np.mean([st[i][:, 2].mean() for i in roles if roles[i].startswith("eye")]) if any(r.startswith("eye") for r in roles.values()) else None
    rest = [i for i in rest if i not in roles]
    if eye_z is not None:
        mouth = [i for i in rest if st[i][:, 2].mean() < eye_z - 0.03]
        if mouth:
            teeth = max(mouth, key=lambda i: len(st[i]))
            roles[teeth] = "teeth"
            for i in mouth:
                if i != teeth and len(st[i]) > 300 and abs(st[i][:, 0].mean()) < 0.01:
                    roles[i] = "tongue"
        band = [i for i in rest if i not in roles and abs(st[i][:, 2].mean() - eye_z) < 0.03]
        if band:
            roles[max(band, key=lambda i: len(st[i]))] = "lashes"
    for i in st:
        roles.setdefault(i, "drop")
    return roles


# ------------------------------------------------------------------------------------------------ eyebrows
def bake_brow_mask(face, brows, size, skin_slot, preview=None):
    """Mask of the brow cards in the skin UV (selected-to-active emission bake). Only the skin slot writes to the mask:
    eyes, teeth and lash cards reuse the same UV square and would stamp their hits onto unrelated skin texels."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.device = "CPU"
    mask = bpy.data.images.new("brow_mask", size, size, alpha=False)
    mask.generated_color = (0, 0, 0, 1)
    saved = [s.material for s in face.material_slots]
    tgt = bpy.data.materials.new("bake_target")
    node = tgt.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = mask
    tgt.node_tree.nodes.active = node
    dummy = bpy.data.materials.new("bake_dummy")
    dn = dummy.node_tree.nodes.new("ShaderNodeTexImage")
    dn.image = bpy.data.images.new("bake_dummy", 8, 8)
    dummy.node_tree.nodes.active = dn
    for i, s in enumerate(face.material_slots):
        s.material = tgt if i == skin_slot else dummy
    em = bpy.data.materials.new("brow_emit")
    nt = em.node_tree
    for n in list(nt.nodes):
        if n.type != "OUTPUT_MATERIAL":
            nt.nodes.remove(n)
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (1, 1, 1, 1)
    nt.links.new(e.outputs["Emission"], next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL").inputs["Surface"])
    brows.data.materials.clear()
    brows.data.materials.append(em)
    set_active(face)
    brows.select_set(True)
    wb = lambda o: np.array([tuple(o.matrix_world @ Vector(c)) for c in o.bound_box])  # noqa: E731
    log("brow bake: face bbox", wb(face).min(0).round(3), wb(face).max(0).round(3), "brows bbox", wb(brows).min(0).round(3), wb(brows).max(0).round(3),
        "selected", [o.name for o in bpy.context.selected_objects], "active", bpy.context.view_layer.objects.active.name)
    # the cards float 0.1-4.6 mm above the skin: short rays only, so nothing else on the face reaches them
    bpy.ops.object.bake(type="EMIT", use_selected_to_active=True, cage_extrusion=0.008, max_ray_distance=0.014, margin=2,
                        use_clear=True)
    if preview:
        mask.filepath_raw = preview
        mask.file_format = "PNG"
        mask.save()
    for s, m in zip(face.material_slots, saved):
        s.material = m
    a = np.empty(size * size * 4, np.float32)
    mask.pixels.foreach_get(a)
    m = a.reshape(size, size, 4)[..., 0]
    log(f"brow mask: {int((m > 0.5).sum())} texels of {size * size}")
    return m


def blur(x, r=2):
    k = 2 * r + 1
    pad = np.pad(x, r, mode="edge")
    c = np.cumsum(np.cumsum(pad, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def brow_strokes(mask, seed=7):
    """Hair-by-hair brow inside the baked card area (rows grow upward on the face: Blender pixel order).

    Inner heads: short hairs pointing up; body: hairs lying outward and slightly up; tails: outward and down.
    Strokes are splatted with bilinear weights, so they stay thin and anti-aliased."""
    rng = np.random.default_rng(seed)
    size = mask.shape[0]
    inside = blur(mask, 3) > 0.6
    ys, xs = np.nonzero(inside)
    if not len(xs):
        return np.zeros_like(mask)
    centre = size / 2
    span = np.abs(xs - centre).max()
    inner = np.abs(xs - centre).min()
    out = np.zeros_like(mask)
    n = int(len(xs) / 14)
    pick = rng.integers(0, len(xs), n)
    scale = size / 2048
    for x0, y0 in zip(xs[pick].astype(np.float32), ys[pick].astype(np.float32)):
        side = 1.0 if x0 > centre else -1.0
        t = (abs(x0 - centre) - inner) / max(span - inner, 1)    # 0 inner head .. 1 tail
        angle = np.interp(t, [0, 0.18, 0.5, 0.8, 1], [80, 45, 18, 0, -18]) + rng.normal(0, 8)
        length = (np.interp(t, [0, 0.2, 1], [14, 26, 20]) + rng.normal(0, 4)) * scale
        a = np.radians(angle)
        d = np.array([side * np.cos(a), np.sin(a)])
        bend = rng.normal(0, 0.15)
        steps = max(int(length * 2), 4)
        for k in range(steps):
            u = k / (steps - 1)
            dd = d + np.array([-d[1], d[0]]) * bend * u
            px, py = x0 + dd[0] * length * u, y0 + dd[1] * length * u
            ix, iy = int(px), int(py)
            if not (0 <= ix < size - 1 and 0 <= iy < size - 1):
                break
            fx, fy = px - ix, py - iy
            w = 0.35 * (1 - 0.6 * u)                          # hairs thin out towards the tip
            out[iy, ix] += w * (1 - fx) * (1 - fy)
            out[iy, ix + 1] += w * fx * (1 - fy)
            out[iy + 1, ix] += w * (1 - fx) * fy
            out[iy + 1, ix + 1] += w * fx * fy
    return 1 - np.exp(-out * 2.2)


def head_with_brows(head_png, mask, size):
    bc = pil_rgba(head_png)
    if bc.shape[0] != size:
        k = bc.shape[0] // size
        bc = bc.reshape(size, k, size, k, 4).mean(axis=(1, 3)) if k > 1 else bc
    hairs = brow_strokes(mask)
    tint = blur(np.clip((blur(mask, 6) - 0.35) / 0.4, 0, 1), 3) * 0.18     # faint shadow of skin under the hairs
    a = np.clip(np.maximum(hairs * 0.85, tint), 0, 0.9)[..., None]
    out = bc.copy()
    out[..., :3] = bc[..., :3] * (1 - a) + np.array(BROW_SRGB)[None, None, :] * a
    out[..., 3] = 1
    return out


# ------------------------------------------------------------------------------------------------ hair
def hair_atlas(path, size):
    atlas = pil_rgba(path)
    cov = atlas[..., 0]
    out = np.zeros_like(atlas)
    shade = 0.7 + 0.6 * atlas[..., 1:2]
    out[..., :3] = np.array(HAIR_SRGB)[None, None, :] * shade
    out[..., 3] = np.clip(cov * 1.8, 0, 1)
    if out.shape[0] > size:
        k = out.shape[0] // size
        out = out.reshape(size, k, size, k, 4).mean(axis=(1, 3))
    return out


def ponytail_chain(arm, hair, n_bones=4):
    """Bones along the hanging tail + smooth weights; the rest of the hair stays on the head."""
    head_pos = arm.matrix_world @ arm.data.bones["head"].head_local
    co = np.array([tuple(hair.matrix_world @ v.co) for v in hair.data.vertices])
    # the tail: behind the head joint and below the occiput (character faces -Y, so behind = +Y)
    tail = (co[:, 1] > head_pos.y + 0.03) & (co[:, 2] < head_pos.z + 0.02)
    if tail.sum() < 50:
        log("no ponytail found; hair stays rigid on the head")
        return
    pts = co[tail]
    top = pts[pts[:, 2] > np.percentile(pts[:, 2], 92)].mean(0)
    bottom = pts[pts[:, 2] < np.percentile(pts[:, 2], 5)].mean(0)
    joints = [top + (bottom - top) * t for t in np.linspace(0, 1, n_bones + 1)]
    set_active(arm)
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = arm.data.edit_bones
    inv = arm.matrix_world.inverted()
    parent = ebs["head"]
    names = []
    for k in range(n_bones):
        b = ebs.new(f"hair_ponytail_{k + 1:02d}")
        b.head = inv @ Vector(joints[k])
        b.tail = inv @ Vector(joints[k + 1])
        b.parent = parent
        b.use_connect = k > 0
        b.use_deform = True
        parent = b
        names.append(b.name)
    bpy.ops.object.mode_set(mode="OBJECT")
    axis = (bottom - top) / np.linalg.norm(bottom - top)
    length = np.linalg.norm(bottom - top)
    t = np.clip(((co - top) @ axis) / length, 0, 1)
    head_g = hair.vertex_groups["head"]
    groups = [hair.vertex_groups.new(name=n) for n in names]
    for vi in range(len(co)):
        if not tail[vi]:
            continue
        s = t[vi] * n_bones                              # 0..n along the chain
        k = min(int(s), n_bones - 1)
        f = s - k
        root_blend = np.clip(t[vi] / 0.08, 0, 1)         # tie point stays on the head
        w_cur = (1 - f) * root_blend
        w_next = f * root_blend
        head_g.add([vi], 1 - root_blend, "REPLACE")
        groups[k].add([vi], w_cur + (w_next if k == n_bones - 1 else 0), "REPLACE")
        if k + 1 < n_bones and w_next > 0:
            groups[k + 1].add([vi], w_next, "REPLACE")
    log(f"ponytail: {int(tail.sum())} of {len(co)} hair vertices on a {n_bones}-bone chain, length {length:.3f} m")


# ------------------------------------------------------------------------------------------------ main
def main():
    a = args()
    src, tex = a.src, os.path.join(a.src, "Textures")
    name = next(f[:-len("_Body.fbx")] for f in os.listdir(src) if f.endswith("_Body.fbx"))
    work = tempfile.mkdtemp(prefix="mh2_")
    frames = [(int(k), v[0].lower() + v[1:]) for k, v in json.load(open(os.path.join(src, "arkit_frames.json"))).items()
              if 1 <= int(k) <= 51]
    bpy.ops.wm.read_factory_settings(use_empty=True)

    body_objs = import_fbx(os.path.join(src, f"{name}_Body.fbx"))
    arm, body = of_type(body_objs, "ARMATURE")[0], of_type(body_objs, "MESH")[0]
    arm.name, arm.data.name, body.name = "MetaHuman", "MetaHuman", "Body"
    face_objs = import_fbx(os.path.join(src, f"{name}_Face_LOD0.fbx"))
    face_arm, face = of_type(face_objs, "ARMATURE")[0], of_type(face_objs, "MESH")[0]
    face.name = "Head"
    bake_arkit_shapes(face, face_arm, os.path.join(src, f"{name}_Face_ARKit.fbx"), frames)
    copy_facial_bones(arm, face_arm)
    reskin(face, arm)

    outfit_objs = import_fbx(os.path.join(src, f"{name}_Outfit.fbx"))
    outfit = of_type(outfit_objs, "MESH")[0]
    outfit.name = "Outfit"
    reskin(outfit, arm)

    hair_dir = os.path.join(src, "Hair")
    ponytail = brows = None
    for f in sorted(os.listdir(hair_dir)):
        if not f.lower().endswith(".fbx"):
            continue
        m = of_type(import_fbx(os.path.join(hair_dir, f)), "MESH")[0]
        if "brow" in f.lower():
            brows = m
            m.name = "BrowCards"
        else:
            ponytail = m
            m.name = "Hair"
            g = m.vertex_groups.new(name="head")
            g.add(range(len(m.data.vertices)), 1.0, "REPLACE")
            reskin(m, arm)
    for o in face_objs + outfit_objs + body_objs:
        if o.name in bpy.data.objects and o not in (arm, body, face, outfit) and o.type in ("ARMATURE", "EMPTY") and o is not arm.parent:
            bpy.data.objects.remove(o, do_unlink=True)

    # imported FBX takes carry animation (body take, ARKit take): export the rest pose with all shapes at 0
    for o in bpy.data.objects:
        if o.animation_data:
            o.animation_data_clear()
        if o.type == "ARMATURE":
            for pb in o.pose.bones:
                pb.matrix_basis.identity()
    for k in bpy.data.shape_keys:
        if k.animation_data:
            k.animation_data_clear()
        for kb in k.key_blocks[1:]:
            kb.value = 0.0

    # ------------------------------------------------------------ units / facing (metres; character faces -Y)
    meshes = [body, face, outfit] + ([ponytail] if ponytail else [])
    root_empty = arm.parent
    set_active(arm)
    mw = arm.matrix_world.copy()
    arm.parent = None
    arm.matrix_world = mw
    if root_empty:
        bpy.data.objects.remove(root_empty, do_unlink=True)
    for o in bpy.context.view_layer.objects:
        o.select_set(o is arm or o in meshes or o is brows)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    if brows:
        set_active(brows)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    head = arm.matrix_world @ arm.data.bones["head"].head_local
    co = np.array([tuple(face.matrix_world @ v.co) for v in face.data.vertices])
    if abs(co[:, 1].max() - head.y) > abs(co[:, 1].min() - head.y):   # nose must be at -Y (= +Z in glTF)
        log("character faces +Y; rotating 180 degrees")
        for o in bpy.context.view_layer.objects:
            o.select_set(o is arm or o in meshes or o is brows)
        bpy.context.view_layer.objects.active = arm
        arm.rotation_euler[2] += math.pi
        if brows:
            brows.rotation_euler[2] += math.pi
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # ------------------------------------------------------------ materials
    S, FS = a.texture_size, a.face_texture_size
    T = lambda *p: os.path.join(tex, *p)  # noqa: E731
    roles = classify_face_slots_v2(face)
    log("face slot roles (export labels are shifted; roles from geometry):", {i: (face.material_slots[i].material.name, r) for i, r in roles.items()})
    skin_slot = next(i for i, r in roles.items() if r == "skin")
    brow_mask = bake_brow_mask(face, brows, FS, skin_slot, os.path.join(os.path.dirname(os.path.abspath(a.out)), "brow_mask_check.png")) if brows else np.zeros((FS, FS), np.float32)
    head_bc = image_from_array(head_with_brows(T("Head", f"T_{name}_Head_Basecolor.png"), brow_mask, FS), "MH_HeadBC", os.path.join(work, "head_bc.png"))
    skin_face = pbr("MH_Face", head_bc, load_image(T("Head", f"T_{name}_Head_Normal.png"), FS, non_color=True), roughness=0.5)
    skin_body = pbr("MH_Body", load_image(T("Body", f"T_{name}_Body_Basecolor.png"), FS), load_image(T("Body", f"T_{name}_Body_Normal.png"), S, non_color=True), roughness=0.55)
    teeth = pbr("MH_Teeth", load_image(T("Teeth", f"T_{name}_Teeth_Color.png"), 512), load_image(T("Teeth", f"T_{name}_Teeth_Normal.png"), 512, non_color=True), roughness=0.3)
    eye = pbr("MH_Eye", load_image(T("Eyes", f"T_{name}_Eyes_Color.png"), 1024), load_image(T("Eyes", f"T_{name}_Eyes_Normal.png"), 512, non_color=True), roughness=0.12)
    lash_px = pil_rgba(T("Hair", f"T_{name}_Eyelashes_Color.png"))
    lash_px[..., 3] = np.clip(lash_px[..., 0] * 1.5, 0, 1)
    lash_px[..., :3] = 0.02
    k = lash_px.shape[0] // 1024
    lash_px = lash_px.reshape(1024, k, 1024, k, 4).mean(axis=(1, 3)) if k > 1 else lash_px
    lashes = pbr("MH_Eyelashes", image_from_array(lash_px, "MH_Lashes", os.path.join(work, "lashes.png")), roughness=0.7, cutoff=0.3,
                 double_sided=True, alpha_from=True)
    tongue = pbr("MH_Tongue", colour=srgb_to_lin((0.55, 0.25, 0.26)), roughness=0.35)
    by_role = {"skin": skin_face, "teeth": teeth, "tongue": tongue, "eye_l": eye, "eye_r": eye, "lashes": lashes}
    for i, role in roles.items():
        if role in by_role:
            face.data.materials[i] = by_role[role]
    removed = delete_slot_faces(face, [i for i, r in roles.items() if r == "drop"])
    log(f"removed {removed} faces of transparent Unreal layers (eye shell, lacrimal edge, saliva)")
    body.data.materials.clear()
    body.data.materials.append(skin_body)
    uv = body.data.uv_layers[0]
    uvs = np.empty(len(uv.data) * 2, np.float32)
    uv.data.foreach_get("uv", uvs)
    uvs[0::2] -= np.floor(uvs[0::2].min())                # UDIM tile 1002 -> 0..1
    uv.data.foreach_set("uv", uvs)

    def garment(kind, colour):
        ao = pil_rgba(T("Outfit", f"T_{name}_Outfit_DG_bodyShapeB_{kind}_AO.png"))
        k = ao.shape[0] // S
        ao = ao.reshape(S, k, S, k, 4).mean(axis=(1, 3)) if k > 1 else ao
        lin = np.array(colour)
        srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
        px = np.empty_like(ao)
        px[..., :3] = srgb[None, None, :] * (0.45 + 0.55 * ao[..., :1])
        px[..., 3] = 1
        return pbr(f"MH_{kind}", image_from_array(px, f"MH_{kind}BC", os.path.join(work, f"{kind}.png")),
                   load_image(T("Outfit", f"T_{name}_Outfit_DG_bodyShapeB_{kind}_Normal.png"), S, non_color=True), roughness=0.85)

    shirt, shorts = garment("Shirt", SHIRT), garment("Short", SHORTS)
    for i, slot in enumerate(outfit.material_slots):
        outfit.data.materials[i] = shirt if "shirt" in (slot.material.name if slot.material else "").lower() else shorts
    if ponytail:
        uvh = ponytail.data.uv_layers.get("UVmap_0")
        if uvh:
            ponytail.data.uv_layers.active = uvh
            for other in [u.name for u in ponytail.data.uv_layers if u.name != "UVmap_0"]:
                ponytail.data.uv_layers.remove(ponytail.data.uv_layers[other])
        atlas = T("Hair", f"T_{name}_Hair_Hair_S_LowPonytail_CardsAtlas_Attribute.png")
        hair_mat = pbr("MH_Hair", image_from_array(hair_atlas(atlas, 1024), "MH_HairCards", os.path.join(work, "hair.png")), roughness=0.5,
                       cutoff=0.3, double_sided=True, alpha_from=True)
        ponytail.data.materials.clear()
        ponytail.data.materials.append(hair_mat)
    if brows:
        bpy.data.objects.remove(brows, do_unlink=True)
    for m in (body, face, outfit):
        keep = m.data.uv_layers[0].name
        for other in [u.name for u in m.data.uv_layers if u.name != keep]:
            m.data.uv_layers.remove(m.data.uv_layers[other])

    # ------------------------------------------------------------ skeleton
    if ponytail:
        ponytail_chain(arm, ponytail)
    add_finger_tips(arm, body)
    prune(arm, meshes)
    head = arm.matrix_world @ arm.data.bones["head"].head_local
    co = np.array([tuple(face.matrix_world @ v.co) for v in face.data.vertices])
    log(f"height {co[:, 2].max():.3f} m, head joint {head.z:.3f} m, nose in front: {co[:, 1].min() < head.y}")
    for m in meshes:
        log(f"{m.name}: {len(m.data.vertices)} verts, materials {[s.material.name for s in m.material_slots]}, shape keys {len(m.data.shape_keys.key_blocks) - 1 if m.data.shape_keys else 0}")
    if a.blend:
        bpy.ops.wm.save_as_mainfile(filepath=a.blend)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    kwargs = dict(filepath=a.out, export_format="GLB", use_selection=False, export_apply=False, export_animations=False,
                  export_skins=True, export_morph=True, export_morph_normal=False, export_yup=True, export_cameras=False,
                  export_lights=False, export_image_format="WEBP")
    for extra in ({"export_image_quality": 88, "export_try_sparse_sk": True}, {"export_image_quality": 88}, {}):
        try:
            bpy.ops.export_scene.gltf(**kwargs, **extra)
            break
        except TypeError:
            continue
    log(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
