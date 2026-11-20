"""Convert a MetaHuman FBX export (body + face + outfit + hair cards) into one web GLB.

Run with Blender (tested with Blender 5.1) in background mode:

    blender -b --factory-startup --python scripts/metahuman/convert_metahuman.py -- \
        --src  /path/to/MetaHuman_Export  --out .runtime/avatar-candidates/MetaHuman.glb

Expected export layout (MetaHuman Creator "FBX export" of the baked LOD):
    <Name>_Body.fbx, <Name>_Face.fbx, <Name>_Outfit.fbx, Hair/*.fbx
    Textures/Body_Baked/*.png, Textures/Face_Baked/*.png, Textures/Grooms_Textures/*.png

What the script does (all on one skeleton, the body's UE5 skeleton):
* the face's FACIAL_* joints are copied into the body armature under their original parents;
  the face mesh, outfit and hair are re-skinned to that single armature (hair/brow cards
  are rigid on `head`);
* finger tip joints `<finger>_04_<side>` are added at the distal end of each finger mesh so
  the web retargeter can measure every phalanx (MetaHuman has no tip joints);
* joints with no skin weight and no needed descendant are pruned (the web player only
  needs the deforming joints);
* PBR materials are rebuilt from the baked textures (base colour, normal, SRMF where
  G = roughness and B = metallic, the glTF channel layout);
* the character is scaled to metres, faces +Z in glTF (towards the Three.js camera) and is
  exported with WebP textures.

The MetaHuman asset itself is NOT part of this repository: see docs/avatar/METAHUMAN.md.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Matrix, Vector

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SIDES = ("l", "r")
# Joints the web player addresses by name (retarget profile + face driver); never pruned.
KEEP = {
    "pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05", "neck_01", "neck_02", "head",
    *[f"{b}_{s}" for s in SIDES for b in ("clavicle", "upperarm", "lowerarm", "hand")],
    *[f"{f}_{i:02d}_{s}" for s in SIDES for f in FINGERS for i in (1, 2, 3, 4)],
    *[f"{f}_metacarpal_{s}" for s in SIDES for f in ("index", "middle", "ring", "pinky")],
}
FACE_DRIVER_PREFIXES = ("FACIAL_C_Jaw", "FACIAL_L_Eyelid", "FACIAL_R_Eyelid", "FACIAL_L_Brow", "FACIAL_R_Brow",
                        "FACIAL_C_Brow", "FACIAL_L_Eye", "FACIAL_R_Eye", "FACIAL_C_LowerLip", "FACIAL_C_UpperLip",
                        "FACIAL_L_LipCorner", "FACIAL_R_LipCorner", "FACIAL_L_ForeheadIn", "FACIAL_R_ForeheadIn",
                        "FACIAL_L_ForeheadMid", "FACIAL_R_ForeheadMid", "FACIAL_L_ForeheadOut", "FACIAL_R_ForeheadOut")
# Outfit textures are not part of the MetaHuman baked export: plain, dark, matte clothing
# (the usual interpreter convention) keeps the hands readable against the torso.
OUTFIT_COLOURS = {"shirt": (0.016, 0.018, 0.026), "short": (0.022, 0.022, 0.025)}
HAIR_COLOUR = (0.12, 0.075, 0.045)  # sRGB; sampled from the painted hair cap of the head texture


def log(*a):
    print("[metahuman]", *a, flush=True)


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default=None, help="file prefix, e.g. FrenchWoman (auto-detected)")
    p.add_argument("--texture-size", type=int, default=1024)
    p.add_argument("--hair-texture-size", type=int, default=1024)
    p.add_argument("--keep-all-joints", action="store_true")
    return p.parse_args(argv)


def import_fbx(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=False, ignore_leaf_bones=False)
    return [o for o in bpy.data.objects if o not in before]


def of_type(objs, kind):
    return [o for o in objs if o.type == kind]


def set_active(obj):
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def parent_keep(child, parent):
    mw = child.matrix_world.copy()
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()
    child.matrix_world = mw


def reskin(mesh, armature):
    parent_keep(mesh, armature)
    mods = [m for m in mesh.modifiers if m.type == "ARMATURE"]
    if not mods:
        mods = [mesh.modifiers.new("Armature", "ARMATURE")]
    for m in mods[1:]:
        mesh.modifiers.remove(m)
    mods[0].object = armature


def copy_facial_bones(body_arm, face_arm):
    """Copy every face-armature bone missing from the body armature, keeping parents."""
    set_active(face_arm)
    bpy.ops.object.mode_set(mode="EDIT")
    to_body = body_arm.matrix_world.inverted() @ face_arm.matrix_world
    src = {}
    for eb in face_arm.data.edit_bones:
        src[eb.name] = (to_body @ eb.matrix, eb.length, eb.parent.name if eb.parent else None, eb.use_deform)
    shared = {}
    for name in ("head", "neck_02", "spine_05", "clavicle_l"):
        if name in face_arm.data.edit_bones:
            shared[name] = (to_body @ face_arm.data.edit_bones[name].head)
    bpy.ops.object.mode_set(mode="OBJECT")
    set_active(body_arm)
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = body_arm.data.edit_bones
    for name, pos in shared.items():
        if name in ebs:
            log(f"rest mismatch face vs body at {name}: {(ebs[name].head - pos).length:.4f} (armature units)")
    added = []
    pending = [n for n in src if n not in ebs]
    # parents first
    while pending:
        progress = False
        for name in list(pending):
            matrix, length, parent, deform = src[name]
            if parent and parent not in ebs:
                continue
            eb = ebs.new(name)
            eb.length = max(length, 1e-3)
            eb.matrix = matrix
            eb.use_deform = deform
            if parent:
                eb.parent = ebs[parent]
            pending.remove(name)
            added.append(name)
            progress = True
        if not progress:
            raise RuntimeError(f"orphan face bones: {pending[:5]}")
    bpy.ops.object.mode_set(mode="OBJECT")
    log(f"copied {len(added)} face joints into the body skeleton")


def weighted_groups(mesh):
    me = mesh.data
    names = {g.index: g.name for g in mesh.vertex_groups}
    used = set()
    for v in me.vertices:
        for g in v.groups:
            if g.weight > 1e-4:
                used.add(names[g.group])
    return used


def add_finger_tips(arm, body_mesh):
    """Tip joint at the distal end of each finger, measured on the skinned body mesh."""
    groups = {g.name: g.index for g in body_mesh.vertex_groups}
    to_arm = arm.matrix_world.inverted() @ body_mesh.matrix_world
    verts = body_mesh.data.vertices
    set_active(arm)
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = arm.data.edit_bones
    for side in SIDES:
        for finger in FINGERS:
            d3, d2 = f"{finger}_03_{side}", f"{finger}_02_{side}"
            if d3 not in ebs or d2 not in ebs or f"{finger}_04_{side}" in ebs:
                continue
            base = ebs[d3].head.copy()
            direction = (ebs[d3].head - ebs[d2].head).normalized()
            gi = groups.get(d3)
            best = None
            if gi is not None:
                pts = [to_arm @ v.co for v in verts if any(g.group == gi and g.weight > 0.5 for g in v.groups)]
                if pts:
                    best = max((p - base).dot(direction) for p in pts)
            length = best if best and best > 1e-3 else (ebs[d3].head - ebs[d2].head).length * 0.8
            tip = ebs.new(f"{finger}_04_{side}")
            tip.head = base + direction * length
            tip.tail = tip.head + direction * max(length * 0.25, 0.2)
            tip.roll = ebs[d3].roll
            tip.parent = ebs[d3]
            tip.use_deform = False
    bpy.ops.object.mode_set(mode="OBJECT")


def prune(arm, meshes, keep_all=False):
    used = set().union(*(weighted_groups(m) for m in meshes))
    keep = set(KEEP) | used | {b.name for b in arm.data.bones if b.name.startswith(FACE_DRIVER_PREFIXES)}
    bones = arm.data.bones
    closure = set()
    for name in keep:
        b = bones.get(name)
        while b is not None and b.name not in closure:
            closure.add(b.name)
            b = b.parent
    if keep_all:
        closure = {b.name for b in bones}
    drop = [b.name for b in bones if b.name not in closure]
    set_active(arm)
    bpy.ops.object.mode_set(mode="EDIT")
    for name in drop:
        arm.data.edit_bones.remove(arm.data.edit_bones[name])
    bpy.ops.object.mode_set(mode="OBJECT")
    for m in meshes:
        for g in list(m.vertex_groups):
            if g.name not in closure:
                m.vertex_groups.remove(g)
    log(f"joints kept {len(closure)} / dropped {len(drop)}")
    return closure


# ---------------------------------------------------------------- textures / materials
def load_image(path, size=None, name=None, non_color=False):
    img = bpy.data.images.load(path, check_existing=False)
    if name:
        img.name = name
    if size and max(img.size) > size:
        img.scale(size, size)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def pil_rgba(path):
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    a = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(a)
    bpy.data.images.remove(img)
    return a.reshape(h, w, 4)


def image_from_array(arr, name, path):
    h, w = arr.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.pixels.foreach_set(np.ascontiguousarray(arr, dtype=np.float32).ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    return img


def pbr(name, base=None, normal=None, srm=None, colour=None, roughness=0.6, alpha=None, cutoff=None,
        double_sided=False, alpha_from=None):
    mat = bpy.data.materials.new(name)
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    if colour:
        bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    if base:
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = base
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
        if alpha_from:
            nt.links.new(t.outputs["Alpha"], bsdf.inputs["Alpha"])
    if normal:
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = normal
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(t.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if srm:
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = srm
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(t.outputs["Color"], sep.inputs["Color"])
        nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
        nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    if cutoff is not None:
        # glTF exporter reads alpha clip from a Math "round"/greater-than node or the
        # material's blend settings depending on version; set both for robustness.
        try:
            mat.blend_method = "CLIP"
            mat.alpha_threshold = cutoff
        except AttributeError:
            pass
        mat["gltf_alpha_cutoff"] = cutoff
    mat.use_backface_culling = not double_sided
    return mat


def eye_texture(tex, side, size):
    sclera = pil_rgba(os.path.join(tex, f"T_EyeSclera{side}_BC.png"))
    iris = pil_rgba(os.path.join(tex, f"T_EyeIris{side}_BC.png"))
    n = sclera.shape[0]
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1) - 0.5
    r = np.sqrt(xx ** 2 + yy ** 2)
    # Iris limbus sits at ~0.43 of the iris texture; place it at radius 0.17 of the eye UV.
    scale = 0.43 / 0.17
    ui = np.clip(((xx * scale) + 0.5) * (iris.shape[1] - 1), 0, iris.shape[1] - 1).astype(int)
    vi = np.clip(((yy * scale) + 0.5) * (iris.shape[0] - 1), 0, iris.shape[0] - 1).astype(int)
    iris_s = iris[vi, ui]
    mask = np.clip((0.175 - r) / 0.012, 0, 1)[..., None]
    out = sclera * (1 - mask) + iris_s * mask
    out[..., 3] = 1
    return out


def hair_texture(path, size):
    atlas = pil_rgba(path)
    cov = atlas[..., 0]
    out = np.zeros_like(atlas)
    shade = 0.75 + 0.5 * atlas[..., 1:2]  # tangent channel adds strand variation
    out[..., :3] = np.array(HAIR_COLOUR)[None, None, :] * shade
    out[..., 3] = np.clip(cov * 1.6, 0, 1)
    if out.shape[0] > size:
        k = out.shape[0] // size
        out = out.reshape(size, k, size, k, 4).mean(axis=(1, 3))
    return out


def assign(mesh, mapping):
    for i, slot in enumerate(mesh.material_slots):
        name = slot.material.name if slot.material else ""
        for key, mat in mapping.items():
            if key.lower() in name.lower():
                mesh.data.materials[i] = mat
                break


def classify_face_slots(face):
    """Assign roles to the face's material slots from geometry.

    The FBX material labels of the baked MetaHuman face are shifted (the skin slot is
    labelled "Teeth"), so labels are not trusted. Roles: skin = most faces; eyes = the two
    compact parts entirely on one side (+X is the character's left in Blender); teeth = the
    lowest remaining part; lashes = the remaining part reaching furthest forward (-Y); the
    rest (eye shell, lacrimal fluid: transparent UE layers) is dropped."""
    parts = {}
    for p in face.data.polygons:
        parts.setdefault(p.material_index, []).append(face.matrix_world @ p.center)
    stats = {i: (len(c), np.array([tuple(v) for v in c])) for i, c in parts.items()}
    roles = {}
    skin = max(stats, key=lambda i: stats[i][0])
    roles[skin] = "skin"
    rest = [i for i in stats if i != skin]
    eye_z = None
    for i in rest:
        xs = stats[i][1][:, 0]
        if xs.min() > 0:
            roles[i] = "eye_l"
        elif xs.max() < 0:
            roles[i] = "eye_r"
        if i in roles:
            eye_z = stats[i][1][:, 2].mean()
    rest = [i for i in rest if i not in roles]
    if rest:
        teeth = min(rest, key=lambda i: stats[i][1][:, 2].mean())
        if eye_z is None or stats[teeth][1][:, 2].mean() < eye_z - 0.03:
            roles[teeth] = "teeth"
            rest.remove(teeth)
    if rest:
        lashes = min(rest, key=lambda i: stats[i][1][:, 1].min())
        roles[lashes] = "lashes"
        rest.remove(lashes)
    for i in rest:
        roles[i] = "drop"
    return roles


def face_with_brows(face_dir):
    """LOD3 skin colour plus the brows painted only in the LOD5-7 texture (brow cards are dropped)."""
    hi = pil_rgba(os.path.join(face_dir, "T_Head_LOD3_BC_VT.png"))
    lo = pil_rgba(os.path.join(face_dir, "T_Head_LOD5to7_BC_VT.png"))
    k = hi.shape[0] // lo.shape[0]
    lo_up = np.repeat(np.repeat(lo, k, axis=0), k, axis=1)
    lum = lambda x: x[..., :3] @ np.array([0.2126, 0.7152, 0.0722])  # noqa: E731
    mask = np.clip((lum(hi) - lum(lo_up) - 0.04) / 0.06, 0, 1)[..., None]
    out = hi * (1 - mask) + lo_up * mask
    out[..., 3] = 1
    return out


def delete_slot_faces(mesh, idx, where=None):
    me = mesh.data
    idx = set(idx)
    if not idx:
        return 0
    import bmesh
    bm = bmesh.new(); bm.from_mesh(me)
    faces = [f for f in bm.faces if f.material_index in idx
             and (where is None or where(mesh.matrix_world @ f.calc_center_median()))]
    bmesh.ops.delete(bm, geom=faces, context="FACES")
    bm.to_mesh(me); bm.free()
    return len(faces)


def main():
    a = args()
    src = a.src
    name = a.name or next(f[:-len("_Body.fbx")] for f in os.listdir(src) if f.endswith("_Body.fbx"))
    tex = os.path.join(src, "Textures")
    work = tempfile.mkdtemp(prefix="mh_")
    bpy.ops.wm.read_factory_settings(use_empty=True)

    body_objs = import_fbx(os.path.join(src, f"{name}_Body.fbx"))
    arm = of_type(body_objs, "ARMATURE")[0]
    body = of_type(body_objs, "MESH")[0]
    arm.name, arm.data.name, body.name = "MetaHuman", "MetaHuman", "Body"

    face_objs = import_fbx(os.path.join(src, f"{name}_Face.fbx"))
    face_arm = of_type(face_objs, "ARMATURE")[0]
    face = of_type(face_objs, "MESH")[0]
    face.name = "Head"
    copy_facial_bones(arm, face_arm)
    reskin(face, arm)

    outfit_objs = import_fbx(os.path.join(src, f"{name}_Outfit.fbx"))
    outfit = of_type(outfit_objs, "MESH")[0]
    outfit.name = "Outfit"
    missing = [g.name for g in outfit.vertex_groups if g.name not in arm.data.bones]
    if missing:
        log("outfit groups without body joint:", missing[:10])
    reskin(outfit, arm)

    hair = []
    hair_dir = os.path.join(src, "Hair")
    for f in sorted(os.listdir(hair_dir)) if os.path.isdir(hair_dir) else []:
        if f.lower().endswith(".fbx"):
            for m in of_type(import_fbx(os.path.join(hair_dir, f)), "MESH"):
                m.name = "Eyebrows" if "brow" in f.lower() else f"Hair{len([h for h in hair if h.name.startswith('Hair')])}"
                g = m.vertex_groups.new(name="head")
                g.add(range(len(m.data.vertices)), 1.0, "REPLACE")
                reskin(m, arm)
                hair.append(m)

    # drop helper armatures/empties from the other imports
    for o in face_objs + outfit_objs + body_objs:
        if o.name in bpy.data.objects and o not in (arm, body, face, outfit) and o.type in ("ARMATURE", "EMPTY"):
            if o.type == "EMPTY" and o is arm.parent:
                continue
            bpy.data.objects.remove(o, do_unlink=True)

    # ------------------------------------------------------------- materials
    S = a.texture_size
    body_dir, face_dir, groom_dir = (os.path.join(tex, d) for d in ("Body_Baked", "Face_Baked", "Grooms_Textures"))
    L = lambda d, f, nc=False, size=S: load_image(os.path.join(d, f), size, non_color=nc)  # noqa: E731
    skin_body = pbr("MH_Body", L(body_dir, "T_Body_BC_VT.png"), L(body_dir, "T_Body_N_VT.png", True), L(body_dir, "T_Body_SRMF_VT.png", True))
    face_bc = image_from_array(face_with_brows(face_dir), "MH_FaceBC", os.path.join(work, "face_bc.png"))
    skin_face = pbr("MH_Face", face_bc, L(face_dir, "T_Head_LOD3_N_VT.png", True), L(face_dir, "T_Head_LOD3_SRMF_VT.png", True))
    teeth = pbr("MH_Teeth", L(face_dir, "T_Teeth_BC.png", size=512), L(face_dir, "T_Teeth_N.png", True, 512), L(face_dir, "T_Teeth_SRM.png", True, 512))
    eyes = {}
    for side in ("L", "R"):
        img = image_from_array(eye_texture(face_dir, side, 512), f"MH_Eye{side}", os.path.join(work, f"eye{side}.png"))
        eyes[side] = pbr(f"MH_Eye{side}", img, roughness=0.15)
    lashes = pbr("MH_Eyelashes", colour=(0.01, 0.008, 0.007), roughness=0.7, double_sided=True)
    assign(body, {"Body": skin_body})
    roles = classify_face_slots(face)
    log("face slot roles (FBX slot labels are not reliable):", roles)
    by_role = {"skin": skin_face, "teeth": teeth, "eye_l": eyes["L"], "eye_r": eyes["R"], "lashes": lashes}
    for i, role in roles.items():
        if role in by_role:
            face.data.materials[i] = by_role[role]
    removed = delete_slot_faces(face, [i for i, r in roles.items() if r == "drop"])
    lashes_slot = next((i for i, r in roles.items() if r == "lashes"), None)
    eye_slots = [i for i, r in roles.items() if r in ("eye_l", "eye_r")]
    if lashes_slot is not None and eye_slots:
        # untextured lower-lash cards read as grey slabs under the eyes on the web: keep upper lashes only
        eye_z = np.mean([(face.matrix_world @ p.center).z for p in face.data.polygons if p.material_index in eye_slots])
        removed += delete_slot_faces(face, [lashes_slot], lambda c: c.z < eye_z)
    log(f"removed {removed} transparent eye-shell/lacrimal faces (no web equivalent)")
    shirt = pbr("MH_Shirt", colour=OUTFIT_COLOURS["shirt"], roughness=0.85)
    short = pbr("MH_Shorts", colour=OUTFIT_COLOURS["short"], roughness=0.85)
    assign(outfit, {"Shirt": shirt, "Short": short})
    groom = [f for f in os.listdir(groom_dir) if f.endswith(".png")] if os.path.isdir(groom_dir) else []
    atlas = next((f for f in groom if "Attribute" in f), groom[0] if groom else None)
    hair_mat = None
    if atlas:
        img = image_from_array(hair_texture(os.path.join(groom_dir, atlas), a.hair_texture_size), "MH_HairCards", os.path.join(work, "hair.png"))
        hair_mat = pbr("MH_Hair", img, roughness=0.55, cutoff=0.35, double_sided=True, alpha_from=True)
    kept = []
    for m in hair:
        if m.name == "Eyebrows":
            # the brow cards carry no atlas UV: the brows are already painted in the skin
            # texture, so the cards are dropped rather than rendered as solid strips.
            bpy.data.objects.remove(m, do_unlink=True)
            continue
        uv = m.data.uv_layers.get("UVmap_0")
        if uv:
            m.data.uv_layers.active = uv
            for other in [u.name for u in m.data.uv_layers if u.name != "UVmap_0"]:
                m.data.uv_layers.remove(m.data.uv_layers[other])
        m.data.materials.clear()
        m.data.materials.append(hair_mat or pbr("MH_HairFlat", colour=HAIR_COLOUR, roughness=0.6, double_sided=True))
        kept.append(m)
    hair = kept
    for m in (body, face, outfit):
        keep_uv = m.data.uv_layers[0].name
        for other in [u.name for u in m.data.uv_layers if u.name != keep_uv]:
            m.data.uv_layers.remove(m.data.uv_layers[other])

    # ------------------------------------------------------------- skeleton
    add_finger_tips(arm, body)
    meshes = [body, face, outfit, *hair]
    prune(arm, meshes, a.keep_all_joints)

    # ------------------------------------------------------------- units / facing
    root_empty = arm.parent
    set_active(arm)
    mw = arm.matrix_world.copy()
    arm.parent = None
    arm.matrix_world = mw
    if root_empty:
        bpy.data.objects.remove(root_empty, do_unlink=True)
    for o in bpy.context.view_layer.objects:
        o.select_set(o is arm or o in meshes)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    # face +Z in glTF == -Y in Blender: the nose must be in front (-Y) of the head joint
    head = arm.matrix_world @ arm.data.bones["head"].head_local
    co = np.array([face.matrix_world @ v.co for v in face.data.vertices])
    front = co[np.argmin(co[:, 1])] if (co[:, 1].min() - head.y) ** 2 > (co[:, 1].max() - head.y) ** 2 else co[np.argmax(co[:, 1])]
    if front[1] > head.y:
        log("character faces +Y; rotating 180 degrees")
        arm.rotation_euler[2] += math.pi
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    zs = co[:, 2]
    log(f"height {max(zs):.3f} m, head joint at {head.z:.3f} m")
    for m in meshes:
        log(f"{m.name}: {len(m.data.vertices)} verts, {len(m.data.polygons)} faces, materials {[s.material.name for s in m.material_slots]}")
    facial = sorted(b.name for b in arm.data.bones if b.name.startswith("FACIAL"))
    log(f"FACIAL joints kept: {len(facial)}")
    log("driver joints:", [b for b in facial if b.startswith(FACE_DRIVER_PREFIXES)][:60])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    kwargs = dict(filepath=a.out, export_format="GLB", use_selection=False, export_apply=False,
                  export_animations=False, export_skins=True, export_morph=False, export_yup=True,
                  export_cameras=False, export_lights=False, export_image_format="WEBP")
    try:
        bpy.ops.export_scene.gltf(**kwargs, export_image_quality=85)
    except TypeError:
        bpy.ops.export_scene.gltf(**kwargs)
    log(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
