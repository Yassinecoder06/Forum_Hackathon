"""Convert a MetaHuman FBX export into a Studio-compatible GLB avatar.

The Studio retargeter (frontend/src/utils/retargetHumanoid.ts) drives a
Ready-Player-Me-style skeleton (Hips, Spine1/2, Neck, Head, LeftArm,
LeftForeArm, LeftHand, LeftHandIndex1..4, ...) and ARKit-named blendshapes
(jawOpen, eyeBlinkLeft/Right, browInnerUp, browOuterUpLeft/Right,
browDownLeft/Right). This script adapts the asset, not the Studio code:

1. merges body, face, outfit, hair and eyebrow FBX files onto one armature;
2. removes twist/corrective/helper bones (their skin weights go to the nearest
   kept ancestor) so every chain bone has a single bone child, and drops
   unweighted facial joints;
3. adds fingertip and head-end bones so the last segments have a direction;
4. renames core bones to the Studio names;
5. bakes facial blendshapes: jaw and eyelids from the MetaHuman facial joints,
   brows as a smooth displacement field (no brow joints are weighted at this LOD);
6. builds glTF materials from the baked textures (iris composited into the
   sclera, hair opacity from the card atlas coverage channel);
7. exports one GLB.

Run with Blender's Python module (isolated tool environment):

    uv venv .runtime/tools/bpy-venv --python 3.11 && uv pip install --python .runtime/tools/bpy-venv/bin/python bpy==4.2.0
    .runtime/tools/bpy-venv/bin/python scripts/build-metahuman-avatar.py \
        .runtime/avatar-sources/FrenchWoman .runtime/avatar-candidates/FrenchWoman.glb
"""
import math
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np

SRC, OUT = Path(sys.argv[-2]).resolve(), Path(sys.argv[-1]).resolve()
TEX = SRC / "Textures"
HAIR_COLOUR = (0.055, 0.035, 0.025)          # dark brown; Unreal computes hair colour procedurally
OUTFIT_COLOURS = {"Shirt": (0.035, 0.035, 0.04), "Short": (0.09, 0.09, 0.1)}   # dark top keeps hands legible

FINGERS = {"thumb": "Thumb", "index": "Index", "middle": "Middle", "ring": "Ring", "pinky": "Pinky"}
RENAME = {"pelvis": "Hips", "spine_01": "Spine", "spine_03": "Spine1", "spine_05": "Spine2",
          "neck_01": "Neck", "head": "Head"}
for s, side in (("l", "Left"), ("r", "Right")):
    RENAME.update({f"clavicle_{s}": f"{side}Shoulder", f"upperarm_{s}": f"{side}Arm", f"lowerarm_{s}": f"{side}ForeArm",
                   f"hand_{s}": f"{side}Hand", f"thigh_{s}": f"{side}UpLeg", f"calf_{s}": f"{side}Leg",
                   f"foot_{s}": f"{side}Foot", f"ball_{s}": f"{side}ToeBase"})
    for f, F in FINGERS.items():
        for i in (1, 2, 3):
            RENAME[f"{f}_0{i}_{s}"] = f"{side}Hand{F}{i}"
CORE = set(RENAME) | {"root", "spine_02", "spine_04", "neck_02"} | {
    f"{f}_metacarpal_{s}" for f in ("index", "middle", "ring", "pinky") for s in "lr"}


def log(*a):
    print("[metahuman]", *a, flush=True)


def activate(obj, mode="OBJECT"):
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    if mode != "OBJECT":
        bpy.ops.object.mode_set(mode=mode)


def import_fbx(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(path))
    new = [o for o in bpy.data.objects if o not in before]
    for o in new:                                    # flatten FBX helper empties, keep world transforms
        if o.parent and o.parent.type == "EMPTY":
            m = o.matrix_world.copy(); o.parent = None; o.matrix_world = m
    keep = [o for o in new if o.type != "EMPTY"]
    for o in [o for o in new if o.type == "EMPTY"]:
        bpy.data.objects.remove(o)
    return keep


def only(objs, kind):
    return next(o for o in objs if o.type == kind)


def rebind(mesh, armature):
    m = mesh.matrix_world.copy()
    mesh.parent = armature; mesh.matrix_world = m
    mods = [x for x in mesh.modifiers if x.type == "ARMATURE"] or [mesh.modifiers.new("Armature", "ARMATURE")]
    mods[0].object = armature


def merge_weights(mesh, redirect):
    """Add the weights of removed bones' vertex groups to their kept ancestor, then drop the groups."""
    names = {g.index: g.name for g in mesh.vertex_groups}
    sums = {}
    for v in mesh.data.vertices:
        for g in v.groups:
            target = redirect.get(names[g.group])
            if target:
                sums.setdefault(target, {}).setdefault(v.index, 0.0)
                sums[target][v.index] += g.weight
    for target, weights in sums.items():
        group = mesh.vertex_groups.get(target) or mesh.vertex_groups.new(name=target)
        for idx, w in weights.items():
            group.add([idx], w, "ADD")
    for name in redirect:
        if name in mesh.vertex_groups:
            mesh.vertex_groups.remove(mesh.vertex_groups[name])


def image(path, colour=True):
    img = bpy.data.images.load(str(path), check_existing=True)
    img.colorspace_settings.name = "sRGB" if colour else "Non-Color"
    return img


def pbr(name, base=None, normal=None, srm=None, colour=None, alpha=None, roughness=0.6):
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    nt = mat.node_tree; bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    if colour:
        bsdf.inputs["Base Color"].default_value = (*colour, 1)
    if base:
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = image(base)
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    if alpha:
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = image(alpha, False)
        nt.links.new(t.outputs["Color"], bsdf.inputs["Alpha"])
        mat.blend_method = "CLIP"; mat.alpha_threshold = 0.35
        mat.use_backface_culling = False
    if normal:
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = image(normal, False)
        n = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(t.outputs["Color"], n.inputs["Color"]); nt.links.new(n.outputs["Normal"], bsdf.inputs["Normal"])
    if srm:                                          # MetaHuman SRM(F): G = roughness, B = metallic (glTF layout)
        t = nt.nodes.new("ShaderNodeTexImage"); t.image = image(srm, False)
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(t.outputs["Color"], sep.inputs["Color"])
        nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
        nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    return mat


def composite_eye(side, out_dir):
    """Iris scaled into the sclera's dark disk with a soft edge (what the MetaHuman eye shader does)."""
    def load(n):
        img = bpy.data.images.load(str(TEX / "Face_Baked" / f"{n}.png"))
        a = np.array(img.pixels[:], np.float32).reshape(img.size[1], img.size[0], img.channels)
        bpy.data.images.remove(img)
        return a
    sclera, iris = load(f"T_EyeSclera{side}_BC"), load(f"T_EyeIris{side}_BC")
    h, w = sclera.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = (xx + .5) / w - .5, (yy + .5) / h - .5
    r = np.hypot(u, v)
    iris_r, iris_tex_r = 0.165, 0.42                 # disk radius in sclera UV; iris radius in its own texture
    su = np.clip(((u * iris_tex_r / iris_r) + .5) * iris.shape[1], 0, iris.shape[1] - 1).astype(int)
    sv = np.clip(((v * iris_tex_r / iris_r) + .5) * iris.shape[0], 0, iris.shape[0] - 1).astype(int)
    mask = np.clip((iris_r - r) / 0.012 + .5, 0, 1)[..., None]
    out = sclera.copy(); out[..., :3] = sclera[..., :3] * (1 - mask) + iris[sv, su, :3] * mask
    img = bpy.data.images.new(f"T_Eye{side}_Composite", w, h, alpha=True)
    img.pixels[:] = out.ravel(); img.filepath_raw = str(out_dir / f"T_Eye{side}_Composite.png"); img.file_format = "PNG"; img.save()
    return out_dir / f"T_Eye{side}_Composite.png"


def hair_alpha(out_dir):
    """Card atlas red channel = strand coverage -> grey opacity map."""
    img = bpy.data.images.load(str(TEX / "Grooms_Textures" / "Hair_L_StraightBangs_CardsAtlas_Attribute.png"))
    a = np.array(img.pixels[:], np.float32).reshape(img.size[1], img.size[0], img.channels)
    cov = np.clip(a[..., 0] * 1.6, 0, 1)
    out = np.stack([cov, cov, cov, np.ones_like(cov)], -1)
    new = bpy.data.images.new("T_Hair_Coverage", img.size[0], img.size[1], alpha=True)
    new.pixels[:] = out.ravel(); new.filepath_raw = str(out_dir / "T_Hair_Coverage.png"); new.file_format = "PNG"; new.save()
    return out_dir / "T_Hair_Coverage.png"


def classify_face_slots(face):
    """Role of each face material slot from its geometry, not its name.

    MetaHuman FBX exports can carry material names shifted by one slot (the head
    skin arrives labelled "Teeth"), so roles are read from where polygons sit:
    skin = largest slot; eyes = two small spheres left/right of the midline;
    teeth = centred slot below the eyes; eyelashes = the most forward remaining
    slot; anything else (eye shell, tear film, eye edge) is dropped."""
    co = np.array([face.matrix_world @ v.co for v in face.data.vertices])
    slots = {}
    for p in face.data.polygons:
        slots.setdefault(p.material_index, set()).update(p.vertices)
    count = {i: sum(1 for p in face.data.polygons if p.material_index == i) for i in slots}
    box = {i: (co[list(v)].min(0), co[list(v)].max(0)) for i, v in slots.items()}
    roles = {max(count, key=count.get): "skin"}
    eyes = [i for i in slots if i not in roles and box[i][1][2] - box[i][0][2] >= .02
            and box[i][1][0] - box[i][0][0] <= .04 and abs((box[i][0][0] + box[i][1][0]) / 2) > .015]
    for i in eyes:
        roles[i] = "eyeL" if (box[i][0][0] + box[i][1][0]) / 2 > 0 else "eyeR"   # character's left is +X
    eye_z = min(((box[i][0][2] + box[i][1][2]) / 2 for i in eyes), default=1e9)
    for i in slots:
        if i not in roles and abs((box[i][0][0] + box[i][1][0]) / 2) < .01 and box[i][1][2] < eye_z - .01:
            roles[i] = "teeth"
    rest = [i for i in slots if i not in roles]
    if rest:
        roles[min(rest, key=lambda i: box[i][0][1])] = "lashes"                 # Blender front is -Y
    for i in slots:
        roles.setdefault(i, "drop")
    log("face slots:", {face.data.materials[i].name: r for i, r in sorted(roles.items())})
    return roles


def world_head(arm, name):
    return arm.matrix_world @ arm.data.bones[name].head_local


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    work = OUT.parent / f".{OUT.stem}-build"; work.mkdir(parents=True, exist_ok=True)

    body_objs = import_fbx(SRC / "FrenchWoman_Body.fbx")
    face_objs = import_fbx(SRC / "FrenchWoman_Face.fbx")
    outfit_objs = import_fbx(SRC / "FrenchWoman_Outfit.fbx")
    hair = [m for f in ("FrenchWoman_Hair_Cards_Group0", "FrenchWoman_Hair_Cards_Group1")
            for m in import_fbx(SRC / "Hair" / f"{f}.fbx") if m.type == "MESH"]
    brows = [m for m in import_fbx(SRC / "Hair" / "FrenchWoman_Eyebrows_Cards.fbx") if m.type == "MESH"]
    arm, body = only(body_objs, "ARMATURE"), only(body_objs, "MESH")
    face_arm, face = only(face_objs, "ARMATURE"), only(face_objs, "MESH")
    outfit_arm, outfit = only(outfit_objs, "ARMATURE"), only(outfit_objs, "MESH")
    arm.name, body.name, face.name, outfit.name = "Armature", "Body", "Face", "Outfit"
    for i, m in enumerate(hair):
        m.name = f"Hair{i}"
    brows[0].name = "Eyebrows"

    # ---- facial joints that the face mesh actually uses (+ ancestors)
    weighted = {g.name for g in face.vertex_groups if g.name.startswith("FACIAL")}
    keep_facial = set()
    for name in weighted:
        b = face_arm.data.bones.get(name)
        while b is not None and b.name.startswith("FACIAL"):
            keep_facial.add(b.name); b = b.parent
    # ---- move facial joints into the body armature
    activate(face_arm, "EDIT")
    for eb in list(face_arm.data.edit_bones):
        if eb.name not in keep_facial:
            face_arm.data.edit_bones.remove(eb)
    activate(face_arm)
    activate(arm)
    face_arm.select_set(True)
    bpy.ops.object.join()                            # face_arm merges into arm
    activate(arm, "EDIT")
    eb = arm.data.edit_bones
    for b in eb:
        if b.name.startswith("FACIAL") and (b.parent is None or not b.parent.name.startswith("FACIAL")):
            b.parent = eb["head"]
    activate(arm)
    rebind(face, arm); rebind(outfit, arm)
    bpy.data.objects.remove(outfit_arm)

    # ---- remove helper bones, merging their weights into the nearest kept ancestor
    def kept(b):
        return b.name in CORE or b.name.startswith("FACIAL")
    redirect = {}
    for b in arm.data.bones:
        if not kept(b):
            a = b.parent
            while a is not None and not kept(a):
                a = a.parent
            redirect[b.name] = a.name if a else "root"
    for mesh in (body, face, outfit):
        merge_weights(mesh, redirect)
    activate(arm, "EDIT")
    eb = arm.data.edit_bones
    for name in redirect:
        bone = eb.get(name)
        if bone:
            for child in list(bone.children):
                child.parent = bone.parent
            eb.remove(bone)
    # ---- fingertip and head-end bones (directions for the last segments)
    for s, side in (("l", "Left"), ("r", "Right")):
        for f, F in FINGERS.items():
            b2, b3 = eb[f"{f}_02_{s}"], eb[f"{f}_03_{s}"]
            d = b3.head - b2.head
            tip = eb.new(f"{side}Hand{F}4"); tip.head = b3.head + d * 0.8; tip.tail = tip.head + d.normalized() * 0.01
            tip.parent = b3; tip.use_deform = False
    head = eb["head"]
    up = (arm.matrix_world.inverted().to_3x3() @ mathutils.Vector((0, 0, 1))).normalized()
    end = eb.new("HeadTop_End"); end.head = head.head + up * 0.17; end.tail = end.head + up * 0.02
    end.parent = head; end.use_deform = False
    activate(arm)
    # ---- Studio bone names (vertex groups renamed alongside)
    for old, new in RENAME.items():
        if old in arm.data.bones:
            arm.data.bones[old].name = new
            for mesh in (body, face, outfit):
                if old in mesh.vertex_groups:
                    mesh.vertex_groups[old].name = new
    log("bones after cleanup:", len(arm.data.bones))

    # ---- hair and eyebrows: rigid to the head
    for m in hair + brows:
        g = m.vertex_groups.new(name="Head"); g.add(list(range(len(m.data.vertices))), 1.0, "REPLACE")
        rebind(m, arm)

    # ---- remove the transparent eye shell and tear film (render as grey shells outside Unreal)
    roles = classify_face_slots(face)
    drop = [i for i, r in roles.items() if r == "drop"]
    activate(face, "EDIT"); bpy.ops.mesh.select_all(action="DESELECT"); activate(face)
    for p in face.data.polygons:
        p.select = p.material_index in drop
    activate(face, "EDIT"); bpy.ops.mesh.delete(type="FACE"); activate(face)

    build_blendshapes(arm, face, brows[0], roles)
    assign_materials(body, face, outfit, hair, brows[0], work, roles)

    for o in list(bpy.data.objects):
        if o not in (arm, body, face, outfit, *hair, *brows):
            bpy.data.objects.remove(o)
    activate(arm)
    bpy.ops.export_scene.gltf(filepath=str(OUT), export_format="GLB", export_skins=True, export_morph=True,
                              export_morph_normal=False, export_animations=False, export_yup=True,
                              export_def_bones=False, export_image_format="JPEG", export_jpeg_quality=88)
    log("wrote", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")


def evaluated_coords(obj):
    dg = bpy.context.evaluated_depsgraph_get(); dg.update()
    ev = obj.evaluated_get(dg); me = ev.to_mesh()
    co = np.array([obj.matrix_world @ v.co for v in me.vertices]); ev.to_mesh_clear()
    return co


def pose_shapekey(face, arm, name, pose):
    """Pose facial joints, store the deformation as a shape key, reset the pose."""
    activate(arm, "POSE")
    pose(arm.pose.bones)
    activate(face)
    bpy.ops.object.modifier_apply_as_shapekey(keep_modifier=True, modifier=face.modifiers[0].name)
    face.data.shape_keys.key_blocks[-1].name = name
    activate(arm, "POSE")
    for pb in arm.pose.bones:
        pb.location = (0, 0, 0); pb.rotation_quaternion = (1, 0, 0, 0); pb.rotation_euler = (0, 0, 0); pb.scale = (1, 1, 1)
    activate(arm)


def build_blendshapes(arm, face, brows, roles):
    if not face.data.shape_keys:
        face.shape_key_add(name="Basis")
    rest = evaluated_coords(face)
    jaw_verts = [v.index for v in face.data.vertices
                 if any(face.vertex_groups[g.group].name == "FACIAL_C_Jaw" and g.weight > .5 for g in v.groups)]

    # jaw: choose the local axis/sign that moves the jaw-weighted vertices down the most
    best = None
    for axis in range(3):
        for sign in (1, -1):
            def pose(pbs, axis=axis, sign=sign):
                pb = pbs["FACIAL_C_Jaw"]; pb.rotation_mode = "XYZ"
                e = [0, 0, 0]; e[axis] = sign * math.radians(18); pb.rotation_euler = e
            activate(arm, "POSE"); pose(arm.pose.bones); activate(arm)
            drop = (rest[jaw_verts, 2] - evaluated_coords(face)[jaw_verts, 2]).mean()
            arm.pose.bones["FACIAL_C_Jaw"].rotation_euler = (0, 0, 0)
            if best is None or drop > best[0]:
                best = (drop, axis, sign)
    log(f"jaw axis {'XYZ'[best[1]]} sign {best[2]}: chin drops {best[0] * 100:.1f} cm at 18 deg")

    def open_jaw(pbs, degrees=22):
        """Rotate the jaw and carry the lower-lip joints with it (Unreal's rig logic does this at runtime)."""
        jaw = pbs["FACIAL_C_Jaw"]; jaw.rotation_mode = "XYZ"
        jaw.rotation_euler = [best[2] * math.radians(degrees) if i == best[1] else 0 for i in range(3)]
        bpy.context.view_layer.update()
        rest_jaw = arm.data.bones["FACIAL_C_Jaw"].matrix_local
        moved = jaw.matrix @ rest_jaw.inverted()          # rest -> posed transform of the jaw, armature space
        for pb in pbs:
            if "LipLower" in pb.name or "LipCorner" in pb.name:
                if pb.bone.parent and pb.bone.parent.name == "FACIAL_C_Jaw":
                    continue                              # already carried by the jaw
                weight = 0.5 if "LipCorner" in pb.name else 1.0
                rest = arm.data.bones[pb.name].matrix_local
                target = (moved @ rest).translation
                m = pb.matrix.copy(); m.translation = rest.translation.lerp(target, weight); pb.matrix = m
                bpy.context.view_layer.update()
    pose_shapekey(face, arm, "jawOpen", open_jaw)

    # blinks: upper eyelid joints move onto their lower-lid counterparts
    forward = (arm.matrix_world.inverted().to_3x3() @ mathutils.Vector((0, -1, 0))).normalized()   # Blender front is -Y
    for s, side in (("L", "Left"), ("R", "Right")):
        def pose(pbs, s=s):
            for row in ("A", "B"):
                for i in (1, 2, 3):
                    up, lo = f"FACIAL_{s}_EyelidUpper{row}{i}", f"FACIAL_{s}_EyelidLower{row}{i}"
                    if up in pbs and lo in pbs:
                        delta = (arm.data.bones[lo].head_local - arm.data.bones[up].head_local) * 1.0
                        # bulge forward so the closing lid wraps over the eyeball instead of sliding behind it
                        delta = delta + forward * (0.009 if i == 2 else 0.006)
                        m = pbs[up].matrix.copy(); m.translation = m.translation + delta; pbs[up].matrix = m
                        bpy.context.view_layer.update()
            for i in (1, 2, 3):                      # lashes follow the upper lid
                name = f"FACIAL_{s}_EyelashesUpperA{i}"
                lid = f"FACIAL_{s}_EyelidUpperA{i}"
                if name in pbs and lid in pbs:
                    delta = (arm.data.bones[f"FACIAL_{s}_EyelidLowerA{i}"].head_local - arm.data.bones[lid].head_local) * 0.9
                    m = pbs[name].matrix.copy(); m.translation = m.translation + delta; pbs[name].matrix = m
                    bpy.context.view_layer.update()
        pose_shapekey(face, arm, f"eyeBlink{side}", pose)

    # brows: smooth displacement field above each eye (no brow joints are weighted at this LOD)
    eyes = {s: world_head(arm, f"FACIAL_{s}_EyeParallel") for s in ("L", "R")}
    skin = {i for i, r in roles.items() if r == "skin"}
    def field(co, s, part):
        e = eyes[s]; xs = e.x * (0.45 if part == "inner" else 1.35)
        zb = e.z + 0.022
        w = np.exp(-((co[:, 0] - xs) ** 2 / (2 * .013 ** 2) + (co[:, 2] - zb) ** 2 / (2 * .014 ** 2)))
        w[co[:, 2] < e.z + 0.006] = 0
        w[co[:, 1] > e.y + 0.015] = 0                # front of the face only
        return w
    shapes = {"browInnerUp": [("L", "inner", 0.006), ("R", "inner", 0.006)],
              "browOuterUpLeft": [("L", "outer", 0.006)], "browOuterUpRight": [("R", "outer", 0.006)],
              "browDownLeft": [("L", "inner", -0.004), ("L", "outer", -0.003)],
              "browDownRight": [("R", "inner", -0.004), ("R", "outer", -0.003)]}
    skin_verts = np.zeros(len(face.data.vertices), bool)
    for p in face.data.polygons:
        if p.material_index in skin:
            skin_verts[list(p.vertices)] = True
    for obj, mask in ((face, skin_verts), (brows, None)):
        if not obj.data.shape_keys:
            obj.shape_key_add(name="Basis")
        world = np.array([obj.matrix_world @ v.co for v in obj.data.vertices])
        to_local = obj.matrix_world.inverted().to_3x3()
        for name, parts in shapes.items():
            dz = np.zeros(len(world))
            for s, part, amount in parts:
                dz += field(world, s, part) * amount
            if mask is not None:
                dz[~mask] = 0
            key = obj.shape_key_add(name=name, from_mix=False)
            local = np.array(to_local @ mathutils.Vector((0, 0, 1)))
            co = np.array([v.co for v in key.data]) + dz[:, None] * local[None]
            key.data.foreach_set("co", co.ravel())
    log("blendshapes:", face.data.shape_keys.key_blocks.keys())


def assign_materials(body, face, outfit, hair, brows, work, roles):
    fb, bb = TEX / "Face_Baked", TEX / "Body_Baked"
    body.data.materials.clear()
    body.data.materials.append(pbr("Body", bb / "T_Body_BC_VT.png", bb / "T_Body_N_VT.png", bb / "T_Body_SRMF_VT.png"))
    eyes = {s: composite_eye(s, work) for s in ("L", "R")}
    by_role = {
        "skin": lambda: pbr("FaceSkin", fb / "T_Head_LOD3_BC_VT.png", fb / "T_Head_LOD3_N_VT.png", fb / "T_Head_LOD3_SRMF_VT.png"),
        "teeth": lambda: pbr("Teeth", fb / "T_Teeth_BC.png", fb / "T_Teeth_N.png", fb / "T_Teeth_SRM.png"),
        "eyeL": lambda: pbr("EyeL", eyes["L"], roughness=0.15),
        "eyeR": lambda: pbr("EyeR", eyes["R"], roughness=0.15),
        "lashes": lambda: pbr("Eyelashes", colour=(0.02, 0.015, 0.012), roughness=0.7),
        "drop": lambda: pbr("Unused", colour=(0.5, 0.5, 0.5)),
    }
    for i in range(len(face.data.materials)):
        face.data.materials[i] = by_role[roles.get(i, "drop")]()
    for i, mat in enumerate(outfit.data.materials):
        kind = "Shirt" if "Shirt" in mat.name else "Short"
        outfit.data.materials[i] = pbr(f"Outfit{kind}", colour=OUTFIT_COLOURS[kind], roughness=0.8)
    alpha = hair_alpha(work)
    hair_mat = pbr("Hair", colour=HAIR_COLOUR, alpha=alpha, roughness=0.55)
    for m in hair + [brows]:
        m.data.materials.clear(); m.data.materials.append(hair_mat)


if __name__ == "__main__":
    main()
