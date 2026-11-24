import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { EmotionWindow, MotionFile } from "../../types/contract";
import { interpolateJointPositions } from "../../utils/motionInterpolator";
import { HumanoidRetargeter } from "../../utils/retargetHumanoid";
import { browWeights, faceCalibration, frameInterval, sampleFace } from "../../utils/avatarPlayback";
import { IdleLife } from "../../utils/idleLife";
import { rigBoneName } from "../../utils/rigProfiles";
import { MetaHumanFaceDriver } from "../../utils/metahumanFace";
import { engineFace, turnHead } from "../../utils/engineFace";
import { PerformanceLayer } from "../../utils/performanceLayer";
import { EmotionStyle, emotionFace } from "../../utils/emotionStyle";
import { polishMaterials } from "../../utils/renderPolish";
import { EMOTION_KEYS, emotionToBlendshapes, sampleEmotion, smoothWeights } from "../../utils/emotionFace";
import { addLook, createPointingRig, pointAt, pointingAt, pointLook, type PointCommand, type PointingRig } from "../../utils/pointing";
import { applyRoleShift, createRoleShiftRig, type RoleShiftPose, type RoleShiftRig } from "../../utils/roleShift";

/** Largest head nod on a fully emphasised sign; small, because head movement is LSF grammar. */
const NOD_MAX_DEG = 6;

/** What the avatar shows this frame (the AvatarCanvas props that drive the pose). */
export type AvatarFrame = {
  motion: MotionFile | null;
  currentTimeMs: number;
  headRotation?: number[];
  faceBlendshapes?: Record<string, number>;
  expressiveModulation?: boolean;
  emotion?: Record<string, number | string> | null;
  emotionTimeline?: EmotionWindow[] | null;
  emotionIntensity?: number;
  emotionClock?: () => number;
  /** Only pose and draw when something changed (export / still views). */
  renderOnChange?: boolean;
  prewarm?: MotionFile[];
  onPrewarmed?: (ready: number) => void;
  /** Turn of torso / head / gaze toward a side (film speakers, the companion looking at an object). */
  roleShift?: RoleShiftPose | null;
  /** [pointing] Explicit point (AR companion: a world position, or a locus side). Without it, motion.pointing is played. */
  point?: PointCommand | PointCommand[] | null;
  /** With prewarm: derive at most this many ms per frame (sliced, no stall) instead of one whole motion per frame. */
  prewarmBudgetMs?: number;
  /** Called with each motion whose derivation is cached (it will start without any stall). */
  onPrepared?: (motion: MotionFile) => void;
  /** Cross-fade (ms) from the shown pose whenever the motion changes (sentence -> rest -> next sentence). 0 = cut. */
  blendMs?: number;
};

export class AvatarLoadError extends Error {
  constructor(readonly kind: "model" | "skeleton") {
    super(kind === "skeleton" ? "Le squelette du modèle ne possède pas les articulations requises." : "Personnage 3D indisponible.");
  }
}

/** Free the GPU resources of a loaded model. */
export function disposeModel(root: THREE.Object3D) {
  const materials = new Set<THREE.Material>();
  const textures = new Set<THREE.Texture>();
  root.traverse((o) => {
    if (o instanceof THREE.Mesh) {
      o.geometry.dispose();
      for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
        materials.add(m);
        for (const v of Object.values(m)) if (v instanceof THREE.Texture) textures.add(v);
      }
    }
  });
  materials.forEach((m) => m.dispose());
  textures.forEach((t) => t.dispose());
}

/** Load the avatar GLB and build its rig. `isDisposed` lets a caller that unmounted meanwhile drop the model. */
export function loadAvatarRig(url = "/api/avatar/model", isDisposed: () => boolean = () => false): Promise<AvatarRig | null> {
  return new Promise((resolve, reject) => {
    new GLTFLoader().load(
      url,
      (gltf) => {
        if (isDisposed()) {
          disposeModel(gltf.scene);
          resolve(null);
          return;
        }
        polishMaterials(gltf.scene);
        gltf.scene.updateMatrixWorld(true);
        try {
          resolve(new AvatarRig(gltf.scene));
        } catch (e) {
          reject(e instanceof AvatarLoadError ? e : new AvatarLoadError("skeleton"));
        }
      },
      undefined,
      () => reject(new AvatarLoadError("model")),
    );
  });
}

/**
 * The avatar runtime shared by every view (Studio canvas, film, conversation, AR companion): the MetaHuman, the
 * retargeter with hand limits, the interpreter layers (idle life, performance, emotion style, role shift) and the
 * face (engine face, grammatical non-manuals, emotion, blinks). A view owns its renderer, camera and scene, adds
 * `rig.model` where it wants, calls `pose()` before drawing and `afterDraw()` right after.
 */
export class AvatarRig {
  readonly retarget: HumanoidRetargeter;
  private idle: IdleLife;
  private performer: PerformanceLayer;
  private style: EmotionStyle;
  private shiftRig: RoleShiftRig | null;
  private pointRig: PointingRig | null;
  private faceDriver: MetaHumanFaceDriver | null;
  private morphGain: Record<string, number> | undefined;
  private morphs: THREE.Mesh[] = [];
  private head: THREE.Bone | undefined;
  private bindHead: THREE.Quaternion | undefined;
  // Interpreter-style layers: wall clock for presentation easing, motion clock for phrase-aware blinks.
  private lastWall = -1;
  private lastMotionTime = -1;
  private emotionLevel = 0;
  private emotionLabel: string | undefined;
  // Channel values in ARKit names; rigs without blendshapes read them via faceDriver.
  private faceWeights: Record<string, number> = {};
  // Values set this frame, so the emotion layer can add on top of captured/base weights.
  private applied = new Map<string, number>();
  private emotionWeights: Record<string, number> = {};
  private lastEmotionAt: number | undefined;
  private nodRestore: THREE.Quaternion | undefined;
  // Cross-fade between motions (frame.blendMs): the pose shown last frame, and the pose the fade starts from.
  private fade: { bones: THREE.Object3D[]; shown: { q: THREE.Quaternion[]; p: THREE.Vector3[]; m: number[][] };
    from: { q: THREE.Quaternion[]; p: THREE.Vector3[]; m: number[][] } | null; start: number; key: MotionFile | null | undefined } | null = null;
  // renderOnChange bookkeeping
  private last: Partial<AvatarFrame> & { set?: boolean } = {};

  constructor(readonly model: THREE.Object3D) {
    try {
      this.retarget = new HumanoidRetargeter(model);
    } catch {
      throw new AvatarLoadError("skeleton");
    }
    const profile = this.retarget.profile;
    this.morphGain = profile.morphGain;
    this.idle = new IdleLife(model, profile);
    // Interpreter performance (twist bones, springs, moving holds, hair) and emotion style (posture blendspace).
    // The retargeter's hand limits are re-applied after the springs, so no layer can bend a finger out of range.
    this.performer = new PerformanceLayer(model, profile, this.retarget.hands);
    this.style = new EmotionStyle(model, profile);
    this.shiftRig = createRoleShiftRig(model, profile);
    this.pointRig = createPointingRig(model, profile, this.retarget.hands);
    const style = this.style;
    fetch("/api/avatar/emotion-idles")
      .then((r) => (r.ok ? r.json() : []))
      .then((list: { label: string; url: string }[]) => style.loadIdles(list.map((i) => ({ ...i, url: "/api" + i.url }))))
      .catch(() => undefined);
    const headName = rigBoneName(profile, "Head");
    model.traverse((o) => {
      if (o instanceof THREE.Bone && o.name === headName) {
        this.head = o;
        this.bindHead = o.quaternion.clone();
      }
      if (o instanceof THREE.Mesh) {
        o.frustumCulled = false;
        if (o.morphTargetDictionary) this.morphs.push(o);
      }
    });
    // MetaHuman without ARKit shapes (v1 conversion): drive jaw/lids/brows/gaze through FACIAL_* joints instead.
    // With baked ARKit shapes (v2) the joints stay at rest, else every expression would be applied twice.
    this.faceDriver = this.morphs.some((m) => m.morphTargetDictionary?.jawOpen !== undefined) ? null : MetaHumanFaceDriver.create(model);
  }

  private morph(name: string, value: number) {
    this.faceWeights[name] = THREE.MathUtils.clamp(value, 0, 1);
    // faceWeights keep the requested weight (the v1 joint driver reads them); the mesh gets the rig's display gain.
    const shown = THREE.MathUtils.clamp(value, 0, 1) * (this.morphGain?.[name] ?? 1);
    this.applied.set(name, THREE.MathUtils.clamp(value, 0, 1));
    for (const mesh of this.morphs) {
      const i = mesh.morphTargetDictionary?.[name];
      if (i !== undefined && mesh.morphTargetInfluences) mesh.morphTargetInfluences[i] = shown;
    }
  }

  /** Film mode: derive the next not-yet-prepared motion of the queue (one per frame). Returns true if it did. */
  prewarm(frame: AvatarFrame) {
    const queue = frame.prewarm;
    if (!queue?.length) return false;
    const pending = queue.find((m) => !this.retarget.isPrepared(m));
    if (!pending) return false;
    if (frame.prewarmBudgetMs) {
      if (this.retarget.primeSlice(pending, frame.prewarmBudgetMs)) frame.onPrepared?.(pending);
      return true;
    }
    this.retarget.prime(pending);
    frame.onPrepared?.(pending);
    frame.onPrewarmed?.(queue.filter((m) => this.retarget.isPrepared(m)).length);
    return true;
  }

  /**
   * Pose the avatar for this frame. With `renderOnChange`, returns false (and does nothing) when neither the inputs
   * nor `dirty` changed, so the caller can skip the draw. `debugHost` receives the retargeter's diagnostics.
   */
  pose(frame: AvatarFrame, dirty = true, debugHost?: HTMLElement): boolean {
    const { motion, currentTimeMs, expressiveModulation, faceBlendshapes, headRotation, emotion, emotionTimeline,
      emotionIntensity, emotionClock, renderOnChange } = frame;
    const last = this.last;
    if (renderOnChange && !dirty && last.set && last.motion === motion && last.currentTimeMs === currentTimeMs &&
      last.faceBlendshapes === faceBlendshapes && last.headRotation === headRotation && last.roleShift === frame.roleShift && last.point === frame.point) return false;
    this.last = { set: true, motion, currentTimeMs, faceBlendshapes, headRotation, roleShift: frame.roleShift, point: frame.point };
    const { retarget, head, bindHead } = this;
    const morph = this.morph.bind(this);
    this.applied.clear();
    // Anatomical neutral stance; old synthetic fixtures use the legacy mirrored X convention.
    if (!motion) {
      const joints = interpolateJointPositions(null, currentTimeMs);
      for (const side of ["left", "right"]) {
        const sign = side === "left" ? 1 : -1;
        joints[`${side}_shoulder`] = [sign * 0.18, 1.48, 0];
        joints[`${side}_elbow`] = [sign * 0.25, 1.2, 0.04];
        // Hands resting loosely together without the fingers interpenetrating.
        joints[`${side}_wrist`] = [sign * 0.17, 1.02, 0.1];
      }
      retarget.apply(joints);
    } else {
      const tracking = retarget.applyMotion(motion, currentTimeMs);
      if (debugHost) {
        debugHost.dataset.handTracking = JSON.stringify(tracking);
        debugHost.dataset.rigDiagnostics = JSON.stringify(retarget.diagnostics);
      }
    }
    // Interpreter style on top of the retargeted pose: emotion posture, then springs / twist / holds / hair.
    const nowSec = performance.now() / 1000;
    const wallDt = this.lastWall < 0 ? 0 : Math.min(0.1, nowSec - this.lastWall);
    this.lastWall = nowSec;
    const feeling = motion?.generation?.emotion;
    if (feeling && feeling.label !== "neutral") this.emotionLabel = feeling.label;
    this.emotionLevel += ((feeling && feeling.label !== "neutral" ? feeling.confidence : 0) - this.emotionLevel) * Math.min(1, wallDt * 2);
    this.style.apply(nowSec, this.emotionLabel, this.emotionLevel, wallDt);
    this.performer.update(motion ? currentTimeMs : undefined, Boolean(motion));
    // Phrase-aware blinks: at the end of each grammatical clause span, during normal playback.
    const lastMotionTime = this.lastMotionTime;
    if (motion?.nonmanual && lastMotionTime >= 0 && currentTimeMs > lastMotionTime && currentTimeMs - lastMotionTime < 250)
      for (const span of motion.nonmanual) if (span.end_ms > lastMotionTime && span.end_ms <= currentTimeMs) this.idle.requestBlink(nowSec);
    this.lastMotionTime = motion ? currentTimeMs : -1;
    // Display-only life layer on top of the reset-every-frame retargeted pose.
    const life = renderOnChange ? undefined : this.idle.update(nowSec, Boolean(motion));
    morph("eyeBlinkLeft", life?.blink ?? 0);
    morph("eyeBlinkRight", life?.blink ?? 0);
    // Pointing (index): the frame's explicit point, else the motion's scheduled points. Head and eyes turn toward it too.
    const point = frame.point ?? (motion?.pointing ? pointingAt(motion.pointing, currentTimeMs) : null);
    const roleShift = addLook(frame.roleShift, pointLook(this.pointRig, point));
    const gx = (life?.gaze.x ?? 0) + (roleShift?.gazeX ?? 0), gy = life?.gaze.y ?? 0; // gaze toward the speaker / object
    morph("eyeLookOutLeft", Math.max(0, gx));
    morph("eyeLookInRight", Math.max(0, gx));
    morph("eyeLookInLeft", Math.max(0, -gx));
    morph("eyeLookOutRight", Math.max(0, -gx));
    morph("eyeLookUpLeft", Math.max(0, gy));
    morph("eyeLookUpRight", Math.max(0, gy));
    morph("eyeLookDownLeft", Math.max(0, -gy));
    morph("eyeLookDownRight", Math.max(0, -gy));
    const face = sampleFace(motion, currentTimeMs);
    const calibration = motion ? faceCalibration(motion) : undefined;
    const browLimit = .18 + .47 * THREE.MathUtils.clamp(face?.brow_reference_blend ?? (calibration ? 1 : 0), 0, 1);
    // Voice-emotion layer: full face at rest, capped (brows only nudged) while signing.
    const now = performance.now();
    const emotionTarget = expressiveModulation
      ? emotionToBlendshapes(
          sampleEmotion(emotionTimeline, emotion, emotionClock ? emotionClock() : currentTimeMs),
          motion ? "signing" : "rest",
          emotionIntensity ?? 1,
        )
      : {};
    this.emotionWeights = smoothWeights(this.emotionWeights, emotionTarget, this.lastEmotionAt === undefined ? 1e6 : now - this.lastEmotionAt);
    this.lastEmotionAt = now;
    const emotionWeights = this.emotionWeights, faceWeights = this.faceWeights;
    const negative = ["mouthFrownLeft", "mouthPressLeft", "noseSneerLeft", "mouthStretchLeft"]
      .reduce((sum, key) => sum + (emotionWeights[key] ?? 0), 0);
    const restSmile = 0.22 * (1 - THREE.MathUtils.clamp(negative * 2, 0, 1));
    morph("jawOpen", face && Number.isFinite(face.mouth_open) ? face.mouth_open * 2 : 0);
    for (const side of ["Left", "Right"]) {
      const key = side === "Left" ? "brow_left" : "brow_right";
      const brow = face?.[key];
      const weights = browWeights(brow, face?.[`${key}_neutral`] ?? calibration?.[key], browLimit);
      morph(`browOuterUp${side}`, brow !== undefined ? weights.up : 0);
      morph(`browDown${side}`, brow !== undefined ? weights.down : 0);
      // A soft welcoming smile only at rest; never added over signed content.
      morph(`mouthSmile${side}`, face || motion ? 0 : restSmile);
      morph(`mouthFrown${side}`, 0);
    }
    morph("browInnerUp", face ? (browWeights(face.brow_left, face.brow_left_neutral ?? calibration?.brow_left, browLimit).up + browWeights(face.brow_right, face.brow_right_neutral ?? calibration?.brow_right, browLimit).up) * .2 : 0);
    // LSF engines: lip shape, lip corners and eye opening relative to the clip's neutral, plus the
    // grammatical non-manual layer (question brows, negation head shake), which emotion never writes.
    const engine = motion?.generation ? engineFace(face, calibration) : null;
    if (engine) {
      for (const [name, value] of Object.entries(engine.weights))
        morph(name, name.startsWith("eyeBlink") ? Math.max(value, faceWeights[name] ?? 0) : value);
      const nm = engine.nonmanual;
      if (nm.browRaise || nm.browFurrow) {
        morph("browInnerUp", Math.max(faceWeights.browInnerUp ?? 0, nm.browRaise * 0.9));
        for (const side of ["Left", "Right"]) {
          morph(`browOuterUp${side}`, Math.max(faceWeights[`browOuterUp${side}`] ?? 0, nm.browRaise * 0.7));
          morph(`browDown${side}`, Math.max(faceWeights[`browDown${side}`] ?? 0, nm.browFurrow * 0.8));
        }
      }
    }
    // Emotion face (FACS presets), additive and below grammar: emotional brows are cut while a marker is active.
    const grammar = Boolean(engine && (engine.nonmanual.browRaise > 0.05 || engine.nonmanual.browFurrow > 0.05));
    for (const [name, value] of Object.entries(emotionFace(this.emotionLabel, this.emotionLevel, grammar)))
      morph(name, Math.min(1, (faceWeights[name] ?? 0) + value));
    // Additive on top of captured/base values; every emotion key is rewritten each frame.
    for (const key of EMOTION_KEYS) morph(key, (this.applied.get(key) ?? 0) + (emotionWeights[key] ?? 0));
    if (faceBlendshapes) {
      for (const name of [
        "jawOpen", "eyeBlinkLeft", "eyeBlinkRight", "browInnerUp",
        "browOuterUpLeft", "browOuterUpRight", "browDownLeft", "browDownRight",
        "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
      ]) {
        const weight = faceBlendshapes[name];
        if (Number.isFinite(weight)) morph(name, THREE.MathUtils.clamp(weight, 0, 1));
      }
    }
    this.faceDriver?.apply(faceWeights);
    if (engine && head && (engine.nonmanual.headYaw || engine.nonmanual.headPitch))
      turnHead(head, engine.nonmanual.headYaw, engine.nonmanual.headPitch);
    // Role shift: torso/head yaw toward the speaker (or the object), added on top of the grammatical head motion.
    applyRoleShift(this.shiftRig, roleShift);
    pointAt(this.pointRig, point);
    if (head?.parent && bindHead && headRotation?.length === 9 && headRotation.every(Number.isFinite)) {
      const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = headRotation;
      const rotation = new THREE.Matrix4().set(xx, xy, xz, 0, yx, yy, yz, 0, zx, zy, zz, 0, 0, 0, 0, 1);
      const parent = head.parent.getWorldQuaternion(new THREE.Quaternion());
      const delta = parent.clone().invert().multiply(new THREE.Quaternion().setFromRotationMatrix(rotation)).multiply(parent);
      head.quaternion.copy(bindHead).premultiply(delta);
      head.updateWorldMatrix(false, true);
    }
    this.crossFade(motion, frame.blendMs ?? 0);
    // Emphasis nod: additive head pitch on stressed signs (<= NOD_MAX_DEG), undone by afterDraw() so it never
    // accumulates into the retargeted pose.
    this.nodRestore = undefined;
    if (expressiveModulation && motion?.frames.length && head?.parent) {
      const { lo, next, alpha } = frameInterval(motion.frames, currentTimeMs);
      const nod = (motion.frames[lo].nod ?? 0) * (1 - alpha) + (motion.frames[next].nod ?? 0) * alpha;
      if (nod > 1e-3) {
        this.nodRestore = head.quaternion.clone();
        const parent = head.parent.getWorldQuaternion(new THREE.Quaternion());
        const lateral = new THREE.Vector3(1, 0, 0).applyQuaternion(parent.invert());
        head.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(lateral, THREE.MathUtils.degToRad(NOD_MAX_DEG) * Math.min(1, nod)));
        head.updateWorldMatrix(false, true);
      }
    }
    return true;
  }

  /** Ease from the last shown pose into this frame's pose for `ms` after each change of motion (null = rest). */
  private crossFade(motion: MotionFile | null, ms: number) {
    if (ms <= 0) { this.fade = null; return; }
    if (!this.fade) {
      const bones: THREE.Object3D[] = [];
      this.model.traverse((o) => { if ((o as THREE.Bone).isBone) bones.push(o); });
      this.fade = { bones, shown: { q: [], p: [], m: [] }, from: null, start: 0, key: motion };
    }
    const f = this.fade, now = performance.now();
    if (motion !== f.key && f.shown.q.length) {
      f.from = { q: f.shown.q.map((q) => q.clone()), p: f.shown.p.map((v) => v.clone()), m: f.shown.m.map((a) => a.slice()) };
      f.start = now;
    }
    f.key = motion;
    if (f.from) {
      const t = (now - f.start) / ms;
      if (t >= 1) f.from = null;
      else {
        const e = t * t * t * (10 - 15 * t + 6 * t * t); // minimum-jerk (human reaching): no jump in speed or acceleration
        const target = new THREE.Quaternion();
        f.bones.forEach((b, i) => {
          b.quaternion.copy(f.from!.q[i].clone().slerp(target.copy(b.quaternion), e));
          b.position.lerpVectors(f.from!.p[i], b.position.clone(), e);
        });
        this.morphs.forEach((mesh, i) => {
          const inf = mesh.morphTargetInfluences, a = f.from!.m[i];
          if (inf && a) for (let k = 0; k < inf.length; k++) inf[k] = a[k] + (inf[k] - a[k]) * e;
        });
        this.model.updateMatrixWorld(true);
      }
    }
    f.bones.forEach((b, i) => { (f.shown.q[i] ??= new THREE.Quaternion()).copy(b.quaternion); (f.shown.p[i] ??= new THREE.Vector3()).copy(b.position); });
    this.morphs.forEach((mesh, i) => { f.shown.m[i] = Array.from(mesh.morphTargetInfluences ?? []); });
  }

  /** Undo the display-only emphasis nod after the draw. */
  afterDraw() {
    if (this.nodRestore && this.head) {
      this.head.quaternion.copy(this.nodRestore);
      this.head.updateWorldMatrix(false, true);
    }
    this.nodRestore = undefined;
  }

  dispose() {
    this.model.removeFromParent();
    disposeModel(this.model);
  }
}
