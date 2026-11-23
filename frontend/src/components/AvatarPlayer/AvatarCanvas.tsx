import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { MotionFile } from "../../types/contract";
import { interpolateJointPositions } from "../../utils/motionInterpolator";
import { HumanoidRetargeter } from "../../utils/retargetHumanoid";
import { browWeights, faceCalibration, frameInterval, sampleFace } from "../../utils/avatarPlayback";
import { IdleLife } from "../../utils/idleLife";
import { createCafeBackdrop } from "../../utils/stageBackdrop";
import { rigBoneName } from "../../utils/rigProfiles";
import { MetaHumanFaceDriver } from "../../utils/metahumanFace";
import { engineFace, turnHead } from "../../utils/engineFace";
import { PerformanceLayer } from "../../utils/performanceLayer";
import { EmotionStyle, emotionFace } from "../../utils/emotionStyle";
import { createComposer, enableSoftShadows, polishMaterials } from "../../utils/renderPolish";
import { EMOTION_KEYS, emotionToBlendshapes, sampleEmotion, smoothWeights } from "../../utils/emotionFace";
import type { EmotionWindow } from "../../types/contract";
/** Largest head nod on a fully emphasised sign; small, because head movement is LSF grammar. */
const NOD_MAX_DEG = 6;
type Props = {
  avatarCredit?: string;
  fitNarrowViewport?: boolean;
  renderOnChange?: boolean;
  motion: MotionFile | null;
  currentTimeMs: number;
  headRotation?: number[];
  faceBlendshapes?: Record<string, number>;
  expressiveModulation?: boolean;
  showSkeletonDebug?: boolean;
  emotion?: Record<string, number | string> | null;
  /** Time-varying emotion2vec windows; falls back to `emotion` when absent. */
  emotionTimeline?: EmotionWindow[] | null;
  /** 0–1 multiplier for the voice-emotion face layer. */
  emotionIntensity?: number;
  /** Audio time (ms) used to sample the emotion timeline; defaults to currentTimeMs. */
  emotionClock?: () => number;
  onCanvasReady?: (canvas: HTMLCanvasElement) => void;
  /** Motions to derive ahead of playback (film mode), one per animation frame while idle. */
  prewarm?: MotionFile[];
  /** Called after each prewarmed motion with the number of prewarm motions ready. */
  onPrewarmed?: (ready: number) => void;
};
export function AvatarCanvas(props: Props) {
  const mount = useRef<HTMLDivElement>(null),
    latest = useRef(props);
  latest.current = props;
  const [status, setStatus] = useState("Chargement du personnage articulé…"),
    [ready, setReady] = useState(false);
  useEffect(() => {
    const host = mount.current;
    if (!host) return;
    let disposed = false,
      frameId = 0,
      model: THREE.Object3D | null = null,
      retarget: HumanoidRetargeter | null = null,
      idle: IdleLife | null = null,
      faceDriver: MetaHumanFaceDriver | null = null,
      performer: PerformanceLayer | null = null,
      style: EmotionStyle | null = null;
    // Interpreter-style layers: wall clock for presentation easing, motion clock for phrase-aware blinks.
    let lastWall = -1, lastMotionTime = -1, emotionLevel = 0, emotionLabel: string | undefined;
    // Channel values in ARKit names; rigs without blendshapes read them via faceDriver.
    const faceWeights: Record<string, number> = {};
    const morphs: THREE.Mesh[] = [];
    let morphGain: Record<string, number> | undefined;
    let head: THREE.Bone | undefined;
    let bindHead: THREE.Quaternion | undefined;
    let redraw = true;
    let lastMotion: MotionFile | null = null;
    let lastTime = -1;
    let lastFace: Props["faceBlendshapes"];
    let lastHead: Props["headRotation"];
    const scene = new THREE.Scene();
    const backdrop = createCafeBackdrop();
    scene.background = backdrop;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false,
        preserveDrawingBuffer: true,
      });
    } catch {
      backdrop.dispose();
      setStatus("WebGL est indisponible dans ce navigateur.");
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x6b4232);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    host.appendChild(renderer.domElement);
    const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 30);
    camera.position.set(0, 1.43, 1.95);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.4, 0.02);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.minDistance = 1.4;
    controls.maxDistance = 4;
    controls.maxPolarAngle = 2;
    controls.update();
    // Café lighting: warm tungsten key, soft fill, amber rims to lift hair and
    // shoulders off the wall. Faces and hands stay evenly lit for legibility.
    scene.add(new THREE.HemisphereLight(0xffe8d2, 0x4a2c20, 1.7));
    let keyLight: THREE.DirectionalLight | undefined;
    for (const [x, y, z, power, color] of [
      [-1.8, 2.6, 3.2, 3.1, 0xffd9b0],
      [2.2, 1.6, 2.2, 1.2, 0xfff0e2],
      [1.3, 2.4, -2.4, 3.4, 0xffb46e],
      [-1.5, 2.1, -2.2, 2.0, 0xffc893],
    ]) {
      const light = new THREE.DirectionalLight(color, power);
      light.position.set(x, y, z);
      scene.add(light);
      keyLight ??= light;
      if (light === keyLight) scene.add(light.target);
    }
    // Rendering polish: soft key-light shadows, ambient occlusion, light bloom, SMAA (renderPolish.ts).
    enableSoftShadows(renderer, keyLight!);
    const polished = createComposer(renderer, scene, camera);
    const resize = () => {
      if (!host.clientWidth || !host.clientHeight) return;
      renderer.setSize(host.clientWidth, host.clientHeight);
      polished.setSize(host.clientWidth, host.clientHeight);
      camera.aspect = host.clientWidth / host.clientHeight;
      if (latest.current.fitNarrowViewport) {
        const halfFov = THREE.MathUtils.degToRad(15);
        camera.fov = THREE.MathUtils.radToDeg(
          2 * Math.atan(Math.tan(halfFov) * Math.max(1, 1.2 / camera.aspect)),
        );
      }
      camera.updateProjectionMatrix();
      redraw = true;
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();
    const disposeModel = (root: THREE.Object3D) => {
      const materials = new Set<THREE.Material>();
      const textures = new Set<THREE.Texture>();
      root.traverse((o) => {
        if (o instanceof THREE.Mesh) {
          o.geometry.dispose();
          for (const m of Array.isArray(o.material)
            ? o.material
            : [o.material]) {
            materials.add(m);
            for (const v of Object.values(m))
              if (v instanceof THREE.Texture) textures.add(v);
          }
        }
      });
      materials.forEach((m) => m.dispose());
      textures.forEach((t) => t.dispose());
    };
    new GLTFLoader().load(
      "/api/avatar/model",
      (gltf) => {
        if (disposed) {
          disposeModel(gltf.scene);
          return;
        }
        model = gltf.scene;
        polishMaterials(model);
        scene.add(model);
        model.updateMatrixWorld(true);
        try {
          retarget = new HumanoidRetargeter(model);
        } catch {
          setStatus(
            "Le squelette du modèle ne possède pas les articulations requises.",
          );
          return;
        }
        const profile = retarget.profile;
        morphGain = profile.morphGain;
        idle = new IdleLife(model, profile);
        // Interpreter performance (twist bones, springs, moving holds, hair) and emotion style (posture blendspace).
        // The retargeter's hand limits are re-applied after the springs, so no layer can bend a finger out of range.
        performer = new PerformanceLayer(model, profile, retarget.hands);
        style = new EmotionStyle(model, profile);
        const currentStyle = style;
        fetch("/api/avatar/emotion-idles")
          .then((r) => (r.ok ? r.json() : []))
          .then((list: { label: string; url: string }[]) => currentStyle.loadIdles(list.map((i) => ({ ...i, url: "/api" + i.url }))))
          .catch(() => undefined);
        const headName = rigBoneName(profile, "Head");
        model.traverse((o) => {
          if (o instanceof THREE.Bone && o.name === headName) {
            head = o;
            bindHead = o.quaternion.clone();
          }
          if (o instanceof THREE.Mesh) {
            o.frustumCulled = false;
            if (o.morphTargetDictionary) morphs.push(o);
          }
        });
        // MetaHuman without ARKit shapes (v1 conversion): drive jaw/lids/brows/gaze through FACIAL_* joints instead.
        // With baked ARKit shapes (v2) the joints stay at rest, else every expression would be applied twice.
        faceDriver = morphs.some((m) => m.morphTargetDictionary?.jawOpen !== undefined) ? null : MetaHumanFaceDriver.create(model);
        setReady(true);
        redraw = true;
        setStatus("");
        latest.current.onCanvasReady?.(renderer.domElement);
      },
      undefined,
      () => {
        if (!disposed)
          setStatus(
            "Personnage 3D indisponible. Lancez scripts/setup-avatar.py puis rechargez.",
          );
      },
    );
    // Values set this frame, so the emotion layer can add on top of captured/base weights.
    const applied = new Map<string, number>();
    let emotionWeights: Record<string, number> = {};
    let lastEmotionAt: number | undefined;
    function morph(name: string, value: number) {
      faceWeights[name] = THREE.MathUtils.clamp(value, 0, 1);
      // faceWeights keep the requested weight (the v1 joint driver reads them); the mesh gets the rig's display gain.
      const shown = THREE.MathUtils.clamp(value, 0, 1) * (morphGain?.[name] ?? 1);
      applied.set(name, THREE.MathUtils.clamp(value, 0, 1));
      for (const mesh of morphs) {
        const i = mesh.morphTargetDictionary?.[name];
        if (i !== undefined && mesh.morphTargetInfluences)
          mesh.morphTargetInfluences[i] = shown;
      }
    }
    function render() {
      if (disposed) return;
      frameId = requestAnimationFrame(render);
      const {
        motion,
        currentTimeMs,
        expressiveModulation,
        faceBlendshapes,
        headRotation,
        emotion,
        emotionTimeline,
        emotionIntensity,
        emotionClock,
        renderOnChange,
      } = latest.current;
      applied.clear();
      // Film mode: derive upcoming motions now, so playback never stalls on a first use.
      const queue = latest.current.prewarm;
      if (retarget && queue?.length) {
        const pending = queue.find((m) => !retarget!.isPrepared(m));
        if (pending) {
          retarget.prime(pending);
          redraw = true;
          latest.current.onPrewarmed?.(queue.filter((m) => retarget!.isPrepared(m)).length);
        }
      }
      const cameraMoved = controls.update();
      if (renderOnChange && !redraw && !cameraMoved && lastMotion === motion && lastTime === currentTimeMs && lastFace === faceBlendshapes && lastHead === headRotation) return;
      redraw = false;
      lastMotion = motion;
      lastTime = currentTimeMs;
      lastFace = faceBlendshapes;
      lastHead = headRotation;
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
        retarget?.apply(joints);
      }
      if (motion && host) {
        const tracking = retarget?.applyMotion(motion, currentTimeMs);
        host.dataset.handTracking = JSON.stringify(tracking);
        host.dataset.rigDiagnostics = JSON.stringify(retarget?.diagnostics);
      }
      // Interpreter style on top of the retargeted pose: emotion posture, then springs / twist / holds / hair.
      const nowSec = performance.now() / 1000;
      const wallDt = lastWall < 0 ? 0 : Math.min(0.1, nowSec - lastWall);
      lastWall = nowSec;
      const feeling = motion?.generation?.emotion;
      if (feeling && feeling.label !== "neutral") emotionLabel = feeling.label;
      emotionLevel += ((feeling && feeling.label !== "neutral" ? feeling.confidence : 0) - emotionLevel) * Math.min(1, wallDt * 2);
      if (model) {
        style?.apply(nowSec, emotionLabel, emotionLevel, wallDt);
        performer?.update(motion ? currentTimeMs : undefined, Boolean(motion));
      }
      // Phrase-aware blinks: at the end of each grammatical clause span, during normal playback.
      if (motion?.nonmanual && lastMotionTime >= 0 && currentTimeMs > lastMotionTime && currentTimeMs - lastMotionTime < 250)
        for (const span of motion.nonmanual) if (span.end_ms > lastMotionTime && span.end_ms <= currentTimeMs) idle?.requestBlink(nowSec);
      lastMotionTime = motion ? currentTimeMs : -1;
      // Display-only life layer on top of the reset-every-frame retargeted pose.
      const life = renderOnChange ? undefined : idle?.update(nowSec, Boolean(motion));
      morph("eyeBlinkLeft", life?.blink ?? 0);
      morph("eyeBlinkRight", life?.blink ?? 0);
      const gx = life?.gaze.x ?? 0, gy = life?.gaze.y ?? 0;
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
      emotionWeights = smoothWeights(emotionWeights, emotionTarget, lastEmotionAt === undefined ? 1e6 : now - lastEmotionAt);
      lastEmotionAt = now;
      const negative = ["mouthFrownLeft", "mouthPressLeft", "noseSneerLeft", "mouthStretchLeft"]
        .reduce((sum, key) => sum + (emotionWeights[key] ?? 0), 0);
      const restSmile = 0.22 * (1 - THREE.MathUtils.clamp(negative * 2, 0, 1));
      morph(
        "jawOpen",
        face && Number.isFinite(face.mouth_open) ? face.mouth_open * 2 : 0,
      );
      for (const side of ["Left", "Right"]) {
        const key = side === "Left" ? "brow_left" : "brow_right";
        const brow = face?.[key];
        const weights = browWeights(brow, face?.[`${key}_neutral`] ?? calibration?.[key], browLimit);
        morph(
          `browOuterUp${side}`,
          brow !== undefined ? weights.up : 0,
        );
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
      for (const [name, value] of Object.entries(emotionFace(emotionLabel, emotionLevel, grammar)))
        morph(name, Math.min(1, (faceWeights[name] ?? 0) + value));
      // Additive on top of captured/base values; every emotion key is rewritten each frame.
      for (const key of EMOTION_KEYS) morph(key, (applied.get(key) ?? 0) + (emotionWeights[key] ?? 0));
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
      faceDriver?.apply(faceWeights);
      if (engine && head && (engine.nonmanual.headYaw || engine.nonmanual.headPitch))
        turnHead(head, engine.nonmanual.headYaw, engine.nonmanual.headPitch);
      if (head?.parent && bindHead && headRotation?.length === 9 && headRotation.every(Number.isFinite)) {
        const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = headRotation;
        const rotation = new THREE.Matrix4().set(xx, xy, xz, 0, yx, yy, yz, 0, zx, zy, zz, 0, 0, 0, 0, 1);
        const parent = head.parent.getWorldQuaternion(new THREE.Quaternion());
        const delta = parent.clone().invert().multiply(new THREE.Quaternion().setFromRotationMatrix(rotation)).multiply(parent);
        head.quaternion.copy(bindHead).premultiply(delta);
        head.updateWorldMatrix(false, true);
      }
      // Emphasis nod: additive head pitch on stressed signs (<= NOD_MAX_DEG), undone after the
      // draw so it never accumulates into the retargeted pose.
      let nodRestore: THREE.Quaternion | undefined;
      if (expressiveModulation && motion?.frames.length && head?.parent) {
        const { lo, next, alpha } = frameInterval(motion.frames, currentTimeMs);
        const nod = (motion.frames[lo].nod ?? 0) * (1 - alpha) + (motion.frames[next].nod ?? 0) * alpha;
        if (nod > 1e-3) {
          nodRestore = head.quaternion.clone();
          const parent = head.parent.getWorldQuaternion(new THREE.Quaternion());
          const lateral = new THREE.Vector3(1, 0, 0).applyQuaternion(parent.invert());
          head.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(lateral, THREE.MathUtils.degToRad(NOD_MAX_DEG) * Math.min(1, nod)));
          head.updateWorldMatrix(false, true);
        }
      }
      polished.composer.render();
      if (nodRestore && head) {
        head.quaternion.copy(nodRestore);
        head.updateWorldMatrix(false, true);
      }
    }
    render();
    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      observer.disconnect();
      controls.dispose();
      if (model) disposeModel(model);
      backdrop.dispose();
      polished.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);
  return (
    <div className="avatar-stage" data-avatar-ready={ready}>
      {status && (
        <span role="status" className="avatar-loading">
          {status}
        </span>
      )}
      <div ref={mount} aria-label="Avatar 3D articulé">
        <span className="hand-tracking-note" aria-live="off" />
      </div>
    </div>
  );
}
