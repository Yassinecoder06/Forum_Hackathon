import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { MotionFile } from "../../types/contract";
import { interpolateJointPositions } from "../../utils/motionInterpolator";
import { HumanoidRetargeter } from "../../utils/retargetHumanoid";
import { browWeights, faceCalibration, sampleFace } from "../../utils/avatarPlayback";
import { IdleLife } from "../../utils/idleLife";
import { createCafeBackdrop } from "../../utils/stageBackdrop";
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
  onCanvasReady?: (canvas: HTMLCanvasElement) => void;
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
      idle: IdleLife | null = null;
    const morphs: THREE.Mesh[] = [];
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
    for (const [x, y, z, power, color] of [
      [-1.8, 2.6, 3.2, 3.1, 0xffd9b0],
      [2.2, 1.6, 2.2, 1.2, 0xfff0e2],
      [1.3, 2.4, -2.4, 3.4, 0xffb46e],
      [-1.5, 2.1, -2.2, 2.0, 0xffc893],
    ]) {
      const light = new THREE.DirectionalLight(color, power);
      light.position.set(x, y, z);
      scene.add(light);
    }
    const resize = () => {
      if (!host.clientWidth || !host.clientHeight) return;
      renderer.setSize(host.clientWidth, host.clientHeight);
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
        idle = new IdleLife(model);
        model.traverse((o) => {
          if (o instanceof THREE.Bone && o.name === "Head") {
            head = o;
            bindHead = o.quaternion.clone();
          }
          if (o instanceof THREE.Mesh) {
            o.frustumCulled = false;
            if (o.morphTargetDictionary) morphs.push(o);
          }
        });
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
    function morph(name: string, value: number) {
      for (const mesh of morphs) {
        const i = mesh.morphTargetDictionary?.[name];
        if (i !== undefined && mesh.morphTargetInfluences)
          mesh.morphTargetInfluences[i] = THREE.MathUtils.clamp(value, 0, 1);
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
        renderOnChange,
      } = latest.current;
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
        const note = host.querySelector<HTMLElement>(".hand-tracking-note");
        if (note) {
          // Warn only for real tracking gaps (hand held or faded to rest), not for the planned
          // bridges between signs or sub-180 ms interpolation, which would make it flicker constantly.
          const uncertain = tracking && Object.values(tracking).some((s) => s === "held" || s === "untracked");
          note.textContent = uncertain ? "Suivi des mains incomplet · pose estimée à vérifier" : "";
        }
      }
      // Display-only life layer on top of the reset-every-frame retargeted pose.
      const life = renderOnChange ? undefined : idle?.update(performance.now() / 1000, Boolean(motion));
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
      const useEmotionMorphs = expressiveModulation && !motion;
      const joy = useEmotionMorphs ? Number(emotion?.happy || 0) : 0,
        sad = useEmotionMorphs ? Number(emotion?.sad || 0) : 0,
        angry = useEmotionMorphs ? Number(emotion?.angry || 0) : 0;
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
          brow !== undefined ? weights.up : joy * 0.15,
        );
        morph(`browDown${side}`, brow !== undefined ? weights.down : angry * 0.2);
        // A soft welcoming smile only at rest; never added over signed content.
        morph(`mouthSmile${side}`, face ? 0 : Math.max(joy * 0.2, motion ? 0 : 0.22));
        morph(`mouthFrown${side}`, face ? 0 : sad * 0.15);
      }
      morph("browInnerUp", face ? (browWeights(face.brow_left, face.brow_left_neutral ?? calibration?.brow_left, browLimit).up + browWeights(face.brow_right, face.brow_right_neutral ?? calibration?.brow_right, browLimit).up) * .2 : 0);
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
      if (head?.parent && bindHead && headRotation?.length === 9 && headRotation.every(Number.isFinite)) {
        const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = headRotation;
        const rotation = new THREE.Matrix4().set(xx, xy, xz, 0, yx, yy, yz, 0, zx, zy, zz, 0, 0, 0, 0, 1);
        const parent = head.parent.getWorldQuaternion(new THREE.Quaternion());
        const delta = parent.clone().invert().multiply(new THREE.Quaternion().setFromRotationMatrix(rotation)).multiply(parent);
        head.quaternion.copy(bindHead).premultiply(delta);
        head.updateWorldMatrix(false, true);
      }
      renderer.render(scene, camera);
    }
    render();
    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      observer.disconnect();
      controls.dispose();
      if (model) disposeModel(model);
      backdrop.dispose();
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
      <span className="avatar-view-note">
        {props.avatarCredit ?? "Avatar"} · UPF-GTI / Ready Player Me
      </span>
    </div>
  );
}
