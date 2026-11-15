import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { MotionFile } from "../../types/contract";
import { interpolateJointPositions } from "../../utils/motionInterpolator";
import { HumanoidRetargeter } from "../../utils/retargetHumanoid";
import { browWeights, faceCalibration, sampleFace } from "../../utils/avatarPlayback";
type Props = {
  motion: MotionFile | null;
  currentTimeMs: number;
  expressiveModulation?: boolean;
  energyFactor?: number;
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
      retarget: HumanoidRetargeter | null = null;
    const morphs: THREE.Mesh[] = [];
    const scene = new THREE.Scene();
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0xeff3ec);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
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
    scene.add(new THREE.HemisphereLight(0xfff7ec, 0xb9c7cb, 2.2));
    for (const [x, y, z, power, color] of [
      [-2, 3, 4, 3.5, 0xfff4e5],
      [2, 2, 1, 2, 0xe1edff],
      [0, 3, -2, 3, 0xffffff],
    ]) {
      const light = new THREE.DirectionalLight(color, power);
      light.position.set(x, y, z);
      scene.add(light);
    }
    const resize = () => {
      if (!host.clientWidth || !host.clientHeight) return;
      renderer.setSize(host.clientWidth, host.clientHeight);
      camera.aspect = host.clientWidth / host.clientHeight;
      camera.updateProjectionMatrix();
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
        model.traverse((o) => {
          if (o instanceof THREE.Mesh) {
            o.frustumCulled = false;
            if (o.morphTargetDictionary) morphs.push(o);
          }
        });
        setReady(true);
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
      const { motion, currentTimeMs, expressiveModulation, emotion } =
        latest.current;
      const joints = interpolateJointPositions(motion, currentTimeMs);
      // Anatomical neutral stance; old synthetic fixtures use the legacy mirrored X convention.
      if (!motion) {
        for (const side of ["left", "right"]) {
          const sign = side === "left" ? 1 : -1;
          joints[`${side}_shoulder`] = [sign * 0.18, 1.48, 0];
          joints[`${side}_elbow`] = [sign * 0.25, 1.2, 0.04];
          joints[`${side}_wrist`] = [sign * 0.16, 1.02, 0.09];
        }
      }
      if (motion && host) {
        const tracking = retarget?.applyMotion(motion, currentTimeMs);
        host.dataset.handTracking = JSON.stringify(tracking);
        host.dataset.rigDiagnostics = JSON.stringify(retarget?.diagnostics);
        const note = host.querySelector<HTMLElement>(".hand-tracking-note");
        if (note) {
          const uncertain = tracking && Object.values(tracking).some((s) => s !== "observed");
          note.textContent = uncertain ? "Suivi des mains incomplet · pose estimée à vérifier" : "";
        }
      } else retarget?.apply(joints);
      const face = sampleFace(motion, currentTimeMs);
      const calibration = motion ? faceCalibration(motion) : undefined;
      const browLimit = .18 + .47 * THREE.MathUtils.clamp(face?.brow_reference_blend ?? (calibration ? 1 : 0), 0, 1);
      const joy = expressiveModulation ? Number(emotion?.happy || 0) : 0,
        sad = expressiveModulation ? Number(emotion?.sad || 0) : 0,
        angry = expressiveModulation ? Number(emotion?.angry || 0) : 0;
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
        morph(`mouthSmile${side}`, face ? 0 : joy * 0.2);
        morph(`mouthFrown${side}`, face ? 0 : sad * 0.15);
      }
      morph("browInnerUp", face ? (browWeights(face.brow_left, face.brow_left_neutral ?? calibration?.brow_left, browLimit).up + browWeights(face.brow_right, face.brow_right_neutral ?? calibration?.brow_right, browLimit).up) * .2 : 0);
      controls.update();
      renderer.render(scene, camera);
    }
    render();
    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      observer.disconnect();
      controls.dispose();
      if (model) disposeModel(model);
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
        Avatar Noa · UPF-GTI / Ready Player Me
      </span>
    </div>
  );
}
