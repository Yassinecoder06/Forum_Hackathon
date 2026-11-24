import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { MotionFile } from "../../types/contract";
import { createCafeBackdrop } from "../../utils/stageBackdrop";
import { createComposer, enableSoftShadows } from "../../utils/renderPolish";
import type { EmotionWindow } from "../../types/contract";
import type { RoleShiftPose } from "../../utils/roleShift"; // [film-speakers]
import { AvatarLoadError, loadAvatarRig, type AvatarRig } from "./avatarRig";
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
  /** [film-speakers] Role shift toward the current speaker's locus (roleShift.ts), additive after retargeting. */
  roleShift?: RoleShiftPose | null;
  /** With prewarm: derive at most this many ms per frame (sliced, no stall). */
  prewarmBudgetMs?: number;
  /** Called with each prewarmed motion once it is ready to start without a stall. */
  onPrepared?: (motion: MotionFile) => void;
  /** Cross-fade (ms) from the shown pose whenever the motion changes; 0 = cut (default). */
  blendMs?: number;
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
      rig: AvatarRig | null = null;
    let redraw = true;
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
    // The avatar runtime (retargeting, hand limits, interpreter layers, face) is shared with the AR companion: avatarRig.ts.
    loadAvatarRig("/api/avatar/model", () => disposed)
      .then((loaded) => {
        if (!loaded) return;
        rig = loaded;
        // Measurement hook for tests only (address with ?rigdebug): the page's rig, to sample poses offline.
        if (new URLSearchParams(location.search).has("rigdebug")) (window as unknown as { __scRig?: unknown }).__scRig = loaded;
        scene.add(rig.model);
        setReady(true);
        redraw = true;
        setStatus("");
        latest.current.onCanvasReady?.(renderer.domElement);
      })
      .catch((e) => {
        if (!disposed)
          setStatus(
            e instanceof AvatarLoadError && e.kind === "skeleton"
              ? "Le squelette du modèle ne possède pas les articulations requises."
              : "Personnage 3D indisponible. Lancez scripts/setup-avatar.py puis rechargez.",
          );
      });
    function render() {
      if (disposed) return;
      frameId = requestAnimationFrame(render);
      const frame = latest.current;
      // Film mode: derive upcoming motions now, so playback never stalls on a first use.
      if (rig?.prewarm(frame)) redraw = true;
      const cameraMoved = controls.update();
      const dirty = redraw || cameraMoved;
      if (rig) {
        if (!rig.pose(frame, dirty, host ?? undefined)) return;
      } else if (frame.renderOnChange && !dirty) return;
      redraw = false;
      polished.composer.render();
      rig?.afterDraw();
    }
    render();
    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      observer.disconnect();
      controls.dispose();
      rig?.dispose();
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
