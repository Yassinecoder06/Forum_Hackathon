import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { MotionFile } from "../../types/contract";
import { interpolateJointPositions } from "../../utils/motionInterpolator";

type Props = {
  motion: MotionFile | null;
  currentTimeMs: number;
  expressiveModulation?: boolean;
  energyFactor?: number;
  showSkeletonDebug?: boolean;
  emotion?: Record<string, number | string> | null;
  onCanvasReady?: (canvas: HTMLCanvasElement) => void;
};
const CONNECTIONS: [string, string, number, "shirt" | "skin"][] = [
  ["left_shoulder", "left_elbow", 0.062, "shirt"],
  ["right_shoulder", "right_elbow", 0.062, "shirt"],
  ["left_elbow", "left_wrist", 0.039, "skin"],
  ["right_elbow", "right_wrist", 0.039, "skin"],
];
const FINGERS = ["thumb", "index", "middle", "ring", "pinky"];
const up = new THREE.Vector3(0, 1, 0);

export function AvatarCanvas(props: Props) {
  const mount = useRef<HTMLDivElement>(null);
  const latest = useRef(props);
  latest.current = props;
  useEffect(() => {
    const host = mount.current;
    if (!host) return;
    const scene = new THREE.Scene();
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0xeff4e8, 0);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.35;
    host.appendChild(renderer.domElement);
    latest.current.onCanvasReady?.(renderer.domElement);
    const camera = new THREE.PerspectiveCamera(31, 1, 0.01, 30);
    camera.position.set(0.1, 1.42, 2.35);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.42, 0);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.minDistance = 1.6;
    controls.maxDistance = 4;
    controls.minPolarAngle = 0.5;
    controls.maxPolarAngle = 2;
    controls.update();
    scene.add(new THREE.HemisphereLight(0xffffff, 0xbdcba9, 3.2));
    const key = new THREE.DirectionalLight(0xfff5e7, 3.5);
    key.position.set(-2, 4, 4);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.bias = -0.0005;
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xdcecd3, 1.7);
    fill.position.set(2, 2, -1);
    scene.add(fill);
    const skin = new THREE.MeshStandardMaterial({
      color: 0xe7b99b,
      roughness: 0.85,
    });
    const shirt = new THREE.MeshStandardMaterial({
      color: 0x638976,
      roughness: 0.95,
    });
    const hair = new THREE.MeshStandardMaterial({
      color: 0x423a33,
      roughness: 0.95,
    });
    const dark = new THREE.MeshStandardMaterial({
      color: 0x3e3b32,
      roughness: 0.8,
    });
    const lip = new THREE.MeshStandardMaterial({
      color: 0x9d6355,
      roughness: 0.9,
    });
    const mesh = (
      geometry: THREE.BufferGeometry,
      material: THREE.Material,
      parent: THREE.Object3D = scene,
    ) => {
      const m = new THREE.Mesh(geometry, material);
      m.castShadow = true;
      m.receiveShadow = true;
      parent.add(m);
      return m;
    };
    const sphere = new THREE.SphereGeometry(1, 24, 18);
    const cylinder = new THREE.CylinderGeometry(1, 1, 1, 20);
    const limbs = CONNECTIONS.map(([a, b, r, mat]) => ({
      a,
      b,
      r,
      mesh: mesh(cylinder, mat === "shirt" ? shirt : skin),
    }));
    const shoulders = [mesh(sphere, shirt), mesh(sphere, shirt)];
    const elbows = [mesh(sphere, skin), mesh(sphere, skin)];
    const torso = mesh(new THREE.CylinderGeometry(0.205, 0.135, 1, 32), shirt);
    const neck = mesh(cylinder, skin);
    const collar = mesh(new THREE.TorusGeometry(0.068, 0.013, 8, 32), shirt);
    collar.rotation.x = Math.PI / 2;
    const head = new THREE.Group();
    scene.add(head);
    const skull = mesh(new THREE.SphereGeometry(0.115, 40, 32), skin, head);
    skull.scale.set(1, 1.27, 0.91);
    const hairCap = mesh(
      new THREE.SphereGeometry(
        0.117,
        32,
        20,
        0,
        Math.PI * 2,
        0,
        Math.PI * 0.46,
      ),
      hair,
      head,
    );
    hairCap.scale.set(1, 1.3, 0.93);
    hairCap.position.y = 0.006;
    for (const sign of [-1, 1]) {
      const ear = mesh(sphere, skin, head);
      ear.scale.set(0.018, 0.032, 0.018);
      ear.position.set(sign * 0.112, -0.005, 0);
      const eye = mesh(sphere, dark, head);
      eye.scale.set(0.008, 0.011, 0.005);
      eye.position.set(sign * 0.041, 0.015, 0.099);
    }
    const nose = mesh(sphere, skin, head);
    nose.scale.set(0.015, 0.023, 0.022);
    nose.position.set(0, -0.006, 0.103);
    const mouth = mesh(
      new THREE.TubeGeometry(
        new THREE.QuadraticBezierCurve3(
          new THREE.Vector3(-0.029, -0.041, 0.104),
          new THREE.Vector3(0, -0.055, 0.112),
          new THREE.Vector3(0.029, -0.041, 0.104),
        ),
        16,
        0.0022,
        6,
        false,
      ),
      lip,
      head,
    );
    const brows = [-1, 1].map((sign) => {
      const brow = mesh(
        new THREE.CapsuleGeometry(0.0025, 0.027, 3, 8),
        hair,
        head,
      );
      brow.rotation.z = Math.PI / 2;
      brow.position.set(sign * 0.042, 0.041, 0.102);
      return brow;
    });
    const palms = new Map<string, THREE.Mesh>();
    const fingers: { a: string; b: string; mesh: THREE.Mesh }[] = [];
    for (const side of ["left", "right"]) {
      const palm = mesh(sphere, skin);
      palms.set(side, palm);
      for (const finger of FINGERS) {
        for (let i = 1; i <= 4; i++) {
          const a = i === 1 ? `${side}_wrist` : `${side}_${finger}_${i - 1}`;
          const b = `${side}_${finger}_${i}`;
          fingers.push({ a, b, mesh: mesh(cylinder, skin) });
          const joint = mesh(sphere, skin);
          joint.scale.setScalar(0.007);
          joint.visible = false;
          fingers.push({ a: b, b, mesh: joint });
        }
      }
    }
    const ground = mesh(
      new THREE.CircleGeometry(0.6, 64),
      new THREE.ShadowMaterial({ color: 0x45622f, opacity: 0.12 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = 0.78;
    ground.castShadow = false;
    const resize = () => {
      const w = host.clientWidth,
        h = host.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();
    const p0 = new THREE.Vector3(),
      p1 = new THREE.Vector3(),
      dir = new THREE.Vector3();
    function placeBone(
      m: THREE.Mesh,
      a: number[],
      b: number[],
      radius: number,
    ) {
      p0.fromArray(a);
      p1.fromArray(b);
      dir.subVectors(p1, p0);
      const length = dir.length();
      m.visible = length > 0.0001;
      if (!m.visible) return;
      m.position.copy(p0).add(p1).multiplyScalar(0.5);
      m.quaternion.setFromUnitVectors(up, dir.normalize());
      m.scale.set(radius, length, radius);
    }
    let id = 0;
    function render() {
      id = requestAnimationFrame(render);
      const {
        motion,
        currentTimeMs,
        expressiveModulation,
        emotion,
        energyFactor,
      } = latest.current;
      const joints = interpolateJointPositions(motion, currentTimeMs);
      for (const item of limbs) {
        const a = joints[item.a],
          b = joints[item.b];
        item.mesh.visible = !!(a && b);
        if (a && b) placeBone(item.mesh, a, b, item.r);
      }
      ["left", "right"].forEach((side, i) => {
        const shoulder = joints[`${side}_shoulder`],
          elbow = joints[`${side}_elbow`],
          wrist = joints[`${side}_wrist`];
        shoulders[i].visible = !!shoulder;
        elbows[i].visible = !!elbow;
        if (shoulder) {
          shoulders[i].position.fromArray(shoulder);
          shoulders[i].scale.setScalar(0.062);
        }
        if (elbow) {
          elbows[i].position.fromArray(elbow);
          elbows[i].scale.setScalar(0.037);
        }
        const palm = palms.get(side)!;
        palm.visible = !!wrist;
        if (wrist) {
          palm.position.fromArray(wrist);
          palm.scale.set(0.037, 0.052, 0.019);
          const index = joints[`${side}_middle_1`];
          const ref =
            index ||
            (elbow
              ? [
                  wrist[0] + (wrist[0] - elbow[0]) * 0.2,
                  wrist[1] + (wrist[1] - elbow[1]) * 0.2,
                  wrist[2] + (wrist[2] - elbow[2]) * 0.2,
                ]
              : [wrist[0], wrist[1] - 0.05, wrist[2]]);
          dir.fromArray(ref).sub(palm.position);
          if (dir.lengthSq() > 0.00001)
            palm.quaternion.setFromUnitVectors(up, dir.normalize());
        }
      });
      for (const f of fingers) {
        const a = joints[f.a],
          b = joints[f.b];
        f.mesh.visible = !!(a && b);
        if (!a || !b) continue;
        if (f.a === f.b) f.mesh.position.fromArray(a);
        else placeBone(f.mesh, a, b, 0.006);
      }
      const hips = joints.hips,
        top = joints.neck,
        nosePoint = joints.head;
      if (hips && top) {
        placeBone(torso, hips, top, 1);
        const length = new THREE.Vector3(...top).distanceTo(
          new THREE.Vector3(...hips),
        );
        torso.scale.set(1, length, 1);
        collar.position.fromArray(top);
        placeBone(neck, top, [top[0], top[1] + 0.1, top[2]], 0.053);
      }
      if (nosePoint) {
        head.position.fromArray(nosePoint);
        head.position.y += 0.025;
      }
      // Model hypotheses only affect the experimental facial layer. Hand and arm
      // trajectories remain measured; there is no arbitrary limb scaling.
      const joy = expressiveModulation ? Number(emotion?.happy || 0) : 0;
      const sadness = expressiveModulation ? Number(emotion?.sad || 0) : 0;
      mouth.scale.y = 1 + joy * 0.6 - sadness * 0.4;
      brows[0].rotation.z =
        Math.PI / 2 +
        (expressiveModulation ? Number(emotion?.angry || 0) * 0.25 : 0);
      brows[1].rotation.z =
        Math.PI / 2 -
        (expressiveModulation ? Number(emotion?.angry || 0) * 0.25 : 0);
      // Facial capture ratios drive mouth opening when actual measurements exist.
      const face = motion?.frames.find(
        (f, i) =>
          f.time_ms <= currentTimeMs &&
          (!motion.frames[i + 1] ||
            motion.frames[i + 1].time_ms > currentTimeMs),
      )?.face;
      mouth.scale.y = face?.mouth_open
        ? 1 + Math.min(2, face.mouth_open * 4)
        : mouth.scale.y;
      key.intensity =
        3.5 + (expressiveModulation ? (energyFactor || 0) * 0.12 : 0);
      controls.update();
      renderer.render(scene, camera);
    }
    render();
    return () => {
      cancelAnimationFrame(id);
      observer.disconnect();
      controls.dispose();
      const geometries = new Set<THREE.BufferGeometry>(),
        materials = new Set<THREE.Material>();
      scene.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          geometries.add(obj.geometry);
          for (const mat of Array.isArray(obj.material)
            ? obj.material
            : [obj.material])
            materials.add(mat);
        }
      });
      geometries.forEach((g) => g.dispose());
      materials.forEach((m) => m.dispose());
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);
  return (
    <div className="avatar-stage">
      <div ref={mount} aria-label="Avatar 3D interactif" />
      <span className="avatar-view-note">
        Glissez pour explorer · Faites défiler pour zoomer
      </span>
    </div>
  );
}
