import React, { useEffect, useRef, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { MotionFile } from '../../types/contract';
import { interpolateJointPositions, InterpolatedJoints } from '../../utils/motionInterpolator';

interface AvatarCanvasProps {
  motion: MotionFile | null;
  currentTimeMs: number;
  expressiveModulation?: boolean;
  energyFactor?: number;
  showSkeletonDebug?: boolean;
  onResetView?: () => void;
}

// Upper-body bone hierarchy for SignaCulture v1 (head, neck, hips, shoulders, elbows, wrists)
// Explicit: No finger data available in current fixtures; body-only prototype.
const BONE_CONNECTIONS: Array<[string, string]> = [
  ['hips', 'neck'],
  ['neck', 'head'],
  ['neck', 'left_shoulder'],
  ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'],
  ['neck', 'right_shoulder'],
  ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'],
];

// Default camera parameters
const DEFAULT_CAMERA_POS = new THREE.Vector3(0, 1.3, 2.2);
const DEFAULT_CAMERA_TARGET = new THREE.Vector3(0, 1.25, 0);

export const AvatarCanvas: React.FC<AvatarCanvasProps> = ({
  motion,
  currentTimeMs,
  expressiveModulation = false,
  energyFactor = 0.5,
  showSkeletonDebug = false,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  // Mesh and resources registries for proper disposal
  const jointMeshesRef = useRef<Map<string, THREE.Mesh>>(new Map());
  const boneMeshesRef = useRef<Map<string, THREE.Mesh>>(new Map());
  const torsoMeshRef = useRef<THREE.Mesh | null>(null);
  const currentJointsRef = useRef<InterpolatedJoints>({});
  const disposablesRef = useRef<Array<THREE.BufferGeometry | THREE.Material>>([]);

  // Setup Three.js scene
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 500;

    // 1. Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0f19); // Rich dark slate
    sceneRef.current = scene;

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 50);
    camera.position.copy(DEFAULT_CAMERA_POS);
    cameraRef.current = camera;

    // 3. Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.copy(DEFAULT_CAMERA_TARGET);
    controls.minDistance = 0.8;
    controls.maxDistance = 6.0;
    controls.maxPolarAngle = Math.PI / 2 + 0.1; // Don't go below floor
    controlsRef.current = controls;

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0xf1f5f9, 0.7);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.4);
    keyLight.position.set(1.5, 3.0, 2.5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 1024;
    keyLight.shadow.mapSize.height = 1024;
    keyLight.shadow.bias = -0.001;
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x38bdf8, 1.0); // Cyan rim light
    rimLight.position.set(-2.0, 2.5, -2.0);
    scene.add(rimLight);

    const fillLight = new THREE.DirectionalLight(0xa78bfa, 0.5); // Lavender fill light
    fillLight.position.set(-1.5, 1.5, 2.0);
    scene.add(fillLight);

    // 6. Ground Studio Grid
    const grid = new THREE.GridHelper(5, 20, 0x6366f1, 0x1e293b);
    grid.position.y = 0.0;
    scene.add(grid);
    disposablesRef.current.push(grid.geometry);
    if (Array.isArray(grid.material)) {
      disposablesRef.current.push(...grid.material);
    } else {
      disposablesRef.current.push(grid.material);
    }

    // Floor disc shadow
    const floorGeo = new THREE.CircleGeometry(1.5, 32);
    const floorMat = new THREE.MeshBasicMaterial({ color: 0x05070f, depthWrite: false });
    const floorDisc = new THREE.Mesh(floorGeo, floorMat);
    floorDisc.rotation.x = -Math.PI / 2;
    floorDisc.position.y = 0.001;
    scene.add(floorDisc);
    disposablesRef.current.push(floorGeo, floorMat);

    // 7. Avatar Materials
    const jointMaterial = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      roughness: 0.25,
      metalness: 0.4,
      emissive: 0x0369a1,
      emissiveIntensity: 0.2,
    });
    const boneMaterial = new THREE.MeshStandardMaterial({
      color: 0x818cf8,
      roughness: 0.35,
      metalness: 0.2,
    });
    const headMaterial = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      roughness: 0.2,
      metalness: 0.3,
      emissive: 0x0284c7,
      emissiveIntensity: 0.25,
    });
    const torsoMaterial = new THREE.MeshStandardMaterial({
      color: 0x4f46e5,
      roughness: 0.4,
      metalness: 0.3,
      transparent: true,
      opacity: 0.85,
    });
    disposablesRef.current.push(jointMaterial, boneMaterial, headMaterial, torsoMaterial);

    // 8. Create Joint Meshes
    const jointKeys = [
      'hips',
      'neck',
      'head',
      'left_shoulder',
      'right_shoulder',
      'left_elbow',
      'right_elbow',
      'left_wrist',
      'right_wrist',
    ];

    const jointMeshes = new Map<string, THREE.Mesh>();
    for (const key of jointKeys) {
      let radius = 0.04;
      let mat = jointMaterial;
      if (key === 'head') {
        radius = 0.09;
        mat = headMaterial;
      } else if (key === 'hips') {
        radius = 0.055;
      } else if (key === 'neck') {
        radius = 0.04;
      } else if (key.includes('shoulder')) {
        radius = 0.045;
      } else if (key.includes('wrist')) {
        radius = 0.038;
      }

      const sphereGeo = new THREE.SphereGeometry(radius, 24, 24);
      disposablesRef.current.push(sphereGeo);
      const mesh = new THREE.Mesh(sphereGeo, mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.name = `joint_${key}`;
      scene.add(mesh);
      jointMeshes.set(key, mesh);
    }
    jointMeshesRef.current = jointMeshes;

    // 9. Create Bone Meshes
    const boneMeshes = new Map<string, THREE.Mesh>();
    const unitCylinder = new THREE.CylinderGeometry(0.026, 0.026, 1, 16);
    unitCylinder.translate(0, 0.5, 0); // Origin at bottom base
    disposablesRef.current.push(unitCylinder);

    for (const [from, to] of BONE_CONNECTIONS) {
      const boneKey = `${from}_to_${to}`;
      const boneMesh = new THREE.Mesh(unitCylinder, boneMaterial);
      boneMesh.castShadow = true;
      boneMesh.receiveShadow = true;
      boneMesh.name = `bone_${boneKey}`;
      scene.add(boneMesh);
      boneMeshes.set(boneKey, boneMesh);
    }
    boneMeshesRef.current = boneMeshes;

    // 10. Torso Chest Volume
    const torsoGeo = new THREE.CylinderGeometry(0.18, 0.12, 0.42, 16);
    disposablesRef.current.push(torsoGeo);
    const torsoMesh = new THREE.Mesh(torsoGeo, torsoMaterial);
    torsoMesh.castShadow = true;
    scene.add(torsoMesh);
    torsoMeshRef.current = torsoMesh;

    // 11. Render Loop
    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // 12. Resize Handler
    const handleResize = () => {
      if (!container || !rendererRef.current || !cameraRef.current) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    // Comprehensive unmount cleanup: dispose controls, geometries, materials, renderer
    return () => {
      cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();

      // Dispose controls
      controls.dispose();

      // Dispose all registered geometries and materials
      disposablesRef.current.forEach((res) => {
        try {
          res.dispose();
        } catch {
          // ignore
        }
      });
      disposablesRef.current = [];

      // Clear registries
      jointMeshesRef.current.clear();
      boneMeshesRef.current.clear();
      torsoMeshRef.current = null;

      // Dispose renderer
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Update Avatar pose whenever currentTimeMs, motion, or expressive modulation changes
  useEffect(() => {
    const interpolated = interpolateJointPositions(motion, currentTimeMs, {
      expressiveModulation,
      energyFactor,
    });
    currentJointsRef.current = interpolated;

    // 1. Update Joint Meshes
    const jointMeshes = jointMeshesRef.current;
    for (const [key, mesh] of jointMeshes.entries()) {
      const coords = interpolated[key];
      if (coords) {
        mesh.position.set(coords[0], coords[1], coords[2]);
        mesh.visible = true;
      } else {
        mesh.visible = false;
      }
    }

    // 2. Update Bones
    const boneMeshes = boneMeshesRef.current;
    const up = new THREE.Vector3(0, 1, 0);

    for (const [from, to] of BONE_CONNECTIONS) {
      const boneKey = `${from}_to_${to}`;
      const boneMesh = boneMeshes.get(boneKey);
      const pFrom = interpolated[from];
      const pTo = interpolated[to];

      if (boneMesh && pFrom && pTo) {
        const vFrom = new THREE.Vector3(...pFrom);
        const vTo = new THREE.Vector3(...pTo);
        const length = vFrom.distanceTo(vTo);

        if (length > 0.001) {
          boneMesh.position.copy(vFrom);
          boneMesh.scale.set(1, length, 1);

          const dir = new THREE.Vector3().subVectors(vTo, vFrom).normalize();
          const quat = new THREE.Quaternion().setFromUnitVectors(up, dir);
          boneMesh.quaternion.copy(quat);
          boneMesh.visible = true;
        } else {
          boneMesh.visible = false;
        }
      } else if (boneMesh) {
        boneMesh.visible = false;
      }
    }

    // 3. Update Torso Position
    if (torsoMeshRef.current && interpolated['hips'] && interpolated['neck']) {
      const vHips = new THREE.Vector3(...interpolated['hips']);
      const vNeck = new THREE.Vector3(...interpolated['neck']);
      const torsoPos = new THREE.Vector3().addVectors(vHips, vNeck).multiplyScalar(0.5);
      torsoMeshRef.current.position.copy(torsoPos);

      const length = vHips.distanceTo(vNeck);
      torsoMeshRef.current.scale.set(1, length / 0.42, 1);

      const dir = new THREE.Vector3().subVectors(vNeck, vHips).normalize();
      const quat = new THREE.Quaternion().setFromUnitVectors(up, dir);
      torsoMeshRef.current.quaternion.copy(quat);
      torsoMeshRef.current.visible = !showSkeletonDebug;
    }
  }, [motion, currentTimeMs, expressiveModulation, energyFactor, showSkeletonDebug]);

  // Reset Camera View
  const handleResetCamera = useCallback(() => {
    if (cameraRef.current && controlsRef.current) {
      cameraRef.current.position.copy(DEFAULT_CAMERA_POS);
      controlsRef.current.target.copy(DEFAULT_CAMERA_TARGET);
      controlsRef.current.update();
    }
  }, []);

  return (
    <div className="relative w-full h-full min-h-[460px] rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 shadow-2xl flex flex-col">
      {/* 3D Canvas Container */}
      <div ref={mountRef} className="w-full flex-1 cursor-grab active:cursor-grabbing outline-none" />

      {/* Explicit limitation notice banner: Body-only prototype (no finger data in fixture) */}
      <div className="absolute top-3 left-3 max-w-sm pointer-events-none z-10">
        <div className="bg-slate-900/90 border border-slate-700/60 rounded-xl px-3 py-1.5 text-[11px] text-slate-300 shadow backdrop-blur leading-tight">
          <span className="font-semibold text-amber-300 block mb-0.5">
            Prototype corps & bras uniquement :
          </span>
          Les fixtures actuelles signaculture-v1 ne comportent pas de données sur les doigts. Ne constitue pas un avatar complet de LSF.
        </div>
      </div>

      {/* Floating View Controls */}
      <div className="absolute top-3 right-3 flex items-center gap-2 z-10">
        <button
          onClick={handleResetCamera}
          title="Recentrer la caméra 3D"
          className="px-2.5 py-1 text-xs font-medium bg-slate-900/80 hover:bg-slate-800 text-slate-300 hover:text-white rounded-lg border border-slate-700/60 shadow backdrop-blur transition-colors"
        >
          Recentrer 3D
        </button>
      </div>

      {/* Skeleton / Joint Debug Coordinates Overlay */}
      {showSkeletonDebug && (
        <div className="absolute bottom-3 left-3 max-w-xs bg-slate-900/90 border border-slate-700/70 rounded-xl p-3 text-[11px] font-mono text-slate-300 backdrop-blur-md shadow-xl z-10 max-h-48 overflow-y-auto">
          <p className="font-semibold text-sky-400 mb-1 border-b border-slate-800 pb-1">
            Repère 3D SignaCulture (Y up, X right, Z fwd)
          </p>
          <div className="space-y-0.5">
            {Object.entries(currentJointsRef.current).map(([name, coords]) => (
              <div key={name} className="flex justify-between gap-2">
                <span className="text-slate-400">{name}:</span>
                <span className="text-slate-200">
                  [{coords[0].toFixed(2)}, {coords[1].toFixed(2)}, {coords[2].toFixed(2)}]
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
