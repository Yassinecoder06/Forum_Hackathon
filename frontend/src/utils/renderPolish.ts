import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { GTAOPass } from "three/examples/jsm/postprocessing/GTAOPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { SMAAPass } from "three/examples/jsm/postprocessing/SMAAPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";

/**
 * Rendering polish for a believable presenter (display only):
 * - skin: MeshPhysicalMaterial with a warm "sheen" (cheap stand-in for subsurface scattering at grazing angles),
 *   slightly lower specular; eyes: clearcoat for a wet cornea; hair: alpha-tested cards keep their material;
 * - soft shadows from the key light, so arms and hands shade the body (depth reads much better);
 * - post-processing: ground-truth ambient occlusion (contact darkening between fingers, under the chin),
 *   a very light bloom on highlights, SMAA anti-aliasing, then the usual tone mapping / sRGB output.
 */
export function polishMaterials(model: THREE.Object3D) {
  model.traverse((o) => {
    if (!(o instanceof THREE.Mesh)) return;
    o.castShadow = true;
    o.receiveShadow = true;
    const list = Array.isArray(o.material) ? o.material : [o.material];
    const out = list.map((m) => {
      if (!(m instanceof THREE.MeshStandardMaterial) || m instanceof THREE.MeshPhysicalMaterial) return m;
      const name = m.name.toLowerCase();
      const skin = /body|face|head|skin/.test(name), eye = /eye(?!lash)/.test(name);
      if (!skin && !eye) return m;
      const p = new THREE.MeshPhysicalMaterial();
      THREE.MeshStandardMaterial.prototype.copy.call(p, m);   // keep maps, skinning/morph flags, colour
      if (skin) {
        p.sheen = 0.35;
        p.sheenColor = new THREE.Color(0.85, 0.42, 0.36);
        p.sheenRoughness = 0.55;
        p.specularIntensity = 0.6;
      }
      if (eye) {
        p.clearcoat = 1;
        p.clearcoatRoughness = 0.04;
      }
      m.dispose();
      return p;
    });
    o.material = Array.isArray(o.material) ? out : out[0];
  });
}

export function enableSoftShadows(renderer: THREE.WebGLRenderer, key: THREE.DirectionalLight) {
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  const cam = key.shadow.camera as THREE.OrthographicCamera;
  cam.left = -0.6; cam.right = 0.6; cam.top = 0.6; cam.bottom = -0.6; cam.near = 0.5; cam.far = 8;
  key.target.position.set(0, 1.3, 0);
  key.shadow.bias = -0.0004;
  key.shadow.normalBias = 0.02;
  key.shadow.radius = 4;
}

export type Polished = { composer: EffectComposer; setSize: (w: number, h: number) => void; dispose: () => void };

export function createComposer(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera): Polished {
  const size = renderer.getSize(new THREE.Vector2());
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const ao = new GTAOPass(scene, camera, size.x, size.y);
  ao.updateGtaoMaterial({ radius: 0.12, distanceExponent: 1.4, thickness: 1, scale: 1 });
  ao.blendIntensity = 0.75;
  composer.addPass(ao);
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(size.x, size.y), 0.12, 0.4, 0.92));
  composer.addPass(new OutputPass());
  const smaa = new SMAAPass(size.x * renderer.getPixelRatio(), size.y * renderer.getPixelRatio());
  composer.addPass(smaa);
  return {
    composer,
    setSize(w, h) {
      composer.setSize(w, h);
      ao.setSize(w, h);
    },
    dispose() {
      composer.dispose();
    },
  };
}
