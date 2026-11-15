import * as THREE from "three";
export type Capsule = { a: THREE.Vector3; b: THREE.Vector3; sourceA: THREE.Vector3; sourceB: THREE.Vector3 };

/** Closest points of bounded segments, including parallel/zero-length segments. */
export function segmentDistance(a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3, d: THREE.Vector3) {
  const u = b.clone().sub(a), v = d.clone().sub(c), w = a.clone().sub(c);
  const aa = u.dot(u), bb = u.dot(v), cc = v.dot(v), dd = u.dot(w), ee = v.dot(w);
  let s = 0, t = 0;
  if (aa < 1e-12 && cc < 1e-12) return { distance: a.distanceTo(c), left: a.clone(), right: c.clone() };
  if (aa < 1e-12) t = THREE.MathUtils.clamp(ee / cc, 0, 1);
  else if (cc < 1e-12) s = THREE.MathUtils.clamp(-dd / aa, 0, 1);
  else {
    const denominator = aa * cc - bb * bb;
    s = denominator > 1e-12 ? THREE.MathUtils.clamp((bb * ee - cc * dd) / denominator, 0, 1) : 0;
    t = (bb * s + ee) / cc;
    if (t < 0) { t = 0; s = THREE.MathUtils.clamp(-dd / aa, 0, 1); }
    else if (t > 1) { t = 1; s = THREE.MathUtils.clamp((bb - dd) / aa, 0, 1); }
  }
  const left = a.clone().addScaledVector(u, s), right = c.clone().addScaledVector(v, t);
  return { distance: left.distanceTo(right), left, right };
}

/** Correct only model-created intersections when the source supports separation.
 * Near source contact disables repulsion; uncertainty is not permission to rewrite a sign. */
export function handContact(left: Capsule[], right: Capsule[]) {
  const correction = new THREE.Vector3();
  let sourceDistance = Infinity, maxPenetration = 0;
  for (const l of left) for (const r of right) {
    sourceDistance = Math.min(sourceDistance, segmentDistance(l.sourceA, l.sourceB, r.sourceA, r.sourceB).distance);
    const hit = segmentDistance(l.a, l.b, r.a, r.b);
    const penetration = Math.max(0, .016 - hit.distance); // Two 8 mm finger proxies.
    if (penetration > maxPenetration) {
      maxPenetration = penetration;
      const normal = hit.left.sub(hit.right);
      if (normal.lengthSq() < 1e-6) normal.copy(l.sourceA).sub(r.sourceA);
      correction.copy(normal.normalize().multiplyScalar(penetration * .5 + .001));
    }
  }
  const contact = sourceDistance < .035;
  if (contact) correction.set(0, 0, 0);
  else correction.multiplyScalar(THREE.MathUtils.smoothstep(sourceDistance, .035, .06)).clampLength(0, .015);
  return { correction, maxPenetration, sourceDistance, contact };
}
