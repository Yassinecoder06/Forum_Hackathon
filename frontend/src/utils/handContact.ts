import * as THREE from "three";
export type Capsule = { a: THREE.Vector3; b: THREE.Vector3; sourceA: THREE.Vector3; sourceB: THREE.Vector3; radius?: number; sourceObserved?: boolean; kind?: "hand" | "forearm" };

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
 * Source contact allows touching, not deep intersection. Untracked hands still
 * have a rendered volume and must participate in collision detection. */
export function handContact(left: Capsule[], right: Capsule[]) {
  const correction = new THREE.Vector3();
  let sourceDistance = Infinity, maxPenetration = 0, maxRawPenetration = 0;
  for (const l of left) for (const r of right) {
    if (l.kind === "forearm" && r.kind === "forearm") continue;
    const measured = l.sourceObserved !== false && r.sourceObserved !== false;
    const sourceGap = measured ? segmentDistance(l.sourceA, l.sourceB, r.sourceA, r.sourceB).distance : Infinity;
    sourceDistance = Math.min(sourceDistance, sourceGap);
    const hit = segmentDistance(l.a, l.b, r.a, r.b);
    maxRawPenetration = Math.max(maxRawPenetration, (l.radius ?? .008) + (r.radius ?? .008) - hit.distance);
    const touching = sourceGap < .035;
    const penetration = Math.max(0, (l.radius ?? .008) + (r.radius ?? .008) - (touching ? .003 : 0) - hit.distance);
    if (penetration > maxPenetration + 1e-6) {
      maxPenetration = penetration;
      const normal = hit.left.sub(hit.right);
      if (normal.lengthSq() < 1e-6) normal.copy(l.sourceA).sub(r.sourceA);
      if (normal.lengthSq() < 1e-6) normal.set(1, 0, .25);
      correction.copy(normal.normalize().multiplyScalar(penetration * .5 + .001));
    }
  }
  const contact = sourceDistance < .035;
  correction.clampLength(0, .015);
  return { correction, maxPenetration, maxRawPenetration, sourceDistance, contact };
}

/**
 * Allocation-free form of `handContact` for the retargeter's hot loop (same arithmetic, same order, same result).
 * Capsules are flat arrays: `p` holds [ax, ay, az, bx, by, bz] per capsule (the posed segment), `n` how many are
 * active. `base` is the unshifted pose (what `sourceA` falls back to when the source joint is missing), `src` the
 * source points of the same frame and `gaps` a per-frame cache of the source distances (NaN = not computed yet;
 * they depend only on the source joints, so they are the same in every call of a frame).
 */
export type CapsuleSpec = { radius: number; forearm: boolean };
export type CapsuleSet = { p: Float64Array; base: Float64Array; n: number };
export type CapsuleSource = {
  /** Per side: source points [ax, ay, az, bx, by, bz] per capsule; `hasA`: source joint A exists; `observed`: A and B exist. */
  left: { p: Float64Array; hasA: Uint8Array; observed: Uint8Array };
  right: { p: Float64Array; hasA: Uint8Array; observed: Uint8Array };
  /** left capsule index * right count + right index -> source distance (NaN until computed). */
  gaps: Float64Array;
  rightCount: number;
  /** Source contact (any visited source pair under 3.5 cm) per (left count, right count), filled on demand. */
  contact?: Map<number, boolean>;
};
export type ContactResult = { correction: THREE.Vector3; maxPenetration: number; maxRawPenetration: number; contact: boolean };

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));   // = THREE.MathUtils.clamp(x, 0, 1)
/** Closest points of two segments (exactly `segmentDistance`); writes the points to `out` [lx, ly, lz, rx, ry, rz]. */
function segmentClosest(ax: number, ay: number, az: number, bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number, dx: number, dy: number, dz: number, out: Float64Array) {
  const ux = bx - ax, uy = by - ay, uz = bz - az, vx = dx - cx, vy = dy - cy, vz = dz - cz, wx = ax - cx, wy = ay - cy, wz = az - cz;
  const aa = ux * ux + uy * uy + uz * uz, bb = ux * vx + uy * vy + uz * vz, cc = vx * vx + vy * vy + vz * vz;
  const dd = ux * wx + uy * wy + uz * wz, ee = vx * wx + vy * wy + vz * wz;
  let s = 0, t = 0;
  if (aa < 1e-12 && cc < 1e-12) {
    out[0] = ax; out[1] = ay; out[2] = az; out[3] = cx; out[4] = cy; out[5] = cz;
    const ex = ax - cx, ey = ay - cy, ez = az - cz;
    return Math.sqrt(ex * ex + ey * ey + ez * ez);
  }
  if (aa < 1e-12) t = clamp01(ee / cc);
  else if (cc < 1e-12) s = clamp01(-dd / aa);
  else {
    const denominator = aa * cc - bb * bb;
    s = denominator > 1e-12 ? clamp01((bb * ee - cc * dd) / denominator) : 0;
    t = (bb * s + ee) / cc;
    if (t < 0) { t = 0; s = clamp01(-dd / aa); }
    else if (t > 1) { t = 1; s = clamp01((bb - dd) / aa); }
  }
  const lx = ax + ux * s, ly = ay + uy * s, lz = az + uz * s, rx = cx + vx * t, ry = cy + vy * t, rz = cz + vz * t;
  out[0] = lx; out[1] = ly; out[2] = lz; out[3] = rx; out[4] = ry; out[5] = rz;
  const ex = lx - rx, ey = ly - ry, ez = lz - rz;
  return Math.sqrt(ex * ex + ey * ey + ez * ez);
}
const closestTmp = new Float64Array(6), normalTmp = new THREE.Vector3();

/** Gap between two capsules' bounding boxes along the axis where it is largest (0 when the boxes overlap). The true
 * segment distance is at least this gap, so a pair whose gap exceeds both radii cannot touch. */
function boxGap(L: Float64Array, li: number, R: Float64Array, rj: number) {
  let gap = 0;
  for (let k = 0; k < 3; k++) {
    const l0 = L[li + k], l1 = L[li + 3 + k], r0 = R[rj + k], r1 = R[rj + 3 + k];
    const lmin = l0 < l1 ? l0 : l1, lmax = l0 < l1 ? l1 : l0, rmin = r0 < r1 ? r0 : r1, rmax = r0 < r1 ? r1 : r0;
    const g = lmin > rmax ? lmin - rmax : rmin > lmax ? rmin - lmax : 0;
    if (g > gap) gap = g;
  }
  return gap;
}
/** Safety margin of the culling (m): far above rounding (~1e-15 m), far below any contact distance. */
const CULL_MARGIN = 1e-7;

/**
 * `sourceDistance < .035` of `handContact` over the pairs it visits (it does not depend on the pose; cached per set
 * size). Source pairs whose boxes are more than 3.5 cm apart cannot be under 3.5 cm, so they are skipped; as
 * `Math.min` turns into NaN when any distance is NaN, a pair with a non-finite source point decides "no contact" first.
 */
function sourceContactOf(src: CapsuleSource, nl: number, nr: number, leftSpec: CapsuleSpec[], rightSpec: CapsuleSpec[]) {
  const key = nl * 1000 + nr, cached = src.contact?.get(key);
  if (cached !== undefined) return cached;
  const finite = (p: Float64Array, i: number) => { for (let k = i * 6; k < i * 6 + 6; k++) if (!Number.isFinite(p[k])) return false; return true; };
  let contact = false, nan = false;
  for (let i = 0; i < nl && !nan; i++) for (let j = 0; j < nr; j++) {
    if (leftSpec[i].forearm && rightSpec[j].forearm) continue;
    if (!src.left.observed[i] || !src.right.observed[j]) continue;
    if (!finite(src.left.p, i) || !finite(src.right.p, j)) {
      // exact fallback: the plain minimum (NaN when a distance is NaN)
      let d = Infinity;
      for (let a = 0; a < nl; a++) for (let b = 0; b < nr; b++)
        if (!(leftSpec[a].forearm && rightSpec[b].forearm) && src.left.observed[a] && src.right.observed[b]) d = Math.min(d, sourceGapOf(src, a, b));
      contact = d < .035; nan = true; break;
    }
    if (!contact && boxGap(src.left.p, i * 6, src.right.p, j * 6) <= .035 + CULL_MARGIN && sourceGapOf(src, i, j) < .035) contact = true;
  }
  (src.contact ??= new Map()).set(key, contact);
  return contact;
}
function sourceGapOf(src: CapsuleSource, i: number, j: number) {
  const k = i * src.rightCount + j;
  let g = src.gaps[k];
  if (g !== g) {
    const a = src.left.p, b = src.right.p, li = i * 6, rj = j * 6;
    src.gaps[k] = g = segmentClosest(a[li], a[li + 1], a[li + 2], a[li + 3], a[li + 4], a[li + 5], b[rj], b[rj + 1], b[rj + 2], b[rj + 3], b[rj + 4], b[rj + 5], closestTmp);
  }
  return g;
}

/**
 * `handContact` on flat capsule sets (see CapsuleSet); `out` is reused. Same result as `handContact` bit for bit:
 * pairs are visited in the same order, and a pair is skipped only when its bounding boxes are farther apart than its
 * two radii (+ CULL_MARGIN), i.e. when `handContact` would find no raw penetration and no penetration for it (such a
 * pair changes nothing but the source distance, which is taken from the per-frame cache).
 */
export function handContactSets(left: CapsuleSet, right: CapsuleSet, leftSpec: CapsuleSpec[], rightSpec: CapsuleSpec[],
  src: CapsuleSource, out: ContactResult) {
  const correction = out.correction.set(0, 0, 0), L = left.p, R = right.p;
  const sl = src.left, sr = src.right;
  let maxPenetration = 0, maxRawPenetration = 0;
  for (let i = 0; i < left.n; i++) {
    const li = i * 6, lr = leftSpec[i].radius, lf = leftSpec[i].forearm;
    for (let j = 0; j < right.n; j++) {
      if (lf && rightSpec[j].forearm) continue;
      const rj = j * 6, rr = rightSpec[j].radius;
      if (boxGap(L, li, R, rj) > lr + rr + CULL_MARGIN) continue;
      // only "is the source gap under 3.5 cm" matters here; boxes farther apart than that answer it without the distance
      const sourceGap = sl.observed[i] && sr.observed[j] && boxGap(sl.p, li, sr.p, rj) <= .035 + CULL_MARGIN ? sourceGapOf(src, i, j) : Infinity;
      const distance = segmentClosest(L[li], L[li + 1], L[li + 2], L[li + 3], L[li + 4], L[li + 5], R[rj], R[rj + 1], R[rj + 2], R[rj + 3], R[rj + 4], R[rj + 5], closestTmp);
      maxRawPenetration = Math.max(maxRawPenetration, lr + rr - distance);
      const touching = sourceGap < .035;
      const penetration = Math.max(0, lr + rr - (touching ? .003 : 0) - distance);
      if (penetration > maxPenetration + 1e-6) {
        maxPenetration = penetration;
        const normal = normalTmp.set(closestTmp[0], closestTmp[1], closestTmp[2]);
        normal.x -= closestTmp[3]; normal.y -= closestTmp[4]; normal.z -= closestTmp[5];
        if (normal.lengthSq() < 1e-6) {
          // sourceA, or the unshifted posed point when the source joint is missing
          const a = sl.hasA[i] ? sl.p : left.base, b = sr.hasA[j] ? sr.p : right.base;
          normal.set(a[li], a[li + 1], a[li + 2]);
          normal.x -= b[rj]; normal.y -= b[rj + 1]; normal.z -= b[rj + 2];
        }
        if (normal.lengthSq() < 1e-6) normal.set(1, 0, .25);
        correction.copy(normal.normalize().multiplyScalar(penetration * .5 + .001));
      }
    }
  }
  correction.clampLength(0, .015);
  out.maxPenetration = maxPenetration; out.maxRawPenetration = maxRawPenetration;
  out.contact = sourceContactOf(src, left.n, right.n, leftSpec, rightSpec);
  return out;
}
