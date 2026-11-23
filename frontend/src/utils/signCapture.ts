/**
 * Webcam signing -> the corpus landmark layout (sign-model/signlib.py PARTS, 157 points), and the hand-pause detector
 * that ends a take on its own. Pure functions: the MediaPipe models run in ConversationMode (utils/signTracker.ts).
 *
 * Layout: upper_pose 23 (MediaPipe pose 0..22), left_hand 21, right_hand 21, lips 40, left/right eyebrow 10 each,
 * left/right eye 16 each. Face points are the vertices of MediaPipe's FACE_LANDMARKS_* connection sets, sorted (the
 * same lists the corpus extraction used, sign-model/extract_poses.py). Hands are assigned to the body side whose pose
 * wrist they touch (webcam/src/capture.mjs assignHands), never by the mirrored handedness label.
 * Coordinates: x / width and y / WIDTH (true aspect ratio, as the server expects), NaN where a part was not seen.
 */

export type Landmark = { x: number; y: number; z?: number; visibility?: number };

export const N_POINTS = 157;
export const PARTS = { upper_pose: [0, 23], left_hand: [23, 44], right_hand: [44, 65], lips: [65, 105],
  left_eyebrow: [105, 115], right_eyebrow: [115, 125], left_eye: [125, 141], right_eye: [141, 157] } as const;

export const FACE_INDEX = {
  lips: [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308,
    310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415],
  left_eyebrow: [276, 282, 283, 285, 293, 295, 296, 300, 334, 336],
  right_eyebrow: [46, 52, 53, 55, 63, 65, 66, 70, 105, 107],
  left_eye: [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466],
  right_eye: [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246],
};
const FACE_ORDER = ["lips", "left_eyebrow", "right_eyebrow", "left_eye", "right_eye"] as const;

/** Sorted unique vertices of a MediaPipe connection list (how the corpus face index lists were made). */
export function connectionVertices(connections: { start: number; end: number }[]) {
  return [...new Set(connections.flatMap((c) => [c.start, c.end]))].sort((a, b) => a - b);
}

const dist = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y);

/** Detected hands -> body sides, by distance of each hand's wrist to the pose wrist (15 left, 16 right). */
export function assignHands(pose: Landmark[] | null, hands: Landmark[][]): { left: Landmark[] | null; right: Landmark[] | null } {
  const out = { left: null as Landmark[] | null, right: null as Landmark[] | null };
  if (!pose?.[16]) return out;
  let best = { count: -1, cost: Infinity, choice: [-1, -1] };
  for (let l = -1; l < hands.length; l++)
    for (let r = -1; r < hands.length; r++) {
      if (l >= 0 && l === r) continue;
      const choice = [l, r];
      let count = 0, cost = 0, valid = true;
      for (let s = 0; s < 2; s++) {
        if (choice[s] < 0) continue;
        const wrist = pose[s ? 16 : 15], gap = dist(wrist, hands[choice[s]][0]);
        if ((wrist.visibility ?? 1) < 0.3 || gap > 0.18) { valid = false; break; }
        count++;
        cost += gap;
      }
      if (valid && (count > best.count || (count === best.count && cost < best.cost))) best = { count, cost, choice };
    }
  if (best.choice[0] >= 0) out.left = hands[best.choice[0]];
  if (best.choice[1] >= 0) out.right = hands[best.choice[1]];
  return out;
}

/** One webcam frame -> 157 x 3 floats (x, y, z), true aspect ratio, NaN where missing. */
export function assembleFrame(pose: Landmark[] | null, left: Landmark[] | null, right: Landmark[] | null,
  face: Landmark[] | null, width: number, height: number): Float32Array {
  const out = new Float32Array(N_POINTS * 3).fill(NaN);
  const k = height / Math.max(1, width);
  const put = (i: number, p: Landmark | undefined) => {
    if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) return;
    out[3 * i] = p.x;
    out[3 * i + 1] = p.y * k;
    out[3 * i + 2] = p.z ?? 0;
  };
  if (pose) for (let i = 0; i < 23; i++) put(i, pose[i]);
  if (left) for (let i = 0; i < 21; i++) put(PARTS.left_hand[0] + i, left[i]);
  if (right) for (let i = 0; i < 21; i++) put(PARTS.right_hand[0] + i, right[i]);
  if (face?.length) {
    let o = PARTS.lips[0];
    for (const part of FACE_ORDER) for (const idx of FACE_INDEX[part]) put(o++, face[idx]);
  }
  return out;
}

/** Frames of a take -> base64 of float32 little-endian (the /sign/recognize `data_b64` field). */
export function encodeTake(frames: Float32Array[]): string {
  const all = new Float32Array(frames.length * N_POINTS * 3);
  frames.forEach((f, i) => all.set(f, i * N_POINTS * 3));
  const bytes = new Uint8Array(all.buffer);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

/**
 * Ends a take by itself: once the person has signed (a hand raised above the lower chest for `minSignMs`), the take
 * stops when the hands have been lowered or out of view for `pauseMs`. Heights are in shoulder widths, as the server's
 * segmentation (sign_recognition.py: wrist above rest + 0.35 = about 0.7 shoulder widths below the shoulders).
 */
export class HandPause {
  signedMs = 0;
  restMs = 0;
  raised = false;
  private last = -1;
  constructor(private opts = { minSignMs: 400, pauseMs: 1400, raiseBelowShoulders: 0.75 }) {}

  /** Returns true when the take should stop. */
  push(frame: Float32Array, tMs: number): boolean {
    const dt = this.last < 0 ? 0 : Math.min(200, tMs - this.last);
    this.last = tMs;
    const lx = frame[33], ly = frame[34], rx = frame[36], ry = frame[37];
    const width = Math.hypot(lx - rx, ly - ry);
    const sy = (ly + ry) / 2;
    let raised = false;
    if (Number.isFinite(width) && width > 1e-3) {
      for (const [w, h] of [[15, PARTS.left_hand[0]], [16, PARTS.right_hand[0]]]) {
        const seen = Number.isFinite(frame[3 * h]);              // the hand itself is tracked
        const wy = frame[3 * w + 1];
        if (seen && Number.isFinite(wy) && (wy - sy) / width < this.opts.raiseBelowShoulders) raised = true;  // y down
      }
    }
    this.raised = raised;
    if (raised) {
      this.signedMs += dt;
      this.restMs = 0;
    } else {
      this.restMs += dt;
    }
    return this.signedMs >= this.opts.minSignMs && this.restMs >= this.opts.pauseMs;
  }
}

/** Upper-body and hand edges for the light skeleton drawn over the webcam preview. */
export const BODY_EDGES: [number, number][] = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
export const HAND_EDGES: [number, number][] = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10],
  [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
