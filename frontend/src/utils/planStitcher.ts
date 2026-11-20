import type { AnimationPlan, AnimationPlanSegment, MotionFile, MotionFrame } from "../types/contract";
import { interpolateJointPositions } from "./motionInterpolator";
import { faceCalibration, frameInterval, hasHand, sampleFace } from "./avatarPlayback";

type Vec = [number, number, number];

function calibratedFace(clip: MotionFile, frame: MotionFrame): MotionFrame["face"] {
  const reference = faceCalibration(clip);
  return frame.face ? { ...frame.face,
    brow_left_neutral: reference?.brow_left ?? frame.face.brow_left_neutral ?? .38,
    brow_right_neutral: reference?.brow_right ?? frame.face.brow_right_neutral ?? .38,
    // A blend weight between calibrated and fallback rendering, not confidence.
    brow_reference_blend: reference || Number.isFinite(frame.face.brow_left_neutral) ? 1 : 0,
  } : frame.face;
}

/** The captured frames of the segment's window (its signing core), rebased to start at 0.
 * Window bounds are captured frame times, so no frame is resampled or rewritten. */
function windowed(clip: MotionFile, segment: AnimationPlanSegment): { frames: MotionFrame[]; duration: number } {
  const start = segment.clip_start_ms ?? 0;
  const end = segment.clip_end_ms ?? clip.duration_ms;
  const frames = clip.frames.filter((f) => f.time_ms >= start && f.time_ms <= end);
  if (frames.length < 2) return { frames: clip.frames, duration: clip.duration_ms };
  const origin = frames[0].time_ms;
  return { frames: frames.map((f) => ({ ...f, time_ms: f.time_ms - origin })), duration: frames[frames.length - 1].time_ms - origin };
}

/** Library playback uses the cached core, while the original file and source video stay intact. */
export function libraryPreview(clip: MotionFile): MotionFile {
  const bounds = clip.playback_window;
  if (!bounds) return clip;
  const part = windowed(clip, {clip_start_ms:bounds.start_ms,clip_end_ms:bounds.end_ms} as AnimationPlanSegment);
  return {...clip,duration_ms:part.duration,frames:part.frames};
}

const sub = (a: Vec, b: Vec): Vec => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const FINGER_JOINT = /^(left|right)_(thumb|index|middle|ring|pinky)_\d+$/;

/** Joints and hand-tracking state at any source time, for audio-retimed playback. A hand counts as
 * tracked only when both surrounding captured frames have it; otherwise its fingers are dropped and
 * it is flagged missing, so the renderer's hold/fade handles the gap instead of blending real
 * fingers with an absent hand (which detached the handshape from the wrist and made it jump). */
export function resampleFrame(source: MotionFile, sourceTime: number): Pick<MotionFrame, "joints" | "hand_tracking"> {
  const joints = interpolateJointPositions(source, sourceTime) as Record<string, Vec>;
  if (!source.frames.length) return { joints };
  const { lo, next } = frameInterval(source.frames, sourceTime);
  const a = source.frames[lo], b = source.frames[next];
  const hand_tracking: NonNullable<MotionFrame["hand_tracking"]> = {};
  for (const side of ["left", "right"]) {
    const tracked = hasHand(a, side) && hasHand(b, side);
    hand_tracking[side] = { observed: tracked, source: tracked ? (a.hand_tracking?.[side]?.source ?? "interpolated") : "missing" };
    if (!tracked) for (const k of Object.keys(joints)) if (k.startsWith(`${side}_`) && FINGER_JOINT.test(k)) delete joints[k];
  }
  return { joints, hand_tracking };
}
const len = (a: Vec) => Math.hypot(a[0], a[1], a[2]);
const quintic = (u: number) => u * u * u * (10 + u * (-15 + 6 * u));

/** Velocity at a clip boundary in metres per bridge, clamped to the bridge's own travel so the
 * curve cannot overshoot (keeps the Hermite segment monotone along the chord). */
function tangent(near: Vec, far: Vec | undefined, dt: number, bridgeMs: number, chord: number): Vec {
  if (!far || dt <= 0) return [0, 0, 0];
  const v = sub(near, far).map((c) => (c / dt) * bridgeMs) as Vec;
  const size = len(v), limit = Math.min(chord, 0.25);
  return size > limit && size > 0 ? (v.map((c) => (c * limit) / size) as Vec) : v;
}

/** Bridge from the last frame of sign A to the first frame of sign B. Arms keep moving through
 * the join (cubic Hermite with the measured end velocities) instead of stopping at both ends;
 * finger points move with their wrist while the hand shape morphs (quintic), so the handshape
 * never trails behind or flies apart mid-bridge. */
function bridgeFrames(aFrames: MotionFrame[], bFrames: MotionFrame[], position: number, transition: number) {
  const a = aFrames[aFrames.length - 1], aPrev = aFrames[aFrames.length - 2];
  const b = bFrames[0], bNext = bFrames[1];
  const dtA = aPrev ? a.time_ms - aPrev.time_ms : 0, dtB = bNext ? bNext.time_ms - b.time_ms : 0;
  const keys = Object.keys(a.joints).filter((k) => b.joints[k]);
  const finger = (k: string) => /_(thumb|index|middle|ring|pinky)_\d$/.test(k);
  const wristOf = (k: string) => `${k.split("_")[0]}_wrist`;
  const curves = new Map<string, (u: number) => Vec>();
  for (const k of keys.filter((k) => !finger(k))) {
    const p = a.joints[k] as Vec, q = b.joints[k] as Vec, chord = len(sub(q, p));
    const m0 = tangent(p, aPrev?.joints[k] as Vec | undefined, dtA, transition, chord);
    const next = bNext?.joints[k] as Vec | undefined;
    const m1 = tangent(next ?? q, q, dtB, transition, chord);
    curves.set(k, (u) => {
      const u2 = u * u, u3 = u2 * u;
      const h00 = 2 * u3 - 3 * u2 + 1, h10 = u3 - 2 * u2 + u, h01 = -2 * u3 + 3 * u2, h11 = u3 - u2;
      return [0, 1, 2].map((i) => h00 * p[i] + h10 * m0[i] + h01 * q[i] + h11 * m1[i]) as Vec;
    });
  }
  const frames: { frame: MotionFrame; blend: number }[] = [];
  for (let t = 1; t < transition; t += 1000 / 30) {
    const u = t / transition, blend = quintic(u);
    const bridge: MotionFrame = { playback_origin: "transition", time_ms: Math.round(position + t), joints: {} };
    for (const k of keys) {
      const curve = curves.get(k);
      if (curve) { bridge.joints[k] = curve(u); continue; }
      const p = a.joints[k] as Vec, q = b.joints[k] as Vec, w = wristOf(k);
      const wa = a.joints[w] as Vec | undefined, wb = b.joints[w] as Vec | undefined, wu = curves.get(w);
      if (wa && wb && wu) {
        const oa = sub(p, wa), ob = sub(q, wb), wrist = wu(u);
        bridge.joints[k] = [0, 1, 2].map((i) => wrist[i] + oa[i] + (ob[i] - oa[i]) * blend) as Vec;
      } else bridge.joints[k] = p.map((v, i) => v + (q[i] - v) * blend) as Vec;
    }
    frames.push({ frame: bridge, blend });
  }
  return frames;
}

/** All source frames retain their measured coordinates. Plans may select each clip's signing
 * core; audio timing only changes when frames play, with explicit holds during measured pauses. */
export function stitchAnimationPlan(
  plan: AnimationPlan,
  clips: Map<string, MotionFile>,
): MotionFile {
  const frames = new Map<number, MotionFrame>();
  const joints = new Set<string>();
  const attributions = [...clips.values()].flatMap((clip) =>
    clip.attribution ? [clip.attribution] : clip.attributions || [],
  );
  let position = 0,
    fps = 10;
  const transition = plan.transition_ms || 0;
  let previous: { clip: MotionFile; frames: MotionFrame[] } | null = null;
  const spans: SourceSpan[] = [];
  for (const segment of plan.segments) {
    const clip = clips.get(segment.motion_id);
    if (!clip) throw new Error(`Mouvement manquant : ${segment.motion_id}`);
    const part = windowed(clip, segment);
    if (previous && transition) {
      // Dedicated time for a bridge: never overwrite either source endpoint.
      const faceA = calibratedFace(previous.clip, previous.frames[previous.frames.length - 1]);
      const faceB = calibratedFace(clip, part.frames[0]);
      for (const { frame, blend } of bridgeFrames(previous.frames, part.frames, position, transition)) {
        if (faceA && faceB) {
          frame.face = {};
          for (const key of Object.keys(faceA))
            if (Number.isFinite(faceB[key])) frame.face[key] = faceA[key] + (faceB[key] - faceA[key]) * blend;
        }
        frames.set(frame.time_ms, frame);
      }
      position += transition;
    }
    clip.joints.forEach((j) => joints.add(j));
    fps = Math.max(fps, clip.fps);
    spans.push({ start: position, end: position + part.duration, segment, centroids: centroids(part.frames), closest: closestToHead(part.frames) });
    for (const frame of part.frames)
      frames.set(position + frame.time_ms, {
        ...frame,
        face: calibratedFace(clip, frame),
        time_ms: position + frame.time_ms,
      });
    position += part.duration;
    previous = { clip, frames: part.frames };
  }
  const source: MotionFile = {
    attributions,
    synthetic: [...clips.values()].some(
      (clip) => clip.synthetic ?? !clip.attribution,
    ),
    rig_version: "signaculture-v1",
    fps,
    duration_ms: position,
    joints: [...joints],
    frames: [...frames.values()].sort((a, b) => a.time_ms - b.time_ms),
  };
  const audio = plan.audio_features;
  if (plan.timing_mode === "audio_words") return wordTimed(source, plan, spans);
  if (plan.timing_mode !== "audio_voiced" || !audio) return source;
  const regions = audio.frames.filter((f) => !f.is_pause);
  const voiced = regions.reduce((sum, f) => sum + f.end_ms - f.start_ms, 0);
  if (!voiced) return source;
  const output: MotionFrame[] = [];
  const times = Array.from(
    { length: Math.ceil(audio.duration_ms / 40) },
    (_, i) => i * 40,
  );
  times.push(audio.duration_ms);
  for (const time of times) {
    let elapsed = 0;
    for (const frame of regions) {
      if (time >= frame.end_ms) elapsed += frame.end_ms - frame.start_ms;
      else if (time > frame.start_ms) {
        elapsed += time - frame.start_ms;
        break;
      } else break;
    }
    const sourceTime = Math.min(position, (elapsed / voiced) * position);
    const face = sampleFace(source, sourceTime);
    output.push({
      playback_origin: "audio_sample",
      time_ms: time,
      ...resampleFrame(source, sourceTime),
      face,
    });
  }
  return { ...source, fps: 25, duration_ms: audio.duration_ms, frames: output };
}

// ---- Voice pace, tonality and emphasis (timing_mode "audio_words") --------------------------

type SourceSpan = { start: number; end: number; segment: AnimationPlanSegment; centroids: Record<string, Vec>; closest: Record<string, number> };

/** Emphasised signs enlarge their movement path by up to this fraction. */
export const EMPHASIS_SCALE = 0.3;
/** Ease-in/out time for the scaling at each end of a sign (shorter = more of the sign at full size). */
export const SCALE_RAMP_MS = 120;
/** Tonality (pitch liveliness 0..1) maps to a path scale of 0.96..1.06. */
export const LIVELINESS_RANGE: [number, number] = [0.96, 1.06];
/** Head nod at the onset of an emphasised sign. */
export const NOD_MS = 350;
const ARM = /^(left|right)_(wrist|elbow)$/;
const FINGER = /^(left|right)_(thumb|index|middle|ring|pinky)_\d+$/;

/** Mean wrist and elbow positions over a sign: the sign's location, which scaling preserves. */
function centroids(frames: MotionFrame[]): Record<string, Vec> {
  const sums: Record<string, { v: Vec; n: number }> = {};
  for (const f of frames)
    for (const [k, v] of Object.entries(f.joints))
      if (ARM.test(k)) {
        const e = (sums[k] ??= { v: [0, 0, 0], n: 0 });
        e.v = [e.v[0] + v[0], e.v[1] + v[1], e.v[2] + v[2]];
        e.n++;
      }
  return Object.fromEntries(Object.entries(sums).map(([k, { v, n }]) => [k, v.map((c) => c / n) as Vec]));
}

/** Enlargement is off within CONTACT_NEAR (m) of the sign's closest hand-to-head approach and back to
 * full CONTACT_FAR beyond it, so the point where the hand touches the face stays exact. Relative to
 * each sign because capture coordinates differ from the avatar's (contact appears after retargeting). */
export const CONTACT_NEAR = 0.04;
export const CONTACT_FAR = 0.14;

/** Closest wrist-to-head distance over a sign, per side. */
function closestToHead(frames: MotionFrame[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const f of frames) {
    const head = f.joints.head as Vec | undefined;
    if (!head) continue;
    for (const side of ["left", "right"]) {
      const w = f.joints[`${side}_wrist`] as Vec | undefined;
      if (w) out[side] = Math.min(out[side] ?? Infinity, len(sub(w, head)));
    }
  }
  return out;
}

/** Enlarge (or shrink) the movement path around the sign's own centroid; fingers follow their wrist,
 * so the handshape is untouched and the sign keeps its location. Near the head the scaling fades to
 * nothing: where a hand touches the face (chin, mouth, temple...) is part of the sign. */
export function scalePath(joints: Record<string, Vec>, centre: Record<string, Vec>, scale: number, closest: Record<string, number> = {}) {
  if (Math.abs(scale - 1) < 1e-6) return joints;
  const out: Record<string, Vec> = { ...joints };
  const shift: Record<string, Vec> = {};
  const head = joints.head;
  const sideScale = (side: string) => {
    const wrist = joints[`${side}_wrist`];
    if (!head || !wrist) return scale;
    const base = closest[side] ?? 0;
    const x = Math.min(1, Math.max(0, (len(sub(wrist, head)) - base - CONTACT_NEAR) / (CONTACT_FAR - CONTACT_NEAR)));
    return 1 + (scale - 1) * x * x * (3 - 2 * x);
  };
  for (const [k, v] of Object.entries(joints)) {
    const c = centre[k];
    if (!ARM.test(k) || !c) continue;
    const s = sideScale(k.split("_")[0]);
    const next = [0, 1, 2].map((i) => c[i] + (v[i] - c[i]) * s) as Vec;
    out[k] = next;
    shift[k] = sub(next, v);
  }
  for (const [k, v] of Object.entries(joints)) {
    const m = FINGER.exec(k), d = m && shift[`${m[1]}_wrist`];
    if (d) out[k] = [v[0] + d[0], v[1] + d[1], v[2] + d[2]];
  }
  return out;
}

/** Path scale target for a sign from its emphasis and pitch liveliness. */
export function signScale(segment: AnimationPlanSegment) {
  const [lo, hi] = LIVELINESS_RANGE;
  const tone = segment.liveliness == null ? 1 : lo + (hi - lo) * segment.liveliness;
  return tone * (1 + EMPHASIS_SCALE * (segment.emphasis ?? 0));
}

/** Ease in/out over ~SCALE_RAMP_MS at each end of a sign (5–20% of it) so scaling never jumps at a
 * bridge while most of the sign plays at full size. */
const envelope = (u: number, durationMs: number) => {
  const edge = Math.min(0.2, Math.max(0.05, SCALE_RAMP_MS / Math.max(1, durationMs)));
  const x = Math.min(1, Math.max(0, Math.min(u, 1 - u) / edge));
  return x * x * (3 - 2 * x);
};

/** Single down-up nod (0..1) at the onset of an emphasised sign. Any detected stress gets at least
 * half the maximum nod, reaching the full nod from emphasis 0.5 upwards, so it reads clearly. */
export function nodAt(sinceStartMs: number, emphasis: number) {
  if (emphasis <= 0 || sinceStartMs < 0 || sinceStartMs > NOD_MS) return 0;
  return Math.min(1, 0.5 + emphasis) * Math.sin((Math.PI * sinceStartMs) / NOD_MS);
}

/** Map every output time to source time: each sign plays over its word-timed slot (hold at its end),
 * bridges play in the gaps between slots. */
function wordTimed(source: MotionFile, plan: AnimationPlan, spans: SourceSpan[]): MotionFile {
  const anchors: [number, number][] = [[0, spans[0]?.start ?? 0]];
  for (const { start, end, segment } of spans) {
    const hold = segment.hold_ms ?? 0;
    const slotEnd = segment.start_ms + segment.duration_ms;
    anchors.push([segment.start_ms, start], [slotEnd - hold, end], [slotEnd, end]);
  }
  const total = Math.max(plan.duration_ms, anchors[anchors.length - 1][0]);
  const toSource = (t: number) => {
    for (let i = 1; i < anchors.length; i++) {
      const [t0, s0] = anchors[i - 1], [t1, s1] = anchors[i];
      if (t <= t1) return t1 > t0 ? s0 + ((t - t0) / (t1 - t0)) * (s1 - s0) : s1;
    }
    return anchors[anchors.length - 1][1];
  };
  const output: MotionFrame[] = [];
  const times = Array.from({ length: Math.ceil(total / 40) }, (_, i) => i * 40);
  times.push(total);
  for (const time of times) {
    const sourceTime = Math.min(source.duration_ms, toSource(time));
    const sampled = resampleFrame(source, sourceTime);
    let joints = sampled.joints as Record<string, Vec>;
    let nod = 0;
    const active = spans.find(({ segment }) => time >= segment.start_ms && time <= segment.start_ms + segment.duration_ms);
    if (active) {
      const { segment } = active;
      const u = (time - segment.start_ms) / Math.max(1, segment.duration_ms);
      joints = scalePath(joints, active.centroids, 1 + (signScale(segment) - 1) * envelope(u, segment.duration_ms), active.closest);
      nod = nodAt(time - segment.start_ms, segment.emphasis ?? 0);
    }
    output.push({
      playback_origin: "audio_sample",
      time_ms: time,
      joints,
      hand_tracking: sampled.hand_tracking,
      face: sampleFace(source, sourceTime),
      ...(nod ? { nod } : {}),
    });
  }
  return { ...source, fps: 25, duration_ms: total, frames: output };
}
