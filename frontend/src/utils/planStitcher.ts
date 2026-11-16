import type { AnimationPlan, AnimationPlanSegment, MotionFile, MotionFrame } from "../types/contract";
import { interpolateJointPositions } from "./motionInterpolator";
import { faceCalibration, sampleFace } from "./avatarPlayback";

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
      joints: interpolateJointPositions(source, sourceTime),
      face,
    });
  }
  return { ...source, fps: 25, duration_ms: audio.duration_ms, frames: output };
}
