import type { AnimationPlan, MotionFile, MotionFrame } from "../types/contract";
import { interpolateJointPositions } from "./motionInterpolator";
import { faceCalibration, sampleFace } from "./avatarPlayback";

function calibratedFace(clip: MotionFile, frame: MotionFrame): MotionFrame["face"] {
  const reference = faceCalibration(clip);
  return frame.face ? { ...frame.face,
    brow_left_neutral: reference?.brow_left ?? frame.face.brow_left_neutral ?? .38,
    brow_right_neutral: reference?.brow_right ?? frame.face.brow_right_neutral ?? .38,
    // A blend weight between calibrated and fallback rendering, not confidence.
    brow_reference_blend: reference || Number.isFinite(frame.face.brow_left_neutral) ? 1 : 0,
  } : frame.face;
}

/** All source frames retain their measured coordinates. Audio timing only changes
 * when those frames play, with explicit holds during measured pauses. */
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
  let previous: MotionFile | null = null;
  for (const segment of plan.segments) {
    const clip = clips.get(segment.motion_id);
    if (!clip) throw new Error(`Mouvement manquant : ${segment.motion_id}`);
    if (previous && transition) {
      const a = previous.frames[previous.frames.length - 1],
        b = clip.frames[0];
      // Dedicated time for a bridge: never overwrite either source endpoint.
      for (let t = 1; t < transition; t += 1000 / 30) {
        const u = t / transition,
          blend = u * u * u * (10 + u * (-15 + 6 * u));
        const bridge: MotionFrame = {
          playback_origin: "transition",
          time_ms: Math.round(position + t),
          joints: {},
        };
        for (const key of Object.keys(a.joints)) {
          const p = a.joints[key],
            q = b.joints[key];
          if (q)
            bridge.joints[key] = p.map((v, i) => v + (q[i] - v) * blend) as [
              number,
              number,
              number,
            ];
        }
        const faceA = calibratedFace(previous, a), faceB = calibratedFace(clip, b);
        if (faceA && faceB) {
          bridge.face = {};
          for (const key of Object.keys(faceA))
            if (Number.isFinite(faceB[key]))
              bridge.face[key] =
                faceA[key] + (faceB[key] - faceA[key]) * blend;
        }
        frames.set(bridge.time_ms, bridge);
      }
      position += transition;
    }
    clip.joints.forEach((j) => joints.add(j));
    fps = Math.max(fps, clip.fps);
    for (const frame of clip.frames)
      frames.set(position + frame.time_ms, {
        ...frame,
        face: calibratedFace(clip, frame),
        time_ms: position + frame.time_ms,
      });
    position += clip.duration_ms;
    previous = clip;
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
