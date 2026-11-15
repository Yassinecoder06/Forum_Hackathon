import type { AnimationPlan, MotionFile, MotionFrame } from "../types/contract";
import { interpolateJointPositions } from "./motionInterpolator";

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
  for (const segment of plan.segments) {
    const clip = clips.get(segment.motion_id);
    if (!clip) throw new Error(`Mouvement manquant : ${segment.motion_id}`);
    clip.joints.forEach((j) => joints.add(j));
    fps = Math.max(fps, clip.fps);
    for (const frame of clip.frames)
      frames.set(position + frame.time_ms, {
        ...frame,
        time_ms: position + frame.time_ms,
      });
    position += clip.duration_ms;
  }
  const source: MotionFile = {
    attributions,
    synthetic: [...clips.values()].some(clip => clip.synthetic ?? !clip.attribution),
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
    const face = source.frames.find(
      (f, i) =>
        f.time_ms <= sourceTime &&
        (!source.frames[i + 1] || source.frames[i + 1].time_ms > sourceTime),
    )?.face;
    output.push({
      time_ms: time,
      joints: interpolateJointPositions(source, sourceTime),
      face,
    });
  }
  return { ...source, fps: 25, duration_ms: audio.duration_ms, frames: output };
}
