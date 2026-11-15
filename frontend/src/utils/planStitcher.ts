import { AnimationPlan, MotionFile, MotionFrame } from '../types/contract';

/**
 * Stitches multiple motion clips across the timeline of an AnimationPlan.
 *
 * Each segment in the plan defines:
 * - motion_id
 * - start_ms
 * - duration_ms
 *
 * Stitched frames have global timestamps (segment.start_ms + frame.time_ms).
 * Shared timestamps at boundaries are deduplicated cleanly.
 */
export function stitchAnimationPlan(
  plan: AnimationPlan,
  motionMap: Map<string, MotionFile>
): MotionFile {
  if (!plan.segments || plan.segments.length === 0) {
    return {
      rig_version: 'signaculture-v1',
      fps: 10,
      duration_ms: plan.duration_ms || 1000,
      joints: [],
      frames: [],
    };
  }

  const allJointsSet = new Set<string>();
  const timeToFrameMap = new Map<number, MotionFrame>();
  let maxFps = 10;

  for (const segment of plan.segments) {
    const motion = motionMap.get(segment.motion_id);
    if (!motion || !motion.frames) continue;

    if (motion.fps && motion.fps > maxFps) {
      maxFps = motion.fps;
    }

    if (motion.joints) {
      motion.joints.forEach((j) => allJointsSet.add(j));
    }

    const segStart = segment.start_ms;
    const segDuration = segment.duration_ms || motion.duration_ms;

    for (const frame of motion.frames) {
      // Offset frame time by segment start_ms
      const globalTimeMs = segStart + frame.time_ms;

      // Don't overshoot segment boundary if specified
      if (frame.time_ms > segDuration) continue;

      if (!timeToFrameMap.has(globalTimeMs)) {
        timeToFrameMap.set(globalTimeMs, {
          time_ms: globalTimeMs,
          joints: { ...frame.joints },
        });
      } else {
        // Boundary transition: merge joints with preference to incoming segment
        const existing = timeToFrameMap.get(globalTimeMs)!;
        existing.joints = {
          ...existing.joints,
          ...frame.joints,
        };
      }
    }
  }

  // Sort frames strictly by time_ms
  const stitchedFrames = Array.from(timeToFrameMap.values()).sort(
    (a, b) => a.time_ms - b.time_ms
  );

  return {
    rig_version: 'signaculture-v1',
    fps: maxFps,
    duration_ms: plan.duration_ms,
    joints: Array.from(allJointsSet),
    frames: stitchedFrames,
  };
}
