import { MotionFile, MotionFrame } from '../types/contract';

export interface InterpolatedJoints {
  [jointName: string]: [number, number, number];
}

/**
 * Linearly interpolates joint positions between frames for a given time in milliseconds.
 *
 * Coordinates follow contract v1:
 * - Meters
 * - Y up
 * - X right
 * - Z forward
 */
export function interpolateJointPositions(
  motion: MotionFile | null,
  timeMs: number,
  options?: {
    expressiveModulation?: boolean;
    energyFactor?: number; // 0.0 to 1.0 (from audio features if available)
  }
): InterpolatedJoints {
  if (!motion || !motion.frames || motion.frames.length === 0) {
    return getDefaultRestPose();
  }

  const frames = motion.frames;
  const duration = motion.duration_ms || frames[frames.length - 1].time_ms;

  // Clamp time within duration
  const clampedTime = Math.max(0, Math.min(timeMs, duration));

  // If time is before or at the first frame
  if (clampedTime <= frames[0].time_ms) {
    return applyModulation(frames[0].joints, options);
  }

  // If time is at or beyond the last frame
  if (clampedTime >= frames[frames.length - 1].time_ms) {
    return applyModulation(frames[frames.length - 1].joints, options);
  }

  // Find surrounding frames
  let prevFrame: MotionFrame = frames[0];
  let nextFrame: MotionFrame = frames[frames.length - 1];

  for (let i = 0; i < frames.length - 1; i++) {
    if (frames[i].time_ms <= clampedTime && frames[i + 1].time_ms >= clampedTime) {
      prevFrame = frames[i];
      nextFrame = frames[i + 1];
      break;
    }
  }

  const delta = nextFrame.time_ms - prevFrame.time_ms;
  const alpha = delta > 0 ? (clampedTime - prevFrame.time_ms) / delta : 0;

  const result: InterpolatedJoints = {};

  // Interpolate each joint
  const jointKeys = Array.from(
    new Set([...Object.keys(prevFrame.joints), ...Object.keys(nextFrame.joints)])
  );

  for (const joint of jointKeys) {
    const p0 = prevFrame.joints[joint];
    const p1 = nextFrame.joints[joint];

    if (p0 && p1) {
      result[joint] = [
        p0[0] + (p1[0] - p0[0]) * alpha,
        p0[1] + (p1[1] - p0[1]) * alpha,
        p0[2] + (p1[2] - p0[2]) * alpha,
      ];
    } else if (p0) {
      result[joint] = [...p0];
    } else if (p1) {
      result[joint] = [...p1];
    }
  }

  return applyModulation(result, options);
}

/**
 * Constrained expressive visual modulation based on acoustic energy.
 * Strictly constrained to avoid altering grammatical handshapes or spatial syntax.
 * Only applies subtle amplitude adjustment (+/- 5%) to peripheral arm motion.
 */
function applyModulation(
  joints: InterpolatedJoints,
  options?: { expressiveModulation?: boolean; energyFactor?: number }
): InterpolatedJoints {
  if (!options?.expressiveModulation) {
    return joints;
  }

  const energy = Math.max(0, Math.min(1, options.energyFactor ?? 0.5));
  // Scale modulation between 0.96 and 1.05 (strictly capped)
  const scale = 0.96 + energy * 0.09;

  const modulated: InterpolatedJoints = {};
  const centerHips = joints['hips'] || [0, 1.0, 0];

  for (const [name, coords] of Object.entries(joints)) {
    // Only wrists and elbows exhibit minor expressive energy amplitude changes
    if (name.includes('wrist') || name.includes('elbow')) {
      const dx = coords[0] - centerHips[0];
      const dy = coords[1] - centerHips[1];
      const dz = coords[2] - centerHips[2];
      modulated[name] = [
        centerHips[0] + dx * scale,
        centerHips[1] + dy * scale,
        centerHips[2] + dz * scale,
      ];
    } else {
      modulated[name] = [coords[0], coords[1], coords[2]];
    }
  }

  return modulated;
}

/**
 * Standard neutral rest pose for SignaCulture v1 upper-body rig.
 * Dimensions are in meters.
 */
export function getDefaultRestPose(): InterpolatedJoints {
  return {
    hips: [0, 1.0, 0],
    neck: [0, 1.5, 0],
    head: [0, 1.7, 0],
    left_shoulder: [-0.2, 1.45, 0],
    right_shoulder: [0.2, 1.45, 0],
    left_elbow: [-0.3, 1.2, 0],
    right_elbow: [0.3, 1.2, 0],
    left_wrist: [-0.3, 1.0, 0],
    right_wrist: [0.3, 1.0, 0],
  };
}
