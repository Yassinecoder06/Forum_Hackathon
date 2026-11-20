import {readFileSync} from "node:fs";
import {stitchAnimationPlan, libraryPreview} from "../frontend/src/utils/planStitcher";
import type {AnimationPlan, MotionFile} from "../frontend/src/types/contract";

const examples = JSON.parse(readFileSync(0, "utf8"));
const results = examples.map(({label, glosses, paths}: {label: string; glosses: string[]; paths: string[]}) => {
  const clips = paths.map(path => libraryPreview(JSON.parse(readFileSync(path, "utf8")) as MotionFile));
  const transition_ms = 480;
  const segments = clips.map((clip, index) => ({
    gloss_id: glosses[index], motion_id: String(index), start_ms: 0, duration_ms: clip.duration_ms,
  }));
  const plan = {transition_ms, segments} as AnimationPlan;
  const motion = stitchAnimationPlan(plan, new Map(clips.map((clip, index) => [String(index), clip])));
  return {label, glosses, clips, motion, transition_ms};
});
process.stdout.write(JSON.stringify(results));
