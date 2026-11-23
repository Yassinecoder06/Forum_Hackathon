import assert from "node:assert/strict";
import {
  LOOK, SCALE_RANGE, easeAngle, facingYaw, fitSize, flipRows, lookAt, lookSteps, pinchPlacement, sceneDifference, toGray,
} from "../src/utils/companion";

const close = (a: number, b: number, eps = 1e-6) => assert(Math.abs(a - b) < eps, `${a} != ${b}`);

// ---- frame for the brain: longest side 512, aspect kept, never upscaled ----------------------------------------
{
  assert.deepEqual(fitSize(1920, 1080), { width: 512, height: 288 });
  assert.deepEqual(fitSize(720, 1280), { width: 288, height: 512 });
  assert.deepEqual(fitSize(320, 240), { width: 320, height: 240 });
  assert.deepEqual(fitSize(0, 240), { width: 0, height: 0 });
  assert.deepEqual(fitSize(4000, 10, 64), { width: 64, height: 1 });
}

// ---- scene watcher: luminance thumbnails, mean absolute difference ---------------------------------------------
{
  const rgba = new Uint8Array([255, 255, 255, 255, 0, 0, 0, 255, 255, 0, 0, 255]);
  const gray = toGray(rgba);
  assert.deepEqual([...gray], [255, 0, 76]);
  const a = new Uint8Array(192).fill(100), b = new Uint8Array(192).fill(100);
  assert.equal(sceneDifference(a, b), 0);
  b.fill(200, 0, 96);                                                // half the picture changed a lot
  close(sceneDifference(a, b), (96 * 100) / (192 * 255));
  assert.equal(sceneDifference(a, null), 1);
  assert.equal(sceneDifference(a, new Uint8Array(10)), 1);
}

// ---- WebGL rows are bottom-up -----------------------------------------------------------------------------------
{
  const px = new Uint8Array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);  // 1 x 3 image
  assert.deepEqual([...flipRows(px, 1, 3)], [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]);
  const even = new Uint8Array([1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]); // 2 x 2
  assert.deepEqual([...flipRows(even, 2, 2)], [3, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0]);
}

// ---- two-finger gestures: pinch scales, twist turns, scale stays in range ---------------------------------------
{
  const start = { x0: 100, y0: 300, x1: 200, y1: 300 };
  const base = { scale: 1, yaw: 0 };
  const wider = pinchPlacement(start, { x0: 50, y0: 300, x1: 250, y1: 300 }, base);
  close(wider.scale, 2);
  close(wider.yaw, 0);
  const huge = pinchPlacement(start, { x0: 0, y0: 300, x1: 1000, y1: 300 }, base);
  assert.equal(huge.scale, SCALE_RANGE[1]);
  const twisted = pinchPlacement(start, { x0: 150, y0: 250, x1: 150, y1: 350 }, base); // a quarter turn on screen
  close(twisted.scale, 1);
  close(Math.abs(twisted.yaw), Math.PI / 2);
  // no distance at the start: scale unchanged
  assert.equal(pinchPlacement({ x0: 5, y0: 5, x1: 5, y1: 5 }, start, base).scale, 1);
}

// ---- the avatar faces the person: yaw toward the phone, eased the short way round -------------------------------
{
  close(facingYaw({ x: 0, z: 0 }, { x: 0, z: 2 }), 0);               // the phone in front (+Z): no turn
  close(facingYaw({ x: 0, z: 0 }, { x: 2, z: 0 }), Math.PI / 2);
  close(facingYaw({ x: 1, z: -1 }, { x: 1, z: -3 }), Math.PI);
  close(easeAngle(0, 1, 0), 0);
  const eased = easeAngle(3, -3, 0.1);                               // across +-pi: goes up through pi, not down
  assert(eased > 3);
  assert(easeAngle(0, 1, 10) > 0.99);
}

// ---- looking at the object the brain talks about (head + eyes, minimal pointing) --------------------------------
{
  assert.deepEqual(lookSteps(null, 3000), []);
  assert.deepEqual(lookSteps("centre", 3000), []);
  assert.equal(lookAt([], 100), null);
  const left = lookSteps("left", 5000);
  const mid = lookAt(left, 900)!;
  close(mid.headYaw, -LOOK.headDeg * Math.PI / 180, 1e-3);           // toward the left of the screen while naming it
  assert(mid.gazeX < -0.5 && Math.abs(mid.torsoYaw) < Math.abs(mid.headYaw));
  const back = lookAt(left, 4800)!;
  close(back.headYaw, 0);                                            // back to the person for the end of the reply
  const right = lookAt(lookSteps("right", 5000), 900)!;
  assert(right.headYaw > 0.5 && right.gazeX > 0);
  // short replies still look long enough to be seen
  const short = lookSteps("right", 2000);
  assert(lookAt(short, 900)!.headYaw > 0.4);
}

console.log("AR companion tests passed");
