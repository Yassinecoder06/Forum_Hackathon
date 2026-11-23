import assert from "node:assert/strict";
import * as THREE from "three";
import { parseSubtitles } from "../src/utils/subtitles";
import { castOf, cueTurns, defaultCast, namePrefix, speakersOf } from "../src/utils/speakers";
import { applyRoleShift, createRoleShiftRig, filmSegments, FRONTAL, locusSide, ROLE_SHIFT, roleShiftAt, roleShiftSteps, roleShiftValue } from "../src/utils/roleShift";
import { turnHead } from "../src/utils/engineFace";
import { RPM_PROFILE } from "../src/utils/rigProfiles";

// ---- Speakers from subtitles ----------------------------------------------------------------------------------
{
  const vtt = `WEBVTT

00:00:01.000 --> 00:00:03.000
<v Marie>Bonjour Paul, tu viens ce soir ?</v>

00:00:03.500 --> 00:00:05.000
<v.loud Paul>Oui, je viens.

00:00:05.200 --> 00:00:06.000
Merci.
`;
  const cues = parseSubtitles(vtt);
  assert.deepEqual(cues.map((c) => [c.speaker, c.text]), [["marie", "Bonjour Paul, tu viens ce soir ?"], ["paul", "Oui, je viens."], [undefined, "Merci."]]);
  assert.equal(cues[1].speakerLabel, "Paul");
  // two voices in one line
  const two = cueTurns(["<v Marie>Tu viens ?</v> <v Paul>Oui.</v>"]);
  assert.deepEqual(two.map((t) => [t.speaker, t.text]), [["marie", "Tu viens ?"], ["paul", "Oui."]]);
}
{
  const srt = `1
00:00:01,000 --> 00:00:03,000
MARIE : Tu as vu le directeur ?

2
00:00:03,200 --> 00:00:05,000
Paul: Oui, il est entré dans la salle

3
00:00:05,100 --> 00:00:06,500
et il a fermé la porte.

4
00:00:09,000 --> 00:00:10,000
Attention : la séance commence.

5
00:00:11,000 --> 00:00:12,000
[rires] [MARIE] Très drôle.
`;
  const cues = parseSubtitles(srt);
  assert.deepEqual(cues.map((c) => c.speaker), ["marie", "paul", "paul", undefined, "marie"]);
  assert.equal(cues[0].text, "Tu as vu le directeur ?");
  assert.equal(cues[0].speakerLabel, "Marie");
  assert.equal(cues[3].text, "Attention : la séance commence.", "an interjection with a colon is no speaker");
  assert.deepEqual(cues[4].sounds, ["rires"]);
  assert.equal(cues[4].text, "Très drôle.");
  assert.equal(namePrefix("12 : 30 à la gare"), null);
  assert.equal(namePrefix("Jean-Pierre : Salut")?.name, "Jean-Pierre");
}
{
  // dash lines: one cue per speaker, time shared by text length, two alternating anonymous voices
  const srt = `1
00:00:10,000 --> 00:00:14,000
- Tu viens avec nous ?
- Non.

2
00:00:15,000 --> 00:00:17,000
- Pourquoi ?
- SOPHIE : Parce que je travaille.
`;
  const cues = parseSubtitles(srt);
  assert.deepEqual(cues.map((c) => [c.speaker, c.text]), [
    ["voix-1", "Tu viens avec nous ?"], ["voix-2", "Non."], ["voix-1", "Pourquoi ?"], ["sophie", "Parce que je travaille."]]);
  assert.equal(cues[0].start_ms, 10000);
  assert.equal(cues[1].end_ms, 14000);
  assert.equal(cues[0].end_ms, cues[1].start_ms);
  assert(cues[0].end_ms - cues[0].start_ms > cues[1].end_ms - cues[1].start_ms, "longer line, longer share");
  // cast sheet: order of first appearance, L / R / C, previous edits kept
  const cast = defaultCast(cues);
  assert.deepEqual(cast.map((c) => [c.id, c.locus]), [["voix-1", "L"], ["voix-2", "R"], ["sophie", "C"]]);
  assert.deepEqual(speakersOf(cues).map((s) => s.first), [0, 1, 3]);
  const edited = defaultCast(cues, [{ ...cast[2], label: "Sophie D.", locus: "R" }]);
  assert.deepEqual([edited[2].label, edited[2].locus], ["Sophie D.", "R"]);
  assert.equal(castOf(edited, "SOPHIE")?.id, "sophie", "a name from the text finds its cast entry");
}
{
  // plain subtitles are unchanged (no speaker, tags removed)
  const cues = parseSubtitles("1\n00:00:01,000 --> 00:00:02,000\n<i>Le musée</i> est fermé.\n{\\an8}Demain.\n");
  assert.deepEqual(cues, [{ start_ms: 1000, end_ms: 2000, text: "Le musée est fermé. Demain." }]);
}

// ---- Role-shift curve -------------------------------------------------------------------------------------------
{
  const segs = [
    { start_ms: 1000, end_ms: 3000, side: locusSide("L") },
    { start_ms: 3200, end_ms: 5000, side: locusSide("R") },
    { start_ms: 5300, end_ms: 6000, side: locusSide("R") },     // same speaker after a short pause: held
    { start_ms: 9000, end_ms: 10000, side: locusSide("N") },    // narrator
  ];
  const steps = roleShiftSteps(segs);
  const v = (t: number) => roleShiftValue(steps, t);
  assert.equal(v(0), 0);
  assert.equal(v(1000), -1, "fully shifted when the line starts");
  assert.equal(v(2800), -1, "held until the turn change starts");
  assert.equal(v(3200), 1);
  assert(v(3200 - ROLE_SHIFT.leadMs) > -0.05 && v(3200 - ROLE_SHIFT.leadMs) < 0.05, "half-way between two speakers");
  assert.equal(v(5150), 1, "held across a short pause of the same speaker");
  assert.equal(v(7000), 0, "back to frontal after a long pause");
  assert.equal(v(9500), 0);
  // smooth: no jump between 1 ms samples, no overshoot
  let maxStep = 0;
  for (let t = 0; t < 11000; t++) {
    maxStep = Math.max(maxStep, Math.abs(v(t + 1) - v(t)));
    assert(Math.abs(v(t)) <= 1);
  }
  assert(maxStep < 0.01, `smooth ease, max step ${maxStep}`);
  // deterministic: a pure function of t, whatever the evaluation order (seek, export)
  const times = Array.from({ length: 200 }, (_, i) => (i * 7919) % 11000);
  const forward = times.map((t) => roleShiftAt(steps, t));
  const backward = [...times].reverse().map((t) => roleShiftAt(steps, t)).reverse();
  assert.deepEqual(forward, backward);
  assert.deepEqual(roleShiftSteps(segs), roleShiftSteps([...segs].reverse()));
  const pose = roleShiftAt(steps, 4000);
  assert(Math.abs(pose.torsoYaw - THREE.MathUtils.degToRad(ROLE_SHIFT.torsoDeg)) < 1e-9);
  assert(pose.headYaw > pose.torsoYaw && pose.gazeX > 0, "head turns more than the torso, gaze follows");
  assert.equal(roleShiftAt(steps, 9500), FRONTAL);
}
{
  // a cue with a quoted clause: sub-segment toward the quoted person
  const segs = filmSegments([{ start_ms: 1000, end_ms: 5000 }, { start_ms: 6000, end_ms: 6000 }], [0, 1],
    [[{ start_ms: 2000, end_ms: 3000, side: -1 }]]);
  assert.deepEqual(segs, [{ start_ms: 1000, end_ms: 3000, side: 0 }, { start_ms: 3000, end_ms: 4000, side: -1 }, { start_ms: 4000, end_ms: 5000, side: 0 }]);
  assert.deepEqual(filmSegments([{ start_ms: 0, end_ms: 100 }], [null]), [], "a cue without motion has no segment");
}

// ---- Role-shift overlay on a skeleton, together with the grammatical head motion ---------------------------------
{
  const build = () => {
    const root = new THREE.Group();
    let parent: THREE.Object3D = root;
    for (const [name, y] of [["Hips", 1], ["Spine", 0.1], ["Spine1", 0.12], ["Spine2", 0.12], ["Neck", 0.15], ["Head", 0.1], ["RightHand", 0]] as const) {
      const bone = new THREE.Bone();
      bone.name = name;
      bone.position.set(name === "RightHand" ? -0.3 : 0, y, name === "RightHand" ? 0.3 : 0);
      parent.add(bone);
      parent = name === "RightHand" ? parent : bone;
    }
    // the hand hangs from Spine2 in this toy rig (stands for the arm chain)
    const hand = root.getObjectByName("RightHand")!;
    root.getObjectByName("Spine2")!.add(hand);
    root.updateMatrixWorld(true);
    return root;
  };
  const headAngles = (root: THREE.Object3D) => {
    const f = new THREE.Vector3(0, 0, 1).applyQuaternion(root.getObjectByName("Head")!.getWorldQuaternion(new THREE.Quaternion()));
    return { yaw: Math.atan2(f.x, f.z), pitch: Math.asin(THREE.MathUtils.clamp(-f.y, -1, 1)) };
  };
  const pose = { torsoYaw: THREE.MathUtils.degToRad(12), headYaw: THREE.MathUtils.degToRad(22), gazeX: 0.3 };
  // zero pose leaves the skeleton untouched
  {
    const root = build(), rig = createRoleShiftRig(root, RPM_PROFILE)!;
    const before = root.getObjectByName("Head")!.getWorldQuaternion(new THREE.Quaternion());
    applyRoleShift(rig, FRONTAL);
    assert(root.getObjectByName("Head")!.getWorldQuaternion(new THREE.Quaternion()).angleTo(before) < 1e-12);
  }
  // torso and head reach the requested headings; the hand turns with the torso
  {
    const root = build(), rig = createRoleShiftRig(root, RPM_PROFILE)!;
    const hand0 = root.getObjectByName("RightHand")!.getWorldPosition(new THREE.Vector3());
    applyRoleShift(rig, pose);
    assert(Math.abs(headAngles(root).yaw - pose.headYaw) < 1e-9);
    const f = new THREE.Vector3(0, 0, 1).applyQuaternion(root.getObjectByName("Spine2")!.getWorldQuaternion(new THREE.Quaternion()));
    assert(Math.abs(Math.atan2(f.x, f.z) - pose.torsoYaw) < 1e-9);
    const hand1 = root.getObjectByName("RightHand")!.getWorldPosition(new THREE.Vector3());
    assert(Math.abs(hand1.y - hand0.y) < 1e-9 && hand1.distanceTo(hand0) > 0.01, "hands travel with the body, same height");
  }
  // negation head shake: still oscillates, around the shifted heading, with the same amplitude
  for (const nmYaw of [-0.17, -0.05, 0, 0.08, 0.17]) {
    const root = build(), rig = createRoleShiftRig(root, RPM_PROFILE)!;
    turnHead(root.getObjectByName("Head")!, nmYaw, 0);
    applyRoleShift(rig, pose);
    assert(Math.abs(headAngles(root).yaw - (pose.headYaw + nmYaw)) < 1e-9, `shake ${nmYaw} preserved`);
  }
  // question head tilt: the pitch relative to the body is unchanged by the role shift
  {
    const plain = build();
    turnHead(plain.getObjectByName("Head")!, 0, 0.1);
    const shifted = build(), rig = createRoleShiftRig(shifted, RPM_PROFILE)!;
    turnHead(shifted.getObjectByName("Head")!, 0, 0.1);
    applyRoleShift(rig, pose);
    assert(Math.abs(headAngles(shifted).pitch - headAngles(plain).pitch) < 1e-9, "question tilt kept");
    assert(Math.abs(headAngles(shifted).yaw - pose.headYaw) < 1e-9);
  }
}

console.log("film speakers + role shift tests passed");
