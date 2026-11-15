/** Local real-source engineering diagnostic; no linguistic accuracy score. */
import fs from "node:fs";
import path from "node:path";
import * as THREE from "three";
import { HumanoidRetargeter } from "../src/utils/retargetHumanoid";
import { handContact, type Capsule } from "../src/utils/handContact";

const root = path.resolve(process.cwd(), "..");
const output = process.argv[2] || path.join(root, "artifacts/avatar-review/measurements.json");

const load = (p: string) => JSON.parse(fs.readFileSync(p, "utf8"));
// Read the real GLB's skeletal graph without textures, GPU, or synthetic rig fallback.
const glb = fs.readFileSync(path.join(root, ".runtime/avatar-candidates/Noa.glb"));
const model = JSON.parse(glb.subarray(20, 20 + glb.readUInt32LE(12)).toString());
function makeRig() {
  const joints = new Set<number>(model.skins.flatMap((skin: { joints: number[] }) => skin.joints));
  const nodes = model.nodes.map((n: { name?: string; translation?: number[]; rotation?: number[]; scale?: number[] }, i: number) => {
    const o = joints.has(i) ? new THREE.Bone() : new THREE.Object3D();
    o.name = n.name || "";
    if (n.translation) o.position.fromArray(n.translation);
    if (n.rotation) o.quaternion.fromArray(n.rotation);
    if (n.scale) o.scale.fromArray(n.scale);
    return o;
  });
  model.nodes.forEach((n: { children?: number[] }, i: number) => n.children?.forEach((j) => nodes[i].add(nodes[j])));
  const rig = new THREE.Group();
  model.scenes[model.scene || 0].nodes.forEach((i: number) => rig.add(nodes[i]));
  rig.updateMatrixWorld(true);
  return rig;
}

import { stitchAnimationPlan } from "../src/utils/planStitcher";
const results=[];
for(let id=1;id<=Number(process.env.CV_CASE_LIMIT || 15);id++) {
 const input=load(path.join(root,`${process.env.CV_MEASURE_INPUT || "artifacts/phrase-review/live-cv-loop1"}/${String(id).padStart(2,"0")}-input.json`));
 const motion=stitchAnimationPlan(input.plan,new Map(input.assets)), rig=makeRig(), retarget=new HumanoidRetargeter(rig);
 let sum=0,max=0,count=0,worst=0,rotationSum=0,rotationMax=0,rotationCount=0,largeRotationSteps=0;
 const previous = new Map<string, THREE.Quaternion>();
 const previousWorld = new Map<string, THREE.Quaternion>();
 const previousWrist = new Map<string, THREE.Vector3>();
 let maxWorldRotation=0,maxFingerRotation=0,maxWristStep=0,largeWristSteps=0;
 for(let t=0;t<=motion.duration_ms;t+=Number(process.env.CV_SAMPLE_MS || 40)) {
  retarget.applyMotion(motion,t);
  rig.traverse(o => {
   if (!(o instanceof THREE.Bone) || !/^(Left|Right)Hand/.test(o.name)) return;
   const before = previous.get(o.name);
   if (before) {
    const degrees = THREE.MathUtils.radToDeg(before.angleTo(o.quaternion));
    rotationSum += degrees; rotationMax = Math.max(rotationMax,degrees); rotationCount++;
    if (degrees > 30) largeRotationSteps++;
    if (o.name !== "LeftHand" && o.name !== "RightHand") maxFingerRotation=Math.max(maxFingerRotation,degrees);
   }
   previous.set(o.name,o.quaternion.clone());
   const world=o.getWorldQuaternion(new THREE.Quaternion()),oldWorld=previousWorld.get(o.name);
   if (oldWorld) maxWorldRotation=Math.max(maxWorldRotation,THREE.MathUtils.radToDeg(oldWorld.angleTo(world)));
   previousWorld.set(o.name,world);
   if (o.name === "LeftHand" || o.name === "RightHand") {
    const position=o.getWorldPosition(new THREE.Vector3()),old=previousWrist.get(o.name);
    if(old){const step=old.distanceTo(position)*1000;maxWristStep=Math.max(maxWristStep,step);if(step>30)largeWristSteps++;}
    previousWrist.set(o.name,position);
   }
  });
  const capsules=(side:string):Capsule[]=>{
   const segment=(a:string,b:string,radius:number)=>{const x=rig.getObjectByName(side+a)!,y=rig.getObjectByName(side+b)!;const pa=x.getWorldPosition(new THREE.Vector3()),pb=y.getWorldPosition(new THREE.Vector3());return {a:pa,b:pb,sourceA:pa,sourceB:pb,sourceObserved:false,radius,kind:a==="ForeArm"?"forearm" as const:"hand" as const};};
   return [segment("ForeArm","Hand",.028),...["Thumb","Index","Middle","Ring","Pinky"].flatMap(f=>[1,2,3].map(i=>segment(`Hand${f}${i}`,`Hand${f}${i+1}`,.008))),...["Index","Middle","Pinky"].map(f=>segment("Hand",`Hand${f}1`,.022))];
  };
  const hit=handContact(capsules("Left"),capsules("Right"));sum+=hit.maxPenetration;if(hit.maxPenetration>max){max=hit.maxPenetration;worst=t;}count++;
 }
 results.push({id,text:input.text,mean_penetration_mm:sum/count*1000,max_penetration_mm:max*1000,worst_time_ms:worst,samples:count,mean_hand_joint_step_degrees:rotationSum/Math.max(1,rotationCount),max_hand_joint_step_degrees:rotationMax,steps_over_30_degrees:largeRotationSteps,max_world_hand_joint_step_degrees:maxWorldRotation,max_finger_local_step_degrees:maxFingerRotation,max_wrist_step_mm:maxWristStep,wrist_steps_over_30mm:largeWristSteps,diagnostics:retarget.diagnostics});
}
fs.writeFileSync(output,JSON.stringify(results,null,2));console.log(JSON.stringify(results));
