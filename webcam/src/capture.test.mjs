import test from 'node:test';
import assert from 'node:assert/strict';
import {assignHands,exportMotion,exportAnimics,positioning} from './capture.mjs';
import {faceROI,handROI,remap} from './roi.mjs';

function fixture(time_ms=0){
  const pose=Array.from({length:33},()=>({x:.5,y:.5,z:0,visibility:1}));
  pose[11]={x:.65,y:.3,z:0,visibility:1};pose[12]={x:.35,y:.3,z:0,visibility:1};
  pose[15]={x:.7,y:.5,z:0,visibility:1};pose[16]={x:.3,y:.5,z:0,visibility:1};
  const left=Array.from({length:21},(_,i)=>({x:.7+i*.001,y:.5,z:-i*.001}));
  const right=Array.from({length:21},(_,i)=>({x:.3-i*.001,y:.5,z:-i*.001}));
  return {time_ms,width:960,height:540,pose,poseWorld:null,hands:assignHands(pose,[right,left]),face:null,blendshapes:{},leftWorld:null,rightWorld:null};
}
test('anatomical hands use one-to-one pose wrist association, not screen left/right',()=>{
  const f=fixture();assert.equal(f.hands.indices.left,1);assert.equal(f.hands.indices.right,0);
  assert.equal(assignHands(null,[f.hands.left]).left,null);
  const far=Array.from({length:21},()=>({x:.99,y:.99,z:0}));assert.equal(assignHands(f.pose,[far]).left,null);
});
test('occluded hands remain sparse in native motion export',()=>{
  const a=fixture(100),b=fixture(200);b.hands=assignHands(b.pose,[]);
  const m=exportMotion([a,b]);assert.equal(m.frames[0].time_ms,0);assert.equal(m.duration_ms,100);
  assert.ok(m.frames[0].joints.left_index_4);assert.equal(m.frames[1].joints.left_index_4,undefined);
  assert.equal(m.quality.left_hand_coverage,.5);assert.equal(m.source_time_offset_ms,100);
});
test('shoulder scale rejects outliers and expires across tracking gaps',()=>{
  const frames=[fixture(0),fixture(40),fixture(80),fixture(120),fixture(500)];
  frames[2].pose[11].x=.90;
  frames[3].pose[11].x=.90;
  frames[3].pose[11].visibility=.2;
  frames[4].pose[11].x=.90;
  const lengths=exportMotion(frames).frames.map(({joints})=>
    Math.hypot(...joints.left_index_4.map((v,i)=>v-joints.left_wrist[i])));
  for(const length of lengths.slice(1,4))assert.ok(Math.abs(length-lengths[0])<1e-12);
  assert.ok(lengths[4]<lengths[0]*.6);
});
test('missing pose frames preserve actual gaps and are reported in coverage',()=>{
  const a=fixture(20),gap={...fixture(100),pose:null},b=fixture(240);
  const m=exportMotion([a,gap,b]);assert.equal(m.duration_ms,220);assert.equal(m.quality.pose_coverage,2/3);
  assert.throws(()=>exportMotion([gap]),/au moins deux/);
});
test('duplicate rounded timestamps do not create invalid native frame ordering',()=>{
  const m=exportMotion([fixture(1),fixture(1),fixture(100)]);assert.equal(m.frames.length,2);assert.equal(m.frames.at(-1).time_ms,m.duration_ms);
});
test('Animics export has millisecond deltas, explicit world axis conversion and no mutation',()=>{
  const a=fixture(),b=fixture(80);a.poseWorld=[{x:1,y:2,z:3}];const before=structuredClone(a);
  const e=exportAnimics([a,b]);assert.equal(e[1].dt,80);assert.equal(e[1].time_unit,'milliseconds');
  assert.deepEqual(e[0].PWLM,[{x:1,y:-2,z:-3}]);assert.deepEqual(a,before);
});
test('cadrage is an image heuristic and missing person is explicit',()=>{
  assert.equal(positioning(fixture()).level,'good');assert.equal(positioning({pose:null}).level,'missing');
});
test('zoom landmarks return to original non-square camera coordinates',()=>{
  assert.deepEqual(remap([{x:.5,y:.5,z:.2}],{x:100,y:20,size:200},1000,500),[{x:.2,y:.24,z:.04}]);
  const f=fixture();
  for(const box of [faceROI(f.pose,960,540),handROI(f.pose,'left',960,540)]){
    assert.ok(box.x>=0&&box.y>=0&&box.x+box.size<=960&&box.y+box.size<=540);
  }
  assert.equal(handROI(null,'left',960,540),null);
});
