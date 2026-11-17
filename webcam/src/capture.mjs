export const BODY={head:0,left_shoulder:11,right_shoulder:12,left_elbow:13,right_elbow:14,left_wrist:15,right_wrist:16};
export const FINGERS={thumb:[1,2,3,4],index:[5,6,7,8],middle:[9,10,11,12],ring:[13,14,15,16],pinky:[17,18,19,20]};
export const BODY_EDGES=[[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24]];
export const HAND_EDGES=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);

export function assignHands(pose,hands){
  const result={left:null,right:null,indices:{left:null,right:null},distances:{left:null,right:null}};
  if(!pose?.[16])return result;
  const sides=['left','right'];let best={count:-1,cost:Infinity};
  for(let l=-1;l<hands.length;l++)for(let r=-1;r<hands.length;r++){
    if(l>=0&&l===r)continue;
    const choices=[l,r];let count=0,cost=0,valid=true;
    for(let s=0;s<2;s++)if(choices[s]>=0){
      const wrist=pose[s?16:15];const gap=distance(wrist,hands[choices[s]][0]);
      if((wrist.visibility??1)<.3||gap>.18){valid=false;break;}
      count++;cost+=gap;
    }
    if(valid&&(count>best.count||(count===best.count&&cost<best.cost)))best={count,cost,choices};
  }
  for(let i=0;i<2;i++){
    const index=best.choices?.[i]??-1;if(index<0)continue;
    result[sides[i]]=hands[index];result.indices[sides[i]]=index;
    result.distances[sides[i]]=distance(pose[i?16:15],hands[index][0]);
  }
  return result;
}

export function positioning(frame){
  const p=frame.pose;
  if(!p?.[24])return {label:'Personne non détectée',level:'missing',shoulderWidth:0};
  const width=distance(p[11],p[12]);
  const visible=[11,12,13,14].every(i=>(p[i].visibility??0)>.5);
  if(!visible)return {label:'Gardez les épaules et les coudes visibles',level:'warning',shoulderWidth:width};
  if(width>.48)return {label:'Reculez légèrement',level:'warning',shoulderWidth:width};
  if(width<.16)return {label:'Rapprochez-vous légèrement',level:'warning',shoulderWidth:width};
  if(Math.abs((p[11].x+p[12].x)/2-.5)>.22)return {label:'Recentrez-vous dans le cadre',level:'warning',shoulderWidth:width};
  return {label:'Buste bien cadré',level:'good',shoulderWidth:width};
}

function faceMeasurements(points,width,height){
  if(!points?.[386])return null;
  const ex=(points[263].x-points[33].x)*width,ey=(points[263].y-points[33].y)*height,d=Math.hypot(ex,ey);
  if(d<1)return null;
  const gap=(a,b)=>Math.abs((points[a].x-points[b].x)*width*(-ey/d)+(points[a].y-points[b].y)*height*(ex/d))/d*width/height;
  return {mouth_open:gap(13,14),brow_left:gap(105,159),brow_right:gap(334,386)};
}

export function frameToMotion(frame){
  const p=frame.pose;if(!p?.[24])return null;
  const width=frame.width,height=frame.height;
  const shoulder=Math.hypot((p[11].x-p[12].x)*width,(p[11].y-p[12].y)*height);
  if(shoulder<3)return null;
  const scale=.4/shoulder,world=frame.poseWorld;
  const origin=world?.[24]?['x','y','z'].map(a=>(world[23][a]+world[24][a])/2):['x','y','z'].map(a=>(p[23][a]+p[24][a])/2);
  const body=i=>world?.[24]?[world[i].x-origin[0],1-(world[i].y-origin[1]),-(world[i].z-origin[2])]:[(p[i].x-origin[0])*width*scale,1-(p[i].y-origin[1])*height*scale,-(p[i].z-origin[2])*width*scale];
  const joints=Object.fromEntries(Object.entries(BODY).map(([name,i])=>[name,body(i)]));
  joints.hips=[0,1,0];joints.neck=body(11).map((v,i)=>(v+body(12)[i])/2);
  for(const side of ['left','right']){
    const hand=frame.hands[side];if(!hand)continue;
    const anchor=joints[side+'_wrist'];
    for(const [finger,indices] of Object.entries(FINGERS))indices.forEach((i,n)=>{
      joints[`${side}_${finger}_${n+1}`]=[anchor[0]+(hand[i].x-hand[0].x)*width*scale,anchor[1]-(hand[i].y-hand[0].y)*height*scale,anchor[2]-(hand[i].z-hand[0].z)*width*scale];
    });
  }
  if(Object.values(joints).some(p=>p.some(v=>!Number.isFinite(v))))return null;
  return {time_ms:Math.round(frame.time_ms),joints,face:faceMeasurements(frame.face,width,height),
    face_blendshapes:frame.blendshapes,
    hand_tracking:Object.fromEntries(['left','right'].map(side=>[side,{observed:!!frame.hands[side],source:frame.hands[side]?'tasks-hand-landmarker-pose-association':'missing',wrist_disagreement_normalized:frame.hands.distances[side]}]))};
}

export function exportMotion(raw){
  const frames=[];
  for(const f of raw){const m=frameToMotion(f);if(m&&(!frames.length||m.time_ms>frames.at(-1).time_ms))frames.push(m);}
  if(frames.length<2)throw Error('Enregistrez au moins deux images avec le buste détecté.');
  const first=frames[0].time_ms;frames.forEach(f=>f.time_ms-=first);
  const duration=frames.at(-1).time_ms;if(duration<=0)throw Error('Enregistrement trop court.');
  const count=raw.length;
  return {rig_version:'signaculture-v1',fps:(frames.length-1)*1000/duration,duration_ms:duration,
    joints:[...Object.keys(BODY),'hips','neck'],tracked_joints:[...new Set(frames.flatMap(f=>Object.keys(f.joints)))],frames,synthetic:false,
    capture_method:'browser-mediapipe-tasks-0.10.32-webcam-v1',face_feature_method:'eye-normal-projection-legacy-scale-v1',
    quality:{sampled_frames:count,pose_frames:frames.length,pose_coverage:frames.length/count,left_hand_coverage:raw.filter(f=>f.hands.left).length/count,right_hand_coverage:raw.filter(f=>f.hands.right).length/count,face_coverage:raw.filter(f=>f.face).length/count},
    source_time_offset_ms:first,warnings:['Profondeur monoculaire approximative. Les mains absentes restent absentes.','Début et fin limités aux images corporelles observées ; les lacunes internes sont conservées.']};
}

export function exportAnimics(raw){
  const world=points=>points?.map(p=>({...p,y:-p.y,z:-p.z}))??null;
  return raw.map((f,i)=>({dt:i?f.time_ms-raw[i-1].time_ms:0,time_ms:f.time_ms,image_width:f.width,image_height:f.height,
    PLM:f.pose,PWLM:world(f.poseWorld),LLM:f.hands.left,RLM:f.hands.right,LWLM:world(f.leftWorld),RWLM:world(f.rightWorld),FLM:f.face,
    blendshapes:f.blendshapes,facial_transformation_matrix:f.faceMatrix,time_unit:'milliseconds',world_axes:'x-right-y-up-z-toward-camera'}));
}
