import { FilesetResolver,PoseLandmarker,HandLandmarker,FaceLandmarker } from '@mediapipe/tasks-vision';
import {assignHands} from './capture.mjs';
import {faceROI,handROI,remap} from './roi.mjs';

let pose,hands,face;
const zoomCanvas=new OffscreenCanvas(256,256),zoomContext=zoomCanvas.getContext('2d');
function crop(image,box){zoomContext.drawImage(image,box.x,box.y,box.size,box.size,0,0,256,256);return zoomCanvas;}
function close(){for(const m of [pose,hands,face])m?.close();pose=hands=face=null;}
async function init(delegate){
  const vision=await FilesetResolver.forVisionTasks('/vendor/mediapipe');
  pose=await PoseLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:'/models/pose.task',delegate},runningMode:'VIDEO',numPoses:1,minPoseDetectionConfidence:.5,minPosePresenceConfidence:.5,minTrackingConfidence:.5});
  hands=await HandLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:'/models/hands.task',delegate},runningMode:'IMAGE',numHands:2,minHandDetectionConfidence:.4,minHandPresenceConfidence:.4,minTrackingConfidence:.4});
  face=await FaceLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:'/models/face.task',delegate},runningMode:'IMAGE',numFaces:1,minFaceDetectionConfidence:.1,minFacePresenceConfidence:.4,outputFaceBlendshapes:true,outputFacialTransformationMatrixes:true});
}
self.onmessage=async({data})=>{
  if(data.type==='init'){
    try{
      let delegate=data.delegate==='CPU'?'CPU':'GPU';
      try{await init(delegate);}catch{close();delegate='CPU';await init(delegate);}
      self.postMessage({type:'ready',delegate});
    }catch(error){close();self.postMessage({type:'error',message:'Chargement des modèles impossible. Exécutez npm run setup dans webcam, puis réessayez. '+error.message});}
  }else if(data.type==='frame'){
    const {image,timestamp}=data;
    try{
      const started=performance.now(),p=pose.detectForVideo(image,timestamp),h=hands.detect(image);
      let f=face.detect(image),faceSource=f.faceLandmarks.length?'full-frame':'missing';
      const landmarks=p.landmarks[0]??null,candidates=[...h.landmarks],worlds=[...h.worldLandmarks],sources=candidates.map(()=> 'full-frame');
      let association=assignHands(landmarks,candidates);
      for(const side of ['left','right']){
        if(association[side])continue;
        const box=handROI(landmarks,side,image.width,image.height);if(!box)continue;
        const detected=hands.detect(crop(image,box));
        for(let i=0;i<detected.landmarks.length;i++){
          const points=remap(detected.landmarks[i],box,image.width,image.height);
          if(candidates.some(c=>Math.hypot(c[0].x-points[0].x,c[0].y-points[0].y)<.025))continue;
          candidates.push(points);worlds.push(detected.worldLandmarks[i]);sources.push('pose-roi');
        }
        association=assignHands(landmarks,candidates);
      }
      if(!f.faceLandmarks.length){
        const box=faceROI(landmarks,image.width,image.height);
        if(box){const detected=face.detect(crop(image,box));if(detected.faceLandmarks.length){f={...detected,faceLandmarks:[remap(detected.faceLandmarks[0],box,image.width,image.height)],facialTransformationMatrixes:[]};faceSource='pose-roi';}}
      }
      self.postMessage({type:'result',timestamp,width:image.width,height:image.height,pose:landmarks,poseWorld:p.worldLandmarks[0]??null,hands:association,
        unassignedHands:candidates.filter((_,i)=>!Object.values(association.indices).includes(i)),
        handSources:Object.fromEntries(['left','right'].map(s=>[s,association.indices[s]!==null?sources[association.indices[s]]:'missing'])),faceSource,
        leftWorld:association.indices.left!==null?worlds[association.indices.left]:null,
        rightWorld:association.indices.right!==null?worlds[association.indices.right]:null,
        face:f.faceLandmarks[0]??null,blendshapes:Object.fromEntries((f.faceBlendshapes[0]?.categories??[]).map(c=>[c.categoryName,c.score])),
        faceMatrix:f.facialTransformationMatrixes[0]?.data?Array.from(f.facialTransformationMatrixes[0].data):null,inferenceMs:performance.now()-started});
    }catch(error){self.postMessage({type:'frame-error',message:error.message});}
    finally{image.close();}
  }
};
