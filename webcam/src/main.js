import './style.css';
import {BODY_EDGES,HAND_EDGES,positioning,exportMotion,exportAnimics} from './capture.mjs';

document.querySelector('#app').innerHTML=`
<header class="topbar"><a class="brand" href="/">S<span>signaculture<span class="brand-light"> / webcam</span></span></a><span class="local-pill"><i></i> Capture locale</span></header>
<main><section class="workspace"><div class="workspace-heading"><div><p class="eyebrow">STUDIO DE CAPTURE</p><h1>Le mouvement, en direct.</h1></div><span id="engine" class="badge">Modèles locaux</span></div>
<div class="stage" id="stage"><div class="feed" id="feed"><video id="camera" autoplay muted playsinline></video><canvas id="overlay"></canvas></div><div id="empty"><div class="camera-icon">◉</div><h2>Votre caméra, votre mouvement.</h2><p>Cadrez votre buste et vos mains.<br>Les points sont estimés directement dans ce navigateur.</p><button class="primary" id="start-empty">Activer la caméra <span>↗</span></button></div><div class="stage-top"><span id="live" class="live-badge" hidden><i></i> EN DIRECT</span><span id="replay-badge" class="live-badge" hidden>RELECTURE</span><span id="recording-badge" class="recording-badge" hidden>● REC <b>00:00</b></span></div><span id="fps" class="fps" hidden></span><span id="position-hint" class="position-hint" hidden></span></div>
<div class="controls"><button id="start" class="secondary">Activer la caméra</button><button id="stop-camera" class="ghost" disabled>Éteindre</button><button id="return-live" class="ghost" hidden>Revenir au direct</button><div class="control-divider"></div><button id="record" class="record" disabled><i></i> Enregistrer</button><span id="record-time">00:00 / 00:30</span><span id="frame-count">0 image</span></div>
<div id="message" role="status" aria-live="polite">Prêt. La caméra s’active uniquement à votre demande.</div>
<section class="take"><div class="take-heading"><div><p class="eyebrow">VOTRE PRISE</p><h2>Enregistrer. Revoir. Exporter.</h2></div><span id="take-state" class="badge">Aucune prise</span></div><p id="take-info">Enregistrez un signe pour revoir ses points et exporter le mouvement.</p><input id="timeline" aria-label="Image de la prise enregistrée" type="range" min="0" max="0" value="0" disabled><div class="export-actions"><button id="play" class="secondary" disabled>▶ Revoir les points</button><button id="motion" class="primary" disabled>Convertir en mouvement ↗</button><button id="raw" class="secondary" disabled>Capture JSON</button><button id="animics" class="secondary" disabled>Landmarks Animics</button><button id="video-export" class="ghost" disabled>Vidéo source</button><button id="clear" class="ghost" disabled>Effacer</button></div><p class="fine">La conversion exporte un squelette SignaCulture, sans validation linguistique ni retargeting d’avatar. Les mains non détectées restent absentes.</p></section></section>
<aside><div class="aside-heading"><span class="eyebrow">ESTIMATEUR</span><h2>Lecture du mouvement</h2></div><section class="panel"><h3>Caméra & aperçu</h3><label class="field-label" for="device">Source vidéo</label><select id="device"><option value="">Caméra par défaut</option></select><label class="field-label" for="acceleration">Calcul</label><select id="acceleration"><option value="CPU">CPU · compatible</option><option value="GPU">GPU · expérimental, repli CPU</option></select><label class="toggle">Afficher les points <input id="show-overlay" type="checkbox" checked></label>
<fieldset class="landmark-layers" id="landmark-layers"><legend>Points à afficher</legend>
<label><input type="checkbox" id="layer-body" checked><i class="body-dot"></i> Corps</label>
<label><input type="checkbox" id="layer-left" checked><i class="left-dot"></i> Main gauche</label>
<label><input type="checkbox" id="layer-right" checked><i class="right-dot"></i> Main droite</label>
<label><input type="checkbox" id="layer-face" checked><i class="face-dot"></i> Visage</label>
</fieldset><label class="toggle">Aperçu miroir <input id="mirror" type="checkbox" checked></label><p class="fine">Le miroir modifie seulement l’aperçu, pas les coordonnées exportées.</p></section>
<section class="panel"><h3>Position dans le cadre</h3><p id="position" class="position">En attente de caméra</p><div class="quality-row"><span>Cadrage du buste</span><b id="body-label">—</b></div><meter id="body-meter" min="0" max="1" value="0"></meter><div class="quality-row"><span>Main gauche</span><b id="left-label">—</b></div><meter id="left-meter" min="0" max="1" value="0"></meter><div class="quality-row"><span>Main droite</span><b id="right-label">—</b></div><meter id="right-meter" min="0" max="1" value="0"></meter><p class="fine" id="tracking-coverage" hidden></p><p class="fine">Cadrage estimé, sans mesure de distance. Les côtés sont ceux de la personne filmée.</p></section>
<section class="panel expression"><div class="expression-heading"><h3>Expressions du visage</h3><span id="face-status">Non détecté</span></div><p class="fine">Coefficients de blendshapes mesurés, pas des émotions reconnues.</p><div id="blends"><div class="no-face">Les coefficients apparaîtront quand un visage sera détecté.</div></div></section><p class="privacy">Aucune image envoyée à un serveur.<br>Pas de microphone. Aucun enregistrement automatique.</p></aside></main>
<dialog id="source-dialog"><button id="close-dialog" class="secondary">Fermer</button><video id="source-preview" controls playsinline></video></dialog>`;

const embedded = window.parent !== window;
const $=id=>document.getElementById(id),video=$('camera'),canvas=$('overlay'),ctx=canvas.getContext('2d');
let stream=null,worker=null,ready=null,active=false,busy=false,token=0,raf=0,lastSent=0,lastVideoTime=-1,lastResult=null,lastTimestamp=0;
let recording=false,recordStart=0,frames=[],recorder=null,chunks=[],videoBlob=null,videoURL=null,hasTake=false;
let finalizingVideo = false;
let replay=false,replayStart=0,replayIndex=0,paintTick=0,lastUI=0,delegate='';
const trackingHistory = [];
const layerIds = ['layer-body', 'layer-left', 'layer-right', 'layer-face'];
const colors = {body: '#6ce9bf', left: '#ffe284', right: '#83bbff', face: '#ff9aca'};
const clock=ms=>`${String(Math.floor(ms/60000)).padStart(2,'0')}:${String(Math.floor(ms/1000)%60).padStart(2,'0')}`;
function message(text,error=false){$('message').textContent=text;$('message').classList.toggle('error',error);}
function exportsEnabled(enabled){for(const id of ['play','motion','raw','animics','clear'])$(id).disabled=!enabled;$('timeline').disabled=!enabled;}
function withTimeout(operation, milliseconds, errorMessage) {
  let timer;
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(errorMessage)), milliseconds);
  });
  return Promise.race([operation, deadline]).finally(() => clearTimeout(timer));
}
function ensureDetector(){
  if(ready)return ready;
  const detector=new Worker(new URL('vendor/detector-worker.js', document.baseURI));
  worker=detector;
  const initialized=new Promise((resolve,reject)=>{
    detector.onmessage=({data})=>{
      if(data.type==='ready'){delegate=data.delegate;$('engine').textContent='MediaPipe · '+delegate;resolve();}
      else if(data.type==='error'){reject(Error(data.message));}
      else if(data.type==='result'){
        busy=false;if(!active)return;$('record').disabled=false;
        lastResult={...data,time_ms:Math.round(data.timestamp-recordStart),observedFps:lastTimestamp?1000/(data.timestamp-lastTimestamp):0};lastTimestamp=data.timestamp;
        trackingHistory.push([Boolean(data.pose), Boolean(data.hands.left), Boolean(data.hands.right), Boolean(data.face)]);
        if(trackingHistory.length>60)trackingHistory.shift();
        if(recording&&data.timestamp>=recordStart){frames.push(structuredClone(lastResult));hasTake=true;if(data.timestamp-recordStart>=30000)stopRecording();}
        if(!replay&&!$('stage').classList.contains('replaying')){draw(lastResult);if(performance.now()-lastUI>150){updateMetrics(lastResult);lastUI=performance.now();}}
      }else if(data.type==='frame-error'){busy=false;message('L’image n’a pas pu être analysée. '+data.message,true);}
    };
    detector.onerror=event=>{busy=false;reject(Error('Le moteur de suivi n’a pas pu démarrer : '+event.message));};
    detector.postMessage({type:'init',delegate:$('acceleration').value});
  });
  ready=withTimeout(initialized, 20000, 'Le chargement des modèles de suivi a expiré.')
    .catch(error=>{
      detector.terminate();
      if(worker===detector){worker=null;ready=null;}
      throw error;
    });
  return ready;
}
async function devices(){
  if(!navigator.mediaDevices?.enumerateDevices)return;
  const list=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
  const selected=$('device').value;$('device').replaceChildren(new Option('Caméra par défaut',''));
  list.forEach((d,i)=>$('device').add(new Option(d.label||`Caméra ${i+1}`,d.deviceId)));
  if([...$('device').options].some(o=>o.value===selected))$('device').value=selected;
}
async function startCamera(){
  if(active)return;
  if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia){message('Ouvrez cette page sur localhost ou en HTTPS pour accéder à la caméra.',true);return;}
  const mine=++token;$('start').disabled=true;$('start-empty').disabled=true;message('Autorisez la caméra, puis patientez pendant le chargement des modèles…');
  let acquired;
  try{
    const device=$('device').value;acquired=await navigator.mediaDevices.getUserMedia({audio:false,video:{width:{ideal:960},height:{ideal:540},frameRate:{ideal:24,max:30},...(device?{deviceId:{exact:device}}:{facingMode:'user'})}});
    if(mine!==token){acquired.getTracks().forEach(t=>t.stop());return;}
    stream=acquired;
    stream.getVideoTracks()[0].onended=()=>{stopCamera();message('La caméra a été déconnectée.',true);};
    video.srcObject=stream;
    $('stop-camera').disabled=false;$('device').disabled=true;$('acceleration').disabled=true;
    message('Accès à la caméra accordé. Ouverture du flux vidéo…');
    await withTimeout(video.play(), 8000, 'La caméra est autorisée, mais aucune image n’arrive. Vérifiez qu’elle est disponible, puis réessayez.');
    if(mine!==token)return;
    stopReplay();trackingHistory.length=0;$('tracking-coverage').hidden=true;active=true;lastVideoTime=-1;lastSent=0;lastTimestamp=0;busy=false;$('stage').classList.add('streaming');
    canvas.width=video.videoWidth;canvas.height=video.videoHeight;
    $('feed').style.aspectRatio=`${video.videoWidth}/${video.videoHeight}`;
    $('empty').hidden=true;$('live').hidden=false;$('fps').hidden=false;$('position-hint').hidden=false;
    $('stop-camera').disabled=false;$('record').disabled=true;$('start').textContent='Caméra active';$('device').disabled=true;$('acceleration').disabled=true;
    void devices().catch(()=>{});
  }catch(error){
    if(mine!==token)return;
    stopCamera();
    const text=error.name==='NotAllowedError'?'Accès refusé. Autorisez la caméra dans les paramètres de ce site, puis réessayez.':error.name==='NotFoundError'?'Aucune caméra trouvée sur cet ordinateur.':error.name==='NotReadableError'?'Caméra occupée ou indisponible. Fermez les autres applications qui l’utilisent.':error.message;
    message(text,true);
    return;
  }
  message('Caméra active. Chargement des modèles de suivi…');
  try {
    await ensureDetector();
    if(mine!==token)return;
    message('Caméra active. Placez vos mains dans le cadre, puis lancez une prise.');
    loop();
  } catch(error) {
    if(mine!==token)return;
    if($('acceleration').value==='GPU') {
      void recoverCPU();
      return;
    }
    message(error.message+' Éteignez la caméra puis réactivez-la pour réessayer.',true);
  }
}
function stopCamera(){
  ++token;if(recording)stopRecording();active=false;cancelAnimationFrame(raf);busy=false;lastResult=null;
  stream?.getTracks().forEach(t=>{t.onended=null;t.stop();});stream=null;video.srcObject=null;
  $('stage').classList.remove('streaming');$('live').hidden=true;$('fps').hidden=true;$('position-hint').hidden=true;$('empty').hidden=false;
  $('stop-camera').disabled=true;$('record').disabled=true;$('device').disabled=false;$('acceleration').disabled=false;
  $('start').disabled=false;$('start-empty').disabled=false;$('start').textContent='Activer la caméra';
  if(!$('stage').classList.contains('replaying')){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    $('tracking-coverage').hidden=true;
  }
  message('Caméra éteinte. Votre prise reste disponible dans cette page.');
}
async function recoverCPU(){
  const mine=token;active=false;cancelAnimationFrame(raf);busy=false;worker?.terminate();worker=null;ready=null;$('acceleration').value='CPU';message('Le GPU répond trop lentement. Passage au suivi CPU…');
  try{await ensureDetector();if(mine!==token)return;active=true;lastVideoTime=-1;lastSent=0;lastTimestamp=0;loop();message('Suivi CPU actif.');}catch(error){if(mine!==token)return;active=true;message(error.message+' Éteignez la caméra puis réactivez-la pour réessayer.',true);}
}
async function loop(){
  if(!active)return;
  const now=performance.now();
  if(busy&&now-lastSent>8000&&delegate==='GPU'){void recoverCPU();return;}
  if(recording){const elapsed=now-recordStart;$('record-time').textContent=clock(elapsed)+' / 00:30';$('recording-badge').querySelector('b').textContent=clock(elapsed);if(elapsed>=30000)stopRecording();}
  if(!busy&&now-lastSent>=1000/15&&video.readyState>=2&&video.currentTime!==lastVideoTime){
    busy=true;lastSent=now;lastVideoTime=video.currentTime;
    try{const image=await createImageBitmap(video);if(!active){image.close();busy=false;return;}worker.postMessage({type:'frame',timestamp:now,image},[image]);}
    catch(error){busy=false;message('Impossible de lire la caméra. '+error.message,true);}
  }
  raf=requestAnimationFrame(loop);
}
function draw(frame) {
  const {width, height} = canvas;
  ctx.clearRect(0, 0, width, height);
  if (!$('show-overlay').checked) return;

  function drawLandmarks(landmarks, edges, color, radius, minVisibility = 0) {
    if (!landmarks) return;
    const points = landmarks.map(point =>
      point && Number.isFinite(point.x) && Number.isFinite(point.y) && (point.visibility ?? 1) >= minVisibility
        ? [point.x * width, point.y * height] : null);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (const [from, to] of edges) {
      if (!points[from] || !points[to]) continue;
      ctx.moveTo(...points[from]);
      ctx.lineTo(...points[to]);
    }
    ctx.stroke();
    ctx.beginPath();
    for (const point of points) {
      if (!point) continue;
      ctx.moveTo(point[0] + radius, point[1]);
      ctx.arc(...point, radius, 0, Math.PI * 2);
    }
    ctx.fill();
  }

  if ($('layer-body').checked) drawLandmarks(frame.pose, BODY_EDGES, colors.body, 3, .5);
  if ($('layer-left').checked) drawLandmarks(frame.hands.left, HAND_EDGES, colors.left, 2.4);
  if ($('layer-right').checked) drawLandmarks(frame.hands.right, HAND_EDGES, colors.right, 2.4);
  if ($('layer-left').checked && $('layer-right').checked) {
    for (const hand of frame.unassignedHands ?? []) drawLandmarks(hand, HAND_EDGES, '#fff', 2.4);
  }
  if ($('layer-face').checked) drawLandmarks(frame.face, [], colors.face, 1);
}
function updateMetrics(frame){
  const coverage = $('tracking-coverage');
  coverage.hidden = $('stage').classList.contains('replaying') || !trackingHistory.length;
  if (!coverage.hidden) {
    const rates = ['Corps', 'Gauche', 'Droite', 'Visage'].map((label, i) => {
      const detected = trackingHistory.filter(result => result[i]).length;
      return `${label} ${Math.round(100 * detected / trackingHistory.length)} %`;
    });
    coverage.textContent = `Détections sur ${trackingHistory.length} images récentes · ${rates.join(' · ')}`;
  }
  const info=positioning(frame);$('position').textContent=info.label;$('position').dataset.level=info.level;
  $('position-hint').textContent=info.label;$('body-label').textContent=info.level==='good'?'Bon':'À ajuster';$('body-meter').value=info.level==='good'?1:info.level==='warning'?.4:0;
  for(const side of ['left','right']){$(side+'-label').textContent=frame.hands[side]?'Détectée':'Absente';$(side+'-meter').value=frame.hands[side]?1:0;}
  $('face-status').textContent=frame.face?'Détecté':'Non détecté';$('fps').textContent=`${(frame.observedFps??0).toFixed(1)} img/s · ${Math.round(frame.inferenceMs)} ms`;
  $('frame-count').textContent=frames.length+' images';
  if(!frame.face){$('blends').innerHTML='<div class="no-face">Aucun visage détecté.</div>';return;}
  const categories=Object.entries(frame.blendshapes).filter(([name])=>name!=='_neutral');
  if(!$('blends').querySelector('[data-blend]')){
    $('blends').replaceChildren(...categories.map(([name])=>{const row=document.createElement('div');row.className='blend';row.dataset.blend=name;const label=document.createElement('span');label.textContent=name[0].toUpperCase()+name.slice(1);const bar=document.createElement('progress');bar.max=1;row.append(label,bar);return row;}));
  }
  for(const [name,value] of categories){const bar=$('blends').querySelector(`[data-blend="${name}"] progress`);if(bar)bar.value=value;}
}
function discardVideo(){if(videoURL)URL.revokeObjectURL(videoURL);videoURL=null;videoBlob=null;$('video-export').disabled=true;$('source-preview').removeAttribute('src');}
function startRecording(){
  if(!active)return;stopReplay();finalizingVideo=false;frames=[];hasTake=false;chunks=[];discardVideo();recordStart=performance.now();recording=true;
  exportsEnabled(false);$('record').classList.add('is-recording');$('record').innerHTML='<i></i> Terminer la prise';$('recording-badge').hidden=false;$('take-state').textContent='Enregistrement';
  if(typeof MediaRecorder!=='undefined'){
    const mime=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm','video/mp4'].find(m=>MediaRecorder.isTypeSupported(m));
    try{
      recorder=new MediaRecorder(stream,mime?{mimeType:mime}:{});const takeChunks=chunks;
      recorder.ondataavailable=e=>{if(e.data.size)takeChunks.push(e.data);};
      recorder.onstop=()=>{
        if(chunks!==takeChunks)return;
        finalizingVideo=false;
        videoBlob=new Blob(takeChunks,{type:recorder.mimeType});
        videoURL=URL.createObjectURL(videoBlob);
        $('video-export').disabled=!videoBlob.size;
        if(embedded)$('motion').disabled=!hasTake;
      };
      recorder.onerror=()=>{finalizingVideo=false;if(embedded)$('motion').disabled=!hasTake;};
      recorder.start(250);
    }catch{recorder=null;message('Prise des points active. L’enregistrement vidéo n’est pas disponible dans ce navigateur.');}
  }
  message('Prise en cours. Enregistrez jusqu’à 30 secondes.');
}
function stopRecording(){
  if(!recording)return;recording=false;if(recorder?.state==='recording'){finalizingVideo=embedded;recorder.stop();}
  $('record').classList.remove('is-recording');$('record').innerHTML='<i></i> Nouvelle prise';$('recording-badge').hidden=true;
  const usable=frames.filter(f=>f.pose).length;hasTake=frames.length>0;
  exportsEnabled(hasTake);if(finalizingVideo)$('motion').disabled=true;$('take-state').textContent=hasTake?'Prise enregistrée':'Prise vide';
  $('timeline').max=Math.max(0,frames.length-1);$('timeline').value=0;
  $('take-info').textContent=`${frames.length} images · ${usable} avec corps détecté · ${clock(frames.at(-1)?.time_ms??0)}. Les mesures restent à relire.`;
  message(hasTake?'Prise conservée dans ce navigateur. Vous pouvez la revoir ou l’exporter.':'Aucune image analysée. Attendez que les points apparaissent avant de réessayer.',!hasTake);
}
function stopReplay(){replay=false;cancelAnimationFrame(paintTick);$('play').textContent='▶ Revoir les points';$('stage').classList.remove('replaying');$('replay-badge').hidden=true;$('return-live').hidden=true;$('live').hidden=!active;$('empty').hidden=active;if(lastResult&&active){canvas.width=video.videoWidth;canvas.height=video.videoHeight;$('feed').style.aspectRatio=`${video.videoWidth}/${video.videoHeight}`;draw(lastResult);updateMetrics(lastResult);}}
function previewFrame(i){const frame=frames[i];if(!frame)return;canvas.width=frame.width;canvas.height=frame.height;$('feed').style.aspectRatio=`${frame.width}/${frame.height}`;$('stage').classList.add('replaying');$('replay-badge').hidden=false;$('return-live').hidden=false;$('live').hidden=true;$('empty').hidden=true;draw(frame);updateMetrics(frame);$('timeline').value=i;}
function playReplay(){
  if(!frames.length)return;if(replay){stopReplay();return;}replay=true;replayIndex=0;replayStart=performance.now();$('play').textContent='■ Arrêter la revue';
  const step=()=>{if(!replay)return;const t=performance.now()-replayStart;while(replayIndex<frames.length-1&&frames[replayIndex+1].time_ms<=t)replayIndex++;previewFrame(replayIndex);if(replayIndex===frames.length-1){replay=false;$('play').textContent='▶ Revoir les points';return;}paintTick=requestAnimationFrame(step);};step();
}
function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1500);}
function saveJSON(value,name){download(new Blob([JSON.stringify(value)],{type:'application/json'}),name);}
$('return-live').onclick=stopReplay;$('start').onclick=$('start-empty').onclick=startCamera;$('stop-camera').onclick=stopCamera;
$('record').onclick=()=>recording?stopRecording():startRecording();
$('show-overlay').onchange = () => {
  $('landmark-layers').disabled = !$('show-overlay').checked;
  redrawOverlay();
};
function redrawOverlay() {
  const frame = $('stage').classList.contains('replaying') ? frames[Number($('timeline').value)] : lastResult;
  if (frame) draw(frame);
}
for (const id of layerIds) $(id).onchange = redrawOverlay;
$('mirror').onchange=()=>{$('feed').classList.toggle('mirrored',$('mirror').checked);};$('feed').classList.add('mirrored');
$('play').onclick=playReplay;$('timeline').oninput=()=>{stopReplay();previewFrame(Number($('timeline').value));};
if (embedded) $('motion').textContent = 'Utiliser dans le Studio';
$('motion').onclick = () => {
  try {
    const motion = exportMotion(frames);
    if (embedded) {
      window.parent.postMessage({ type: 'signaculture:webcam-motion', motion, video: videoBlob }, window.location.origin);
      return;
    }
    saveJSON(motion, 'webcam-motion.json');
    message(`Mouvement converti : ${motion.frames.length} images, ${motion.duration_ms} ms. Importable comme JSON dans SignaCulture.`);
  } catch (error) {
    message(error.message, true);
  }
};
$('raw').onclick=()=>saveJSON({format:'signaculture-webcam-capture-v1',time_unit:'milliseconds',sdk:'@mediapipe/tasks-vision@0.10.32',display_mirrored:$('mirror').checked,coordinates_mirrored:false,frames},'webcam-capture.json');
$('animics').onclick=()=>{saveJSON(exportAnimics(frames),'webcam-animics-milliseconds.json');message('Landmarks exportés au format Animics actuel (dt en millisecondes). Pour le dashboard SignaCulture, utilisez « Convertir en mouvement ».');};
$('video-export').onclick=()=>{if(videoBlob){download(videoBlob,videoBlob.type.includes('mp4')?'webcam-source.mp4':'webcam-source.webm');$('source-preview').src=videoURL;$('source-dialog').showModal();}};
$('close-dialog').onclick=()=>{$('source-preview').pause();$('source-dialog').close();};
$('clear').onclick=()=>{stopReplay();frames=[];chunks=[];hasTake=false;discardVideo();exportsEnabled(false);$('take-state').textContent='Aucune prise';$('take-info').textContent='Enregistrez un signe pour revoir ses points et exporter le mouvement.';$('frame-count').textContent='0 image';if(!active)ctx.clearRect(0,0,canvas.width,canvas.height);message('Prise effacée de cette page.');};
$('acceleration').onchange=()=>{worker?.terminate();worker=null;ready=null;};
$('device').onchange=()=>{if(active){stopCamera();startCamera();}};
window.addEventListener('pagehide',()=>{stopCamera();worker?.terminate();discardVideo();});
devices().catch(()=>{});navigator.mediaDevices?.addEventListener('devicechange',()=>devices().catch(()=>{}));
