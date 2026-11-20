"""Render synchronized body/hand/face XY trajectories; explicitly not a 3D avatar."""
import argparse,json
from pathlib import Path
import numpy as np
from src.data.signverse import BONES

def write_html(path,arrays,title='Experimental 2D pose preview'):
    # Shared timeline for all landmarks; frames with masked landmarks are not drawn.
    xy=np.asarray(arrays['smoothed_xy']);mask=np.asarray(arrays['mask']);frames=np.asarray(arrays['frame_ids'])
    if len(xy)>2400:raise ValueError('Preview budget 2400 frames')
    payload=json.dumps({'xy':xy.round(4).tolist(),'mask':mask.tolist(),'frames':frames.tolist(),'bones':BONES},allow_nan=False).replace('</','<\\/')
    import html
    page='''<!doctype html><meta charset="utf-8"><title>Pose review</title><style>body{background:#15191e;color:white;font:16px sans-serif}canvas{background:#252b33}input{width:700px}</style><h2>'''+html.escape(title)+'''</h2><p>2D reconstruction · pending LSF review · confidence-masked hands and face · no inferred depth</p><canvas id="c" width="900" height="700"></canvas><br><button id="play">Play/pause</button><input id="slider" type="range" min="0"><span id="time"></span><script>const d='''+payload+''';const c=document.querySelector('#c'),ctx=c.getContext('2d'),s=document.querySelector('#slider');s.max=d.xy.length-1;let playing=false;function draw(){let i=+s.value;ctx.clearRect(0,0,900,700);const p=d.xy[i],m=d.mask[i],pt=j=>[450+p[j][0]*150,200+p[j][1]*150];ctx.strokeStyle='#aaa';for(const [a,b] of d.bones){if(!m[a]||!m[b])continue;ctx.beginPath();ctx.moveTo(...pt(a));ctx.lineTo(...pt(b));ctx.stroke();}for(let j=0;j<128;j++){if(!m[j])continue;ctx.fillStyle=j<18?'#fff':j<39?'#43c9ff':j<60?'#ffbb55':'#b7ea85';ctx.beginPath();ctx.arc(...pt(j),j<60?3:1.5,0,7);ctx.fill();}document.querySelector('#time').textContent=(d.frames[i]/24).toFixed(2)+' s';}s.oninput=draw;document.querySelector('#play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){s.value=(+s.value+1)%d.xy.length;draw();}},1000/24);draw();</script>'''
    Path(path).write_text(page)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('npz');p.add_argument('html');a=p.parse_args();write_html(a.html,np.load(a.npz,allow_pickle=False))
