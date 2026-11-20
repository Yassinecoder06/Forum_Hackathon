function square(cx,cy,size,width,height){
  size=Math.min(size,width,height);
  return {x:Math.max(0,Math.min(width-size,cx-size/2)),y:Math.max(0,Math.min(height-size,cy-size/2)),size};
}
export function faceROI(p,width,height){
  if(!p?.[12]||(p[0].visibility??0)<.45)return null;
  const shoulder=Math.hypot((p[11].x-p[12].x)*width,(p[11].y-p[12].y)*height);
  const ears=Math.hypot((p[7].x-p[8].x)*width,(p[7].y-p[8].y)*height);
  return square(p[0].x*width,p[0].y*height,Math.max(96,shoulder*.8,ears*1.9),width,height);
}
export function handROI(p,side,width,height){
  const wi=side==='left'?15:16,ei=side==='left'?13:14;
  if(!p?.[wi]||(p[wi].visibility??0)<.3||p[wi].x<0||p[wi].x>1||p[wi].y<0||p[wi].y>1)return null;
  const wrist={x:p[wi].x*width,y:p[wi].y*height};
  const dx=wrist.x-p[ei].x*width,dy=wrist.y-p[ei].y*height,len=Math.hypot(dx,dy);
  const shoulder=Math.hypot((p[11].x-p[12].x)*width,(p[11].y-p[12].y)*height);
  const size=Math.max(128,shoulder*.85,len*.9);
  return square(wrist.x+(len?dx/len:0)*size*.18,wrist.y+(len?dy/len:0)*size*.18,size,width,height);
}
export function remap(points,box,width,height){
  return points.map(p=>({...p,x:(box.x+p.x*box.size)/width,y:(box.y+p.y*box.size)/height,z:p.z*box.size/width}));
}
