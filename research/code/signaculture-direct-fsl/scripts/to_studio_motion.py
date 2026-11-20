"""Convert an experimental DWPose prediction to the existing Studio motion format."""
import argparse
import json
from pathlib import Path

import numpy as np

BODY={"head":0,"neck":1,"right_shoulder":2,"right_elbow":3,"right_wrist":4,
      "left_shoulder":5,"left_elbow":6,"left_wrist":7}


def convert(xy,mask,fps):
    if xy.shape != (len(xy),128,2) or mask.shape!=xy.shape[:2] or len(xy)<2 or not 0<fps<=240:
        raise ValueError("Unexpected DWPose motion schema")
    duration=round(len(xy)*1000/fps)
    indices=np.arange(len(xy))
    finite=mask & np.isfinite(xy).all(-1)
    body={}
    for name,j in BODY.items():
        observed=indices[finite[:,j]]
        if not len(observed):
            raise ValueError(f"No training support for {name}")
        body[name]=np.stack([np.interp(indices,observed,xy[observed,j,axis]) for axis in (0,1)],axis=1)
    frames=[]
    for t in indices:
        def project(point,z):
            return [float(point[0]*.34),float(1.45-point[1]*.34),z]
        joints={name:project(track[t],.25 if name.endswith("wrist") else .08) for name,track in body.items()}
        joints["hips"]=[0,1,0]
        for side,offset in [("left",18),("right",39)]:
            if not finite[t,offset]:
                continue
            joints[f"{side}_wrist"]=project(xy[t,offset],.25)
            for f,finger in enumerate(("thumb","index","middle","ring","pinky")):
                for j in range(1,5):
                    idx=offset+4*f+j
                    if finite[t,idx]:
                        joints[f"{side}_{finger}_{j}"]=project(xy[t,idx],.25)
        frames.append({"time_ms":round(t*duration/(len(xy)-1)),"joints":joints})
    return {"rig_version":"signaculture-v1","fps":(len(xy)-1)*1000/duration,
            "duration_ms":duration,"frames":frames,"joints":sorted({k for f in frames for k in f["joints"]}),
            "synthetic":True,"capture_method":"experimental-signverse-fsl-text-to-2d",
            "warnings":["Weak French subtitle supervision. Depth is assumed and unsupported body points are interpolated."]}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    with np.load(a.source,allow_pickle=False) as data:
        result=convert(data["normalized_xy"],data["mask"],float(data["fps"]))
    a.output.write_text(json.dumps(result,allow_nan=False))
