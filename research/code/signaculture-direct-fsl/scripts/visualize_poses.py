import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np


def edges(chain):
    return zip(chain,chain[1:])


def draw(ax,xy,mask,hands=False):
    ax.clear()
    ax.set(xlim=(-2,2),ylim=(2,-1.5),aspect="equal")
    ax.axis("off")
    for start,end,color,chains in [
        (0,18,"#237059",[[0,1,2,3,4],[1,5,6,7],[1,8,9,10],[1,11,12,13],[0,14,16],[0,15,17]]),
        (18,39,"#ac5d1d",[[0,i,i+1,i+2,i+3] for i in (1,5,9,13,17)]),
        (39,60,"#3456ad",[[0,i,i+1,i+2,i+3] for i in (1,5,9,13,17)]),
        (60,128,"#805585",[list(range(a,b)) for a,b in [(0,17),(17,22),(22,27),(27,31),(31,36)]]+
            [[*range(a,b),a] for a,b in [(36,42),(42,48),(48,60),(60,68)]])
    ]:
        if hands and start not in (18,39):continue
        points=xy[start:end]
        visible=mask[start:end]
        ax.scatter(*points[visible].T,s=5,color=color)
        for chain in chains:
            for a,b in edges(chain):
                if visible[a] and visible[b]: ax.plot(*points[[a,b]].T,color=color,lw=1)


def render(path, tracks, fps):
    figure,axes=plt.subplots(1,len(tracks),squeeze=False,figsize=(4*len(tracks),4))
    def update(i):
        for ax,(label,xy,mask) in zip(axes[0],tracks):
            draw(ax,xy[i],mask[i])
            ax.set_title(label)
    movie=FuncAnimation(figure,update,frames=len(tracks[0][1]),interval=1000/fps)
    movie.save(path,writer=PillowWriter(fps=fps) if str(path).endswith(".gif") else FFMpegWriter(fps=fps),dpi=80)
    plt.close(figure)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("input",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--row",type=int,default=0)
    p.add_argument("--hands",action="store_true")
    a=p.parse_args()
    with np.load(a.input,allow_pickle=False) as data:
        xy,mask=data["normalized_xy"],data["mask"]
        if a.output.suffix==".png":
            fig,ax=plt.subplots()
            draw(ax,xy[a.row],mask[a.row],a.hands)
            fig.savefig(a.output)
        else:
            render(a.output,[("Observed FSL pose",xy[a.row:a.row+64],mask[a.row:a.row+64])],24)
