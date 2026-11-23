"""C0 refinement decoders, inference only (copy of sign-model/c0_refine_infer.py; training: c0_handdec.py, gates: c0_gates.py).

Self-contained (torch + numpy). Works on the C0 tokenizer modules of c0.py (residual-VQ VQVAE with .rvq.lookup and .dec):

    ref = C0Refiner(handdec_reg.pt, face=facedec_fm.pt or None, dev=...)
    x = ref.decode(models, tb, th, seed)      # (T,157,3) at 25 fps, BEFORE holds

Output: x/y as the tokenizer decode (normalised engine coordinates) with refined hands and face; channel 2 is zero
except on the hands, where it is the finger depth relative to the wrist, towards the camera, in shoulder widths.

Hand refiner (regression, deterministic): same 2D hands as the tokenizer decode plus a small learned correction, and a
finger depth learned from MediaPipe's own hand depth. Face refiner (flow matching, seeded): restores the mouth / brow /
eye motion the tokenizer smooths away. Both read the frozen C0 tokens; the generator and its sampling are unchanged.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BODY = np.r_[0:23, 65:157]
LHs, RHs, FACE = slice(23, 44), slice(44, 65), slice(65, 157)
ARMS = [11, 12, 13, 14, 15, 16]


class Block(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.n = nn.GroupNorm(8, c)
        self.c1 = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil)
        self.c2 = nn.Conv1d(c, c, 1)
        self.film = nn.Linear(c, 2 * c)

    def forward(self, x, temb):
        s, b = self.film(temb)[:, :, None].chunk(2, 1)
        return x + self.c2(F.gelu(self.c1(F.gelu(self.n(x) * (1 + s) + b))))


class Net(nn.Module):
    def __init__(self, d_x, d_c, c=256, dils=(1, 2, 4, 8, 16, 1, 2, 4, 8, 16)):
        super().__init__()
        self.inp = nn.Conv1d(d_x + d_c, c, 3, padding=1)
        self.temb = nn.Sequential(nn.Linear(64, c), nn.GELU(), nn.Linear(c, c))
        self.blocks = nn.ModuleList([Block(c, d) for d in dils])
        self.out = nn.Conv1d(c, d_x, 3, padding=1)

    def forward(self, xt, cond, t):
        f = torch.exp(-math.log(1000) * torch.arange(32, device=t.device) / 32)
        te = self.temb(torch.cat([torch.sin(t[:, None] * 1000 * f), torch.cos(t[:, None] * 1000 * f)], 1))
        h = self.inp(torch.cat([xt, cond], 2).permute(0, 2, 1))
        for b in self.blocks:
            h = b(h, te)
        return self.out(h).permute(0, 2, 1)


def _load(path, d_x, d_c, dev):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = Net(d_x, d_c)
    net.load_state_dict(ck["net"])
    net.to(dev).eval()
    return net, ck["mu"].to(dev), ck["sd"].to(dev), ck["mode"]


def _up25(y):
    """12.5 -> 25 fps: insert the midpoint between consecutive frames (2T-1 frames)."""
    mid = 0.5 * (y[:, :-1] + y[:, 1:])
    return torch.cat([torch.stack([y[:, :-1], mid], 2).reshape(y.shape[0], -1, y.shape[2]), y[:, -1:]], 1)


class C0Refiner:
    def __init__(self, hand_path, face=None, dev="cpu", steps=16, face_temp=1.0):
        self.dev, self.steps, self.face_temp = dev, steps, face_temp
        self.hand = _load(hand_path, 126, 224, dev)
        self.face = _load(face, 184, 248, dev) if face else None

    def _run(self, net, mu, sd, mode, cond, temp, seed):
        T, dx = cond.shape[1], net.out.out_channels
        if mode == "fm":
            g = torch.Generator(device="cpu").manual_seed(seed)
            x = (temp * torch.randn(1, T, dx, generator=g)).to(self.dev)
            for k in range(self.steps):
                x = x + net(x, cond, torch.full((1,), k / self.steps, device=self.dev)) / self.steps
        else:
            x = net(torch.zeros(1, T, dx, device=self.dev), cond, torch.zeros(1, device=self.dev))
        return x * sd + mu

    @torch.no_grad()
    def decode(self, models, tb, th, seed=0):
        def ids(t):
            return torch.from_numpy(np.asarray(t).astype("int64"))[None].to(self.dev)
        mb, mh = models["body"], models["hand"]
        qb, qh = mb.rvq.lookup(ids(tb)), mh.rvq.lookup(ids(th))
        b = mb.dec(qb).permute(0, 2, 1).float()
        h = mh.dec(qh).permute(0, 2, 1).float()
        T = min(b.shape[1], h.shape[1])
        x = torch.zeros(1, T, 157, 2, device=self.dev)
        x[:, :, BODY] = b[:, :T].reshape(1, T, -1, 2)
        hh = h[:, :T].reshape(1, T, 42, 2)
        x[:, :, LHs] = hh[:, :, :21] + x[:, :, 15:16]
        x[:, :, RHs] = hh[:, :, 21:] + x[:, :, 16:17]
        x = _up25(x.reshape(1, T, -1)).reshape(1, -1, 157, 2)
        T25 = x.shape[1]

        def rep(q):                                    # token codes repeated to 25 fps (1 token step = 4 frames)
            return q.repeat_interleave(4, 2)[:, :, :T25].permute(0, 2, 1).float()
        bh = torch.cat([x[:, :, LHs] - x[:, :, 23:24], x[:, :, RHs] - x[:, :, 44:45]], 2).reshape(1, T25, -1)
        cond = torch.cat([bh, x[:, :, ARMS].reshape(1, T25, -1), rep(qh), rep(qb)], 2)
        y = self._run(*self.hand, cond, 1.0, seed)
        out = torch.zeros(1, T25, 157, 3, device=self.dev)
        out[..., :2] = x
        hr = (bh + y[..., :84]).reshape(1, T25, 42, 2)
        z = y[..., 84:].reshape(1, T25, 42)
        out[:, :, LHs, :2] = hr[:, :, :21] + x[:, :, 23:24]
        out[:, :, RHs, :2] = hr[:, :, 21:] + x[:, :, 44:45]
        out[:, :, LHs, 2] = z[:, :, :21]
        out[:, :, RHs, 2] = z[:, :, 21:]
        if self.face is not None:
            fr = (x[:, :, FACE] - x[:, :, 0:1]).reshape(1, T25, -1)
            y = self._run(*self.face, torch.cat([fr, rep(qb)], 2), self.face_temp, seed + 7919)
            out[:, :, FACE, :2] = x[:, :, 0:1] + (fr + y).reshape(1, T25, 92, 2)
        return out[0].cpu().numpy()
