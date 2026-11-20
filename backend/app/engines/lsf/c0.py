"""C0 v2 text-to-pose generator, inference only.

Copied from sign-model/c0_tokenizer.py and c0_generator.py (training code, datasets and
evaluation left out). Model: VQ-VAE tokenizer with two streams (body+face 230 values, both
hands relative to their wrist 84 values), time downsampling x2, 2-level residual VQ; a
6-layer transformer decoder over token steps conditioned on frozen multilingual-e5 token
states, sampled with classifier-free guidance.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config

BODY = np.r_[0:23, 65:157]
LH, RH = np.r_[23:44], np.r_[44:65]
CB, CD = 512, 64              # codebook size, code dim
D_MODEL = 384
BOS = CB
# Best settings of the 2026-11-20 diagnostic (sign-model/MASTER_STATUS.md).
SAMPLING = dict(temp=(1.0, 0.6), topk=0, cfg=3.0, stop_thr=0.2)


# ----------------------------------------------------------------------------- tokenizer
class VQ(nn.Module):
    def __init__(self, n=CB, d=CD):
        super().__init__()
        self.n, self.d = n, d
        self.register_buffer("emb", torch.zeros(n, d))
        self.register_buffer("cs", torch.ones(n))
        self.register_buffer("es", torch.zeros(n, d))


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.Conv1d(c, c, 3, padding=1), nn.GELU(), nn.Conv1d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.b(x)


class RVQ(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.vqs = nn.ModuleList([VQ() for _ in range(levels)])

    def lookup(self, idx):                     # (B,T,L) -> (B,d,T)
        return sum(vq.emb[idx[..., l]] for l, vq in enumerate(self.vqs)).permute(0, 2, 1)


class VQVAE(nn.Module):
    def __init__(self, d_in, c=256, down=4, levels=1):
        super().__init__()
        self.levels = levels
        n = {2: 1, 4: 2}[down]
        enc = [nn.Conv1d(d_in, c, 3, padding=1), Res(c)]
        for _ in range(n):
            enc += [nn.Conv1d(c, c, 4, 2, 1), Res(c)]
        self.enc = nn.Sequential(*enc, Res(c), nn.Conv1d(c, CD, 1))
        if levels == 1:
            self.vq = VQ()
        else:
            self.rvq = RVQ(levels)
        dec = [nn.Conv1d(CD, c, 3, padding=1), Res(c), Res(c)]
        for _ in range(n):
            dec += [nn.ConvTranspose1d(c, c, 4, 2, 1), Res(c)]
        self.dec = nn.Sequential(*dec, nn.Conv1d(c, d_in, 3, padding=1))

    def decode(self, idx):                     # idx (B,t) or (B,t,L)
        q = self.vq.emb[idx].permute(0, 2, 1) if self.levels == 1 else self.rvq.lookup(idx)
        return self.dec(q).permute(0, 2, 1)


# ----------------------------------------------------------------------------- generator
class Gen(nn.Module):
    def __init__(self, levels, maxs):
        super().__init__()
        self.L, self.maxs = levels, maxs
        self.mem = nn.Linear(768, D_MODEL)
        self.eb = nn.Embedding(CB + 1, D_MODEL)
        self.eh = nn.Embedding(CB + 1, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, maxs + 1, D_MODEL))
        lay = nn.TransformerDecoderLayer(D_MODEL, 6, 4 * D_MODEL, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.dec = nn.TransformerDecoder(lay, 6)
        self.norm = nn.LayerNorm(D_MODEL)
        self.hb, self.hh, self.hs = nn.Linear(D_MODEL, CB), nn.Linear(D_MODEL, CB), nn.Linear(D_MODEL, 1)
        if self.L > 1:
            E = lambda n: nn.ModuleList([nn.Embedding(n, D_MODEL) for _ in range(self.L - 1)])  # noqa: E731
            self.ebx, self.ehx, self.cb, self.ch = E(CB + 1), E(CB + 1), E(CB), E(CB)
            self.hbx = nn.ModuleList([nn.Linear(D_MODEL, CB) for _ in range(self.L - 1)])
            self.hhx = nn.ModuleList([nn.Linear(D_MODEL, CB) for _ in range(self.L - 1)])

    def trunk(self, tb_in, th_in, mem, mem_pad):
        T = tb_in.shape[1]
        x = self.eb(tb_in[..., 0]) + self.eh(th_in[..., 0]) + self.pos[:, :T]
        for l in range(1, self.L):
            x = x + self.ebx[l - 1](tb_in[..., l]) + self.ehx[l - 1](th_in[..., l])
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        return self.norm(self.dec(x, self.mem(mem), tgt_mask=causal, memory_key_padding_mask=mem_pad))

    def head(self, h, l, prev_b, prev_h):
        if l == 0:
            return self.hb(h), self.hh(h)
        cb = h + sum(self.cb[k](prev_b[..., k]) for k in range(l))
        ch = h + sum(self.ch[k](prev_h[..., k]) for k in range(l))
        return self.hbx[l - 1](cb), self.hhx[l - 1](ch)


class Text:
    """Frozen multilingual-e5 token states ("query: " prefix), local files only."""

    def __init__(self, dev):
        import os
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise config.EngineUnavailable("Moteur C0 : le paquet 'transformers' n'est pas installé "
                                           "(pip install -r backend/requirements-lsf-engines.txt).") from exc
        source = os.getenv("SIGNACULTURE_LSF_TEXT_ENCODER") or config.E5_ID
        try:
            self.tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
            model = AutoModel.from_pretrained(source, local_files_only=True)
        except OSError as exc:
            raise config.EngineUnavailable(
                f"Moteur C0 : encodeur de texte '{source}' absent localement. Téléchargez-le explicitement une fois "
                "(voir docs/pipeline/LSF_ENGINES.md) ou indiquez un dossier local avec SIGNACULTURE_LSF_TEXT_ENCODER.") from exc
        self.m = (model.to(dev).half() if dev == "cuda" else model.to(dev)).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, texts):
        b = self.tok(["query: " + t for t in texts], padding=True, truncation=True, max_length=96, return_tensors="pt").to(self.dev)
        return self.m(**b).last_hidden_state.float(), b["attention_mask"] == 0


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


class C0Engine:
    def __init__(self, dev: str):
        config.require("c0")
        self.dev = dev
        st = _load(config.path("c0_tokenizer"))
        cfg = st.get("cfg", dict(down=4, levels=1))
        self.down, self.levels = int(cfg["down"]), int(cfg["levels"])
        self.models = {"body": VQVAE(len(BODY) * 2, down=self.down, levels=self.levels),
                       "hand": VQVAE(84, down=self.down, levels=self.levels)}
        for k, m in self.models.items():
            m.load_state_dict(st[k])
            m.to(dev).eval()
        gs = _load(config.path("c0_generator"))
        maxs = gs["pos"].shape[1] - 1
        self.gen = Gen(self.levels, maxs)
        self.gen.load_state_dict(gs)
        self.gen.to(dev).eval()
        self.text = Text(dev)
        self.fps_tokens = 12.5

    @torch.no_grad()
    def sample(self, text, seed=0, temp=SAMPLING["temp"], topk=SAMPLING["topk"], cfg=SAMPLING["cfg"], stop_thr=SAMPLING["stop_thr"]):
        g, dev = self.gen, self.dev
        torch.manual_seed(seed)
        mem, pad = self.text([text])
        L = g.L
        if cfg != 1.0:
            mem_u, pad_u = self.text([""])
        tb = torch.full((1, 1, L), BOS, dtype=torch.long, device=dev)
        th = tb.clone()
        length = g.maxs
        temps = list(temp) if isinstance(temp, (list, tuple)) else [temp] * L

        def pick(lo, tp):
            lo = lo / tp
            if topk:
                v, _ = lo.topk(topk)
                lo[lo < v[:, -1:]] = -1e9
            return torch.multinomial(F.softmax(lo, -1), 1)[:, 0]

        for t in range(g.maxs):
            h = g.trunk(tb, th, mem, pad)[:, -1]
            hu = g.trunk(tb, th, mem_u, pad_u)[:, -1] if cfg != 1.0 else None
            nb = torch.zeros(1, L, dtype=torch.long, device=dev)
            nh = torch.zeros(1, L, dtype=torch.long, device=dev)
            for l in range(L):
                lb, lh = g.head(h, l, nb[:, :l], nh[:, :l])
                if hu is not None:
                    ub, uh = g.head(hu, l, nb[:, :l], nh[:, :l])
                    lb, lh = ub + cfg * (lb - ub), uh + cfg * (lh - uh)
                nb[:, l], nh[:, l] = pick(lb, temps[l]), pick(lh, temps[l])
            tb = torch.cat([tb, nb[:, None]], 1)
            th = torch.cat([th, nh[:, None]], 1)
            if torch.sigmoid(g.hs(h)[:, 0]).item() > stop_thr:
                length = t + 1
                break
        return tb[0, 1:length + 1].cpu().numpy(), th[0, 1:length + 1].cpu().numpy()

    @torch.no_grad()
    def decode(self, tb, th):
        """Tokens -> (T, 157, 2) at 25 fps, normalised engine coordinates."""
        def ids(t):
            t = torch.from_numpy(np.asarray(t).astype("int64"))
            return (t[..., 0] if t.dim() == 2 and t.shape[1] == 1 else t)[None].to(self.dev)
        b = self.models["body"].decode(ids(tb))[0].float().cpu().numpy()
        h = self.models["hand"].decode(ids(th))[0].float().cpu().numpy()
        T = min(len(b), len(h))
        x = np.zeros((T, 157, 2), "float32")
        x[:, BODY] = b[:T].reshape(T, -1, 2)
        hh = h[:T].reshape(T, 42, 2)
        x[:, LH] = hh[:, :21] + x[:, [15]]
        x[:, RH] = hh[:, 21:] + x[:, [16]]
        t = np.arange(2 * T - 1) / 2.0                                     # 12.5 -> 25 fps (linear)
        lo = np.floor(t).astype(int)
        hi = np.minimum(lo + 1, T - 1)
        w = (t - lo)[:, None, None]
        return x[lo] * (1 - w) + x[hi] * w

    def generate(self, text, seed=0):
        tb, th = self.sample(text, seed=seed)
        return self.decode(tb, th), {"tokens": int(len(tb)), "sampling": {**SAMPLING, "temp": list(SAMPLING["temp"]), "seed": seed}}
