import torch
from torch import nn

from .pose_decoder import PoseDecoder
from .text_encoder import FrozenFeatureProjection, TextEncoder


class Model(nn.Module):
    def __init__(self,vocab_size,width,frames,encoder="learned", residual_scale=0.0):
        super().__init__()
        self.encoder=FrozenFeatureProjection(width) if encoder=="frozen_french" else TextEncoder(vocab_size,width)
        self.decoder=PoseDecoder(width,frames)
        self.residual_scale = residual_scale
        if residual_scale:
            self.register_buffer("phase_template", torch.zeros(frames, 128, 2))
            nn.init.zeros_(self.decoder.output.weight)
            nn.init.zeros_(self.decoder.output.bias)

    def forward(self,ids):
        pose, duration = self.decoder(*self.encoder(ids))
        if self.residual_scale:
            pose = self.phase_template + self.residual_scale * pose.tanh()
        return pose, duration
