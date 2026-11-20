import torch
from torch import nn


class PoseDecoder(nn.Module):
    def __init__(self,width,frames):
        super().__init__()
        phase=torch.linspace(0,1,frames)
        self.register_buffer("phase",phase[:,None])
        self.queries=nn.Parameter(torch.randn(frames,width)*.02)
        self.counter=nn.Linear(1,width)
        self.layers=nn.TransformerDecoder(nn.TransformerDecoderLayer(width,4,width*4,batch_first=True,
                                             norm_first=True),3)
        self.output=nn.Linear(width,256)
        self.duration=nn.Linear(width,1)

    def forward(self,memory,padding):
        queries=(self.queries+self.counter(self.phase)).expand(len(memory),-1,-1)
        hidden=self.layers(queries,memory,memory_key_padding_mask=padding)
        valid=(~padding)[...,None]
        pooled=(memory*valid).sum(1)/valid.sum(1)
        return self.output(hidden).reshape(len(memory),-1,128,2), self.duration(pooled).squeeze(-1)
