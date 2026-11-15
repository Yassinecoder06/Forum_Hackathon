"""Paper-inspired endpoint-conditioned residual cGAN, engineering widths explicit."""
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # 20 endpoint features +16noise dimensions; paper does not specify this input.
        self.net=nn.Sequential(nn.Linear(36,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.4),
            nn.Linear(256,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.4),
            nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Linear(128,100))
        # Residual interpolation shortcut is OUR engineering addition, not paper Eq4.
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('t',torch.arange(1,11,dtype=torch.float32).view(1,10,1)/11)
    def forward(self,endpoints,noise=None):
        if noise is None:noise=torch.zeros(endpoints.shape[0],16,device=endpoints.device)
        linear=endpoints[:,0,None,:]*(1-self.t)+endpoints[:,1,None,:]*self.t
        residual=self.net(torch.cat([endpoints.flatten(1),noise],dim=1)).reshape(-1,10,10)
        return linear+residual

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(120,256),nn.LeakyReLU(.2),nn.Dropout(.3),
            nn.Linear(256,128),nn.LeakyReLU(.2),nn.Dropout(.3),
            nn.Linear(128,64),nn.LeakyReLU(.2),nn.Dropout(.3),nn.Linear(64,1))
    def forward(self,endpoints,interior):
        return self.net(torch.cat([endpoints.flatten(1),interior.flatten(1)],dim=1))
