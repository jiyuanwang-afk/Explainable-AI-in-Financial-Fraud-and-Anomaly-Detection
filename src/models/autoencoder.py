import torch
from torch import nn

class TabularAE(nn.Module):
    def __init__(self, in_dim, hidden=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden, in_dim))
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
    def anomaly_score(self, x):
        recon, _ = self.forward(x)
        return ((x - recon)**2).mean(dim=1)
