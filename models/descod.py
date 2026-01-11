# models/descod.py
import torch
import torch.nn as nn
import numpy as np

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResnetBlockDeScoD(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels); self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels); self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.time_emb = nn.Linear(emb_dim, channels); self.act = nn.SiLU()
    def forward(self, x, emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_emb(emb).unsqueeze(-1)
        h = self.act(self.norm2(self.conv2(h)))
        return x + h

class DeScoD_ScoreNet(nn.Module):
    def __init__(self, in_channels=12):
        super().__init__()
        self.name = "DeScoD-ECG"
        self.embed = nn.Sequential(GaussianFourierProjection(128), nn.Linear(128, 128), nn.SiLU())
        self.init_conv = nn.Conv1d(in_channels * 2, 64, 3, padding=1)
        self.down1 = nn.Conv1d(64, 128, 4, stride=2, padding=1); self.res1 = ResnetBlockDeScoD(128, 128)
        self.down2 = nn.Conv1d(128, 256, 4, stride=2, padding=1); self.res2 = ResnetBlockDeScoD(256, 128)
        self.up2 = nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1)
        self.final_conv = nn.Sequential(nn.GroupNorm(8, 64), nn.SiLU(), nn.Conv1d(64, in_channels, 3, padding=1))

    def forward(self, x, t, cond):
        t_emb = self.embed(t)
        h = self.init_conv(torch.cat([x, cond], dim=1))
        h1 = self.res1(self.down1(h), t_emb); h2 = self.res2(self.down2(h1), t_emb)
        h = self.up2(h2) + h1; h = self.up1(h)
        return self.final_conv(h)
    
    @torch.no_grad()
    def predict(self, input_data, steps=20, device='cuda'):
        """输入 (B, 512, 12)，输出 (B, 512, 12)"""
        self.to(device)
        self.eval()
        
        x_cond = torch.as_tensor(input_data, dtype=torch.float32, device=device)
        if x_cond.shape[-1] == 12: x_cond = x_cond.permute(0, 2, 1)
        
        xt = x_cond.clone()
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x_cond.shape[0],), i/steps, device=device)
            # DeScoD forward 顺序: (x, t, cond)
            xt = xt + self.forward(xt, t, x_cond) * dt
            
        return xt.permute(0, 2, 1).cpu().numpy()