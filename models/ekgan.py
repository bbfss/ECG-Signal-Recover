# models/ekgan.py
import torch
import torch.nn as nn

class EKGAN_Generator(nn.Module):
    def __init__(self, in_chan=12, out_chan=12):
        super().__init__()
        self.name = "EKGAN"
        # 编码器：深度残差结构
        def down(ic, oc): return nn.Sequential(nn.Conv1d(ic, oc, 7, 2, 3), nn.BatchNorm1d(oc), nn.LeakyReLU(0.2))
        def up(ic, oc): return nn.Sequential(nn.ConvTranspose1d(ic, oc, 7, 2, 3, 1), nn.BatchNorm1d(oc), nn.ReLU())
        
        self.e1 = down(in_chan, 64); self.e2 = down(64, 128); self.e3 = down(128, 256); self.e4 = down(256, 512)
        self.b = nn.Sequential(nn.Conv1d(512, 512, 3, padding=1), nn.ReLU())
        self.d4 = up(512, 256); self.d3 = up(512, 128); self.d2 = up(256, 64); self.d1 = up(128, 32)
        self.out = nn.Sequential(nn.Conv1d(32, out_chan, 3, padding=1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        d4 = self.d4(self.b(e4))
        d3 = self.d3(torch.cat([d4, e3], 1))
        d2 = self.d2(torch.cat([d3, e2], 1))
        d1 = self.d1(torch.cat([d2, e1], 1))
        return self.out(d1)
    @torch.no_grad()
    def predict(self, input_data, device='cuda'):
        """输入 (B, 512, 12)，输出 (B, 512, 12)"""
        self.to(device)
        self.eval()
        
        # 处理输入维度: (B, 512, 12) -> (B, 12, 512)
        x = torch.as_tensor(input_data, dtype=torch.float32, device=device)
        if x.dim() == 2: x = x.unsqueeze(0) # 补 Batch 维
        if x.shape[-1] == 12: x = x.permute(0, 2, 1)
        
        # 推理
        out = self.forward(x)
        
        # 还原形状并转回 numpy
        return out.permute(0, 2, 1).cpu().numpy()

class EKGAN_Discriminator(nn.Module):
    def __init__(self, in_chan=12):
        super().__init__()
        # 带谱归一化的判别器块
        def block(ic, oc): return nn.Sequential(nn.utils.spectral_norm(nn.Conv1d(ic, oc, 15, 2, 7)), nn.LeakyReLU(0.2), nn.Dropout(0.2))
        self.net = nn.Sequential(
            block(in_chan, 64), block(64, 128), block(128, 256), block(256, 512),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(512, 1)
        )
    def forward(self, x): return self.net(x)
    
