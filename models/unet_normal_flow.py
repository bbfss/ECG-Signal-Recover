import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==============================================================================
# 1. 基础组件 (保持与原架构一致的工具类)
# ==============================================================================
class CoordAtt1D(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_w = nn.AdaptiveAvgPool1d(1)
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv1d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(mip); self.act = nn.Hardswish()
        self.conv_w = nn.Conv1d(mip, oup, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        identity = x; x_w = self.pool_w(x); y = self.act(self.bn1(self.conv1(x_w)))
        a_w = self.conv_w(y).sigmoid()
        return identity * a_w

class WaveletSKFusion(nn.Module):
    def __init__(self, in_chan, reduction=4):
        super().__init__()
        self.in_chan = in_chan
        self.p1 = nn.Sequential(nn.Conv1d(in_chan, in_chan, 3, padding=1, groups=in_chan), nn.BatchNorm1d(in_chan), nn.SiLU())
        self.p2 = nn.Sequential(nn.Conv1d(in_chan, in_chan, 3, padding=2, dilation=2, groups=in_chan), nn.BatchNorm1d(in_chan), nn.SiLU())
        self.p3 = nn.Sequential(nn.Conv1d(in_chan, in_chan, 3, padding=4, dilation=4, groups=in_chan), nn.BatchNorm1d(in_chan), nn.SiLU())
        mid_chan = max(16, in_chan // reduction)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(in_chan, mid_chan), nn.ReLU(True), nn.Linear(mid_chan, in_chan * 3))
        self.last_conv = nn.Conv1d(in_chan, in_chan, 1); self.norm = nn.BatchNorm1d(in_chan)
    def forward(self, skip_x, up_x):
        if up_x.size(2) != skip_x.size(2):
            up_x = F.interpolate(up_x, size=skip_x.size(2), mode='linear', align_corners=False)
        x = skip_x + up_x; u1, u2, u3 = self.p1(x), self.p2(x), self.p3(x)
        s = self.gap(u1 + u2 + u3).squeeze(-1)
        z = self.fc(s).view(x.size(0), 3, self.in_chan); attn = F.softmax(z, dim=1) 
        v = (u1 * attn[:, 0:1, :].transpose(1, 2) + u2 * attn[:, 1:2, :].transpose(1, 2) + u3 * attn[:, 2:3, :].transpose(1, 2))
        return self.norm(self.last_conv(v)) + up_x

# ==============================================================================
# 2. 主体 UNet 架构 (保持不变，作为共同的特征提取骨架)
# ==============================================================================
class AdvancedUNet1D(nn.Module):
    def __init__(self, in_channels=12, out_channels=12):
        super().__init__()
        self.name = "ECG_Unet_Normal_Flow"
        def conv_block(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv1d(in_dim, out_dim, 3, padding=1), nn.BatchNorm1d(out_dim), nn.ReLU(True),
                CoordAtt1D(out_dim, out_dim), 
                nn.Conv1d(out_dim, out_dim, 3, padding=1), nn.BatchNorm1d(out_dim), nn.ReLU(True)
            )
        self.enc1 = conv_block(in_channels, 64); self.pool1 = nn.MaxPool1d(2)
        self.enc2 = conv_block(64, 128); self.pool2 = nn.MaxPool1d(2)
        self.enc3 = conv_block(128, 256); self.pool3 = nn.MaxPool1d(2)
        self.enc4 = conv_block(256, 512); self.pool4 = nn.MaxPool1d(2)
        self.bottleneck = conv_block(512, 1024)
        self.up4 = nn.ConvTranspose1d(1024, 512, 2, 2); self.fuse4 = WaveletSKFusion(512); self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose1d(512, 256, 2, 2); self.fuse3 = WaveletSKFusion(256); self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose1d(256, 128, 2, 2); self.fuse2 = WaveletSKFusion(128); self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose1d(128, 64, 2, 2); self.fuse1 = WaveletSKFusion(64); self.dec1 = conv_block(128, 64)
        self.final_conv = nn.Conv1d(64, out_channels, 1); self.final_act = nn.Tanh()

    def forward(self, x):
        e1 = self.enc1(x.float()); e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat((self.up4(b), self.fuse4(e4, self.up4(b))), dim=1))
        d3 = self.dec3(torch.cat((self.up3(d4), self.fuse3(e3, self.up3(d4))), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), self.fuse2(e2, self.up2(d3))), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), self.fuse1(e1, self.up1(d2))), dim=1))
        return self.final_act(self.final_conv(d1))

    @torch.no_grad()
    def predict(self, x, device='cuda'):
        self.to(device); self.eval()
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        if x.ndim == 2: x = x.unsqueeze(0)
        if x.shape[-1] == 12: x = x.permute(0, 2, 1)
        out = self.forward(x)
        return out.permute(0, 2, 1).cpu().numpy()

# ==============================================================================
# 3. 对照组：NormalFlowNetwork (基础流匹配网络)
# ==============================================================================
class NormalFlowNetwork(nn.Module):
    def __init__(self, channels=12):
        super().__init__()
        # 移除 FiLM，使用最简单的时间映射
        self.t_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        # 基础输入卷积
        self.init_conv = nn.Conv1d(channels * 2, 128, 3, padding=1)
        
        # 基础残差块 (移除 TimeNav Attention)
        self.res_block = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128)
        )
        # 输出层
        self.final_conv = nn.Conv1d(128, channels, 1)

    def forward(self, t, xt, cond):
        t_embed = self.t_mlp(t.unsqueeze(-1)).unsqueeze(-1)
        # 简单的特征拼接
        x = torch.cat([xt, cond], dim=1)
        x = self.init_conv(x)
        # 简单相加注入时间信息
        x = x + t_embed.expand(-1, -1, x.size(-1))
        # 纯卷积处理
        x = x + self.res_block(x)
        return self.final_conv(x)

    @torch.no_grad()
    def predict(self, x_cond, steps=30, device='cuda'):
        self.to(device); self.eval()
        x_cond = torch.as_tensor(x_cond, dtype=torch.float32, device=device)
        if x_cond.ndim == 2: x_cond = x_cond.unsqueeze(0)
        if x_cond.shape[-1] == 12: x_cond = x_cond.permute(0, 2, 1)
        
        xt = x_cond.clone()
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x_cond.shape[0],), i/steps, device=device)
            xt = xt + self.forward(t, xt, x_cond) * dt
        
        bias = xt.mean(dim=-1, keepdim=True) - x_cond.mean(dim=-1, keepdim=True)
        return (xt - bias).permute(0, 2, 1).cpu().numpy()

# ==============================================================================
# 4. 对照组：NormalFlowMatcher (标准 MSE 损失)
# ==============================================================================
class NormalFlowMatcher:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def compute_loss(self, x1, x_recon):
        """
        x1: 真实信号
        x_recon: UNet 重建信号
        """
        batch_size = x1.shape[0]
        t = torch.rand(batch_size, device=self.device)
        t_v = t.view(-1, 1, 1)
        
        # 基础线性路径: xt = (1-t)x0 + tx1
        xt = (1 - t_v) * x_recon + t_v * x1
        target_v = x1 - x_recon
        
        # 预测速度场
        pred_v = self.model(t, xt, x_recon)
        
        # 【核心差异】：只使用最基础的 MSE，不进行峰值加权或梯度约束
        return F.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def sample(self, x_recon, steps=30):
        self.model.eval()
        xt = x_recon.clone()
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x_recon.shape[0],), i/steps, device=self.device)
            xt = xt + self.model(t, xt, x_recon) * dt
        bias = xt.mean(dim=-1, keepdim=True) - x_recon.mean(dim=-1, keepdim=True)
        return xt - bias