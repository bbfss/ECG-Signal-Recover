

# # # models/unet_flow.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class CoordAtt1D(nn.Module):
#     def __init__(self, inp, oup, reduction=32):
#         super().__init__()
#         self.pool_w = nn.AdaptiveAvgPool1d(1)
#         mip = max(8, inp // reduction)
#         self.conv1 = nn.Conv1d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm1d(mip); self.act = nn.Hardswish()
#         self.conv_w = nn.Conv1d(mip, oup, kernel_size=1, stride=1, padding=0)
#     def forward(self, x):
#         identity = x; x_w = self.pool_w(x); y = self.act(self.bn1(self.conv1(x_w)))
#         a_w = self.conv_w(y).sigmoid()
#         return identity * a_w

# class WaveletSKFusion(nn.Module):
#     def __init__(self, in_chan, reduction=4):
#         super().__init__()
#         self.in_chan = in_chan
#         self.p1 = nn.Sequential(nn.Conv1d(in_chan, in_chan, 3, padding=1, groups=in_chan), nn.BatchNorm1d(in_chan), nn.SiLU())
#         self.p2 = nn.Sequential(nn.Conv1d(in_chan, in_chan, 3, padding=2, dilation=2, groups=in_chan), nn.BatchNorm1d(in_chan), nn.SiLU())
#         self.p3 = nn.Sequential(nn.Conv1d(in_chan, in_chan, 3, padding=4, dilation=4, groups=in_chan), nn.BatchNorm1d(in_chan), nn.SiLU())
#         mid_chan = max(16, in_chan // reduction)
#         self.gap = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(nn.Linear(in_chan, mid_chan), nn.ReLU(True), nn.Linear(mid_chan, in_chan * 3))
#         self.last_conv = nn.Conv1d(in_chan, in_chan, 1); self.norm = nn.BatchNorm1d(in_chan)
#     def forward(self, skip_x, up_x):
#         if up_x.size(2) != skip_x.size(2):
#             up_x = F.interpolate(up_x, size=skip_x.size(2), mode='linear', align_corners=False)
#         x = skip_x + up_x; u1, u2, u3 = self.p1(x), self.p2(x), self.p3(x)
#         s = self.gap(u1 + u2 + u3).squeeze(-1)
#         z = self.fc(s).view(x.size(0), 3, self.in_chan); attn = F.softmax(z, dim=1) 
#         v = (u1 * attn[:, 0:1, :].transpose(1, 2) + u2 * attn[:, 1:2, :].transpose(1, 2) + u3 * attn[:, 2:3, :].transpose(1, 2))
#         return self.norm(self.last_conv(v)) + up_x

# class AdvancedUNet1D(nn.Module):
#     def __init__(self, in_channels=12, out_channels=12):
#         super().__init__()
#         self.name = "ECG_Unet_Flow"
#         def conv_block(in_dim, out_dim):
#             return nn.Sequential(
#                 nn.Conv1d(in_dim, out_dim, 3, padding=1), nn.BatchNorm1d(out_dim), nn.ReLU(True),
#                 CoordAtt1D(out_dim, out_dim), 
#                 nn.Conv1d(out_dim, out_dim, 3, padding=1), nn.BatchNorm1d(out_dim), nn.ReLU(True)
#             )
#         self.enc1 = conv_block(in_channels, 64); self.pool1 = nn.MaxPool1d(2)
#         self.enc2 = conv_block(64, 128); self.pool2 = nn.MaxPool1d(2)
#         self.enc3 = conv_block(128, 256); self.pool3 = nn.MaxPool1d(2)
#         self.enc4 = conv_block(256, 512); self.pool4 = nn.MaxPool1d(2)
#         self.bottleneck = conv_block(512, 1024)
#         self.up4 = nn.ConvTranspose1d(1024, 512, 2, 2); self.fuse4 = WaveletSKFusion(512); self.dec4 = conv_block(1024, 512)
#         self.up3 = nn.ConvTranspose1d(512, 256, 2, 2); self.fuse3 = WaveletSKFusion(256); self.dec3 = conv_block(512, 256)
#         self.up2 = nn.ConvTranspose1d(256, 128, 2, 2); self.fuse2 = WaveletSKFusion(128); self.dec2 = conv_block(256, 128)
#         self.up1 = nn.ConvTranspose1d(128, 64, 2, 2); self.fuse1 = WaveletSKFusion(64); self.dec1 = conv_block(128, 64)
#         self.final_conv = nn.Conv1d(64, out_channels, 1); self.final_act = nn.Tanh()
#     def forward(self, x):
#         e1 = self.enc1(x.float()); e2 = self.enc2(self.pool1(e1))
#         e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
#         b = self.bottleneck(self.pool4(e4))
#         d4 = self.dec4(torch.cat((self.up4(b), self.fuse4(e4, self.up4(b))), dim=1))
#         d3 = self.dec3(torch.cat((self.up3(d4), self.fuse3(e3, self.up3(d4))), dim=1))
#         d2 = self.dec2(torch.cat((self.up2(d3), self.fuse2(e2, self.up2(d3))), dim=1))
#         d1 = self.dec1(torch.cat((self.up1(d2), self.fuse1(e1, self.up1(d2))), dim=1))
#         return self.final_act(self.final_conv(d1))
#     # 在 AdvancedUNet1D 类内部添加
#     @torch.no_grad()
#     def predict(self, x, device='cuda'):
#         """
#         输入形状: (Batch, 512, 12) 或 (512, 12)
#         输出形状: (Batch, 512, 12)
#         """
#         self.to(device)
#         self.eval()
        
#         # 1. 维度处理与转换: (B, 512, 12) -> (B, 12, 512)
#         x = torch.as_tensor(x, dtype=torch.float32, device=device)
#         if x.ndim == 2: x = x.unsqueeze(0) # 补齐 Batch 维
#         if x.shape[-1] == 12: x = x.permute(0, 2, 1)
        
#         # 2. 前向传播
#         out = self.forward(x)
        
#         # 3. 还原维度并返回 Numpy: (B, 512, 12)
#         return out.permute(0, 2, 1).cpu().numpy()
    
# class FlowNetwork(nn.Module):
#     def __init__(self, channels=12):
#         super().__init__()
#         self.t_mlp = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 128))
#         self.init_conv = nn.Conv1d(channels * 2, 128, 3, padding=1)
#         self.res_block = nn.Sequential(nn.Conv1d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(), nn.Conv1d(128, 128, 3, padding=1), nn.GroupNorm(8, 128))
#         self.final_conv = nn.Conv1d(128, channels, 1)
#     def forward(self, t, xt, cond):
#         t_embed = self.t_mlp(t.unsqueeze(-1)).unsqueeze(-1)
#         x = self.init_conv(torch.cat([xt, cond], dim=1))
#         x = x + t_embed.expand(-1, -1, x.size(-1)) 
#         return self.final_conv(x + self.res_block(x))
#     # 在 FlowNetwork 类内部添加
#     @torch.no_grad()
#     def predict(self, x_cond, steps=30, device='cuda'):
#         """
#         输入形状: (Batch, 512, 12) -> 来自 UNet 的初步结果
#         输出形状: (Batch, 512, 12) -> 修正后的最终结果
#         """
#         self.to(device)
#         self.eval()
        
#         # 1. 维度转换: (B, 512, 12) -> (B, 12, 512)
#         x_cond = torch.as_tensor(x_cond, dtype=torch.float32, device=device)
#         if x_cond.ndim == 2: x_cond = x_cond.unsqueeze(0)
#         if x_cond.shape[-1] == 12: x_cond = x_cond.permute(0, 2, 1)
        
#         # 2. 迭代采样逻辑 (参考 ConditionalFlowMatcher.sample)
#         xt = x_cond.clone()
#         dt = 1.0 / steps
#         for i in range(steps):
#             t = torch.full((x_cond.shape[0],), i/steps, device=device)
#             # FlowNetwork forward 顺序为 (t, xt, cond)
#             xt = xt + self.forward(t, xt, x_cond) * dt
            
#         # 3. 均值对齐处理
#         bias = xt.mean(dim=-1, keepdim=True) - x_cond.mean(dim=-1, keepdim=True)
#         final_out = xt - bias

#         # 4. 还原维度: (B, 512, 12)
#         return final_out.permute(0, 2, 1).cpu().numpy()
    
    
# class ConditionalFlowMatcher:
#     def __init__(self, model, device, alpha=4.0, loss_weights=None):
#         self.model = model
#         self.device = device
#         self.alpha = alpha
#         self.loss_weights = loss_weights

#     def compute_loss(self, x1, x_recon):
#         batch_size = x1.shape[0]
#         t = torch.rand(batch_size, device=self.device)
#         t_v = t.view(-1, 1, 1)
#         x0 = x_recon
#         xt = (1 - t_v) * x0 + t_v * x1
#         target_v = x1 - x0
#         pred_v = self.model(t, xt, x_recon)
#         mse_diff = (pred_v - target_v)**2
#         if self.loss_weights is not None:
#             mse_diff = mse_diff * self.loss_weights.view(1, 12, 1)
#         bias_loss = torch.mean(torch.abs(pred_v.mean(dim=-1) - target_v.mean(dim=-1)))
#         return torch.mean(mse_diff) + self.alpha * F.l1_loss(pred_v[:, :, 1:]-pred_v[:, :, :-1], target_v[:, :, 1:]-target_v[:, :, :-1]) + 0.5 * bias_loss

#     @torch.no_grad()
#     def sample(self, x_recon, steps=30):
#         self.model.eval()
#         xt = x_recon.clone()
#         dt = 1.0 / steps
#         for i in range(steps):
#             t = torch.full((x_recon.shape[0],), i/steps, device=self.device)
#             xt = xt + self.model(t, xt, x_recon) * dt
#         bias = xt.mean(dim=-1, keepdim=True) - x_recon.mean(dim=-1, keepdim=True)
#         return xt - bias




# -----------------------------------------------------------------------------



#models/unet_flow.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 保留原有的辅助组件 ---
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

# --- 保留原有的 UNet 架构 ---
class AdvancedUNet1D(nn.Module):
    def __init__(self, in_channels=12, out_channels=12):
        super().__init__()
        self.name = "ECG_Unet_Flow"
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
# --- 新增：TimeNav Attention 核心组件 ---
# ==============================================================================
class TimeNavAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.proj_q = nn.Conv1d(channels, channels, 1)
        self.proj_k = nn.Conv1d(channels, channels, 1)
        self.proj_v = nn.Linear(channels * 2, channels) 
        self.output_proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x_mod, cond, t_feat):
        B, C, L = x_mod.shape
        q = self.proj_q(x_mod).view(B, self.num_heads, self.head_dim, L)
        k = self.proj_k(cond).view(B, self.num_heads, self.head_dim, L)
        
        # Time Momentum (V)
        v = self.proj_v(t_feat).unsqueeze(-1).repeat(1, 1, L)
        v = v.view(B, self.num_heads, self.head_dim, L)
        
        # Attention Core
        attn = torch.matmul(q.transpose(-1, -2), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(v, attn.transpose(-1, -2)).reshape(B, C, L)
        return self.output_proj(out)

# ==============================================================================
# --- 升级版：FlowNetwork (完全对应架构图) ---
# ==============================================================================
class FlowNetwork(nn.Module):
    def __init__(self, channels=12):
        super().__init__()
        # Stage 1: FiLM MLP
        self.t_mlp = nn.Sequential(
            nn.Linear(1, channels * 2), nn.SiLU(), nn.Linear(channels * 2, channels * 2)
        )
        # Stage 2: TimeNav Attention
        self.timenav = TimeNavAttention(channels)
        # Stage 3: Residual & Output
        self.ln = nn.LayerNorm(channels)
        self.final_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, t, xt, cond):
        B, C, L = xt.shape
        # Stage 1: FiLM
        t_feat = self.t_mlp(t.unsqueeze(-1))
        gamma, beta = torch.chunk(t_feat, 2, dim=1)
        x_mod = xt * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        
        # Stage 2: Attention
        attn_out = self.timenav(x_mod, cond, t_feat)
        
        # Stage 3: Output
        x = (xt + attn_out).transpose(1, 2)
        x = self.ln(x).transpose(1, 2)
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
# --- 升级版：ConditionalFlowMatcher (峰值加权 Loss) ---
# ==============================================================================
class ConditionalFlowMatcher:
    def __init__(self, model, device, alpha=5.0, gamma=2.0):
        self.model = model
        self.device = device
        self.alpha = alpha  # 梯度锐利度权重
        self.gamma = gamma  # 峰值增强系数

    def compute_loss(self, x1, x_recon):
        """
        x1: 真实信号 (B, 12, 512)
        x_recon: UNet 初步重建信号 (B, 12, 512)
        """
        batch_size = x1.shape[0]
        t = torch.rand(batch_size, device=self.device)
        t_v = t.view(-1, 1, 1)
        
        xt = (1 - t_v) * x_recon + t_v * x1
        target_v = x1 - x_recon
        pred_v = self.model(t, xt, x_recon)
        
        # --- 改进点 1: 峰值加权权重 ---
        # 信号数值越大的地方(波峰)，权重越高
        peak_weights = 1.0 + self.gamma * torch.abs(x1)
        
        # --- 改进点 2: 加权 MSE ---
        mse_loss = torch.mean(((pred_v - target_v)**2) * peak_weights)
        
        # --- 改进点 3: 梯度(锐利度)损失 ---
        # 强制 Flow 模型学习 QRS 波陡峭的上升和下降沿
        grad_loss = F.l1_loss(pred_v[:, :, 1:] - pred_v[:, :, :-1], 
                              target_v[:, :, 1:] - target_v[:, :, :-1])
        
        # --- 改进点 4: 均值偏置约束 ---
        bias_loss = torch.mean(torch.abs(pred_v.mean(dim=-1) - target_v.mean(dim=-1)))
        
        return mse_loss + self.alpha * grad_loss + 0.5 * bias_loss

    @torch.no_grad()
    def sample(self, x_recon, steps=30):
        # 采样逻辑保持与 predict 内部一致
        self.model.eval()
        xt = x_recon.clone()
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x_recon.shape[0],), i/steps, device=self.device)
            xt = xt + self.model(t, xt, x_recon) * dt
        bias = xt.mean(dim=-1, keepdim=True) - x_recon.mean(dim=-1, keepdim=True)
        return xt - bias