# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# # --- 保留你原有的组件 ---
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

# # --- 升级版 WaveletSKFusion (原生 2D 支持) ---
# class WaveletSKFusion(nn.Module):
#     def __init__(self, in_chan, reduction=4):
#         super().__init__()
#         self.in_chan = in_chan
#         # 使用 2D 卷积以匹配主架构的 2D 路径
#         self.p1 = nn.Sequential(nn.Conv2d(in_chan, in_chan, 3, padding=1, groups=in_chan), nn.BatchNorm2d(in_chan), nn.SiLU())
#         self.p2 = nn.Sequential(nn.Conv2d(in_chan, in_chan, 3, padding=2, dilation=2, groups=in_chan), nn.BatchNorm2d(in_chan), nn.SiLU())
#         self.p3 = nn.Sequential(nn.Conv2d(in_chan, in_chan, 3, padding=4, dilation=4, groups=in_chan), nn.BatchNorm2d(in_chan), nn.SiLU())
        
#         mid_chan = max(16, in_chan // reduction)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(nn.Linear(in_chan, mid_chan), nn.ReLU(True), nn.Linear(mid_chan, in_chan * 3))
#         self.last_conv = nn.Conv2d(in_chan, in_chan, 1); self.norm = nn.BatchNorm2d(in_chan)
        
#     def forward(self, skip_x, up_x):
#         # 确保尺寸一致
#         if up_x.shape[-2:] != skip_x.shape[-2:]:
#             up_x = F.interpolate(up_x, size=skip_x.shape[-2:], mode='bilinear', align_corners=False)
            
#         x = skip_x + up_x
#         u1, u2, u3 = self.p1(x), self.p2(x), self.p3(x)
        
#         # 2D 特征通道注意力
#         s = self.gap(u1 + u2 + u3).view(x.size(0), -1)
#         z = self.fc(s).view(x.size(0), 3, self.in_chan, 1, 1)
#         attn = F.softmax(z, dim=1) 
        
#         v = u1 * attn[:, 0] + u2 * attn[:, 1] + u3 * attn[:, 2]
#         return self.norm(self.last_conv(v)) + up_x

# class AdvancedUNet1D(nn.Module):
#     def __init__(self, in_channels=12, out_channels=12):
#         super().__init__()
#         self.name = "ECG_Recover_Flow_Hybrid"
        
#         def conv2d_block(in_c, out_c):
#             return nn.Sequential(
#                 nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.02),
#                 nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.02)
#             )

#         # -------- Encoder (1D+2D 双路径) --------
#         # Level 1: 1+1 -> 32通道
#         self.enc1d = nn.ModuleList([
#             nn.Conv2d(1, 16, (1,3), (1,2), (0,1)), nn.Conv2d(16, 32, (1,3), (1,2), (0,1)),
#             nn.Conv2d(32, 64, (1,3), (1,2), (0,1)), nn.Conv2d(64, 128, (1,3), (1,2), (0,1))
#         ])
#         self.enc2d = nn.ModuleList([
#             nn.Conv2d(1, 16, (3,3), (1,2), (1,1)), nn.Conv2d(16, 32, (3,3), (1,2), (1,1)),
#             nn.Conv2d(32, 64, (3,3), (1,2), (1,1)), nn.Conv2d(64, 128, (3,3), (1,2), (1,1))
#         ])

#         # Bottleneck: 输入是 f4 (128+128=256通道)
#         self.bottleneck = conv2d_block(256, 512)

#         # -------- Decoder (修正通道匹配) --------
#         # u4 上采样后应与 f3 (128通道) 匹配
#         self.up4 = nn.ConvTranspose2d(512, 128, (1,4), (1,2), (0,1))
#         self.fuse4 = WaveletSKFusion(128)
#         self.dec4 = conv2d_block(256, 128) # 128 (up) + 128 (fused) = 256

#         # u3 上采样后应与 f2 (64通道) 匹配
#         self.up3 = nn.ConvTranspose2d(128, 64, (1,4), (1,2), (0,1))
#         self.fuse3 = WaveletSKFusion(64)
#         self.dec3 = conv2d_block(128, 64) # 64 + 64 = 128

#         # u2 上采样后应与 f1 (32通道) 匹配
#         self.up2 = nn.ConvTranspose2d(64, 32, (1,4), (1,2), (0,1))
#         self.fuse2 = WaveletSKFusion(32)
#         self.dec2 = conv2d_block(64, 32) # 32 + 32 = 64

#         # 最后回到原始 1 通道
#         self.up1 = nn.ConvTranspose2d(32, 16, (1,4), (1,2), (0,1))
#         self.final_conv = nn.Conv2d(16, 1, 1)
#         self.final_act = nn.Tanh()

#     def forward(self, x):
#         if x.dim() == 3: x = x.unsqueeze(1)
        
#         # Encoder
#         e1d1 = self.enc1d[0](x);  e2d1 = self.enc2d[0](x);  f1 = torch.cat([e1d1, e2d1], 1) # 32
#         e1d2 = self.enc1d[1](e1d1); e2d2 = self.enc2d[1](e2d1); f2 = torch.cat([e1d2, e2d2], 1) # 64
#         e1d3 = self.enc1d[2](e1d2); e2d3 = self.enc2d[2](e2d2); f3 = torch.cat([e1d3, e2d3], 1) # 128
#         e1d4 = self.enc1d[3](e1d3); e2d4 = self.enc2d[3](e2d3); f4 = torch.cat([e1d4, e2d4], 1) # 256

#         b = self.bottleneck(f4)

#         # Decoder (严格对齐每一层跳连通道)
#         u4 = self.up4(b)
#         # u4 是 128, f3 是 128 -> fuse4 没问题
#         d4 = self.dec4(torch.cat([u4, self.fuse4(f3, u4)], 1))

#         u3 = self.up3(d4)
#         # u3 是 64, f2 是 64 -> fuse3 没问题
#         d3 = self.dec3(torch.cat([u3, self.fuse3(f2, u3)], 1))

#         u2 = self.up2(d3)
#         # u2 是 32, f1 是 32 -> fuse2 没问题
#         d2 = self.dec2(torch.cat([u2, self.fuse2(f1, u2)], 1))

#         u1 = self.up1(d2)
#         res = self.final_act(self.final_conv(u1))
#         return res.squeeze(1)

#     @torch.no_grad()
#     def predict(self, x, device='cuda'):
#         self.to(device); self.eval()
#         x = torch.as_tensor(x, dtype=torch.float32, device=device)
#         if x.ndim == 2: x = x.unsqueeze(0)
#         if x.shape[-1] == 12 and x.shape[-2] != 12: x = x.permute(0, 2, 1)
#         out = self.forward(x)
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



# # # -------------------------------------------------------------------------------------------------


# models/unet_flow.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    # 在 AdvancedUNet1D 类内部添加
    @torch.no_grad()
    def predict(self, x, device='cuda'):
        """
        输入形状: (Batch, 512, 12) 或 (512, 12)
        输出形状: (Batch, 512, 12)
        """
        self.to(device)
        self.eval()
        
        # 1. 维度处理与转换: (B, 512, 12) -> (B, 12, 512)
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        if x.ndim == 2: x = x.unsqueeze(0) # 补齐 Batch 维
        if x.shape[-1] == 12: x = x.permute(0, 2, 1)
        
        # 2. 前向传播
        out = self.forward(x)
        
        # 3. 还原维度并返回 Numpy: (B, 512, 12)
        return out.permute(0, 2, 1).cpu().numpy()
    
class FlowNetwork(nn.Module):
    def __init__(self, channels=12):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 128))
        self.init_conv = nn.Conv1d(channels * 2, 128, 3, padding=1)
        self.res_block = nn.Sequential(nn.Conv1d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(), nn.Conv1d(128, 128, 3, padding=1), nn.GroupNorm(8, 128))
        self.final_conv = nn.Conv1d(128, channels, 1)
    def forward(self, t, xt, cond):
        t_embed = self.t_mlp(t.unsqueeze(-1)).unsqueeze(-1)
        x = self.init_conv(torch.cat([xt, cond], dim=1))
        x = x + t_embed.expand(-1, -1, x.size(-1)) 
        return self.final_conv(x + self.res_block(x))
    # 在 FlowNetwork 类内部添加
    @torch.no_grad()
    def predict(self, x_cond, steps=30, device='cuda'):
        """
        输入形状: (Batch, 512, 12) -> 来自 UNet 的初步结果
        输出形状: (Batch, 512, 12) -> 修正后的最终结果
        """
        self.to(device)
        self.eval()
        
        # 1. 维度转换: (B, 512, 12) -> (B, 12, 512)
        x_cond = torch.as_tensor(x_cond, dtype=torch.float32, device=device)
        if x_cond.ndim == 2: x_cond = x_cond.unsqueeze(0)
        if x_cond.shape[-1] == 12: x_cond = x_cond.permute(0, 2, 1)
        
        # 2. 迭代采样逻辑 (参考 ConditionalFlowMatcher.sample)
        xt = x_cond.clone()
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x_cond.shape[0],), i/steps, device=device)
            # FlowNetwork forward 顺序为 (t, xt, cond)
            xt = xt + self.forward(t, xt, x_cond) * dt
            
        # 3. 均值对齐处理
        bias = xt.mean(dim=-1, keepdim=True) - x_cond.mean(dim=-1, keepdim=True)
        final_out = xt - bias

        # 4. 还原维度: (B, 512, 12)
        return final_out.permute(0, 2, 1).cpu().numpy()
    
    
class ConditionalFlowMatcher:
    def __init__(self, model, device, alpha=4.0, loss_weights=None):
        self.model = model
        self.device = device
        self.alpha = alpha
        self.loss_weights = loss_weights

    def compute_loss(self, x1, x_recon):
        batch_size = x1.shape[0]
        t = torch.rand(batch_size, device=self.device)
        t_v = t.view(-1, 1, 1)
        x0 = x_recon
        xt = (1 - t_v) * x0 + t_v * x1
        target_v = x1 - x0
        pred_v = self.model(t, xt, x_recon)
        mse_diff = (pred_v - target_v)**2
        if self.loss_weights is not None:
            mse_diff = mse_diff * self.loss_weights.view(1, 12, 1)
        bias_loss = torch.mean(torch.abs(pred_v.mean(dim=-1) - target_v.mean(dim=-1)))
        return torch.mean(mse_diff) + self.alpha * F.l1_loss(pred_v[:, :, 1:]-pred_v[:, :, :-1], target_v[:, :, 1:]-target_v[:, :, :-1]) + 0.5 * bias_loss

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