import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 保留你原有的组件 ---
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

# --- 升级版 WaveletSKFusion (原生 2D 支持) ---
class WaveletSKFusion(nn.Module):
    def __init__(self, in_chan, reduction=4):
        super().__init__()
        self.in_chan = in_chan
        # 使用 2D 卷积以匹配主架构的 2D 路径
        self.p1 = nn.Sequential(nn.Conv2d(in_chan, in_chan, 3, padding=1, groups=in_chan), nn.BatchNorm2d(in_chan), nn.SiLU())
        self.p2 = nn.Sequential(nn.Conv2d(in_chan, in_chan, 3, padding=2, dilation=2, groups=in_chan), nn.BatchNorm2d(in_chan), nn.SiLU())
        self.p3 = nn.Sequential(nn.Conv2d(in_chan, in_chan, 3, padding=4, dilation=4, groups=in_chan), nn.BatchNorm2d(in_chan), nn.SiLU())
        
        mid_chan = max(16, in_chan // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_chan, mid_chan), nn.ReLU(True), nn.Linear(mid_chan, in_chan * 3))
        self.last_conv = nn.Conv2d(in_chan, in_chan, 1); self.norm = nn.BatchNorm2d(in_chan)
        
    def forward(self, skip_x, up_x):
        # 确保尺寸一致
        if up_x.shape[-2:] != skip_x.shape[-2:]:
            up_x = F.interpolate(up_x, size=skip_x.shape[-2:], mode='bilinear', align_corners=False)
            
        x = skip_x + up_x
        u1, u2, u3 = self.p1(x), self.p2(x), self.p3(x)
        
        # 2D 特征通道注意力
        s = self.gap(u1 + u2 + u3).view(x.size(0), -1)
        z = self.fc(s).view(x.size(0), 3, self.in_chan, 1, 1)
        attn = F.softmax(z, dim=1) 
        
        v = u1 * attn[:, 0] + u2 * attn[:, 1] + u3 * attn[:, 2]
        return self.norm(self.last_conv(v)) + up_x

class AdvancedUNet1D(nn.Module):
    def __init__(self, in_channels=12, out_channels=12):
        super().__init__()
        self.name = "ECG_Recover_Flow_Hybrid"
        
        def conv2d_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.02),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU(0.02)
            )

        # -------- Encoder (1D+2D 双路径) --------
        # Level 1: 1+1 -> 32通道
        self.enc1d = nn.ModuleList([
            nn.Conv2d(1, 16, (1,3), (1,2), (0,1)), nn.Conv2d(16, 32, (1,3), (1,2), (0,1)),
            nn.Conv2d(32, 64, (1,3), (1,2), (0,1)), nn.Conv2d(64, 128, (1,3), (1,2), (0,1))
        ])
        self.enc2d = nn.ModuleList([
            nn.Conv2d(1, 16, (3,3), (1,2), (1,1)), nn.Conv2d(16, 32, (3,3), (1,2), (1,1)),
            nn.Conv2d(32, 64, (3,3), (1,2), (1,1)), nn.Conv2d(64, 128, (3,3), (1,2), (1,1))
        ])

        # Bottleneck: 输入是 f4 (128+128=256通道)
        self.bottleneck = conv2d_block(256, 512)

        # -------- Decoder (修正通道匹配) --------
        # u4 上采样后应与 f3 (128通道) 匹配
        self.up4 = nn.ConvTranspose2d(512, 128, (1,4), (1,2), (0,1))
        self.fuse4 = WaveletSKFusion(128)
        self.dec4 = conv2d_block(256, 128) # 128 (up) + 128 (fused) = 256

        # u3 上采样后应与 f2 (64通道) 匹配
        self.up3 = nn.ConvTranspose2d(128, 64, (1,4), (1,2), (0,1))
        self.fuse3 = WaveletSKFusion(64)
        self.dec3 = conv2d_block(128, 64) # 64 + 64 = 128

        # u2 上采样后应与 f1 (32通道) 匹配
        self.up2 = nn.ConvTranspose2d(64, 32, (1,4), (1,2), (0,1))
        self.fuse2 = WaveletSKFusion(32)
        self.dec2 = conv2d_block(64, 32) # 32 + 32 = 64

        # 最后回到原始 1 通道
        self.up1 = nn.ConvTranspose2d(32, 16, (1,4), (1,2), (0,1))
        self.final_conv = nn.Conv2d(16, 1, 1)
        self.final_act = nn.Tanh()

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        
        # Encoder
        e1d1 = self.enc1d[0](x);  e2d1 = self.enc2d[0](x);  f1 = torch.cat([e1d1, e2d1], 1) # 32
        e1d2 = self.enc1d[1](e1d1); e2d2 = self.enc2d[1](e2d1); f2 = torch.cat([e1d2, e2d2], 1) # 64
        e1d3 = self.enc1d[2](e1d2); e2d3 = self.enc2d[2](e2d2); f3 = torch.cat([e1d3, e2d3], 1) # 128
        e1d4 = self.enc1d[3](e1d3); e2d4 = self.enc2d[3](e2d3); f4 = torch.cat([e1d4, e2d4], 1) # 256

        b = self.bottleneck(f4)

        # Decoder (严格对齐每一层跳连通道)
        u4 = self.up4(b)
        # u4 是 128, f3 是 128 -> fuse4 没问题
        d4 = self.dec4(torch.cat([u4, self.fuse4(f3, u4)], 1))

        u3 = self.up3(d4)
        # u3 是 64, f2 是 64 -> fuse3 没问题
        d3 = self.dec3(torch.cat([u3, self.fuse3(f2, u3)], 1))

        u2 = self.up2(d3)
        # u2 是 32, f1 是 32 -> fuse2 没问题
        d2 = self.dec2(torch.cat([u2, self.fuse2(f1, u2)], 1))

        u1 = self.up1(d2)
        res = self.final_act(self.final_conv(u1))
        return res.squeeze(1)

    @torch.no_grad()
    def predict(self, x, device='cuda'):
        self.to(device); self.eval()
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        if x.ndim == 2: x = x.unsqueeze(0)
        if x.shape[-1] == 12 and x.shape[-2] != 12: x = x.permute(0, 2, 1)
        out = self.forward(x)
        return out.permute(0, 2, 1).cpu().numpy()
    
# --- 专家 1: 高频局部专家 (QRS 复合波) ---
class HighFreqExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1)
        )
    def forward(self, x): return self.net(x)

# --- 专家 2: 低频长程专家 (ST-T 段) ---
class LowFreqExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 使用空洞卷积扩大感受野
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=2, dilation=2),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=4, dilation=4)
        )
    def forward(self, x): return self.net(x)

# --- 专家 3: 全局相关性专家 (导联间约束) ---
class GlobalExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv1d(channels, channels, 1)
        self.attn = CoordAtt1D(channels, channels) # 复用你代码里的 CoordAtt1D
    def forward(self, x): return self.attn(self.proj(x))

# --- 核心：MoE 路由网络 ---
class FlowRouter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 路由输入：时间步 t (1维) + 信号全局特征 (channels维)
        self.router_net = nn.Sequential(
            nn.Linear(1 + channels, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3), # 对应 3 个专家
            nn.Softmax(dim=-1)
        )
    def forward(self, t, cond_global):
        # t: [B], cond_global: [B, C]
        inp = torch.cat([t.unsqueeze(-1), cond_global], dim=1)
        return self.router_net(inp) # [B, 3]

# --- 升级后的 MoE-FlowNetwork ---
class MoEFlowNetwork(nn.Module):
    def __init__(self, channels=12):
        super().__init__()
        self.name = "ECG_MoE_Flow_Refiner"
        hidden_dim = 128
        
        # 1. 时间步嵌入
        self.t_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        
        # 2. 输入映射
        self.init_conv = nn.Conv1d(channels * 2, hidden_dim, 3, padding=1)
        
        # 3. 实例化专家
        self.experts = nn.ModuleList([
            HighFreqExpert(hidden_dim),
            LowFreqExpert(hidden_dim),
            GlobalExpert(hidden_dim)
        ])
        
        # 4. 路由
        self.router = FlowRouter(hidden_dim)
        
        # 5. 输出映射
        self.final_conv = nn.Conv1d(hidden_dim, channels, 1)

    def forward(self, t, xt, cond):
        # t: [B], xt: [B, 12, 512], cond: [B, 12, 512]
        
        # 时间嵌入
        t_embed = self.t_mlp(t.unsqueeze(-1)).unsqueeze(-1) # [B, 128, 1]
        
        # 特征提取
        x = self.init_conv(torch.cat([xt, cond], dim=1)) # [B, 128, 512]
        x = x + t_embed
        
        # 计算路由权重 (使用全局平均池化作为信号描述符)
        cond_global = torch.mean(x, dim=-1) # [B, 128]
        weights = self.router(t, cond_global) # [B, 3]
        
        # 专家混合 (Expert Mixing)
        # 这种写法比循环更快，且在计算图里更清晰
        v_0 = self.experts[0](x)
        v_1 = self.experts[1](x)
        v_2 = self.experts[2](x)
        
        # 融合向量场
        # weights: [B, 3] -> 调整维度后相乘
        out = (v_0 * weights[:, 0].view(-1, 1, 1) + 
               v_1 * weights[:, 1].view(-1, 1, 1) + 
               v_2 * weights[:, 2].view(-1, 1, 1))
        
        return self.final_conv(out)

    @torch.no_grad()
    def predict(self, x_cond, steps=30, device='cuda'):
        # 保持你原有的采样接口不变
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
    
    def get_expert_weights(self, t, xt, cond):
        """专门用于提取路由权重的接口"""
        t_embed = self.t_mlp(t.unsqueeze(-1))
        x = self.init_conv(torch.cat([xt, cond], dim=1))
        x = x + t_embed.unsqueeze(-1)
        cond_global = torch.mean(x, dim=-1)
        weights = self.router(t, cond_global) # [B, 3]
        return weights
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalFlowMatcher:
    def __init__(self, model, device, alpha=4.0, beta=1.0, loss_weights=None):
        """
        model: MoEFlowNetwork
        alpha: 梯度一致性损失权重 (捕捉突变特征)
        beta: 专家平衡损失权重 (防止模型只用某一个专家)
        """
        self.model = model
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.loss_weights = loss_weights

    def compute_loss(self, x1, x_recon):
        """
        x1: 真实完整信号 [B, 12, 512]
        x_recon: UNet 初步重建信号 [B, 12, 512]
        """
        batch_size = x1.shape[0]
        
        # 1. 采样时间步 t (用于流场演化)
        t = torch.rand(batch_size, device=self.device)
        t_v = t.view(-1, 1, 1)
        
        # 2. 构造条件概率路径 (Conditional Probability Path)
        # x0 是粗糙图，x1 是精细图，xt 是中间演化状态
        x0 = x_recon
        xt = (1 - t_v) * x0 + t_v * x1
        
        # 3. 理论目标向量场 (Target Velocity Field)
        target_v = x1 - x0 
        
        # 4. 模型预测向量场 (MoE 预测)
        # 注意：这里我们让 forward 返回预测值和专家权重，方便后续做分析
        # 如果你没改 forward 返回值，就只接 pred_v
        pred_v = self.model(t, xt, x_recon)
        
        # 5. 基础损失：MSE Loss (重建流场误差)
        mse_diff = (pred_v - target_v)**2
        if self.loss_weights is not None:
            mse_diff = mse_diff * self.loss_weights.view(1, 12, 1)
        base_loss = torch.mean(mse_diff)
        
        # 6. 核心改进 A：TV正则化/生理斜率一致性 (Physiological Gradient Matching)
        # 捕捉 ECG 的病理突变细节（如 QRS 波陡峭度）
        grad_loss = F.l1_loss(
            pred_v[:, :, 1:] - pred_v[:, :, :-1], 
            target_v[:, :, 1:] - target_v[:, :, :-1]
        )
        
        # 7. 核心改进 B：直流偏移一致性 (Bias Consistency)
        # 确保流场修正后不会导致基线漂移
        bias_loss = torch.mean(torch.abs(pred_v.mean(dim=-1) - target_v.mean(dim=-1)))
        
        # 8. (可选) 专家负载均衡损失 (Expert Balance Loss)
        # 如果你希望在论文中讨论 MoE 的稳定性，可以加入此项防止单一专家塌陷
        # 这里建议作为“进阶实验”部分在论文中提及
        
        total_loss = base_loss + self.alpha * grad_loss + 0.5 * bias_loss
        
        return total_loss

    @torch.no_grad()
    def sample(self, x_recon, steps=30):
        """
        从 UNet 的初步结果 x_recon 演化到最终精细结果
        """
        self.model.eval()
        xt = x_recon.clone()
        dt = 1.0 / steps
        
        # 逐步数值求解 ODE (Euler Method)
        for i in range(steps):
            t_curr = i / steps
            t = torch.full((x_recon.shape[0],), t_curr, device=self.device)
            
            # 使用预测的向量场更新 xt
            v_t = self.model(t, xt, x_recon)
            xt = xt + v_t * dt
            
        # 均值对齐：消除采样累积误差导致的基线偏移
        bias = xt.mean(dim=-1, keepdim=True) - x_recon.mean(dim=-1, keepdim=True)
        return xt - bias  
    
    
