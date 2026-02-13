import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 基础组件定义 ---

class Conv1D_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 处理单导联内部时间特征
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.02)
        )
    def forward(self, x): return self.conv(x)

class Conv2D_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 处理跨导联空间特征
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.02)
        )
    def forward(self, x): return self.conv(x)

class Deconv2D_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 对应编码器的上采样层
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.02)
        )
    def forward(self, x): return self.deconv(x)

# --- 核心模型实现 ---

class ECGRecover(nn.Module):
    def __init__(self, **kwargs):
        super(ECGRecover, self).__init__()
        self.name = "ECG-Recover-Integrated"
        
        # 强制使用 1 通道输入逻辑 (B, 1, 12, 512)
        base_channels = 1 
        
        # -------- Encoder (双路径：1D 时间 + 2D 空间) --------
        self.encoder_conv1d = nn.ModuleList([
            Conv1D_layer(base_channels, 16), Conv1D_layer(16, 32),
            Conv1D_layer(32, 64), Conv1D_layer(64, 128)
        ])
        
        self.encoder_conv2d = nn.ModuleList([
            Conv2D_layer(base_channels, 16), Conv2D_layer(16, 32),
            Conv2D_layer(32, 64), Conv2D_layer(64, 128)
        ])
        
        # -------- Bottleneck (特征融合) --------
        # 由于拼接了 1D 和 2D 路径，输入通道为 128 + 128 = 256
        self.transition_block = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02)
        )

        # -------- Decoder --------
        self.decoder_deconv2d = nn.ModuleList([
            Deconv2D_layer(256, 128), 
            Deconv2D_layer(128, 64),  
            Deconv2D_layer(64, 32),   
            Deconv2D_layer(32, 1)     
        ])
        
        # 输出层：映射回原始导联形状并限制范围
        self.final_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Tanh()
        )

    def _prepare_input(self, x):
        """ 维度对齐：将 (B, 12, 512) 转换为 (B, 1, 12, 512) """
        if x.dim() == 3:
            if x.shape[1] == 512 and x.shape[2] == 12:
                x = x.permute(0, 2, 1)
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] > 10:
             x = x.squeeze(0).unsqueeze(1) if x.shape[0] == 1 else x.mean(dim=1, keepdim=True)
        return x

    def forward(self, x):
        # 1. 输入处理
        x = self._prepare_input(x)
        
        # 2. Encoder 路径并行计算
        c1d, c2d = x, x
        for i in range(4):
            c1d = self.encoder_conv1d[i](c1d)
            c2d = self.encoder_conv2d[i](c2d)

        # 3. 瓶颈层特征拼接与转换
        fused = torch.cat((c1d, c2d), dim=1) 
        dec = self.transition_block(fused)
        
        # 4. Decoder 逐步上采样
        for layer in self.decoder_deconv2d:
            dec = layer(dec)
        
        # 5. 返回 (B, 12, 512) 格式
        out = self.final_conv(dec)
        return out.squeeze(1)

    @torch.no_grad()
    def predict(self, input_data, device='cuda'):
        self.to(device)
        self.eval()
        x = torch.as_tensor(input_data, dtype=torch.float32, device=device)
        out = self.forward(x) 
        # 适配 evaluate.py 要求的 (B, 512, 12) 格式
        return out.permute(0, 2, 1).cpu().numpy()