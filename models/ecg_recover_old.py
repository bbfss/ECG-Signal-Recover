import torch
import torch.nn as nn
import numpy as np

# --- 基础层定义 ---

class Conv1D_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1D 路径：处理每个导联内部特征
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.02)
        )
    def forward(self, x): return self.conv(x)

class Conv2D_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 2D 路径：跨导联空间特征
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.02)
        )
    def forward(self, x): return self.conv(x)

class Deconv2D_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.02)
        )
    def forward(self, x): return self.deconv(x)

# --- 最终修复版模型 ---

class ECGRecover(nn.Module):
    def __init__(self, **kwargs):
        """
        使用 **kwargs 接收所有参数（包括 in_channels），
        但我们内部强制使用 1 通道逻辑处理 12 导联图像。
        """
        super(ECGRecover, self).__init__()
        self.name = "ECG-Recover-NoSkip"
        
        # 第一层强制使用 in_channels=1
        base_channels = 1 
        
        # -------- Encoder (双路径) --------
        self.encoder_conv1d = nn.ModuleList([
            Conv1D_layer(base_channels, 16), Conv1D_layer(16, 32),
            Conv1D_layer(32, 64), Conv1D_layer(64, 128)
        ])
        
        self.encoder_conv2d = nn.ModuleList([
            Conv2D_layer(base_channels, 16), Conv2D_layer(16, 32),
            Conv2D_layer(32, 64), Conv2D_layer(64, 128)
        ])
        
        # -------- Bottleneck --------
        self.transition_block = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02)
        )

        # -------- Decoder (无跳连) --------
        self.decoder_deconv2d = nn.ModuleList([
            Deconv2D_layer(256, 128), 
            Deconv2D_layer(128, 64),  
            Deconv2D_layer(64, 32),   
            Deconv2D_layer(32, 1)     
        ])
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Tanh()
        )

    def _prepare_input(self, x):
        """
        自动维度对齐工具：
        将任何 (B, 12, 512) 或 (B, 512, 12) 统一转换为 (B, 1, 12, 512)
        """
        if x.dim() == 3:
            # 如果是 (B, 512, 12) -> (B, 12, 512)
            if x.shape[1] == 512 and x.shape[2] == 12:
                x = x.permute(0, 2, 1)
            # 变为 (B, 1, 12, 512)
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] > 10:
            # 针对报错中出现的 (1, 128, 12, 512) 异常情况进行压缩
            # 这种通常是 batch 被误 unsqueeze 了
            x = x.squeeze(0).unsqueeze(1) if x.shape[0] == 1 else x.mean(dim=1, keepdim=True)
        
        return x

    def forward(self, x):
        # 1. 自动适配维度
        x = self._prepare_input(x)
        
        # 2. Encoder 路径
        c1d_1 = self.encoder_conv1d[0](x); c2d_1 = self.encoder_conv2d[0](x)
        c1d_2 = self.encoder_conv1d[1](c1d_1); c2d_2 = self.encoder_conv2d[1](c2d_1)
        c1d_3 = self.encoder_conv1d[2](c1d_2); c2d_3 = self.encoder_conv2d[2](c2d_2)
        c1d_4 = self.encoder_conv1d[3](c1d_3); c2d_4 = self.encoder_conv2d[3](c2d_3)

        # 3. 瓶颈融合与解码
        fused_4 = torch.cat((c1d_4, c2d_4), dim=1) 
        dec_4 = self.transition_block(fused_4)
        
        dec_3 = self.decoder_deconv2d[0](dec_4)
        dec_2 = self.decoder_deconv2d[1](dec_3)
        dec_1 = self.decoder_deconv2d[2](dec_2)
        dec_0 = self.decoder_deconv2d[3](dec_1)
        
        # 返回结果前，自动还原为 (B, 12, 512) 方便 Trainer 计算 Loss
        return self.final_conv(dec_0).squeeze(1)

    @torch.no_grad()
    def predict(self, input_data, device='cuda'):
        self.to(device); self.eval()
        x = torch.as_tensor(input_data, dtype=torch.float32, device=device)
        out = self.forward(x) # 得到 (B, 12, 512)
        # 最终适配 evaluate.py 要求的 (B, 512, 12)
        return out.permute(0, 2, 1).cpu().numpy()