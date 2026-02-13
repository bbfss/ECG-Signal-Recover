import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks=[9, 19, 39], bottleneck=True):
        super().__init__()
        ks = [k if k % 2 != 0 else k + 1 for k in ks]  # 确保是奇数
        self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False) if bottleneck else nn.Identity()
        self.convs = nn.ModuleList([nn.Conv1d(nf if bottleneck else ni, nf, k, padding=k//2, bias=False) for k in ks])
        self.maxpool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(ni, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(nf * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(x)
        x = torch.cat([conv(x) for conv in self.convs] + [self.maxpool(input_tensor)], dim=1)
        return self.act(self.bn(x))

class ECGClassifier(nn.Module):
    """
    最新最猛的 InceptionTime 架构，针对 12 导联 ECG 优化
    """
    def __init__(self, in_channels=12, num_classes=5, nf=32, depth=6):
        super().__init__()
        self.name = "InceptionTime_Doctor"
        
        # 初始特征提取
        self.first_layer = nn.Sequential(
            nn.Conv1d(in_channels, nf, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(nf),
            nn.ReLU()
        )
        
        # Inception 堆叠层
        layers = []
        curr_ni = nf
        for i in range(depth):
            layers.append(InceptionModule(curr_ni, nf))
            curr_ni = nf * 4
            # 引入残差连接 (每三层做一次)
            if i % 3 == 2:
                layers.append(nn.Identity()) # 占位，简化演示

        self.layers = nn.Sequential(*layers)
        
        # 全局池化和分类
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(nf * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: [batch, 12, 512]
        x = self.first_layer(x)
        x = self.layers(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits