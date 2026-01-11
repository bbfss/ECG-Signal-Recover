# models/maefe.py
import torch
import torch.nn as nn

class MaeFE(nn.Module):
    """
    参考论文: MaeFE: Masked Autoencoders Family of Electrocardiogram
    架构: 基于 ViT (Vision Transformer) 的编码器-解码器结构
    """
    def __init__(self, seq_len=512, patch_size=16, in_chans=12,
                 embed_dim=256, depth=8, num_heads=8,
                 decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4,
                 mlp_ratio=4.):
        super().__init__()
        self.name = "MaeFE_Original"  # 【绑定位置1】
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        
        # --- 1. Encoder 部分 ---
        # 线性投影：将每个 patch 的 12 导联数据投影到 embed_dim
        self.patch_embed = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=int(embed_dim*mlp_ratio), 
            activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # --- 2. Decoder 部分 ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim, nhead=decoder_num_heads, 
            dim_feedforward=int(decoder_embed_dim*mlp_ratio), 
            activation='gelu', batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # 输出层：映射回原始的 patch 大小 * 导联数
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * in_chans, bias=True)
        
        self.initialize_weights()

    def initialize_weights(self):
        """初始化位置编码、CLS Token 和权重"""
        nn.init.normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.decoder_pos_embed, std=.02)
        nn.init.normal_(self.mask_token, std=.02)
        nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        前向传播逻辑
        x 形状: [B, 12, 512]
        """
        # Encoder 流程
        x = self.patch_embed(x)      # [B, embed_dim, 32]
        x = x.transpose(1, 2)        # [B, 32, embed_dim]
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        x = self.encoder(x)
        x = self.encoder_norm(x)
        
        # Decoder 流程
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed
        
        x = self.decoder(x)
        x = self.decoder_norm(x)
        
        # 重建信号并重塑形状
        x = self.decoder_pred(x[:, 1:, :]) # 移除 cls token 并投影
        
        # 重塑回 [B, 12, 512]
        x = x.view(x.shape[0], self.num_patches, 12, self.patch_size)
        x = x.permute(0, 2, 1, 3).contiguous().view(x.shape[0], 12, -1)
        
        return torch.tanh(x) # 激活输出
    
    
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