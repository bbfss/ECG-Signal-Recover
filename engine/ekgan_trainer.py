# engine/gan_trainer.py
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .base_trainer import BaseTrainer
from config import Config
from utils import (
    calculate_mae, 
    calculate_pcc, 
    calculate_rmse, 
    save_comparison_plot, 
    apply_transient_mask
)

class GAN_Trainer(BaseTrainer):
    def __init__(self, generator, discriminator, train_loader, val_loader, demo_sig):
        # 1. 调用父类初始化
        super().__init__(generator, train_loader, val_loader, demo_sig)
        
        # --- 核心补丁：保存必要属性 ---
        self.demo_sig = demo_sig
        self.model_name = "EKGAN"
        self.D = discriminator.to(Config.DEVICE)
        
        # 2. 优化器配置
        self.opt_g = torch.optim.Adam(self.model.parameters(), lr=Config.LR, betas=(0.5, 0.9))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=Config.LR, betas=(0.5, 0.9))
        
        # 3. 损失函数
        self.criterion_pixel = nn.L1Loss()
        self.criterion_gan = nn.BCEWithLogitsLoss()

    def train_epoch(self, epoch):
        # 这里的 20 应该是从 Config 获取，为了不改动你的逻辑暂留硬编码
        is_pretrain = epoch <= Config.PRETRAIN_EPOCHS 
        self.model.train()
        self.D.train()
        
        total_loss_g = 0
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch} [{'Pretrain' if is_pretrain else 'GAN'}]")
        
        for x, y in loop:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            
            # --- 1. 更新生成器 ---
            self.opt_g.zero_grad()
            fake_y = self.model(x)
            loss_pixel = self.criterion_pixel(fake_y, y)
            
            if is_pretrain:
                loss_g = loss_pixel
            else:
                pred_fake = self.D(fake_y)
                # 生成器希望判别器认为 fake_y 是 1 (Real)
                loss_adv = self.criterion_gan(pred_fake, torch.ones_like(pred_fake))
                loss_g = Config.LAMBDA_PIXEL * loss_pixel + loss_adv
            
            loss_g.backward()
            self.opt_g.step()

            # --- 2. 更新判别器 (仅在 GAN 阶段) ---
            if not is_pretrain:
                self.opt_d.zero_grad()
                loss_d_real = self.criterion_gan(self.D(y), torch.ones_like(pred_fake))
                loss_d_fake = self.criterion_gan(self.D(fake_y.detach()), torch.zeros_like(pred_fake))
                loss_d = (loss_d_real + loss_d_fake) / 2
                loss_d.backward()
                self.opt_d.step()
                loop.set_postfix(G=f"{loss_g.item():.3f}", D=f"{loss_d.item():.3f}")
            else:
                loop.set_postfix(Pixel=f"{loss_pixel.item():.4f}")
            
            total_loss_g += loss_g.item()
            
        return total_loss_g / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch=None):
        """补全验证逻辑：生成信号并计算指标"""
        self.model.eval()
        all_mae, all_rmse, all_pcc = [], [], []
        
        for x, y in self.val_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            # 生成还原信号
            fake_y = self.model(x)
            
            pred_np = fake_y.cpu().numpy()
            y_np = y.cpu().numpy()
            
            all_mae.append(calculate_mae(y_np, pred_np))
            all_rmse.append(calculate_rmse(y_np, pred_np))
            all_pcc.append(calculate_pcc(y_np, pred_np))
            
        return np.mean(all_mae), np.mean(all_rmse), np.mean(all_pcc)

    def visualize(self, epoch):
        """补全可视化逻辑"""
        self.model.eval()
        with torch.no_grad():
            # 1. 准备数据
            m_in, mask = apply_transient_mask(self.demo_sig, Config.MISSING_RATIO)
            # 2. 转为 Tensor: (512, 12) -> (1, 12, 512)
            x_tensor = torch.from_numpy(m_in.T).float().unsqueeze(0).to(Config.DEVICE)
            # 3. 生成还原信号
            recon = self.model(x_tensor).cpu().numpy()[0].T # -> (512, 12)
            
            # 4. 绘图
            save_comparison_plot(
                self.demo_sig, m_in, recon, mask, 
                epoch, self.model_name, "EKGAN_Recon"
            )