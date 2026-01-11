# engine/descod_trainer.py
import torch
import numpy as np
from .base_trainer import BaseTrainer
from config import Config
from tqdm import tqdm
from utils import (
    calculate_mae, 
    calculate_pcc, 
    calculate_rmse, 
    save_comparison_plot, 
    apply_transient_mask
)

class DeScoD_Trainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, demo_sig):
        super().__init__(model, train_loader, val_loader, demo_sig)
        # 1. 必须保存 demo_sig，并设置模型名称
        self.demo_sig = demo_sig
        self.model_name = "DeScoD"
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LR)
        # 扩散参数
        self.sigma_min = Config.SIGMA_MIN if hasattr(Config, 'SIGMA_MIN') else 0.01
        self.sigma_max = Config.SIGMA_MAX if hasattr(Config, 'SIGMA_MAX') else 1.0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch} [DeScoD]")
        for x, y in loop:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            # 随机采样时间步 t
            t = torch.rand(y.shape[0], device=Config.DEVICE) * (1 - 1e-5) + 1e-5
            sigma_t = (self.sigma_min * (self.sigma_max / self.sigma_min) ** t).view(-1, 1, 1)
            
            # 加噪过程
            noise = torch.randn_like(y)
            x_noisy = y + noise * sigma_t
            
            # 预测分数 (Score matching)
            score = self.model(x_noisy, t, x)
            
            # 计算损失
            loss = torch.mean(torch.sum((score * sigma_t + noise)**2, dim=(1, 2)))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(score_loss=f"{loss.item():.4f}")
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch=None):
        """DeScoD 验证逻辑：需要通过扩散采样还原信号"""
        self.model.eval()
        all_mae, all_rmse, all_pcc = [], [], []
        
        # 扩散模型验证较慢，这里只取每个 batch 的部分样本或减小步数
        for x, y in self.val_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            
            # 使用 sample 方法还原信号
            # 为了验证速度，这里 steps 可以比训练时小
            pred = self.sample(x, steps=Config.DIFF_STEPS)
            
            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            
            all_mae.append(calculate_mae(y_np, pred_np))
            all_rmse.append(calculate_rmse(y_np, pred_np))
            all_pcc.append(calculate_pcc(y_np, pred_np))
            
            # 验证集通常很大，扩散模型采样慢，建议只跑前几个 batch 就 break
            # if len(all_mae) >= 2: break 

        return np.mean(all_mae), np.mean(all_rmse), np.mean(all_pcc)

    @torch.no_grad()
    def sample(self, cond, steps=50):
        """反向扩散采样过程 (Inference)"""
        self.model.eval()
        # 从纯噪声开始 [Batch, 12, 512]
        xt = torch.randn_like(cond).to(Config.DEVICE) * self.sigma_max
        time_steps = torch.linspace(1.0, 0.01, steps, device=Config.DEVICE)
        dt = 1.0 / steps
        
        for t in time_steps:
            t_vec = torch.full((cond.shape[0],), t, device=Config.DEVICE)
            sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
            
            # 预测当前步的分数
            score = self.model(xt, t_vec, cond)
            
            # 郎之万动力学更新步
            g = sigma_t * np.sqrt(2 * np.log(self.sigma_max / self.sigma_min))
            xt = xt + (g**2) * score * dt
            
            # 加入校正噪声
            if t > 0.01:
                xt = xt + g * np.sqrt(dt) * torch.randn_like(xt)
        return xt

    def visualize(self, epoch):
        """DeScoD 特有的可视化逻辑"""
        self.model.eval()
        with torch.no_grad():
            # 1. 准备遮掩数据
            m_in, mask = apply_transient_mask(self.demo_sig, Config.MISSING_RATIO)
            # 2. 转为 Tensor 并增加 batch 维度
            cond = torch.from_numpy(m_in.T).float().unsqueeze(0).to(Config.DEVICE)
            
            # 3. 扩散采样还原 (这里的还原过程才是模型真正的表现)
            recon_tensor = self.sample(cond, steps=Config.DIFF_STEPS)
            
            # 4. 转回 Numpy 绘图: (1, 12, 512) -> (512, 12)
            recon = recon_tensor.cpu().numpy()[0].T
            
            save_comparison_plot(self.demo_sig, m_in, recon, mask, epoch, self.model_name, "DeScoD_Gen")