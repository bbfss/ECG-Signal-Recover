# engine/ecg_recover_trainer.py
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

class ECG_Recover_Trainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, demo_sig):
        super().__init__(model, train_loader, val_loader, demo_sig)
        
        # --- 核心补丁：保存必要属性 ---
        self.demo_sig = demo_sig
        self.model_name = "ECGRecover"
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LR)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch} [ECG-Recover]")
        for x, y in loop:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            pred = self.model(x)
            loss = self.criterion(pred, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch=None):
        """补全验证逻辑"""
        self.model.eval()
        all_mae, all_rmse, all_pcc = [], [], []
        
        for x, y in self.val_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            pred = self.model(x)
            
            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            
            all_mae.append(calculate_mae(y_np, pred_np))
            all_rmse.append(calculate_rmse(y_np, pred_np))
            all_pcc.append(calculate_pcc(y_np, pred_np))
            
        return np.mean(all_mae), np.mean(all_rmse), np.mean(all_pcc)

    def visualize(self, epoch):
        """补全可视化逻辑"""
        self.model.eval()
        with torch.no_grad():
            # 1. 准备遮掩数据 (使用 numpy)
            m_in, mask = apply_transient_mask(self.demo_sig, Config.MISSING_RATIO)
            
            # 2. 转为 Tensor: (512, 12) -> (1, 12, 512)
            x_tensor = torch.from_numpy(m_in.T).float().unsqueeze(0).to(Config.DEVICE)
            
            # 3. 前向传播得到还原信号: (1, 12, 512) -> (512, 12)
            recon = self.model(x_tensor).cpu().numpy()[0].T
            
            # 4. 调用绘图函数
            save_comparison_plot(
                self.demo_sig, m_in, recon, mask, 
                epoch, self.model_name, "Recover_Recon"
            )