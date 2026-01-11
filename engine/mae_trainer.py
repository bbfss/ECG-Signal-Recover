# engine/mae_trainer.py
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .base_trainer import BaseTrainer
from config import Config
from utils import (
    calculate_mae, 
    calculate_pcc, 
    save_comparison_plot, 
    apply_transient_mask
)

class MAEFETrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, demo_sig):
        # 调用父类初始化
        super().__init__(model, train_loader, val_loader, demo_sig)
        
        # 1. 明确模型名称，供可视化使用
        self.model_name = "MaeFE"
        self.demo_sig = demo_sig
        
        # 2. 特有的优化器和损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LR)
        self.criterion = nn.MSELoss()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc=f"MAEFE Train Ep {epoch}")
        
        for x, y in loop:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            
            pred = self.model(x)
            loss = self.criterion(pred, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch=None):
        """补全验证逻辑：返回 MAE, RMSE, PCC"""
        self.model.eval()
        all_mae, all_rmse, all_pcc = [], [], []
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                pred = self.model(x)
                
                # 转换回 numpy 计算指标
                pred_np = pred.cpu().numpy()
                y_np = y.cpu().numpy()
                
                # 计算指标
                mae = calculate_mae(y_np, pred_np)
                # RMSE 简单实现：MSE 的平方根
                rmse = np.sqrt(np.mean((y_np - pred_np)**2))
                pcc = calculate_pcc(y_np, pred_np)
                
                all_mae.append(mae)
                all_rmse.append(rmse)
                all_pcc.append(pcc)
        
        # 返回平均指标
        return np.mean(all_mae), np.mean(all_rmse), np.mean(all_pcc)

    def visualize(self, epoch):
        """MaeFE 特有的可视化逻辑"""
        self.model.eval()
        with torch.no_grad():
            # 获取一个遮掩样本
            # 注意：如果 self.demo_sig 是 (512, 12)，需要确保 apply_transient_mask 逻辑一致
            m_in, mask = apply_transient_mask(self.demo_sig, Config.MISSING_RATIO)
            
            # 转置并增加 batch 维度：(12, 512) -> (1, 12, 512)
            x_tensor = torch.from_numpy(m_in.T).float().unsqueeze(0).to(Config.DEVICE)
            
            # 模型预测并转回 numpy：(1, 12, 512) -> (12, 512) -> (512, 12)
            recon = self.model(x_tensor).cpu().numpy()[0].T
            
            # 调用 visualizer.py 绘图
            save_comparison_plot(self.demo_sig, m_in, recon, mask, epoch, self.model_name, "MAE_Recon")