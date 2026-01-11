# engine/base_trainer.py
import torch
import os
from config import Config
from utils import calculate_mae, calculate_rmse, calculate_pcc, save_comparison_plot


class BaseTrainer:
    """通用训练器基类，包含模型保存等通用功能"""
    def __init__(self, model, train_loader, val_loader, demo_sig):
        self.model = model.to(Config.DEVICE)
        self.model_name = getattr(model, 'name', 'Unknown_Model')
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.demo_sig_np = demo_sig 
        
        # 2. 准备 Tensor 格式，专门给模型推理用 (形状: [1, 12, 512])
        self.demo_sig_tensor = torch.from_numpy(demo_sig.T).float().unsqueeze(0).to(Config.DEVICE)
        
    def validate(self):
        """通用的验证逻辑：计算 MAE, RMSE, PCC"""
        self.model.eval()
        maes, rmses, pccs = [], [], []
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                pred = self.model(x)
                
                mae = calculate_mae(y, pred)
                rmse = calculate_rmse(y, pred)
                pcc = calculate_pcc(y, pred)
                maes.append(mae); rmses.append(rmse); pccs.append(pcc)
        return sum(maes)/len(maes), sum(rmses)/len(rmses), sum(pccs)/len(pccs)
    
    
    def save_checkpoint(self, epoch):
            """通用的保存逻辑"""
            path = os.path.join(Config.CHECKPOINT_DIR, f"{self.model_name}_ep{epoch}.pth")
            torch.save(self.model.state_dict(), path)    
