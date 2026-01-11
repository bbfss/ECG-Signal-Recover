# engine/unet_flow_trainer.py
import torch
import numpy as np
from tqdm import tqdm
from .base_trainer import BaseTrainer
from config import Config
from models.unet_flow import ConditionalFlowMatcher
from utils import calculate_mae, calculate_pcc, calculate_rmse, save_comparison_plot

class Unet_Flow_Trainer(BaseTrainer):
    def __init__(self, model_unet, flow_net, train_loader, val_loader, demo_sig):
        # 初始化父类
        super().__init__(model_unet, train_loader, val_loader, demo_sig)
        
        self.flow_net = flow_net.to(Config.DEVICE)
        
        # 针对特定导联增加权重
        self.l_weights = torch.ones(12).to(Config.DEVICE).float()
        self.l_weights[2] = 5.0; self.l_weights[5] = 5.0 
        
        # 初始化流匹配器
        self.cfm = ConditionalFlowMatcher(self.flow_net, Config.DEVICE, alpha=4.0, loss_weights=self.l_weights)
        
        # 定义两个独立的优化器
        self.opt_unet = torch.optim.Adam(self.model.parameters(), lr=Config.LR)
        self.opt_flow = torch.optim.Adam(self.flow_net.parameters(), lr=Config.LR)

    def train_epoch(self, epoch):
        # 判定训练阶段
        is_unet_phase = epoch <= Config.UNET_EPOCHS
        
        if is_unet_phase:
            self.model.train(); self.flow_net.eval()
            desc = f"Epoch {epoch} [UNet Stage]"
        else:
            self.model.eval(); self.flow_net.train()
            desc = f"Epoch {epoch} [Flow Stage]"

        total_loss = 0
        loop = tqdm(self.train_loader, desc=desc)
        
        for x, y in loop:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            
            if is_unet_phase:
                # 阶段一：仅训练 UNet MSE
                pred_u = self.model(x)
                loss = torch.mean(((pred_u - y)**2) * self.l_weights.view(1, 12, 1))
                self.opt_unet.zero_grad(); loss.backward(); self.opt_unet.step()
                loop.set_postfix(u_mse=f"{loss.item():.4f}")
            else:
                # 阶段二：冻结 UNet，基于其输出训练 Flow
                with torch.no_grad():
                    pred_u = self.model(x)
                loss = self.cfm.compute_loss(y, pred_u)
                self.opt_flow.zero_grad(); loss.backward(); self.opt_flow.step()
                loop.set_postfix(fm_loss=f"{loss.item():.4f}")
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        """覆盖父类方法：根据阶段切换评估逻辑"""
        self.model.eval(); self.flow_net.eval()
        is_unet_phase = epoch <= Config.UNET_EPOCHS
        
        maes, rmses, pccs = [], [], []
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                
                u_out = self.model(x)
                # 如果进入 Flow 阶段，则需要采样细化
                if is_unet_phase:
                    out = u_out
                else:
                    out = self.cfm.sample(u_out, steps=Config.SAMPLE_STEPS)
                
                maes.append(calculate_mae(y, out))
                rmses.append(calculate_rmse(y, out))
                pccs.append(calculate_pcc(y, out))
                
        return np.mean(maes), np.mean(rmses), np.mean(pccs)

    def visualize(self, epoch):
            """覆盖父类方法：针对 Unet+Flow 的采样推理绘图"""
            self.model.eval()
            self.flow_net.eval()
            is_unet_phase = epoch <= Config.UNET_EPOCHS
            
            from utils.signal_utils import apply_transient_mask
            
            # 1. 准备原始数据 [512, 12]
            clean_data = self.demo_sig_np  
            # 2. 生成带遮掩的数据
            m_trans, mask_t = apply_transient_mask(clean_data, Config.MISSING_RATIO)
            
            # 3. 模型推理
            with torch.no_grad():
                # self.demo_sig_tensor 应该是 [1, 12, 512]
                u_recon = self.model(self.demo_sig_tensor)
                
                if is_unet_phase:
                    # 结果维度: [12, 512]
                    recon = u_recon.cpu().numpy()[0]
                else:
                    # Flow 采样结果维度: [12, 512]
                    recon = self.cfm.sample(u_recon, steps=Config.SAMPLE_STEPS).cpu().numpy()[0]
            
            # 4. 调用绘图工具
            # 注意：recon 是 [12, 512]，而绘图工具需要 [512, 12]，所以需要 .T
            save_comparison_plot(
                clean_data=clean_data, 
                masked_input=m_trans,      # 修正参数名: masked -> masked_input
                recon_data=recon.T,        # 修正参数名: pred -> recon_data 并确保转置
                mask=mask_t, 
                epoch=epoch, 
                model_name=self.model.name,
                mode_name="Transient"      # 修正参数名: prefix -> mode_name
            )