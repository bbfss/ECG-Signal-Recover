# engine/unet_moe_flow_trainer.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .base_trainer import BaseTrainer
from config import Config
from models.unet_flow import ConditionalFlowMatcher 
from utils import calculate_mae, calculate_pcc, calculate_rmse, save_comparison_plot

class Unet_Moe_Flow_Trainer(BaseTrainer):
    def __init__(self, model_unet, moe_flow_net, train_loader, val_loader, demo_sig):
        # 初始化父类
        super().__init__(model_unet, train_loader, val_loader, demo_sig)
        
        # 统一使用 moe_flow_net 名称
        self.moe_flow_net = moe_flow_net.to(Config.DEVICE)
        
        # 导联加权逻辑 (论文创新点：Physiological Weighted Loss)
        self.l_weights = torch.ones(12).to(Config.DEVICE).float()
        self.l_weights[2] = 5.0  # 强化 Lead III
        self.l_weights[5] = 5.0  # 强化 aVF 
        
        # 初始化流匹配器
        self.cfm = ConditionalFlowMatcher(
            self.moe_flow_net, 
            Config.DEVICE, 
            alpha=4.0, 
            loss_weights=self.l_weights
        )
        
        # 定义两个独立的优化器
        self.opt_unet = torch.optim.Adam(self.model.parameters(), lr=Config.LR)
        self.opt_moe_flow = torch.optim.Adam(self.moe_flow_net.parameters(), lr=Config.LR * 0.8)

    def train_epoch(self, epoch):
        is_unet_phase = epoch <= Config.UNET_EPOCHS
        
        if is_unet_phase:
            self.model.train(); self.moe_flow_net.eval()
            desc = f"Epoch {epoch} [UNet Stage]"
        else:
            self.model.eval(); self.moe_flow_net.train()
            desc = f"Epoch {epoch} [MoE-Flow Stage]"

        total_loss = 0
        loop = tqdm(self.train_loader, desc=desc)
        
        for x, y in loop:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            
            if is_unet_phase:
                pred_u = self.model(x)
                # 阶段一：加权 MSE 训练
                loss = torch.mean(((pred_u - y)**2) * self.l_weights.view(1, 12, 1))
                self.opt_unet.zero_grad(); loss.backward(); self.opt_unet.step()
                loop.set_postfix(u_mse=f"{loss.item():.4f}")
            else:
                with torch.no_grad():
                    pred_u = self.model(x)
                # 阶段二：训练 MoE-Flow 细化器 (CFM Loss)
                loss = self.cfm.compute_loss(y, pred_u)
                self.opt_moe_flow.zero_grad(); loss.backward(); self.opt_moe_flow.step()
                loop.set_postfix(moe_loss=f"{loss.item():.4f}")
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval(); self.moe_flow_net.eval()
        is_unet_phase = epoch <= Config.UNET_EPOCHS
        
        maes, rmses, pccs = [], [], []
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                u_out = self.model(x)
                
                if is_unet_phase:
                    out = u_out
                else:
                    out = self.cfm.sample(u_out, steps=Config.SAMPLE_STEPS)
                
                maes.append(calculate_mae(y, out))
                rmses.append(calculate_rmse(y, out))
                pccs.append(calculate_pcc(y, out))
                
        return np.mean(maes), np.mean(rmses), np.mean(pccs)

    def visualize(self, epoch):
        self.model.eval(); self.moe_flow_net.eval()
        is_unet_phase = epoch <= Config.UNET_EPOCHS
        from utils.signal_utils import apply_transient_mask
        
        clean_data = self.demo_sig_np  
        m_trans, mask_t = apply_transient_mask(clean_data, Config.MISSING_RATIO)
        
        with torch.no_grad():
            u_recon = self.model(self.demo_sig_tensor) # [1, 12, 512]
            
            if is_unet_phase:
                recon = u_recon.cpu().numpy()[0]
            else:
                # --- 实现：追踪专家权重 ---
                xt = u_recon.clone()
                steps = Config.SAMPLE_STEPS
                dt = 1.0 / steps
                weights_history = [] 

                for i in range(steps):
                    t_val = i / steps
                    t_tensor = torch.full((1,), t_val, device=Config.DEVICE)
                    
                    # 1. 提取当前步骤的专家权重 (调用我们新增的接口)
                    current_w = self.moe_flow_net.get_expert_weights(t_tensor, xt, u_recon)
                    weights_history.append(current_w.cpu().numpy()[0])
                    
                    # 2. 执行一步 ODE 迭代
                    v_t = self.moe_flow_net(t_tensor, xt, u_recon)
                    xt = xt + v_t * dt
                
                recon = xt.cpu().numpy()[0]
                # 3. 绘制并保存专家演化图
                self._plot_expert_weights(weights_history, epoch)
        
        # 保存波形对比图
        save_comparison_plot(
            clean_data=clean_data, 
            masked_input=m_trans, 
            recon_data=recon.T, 
            mask=mask_t, 
            epoch=epoch, 
            model_name="Unet_MoE_Flow",
            mode_name="MoE_Refined" if not is_unet_phase else "UNet_Stage"
        )

    def _plot_expert_weights(self, weights_history, epoch):
        """私有辅助方法：生成专家权重演化图"""
        weights_history = np.array(weights_history) # [steps, 3]
        steps = weights_history.shape[0]
        t_axis = np.linspace(0, 1, steps)
        
        plt.figure(figsize=(10, 6))
        labels = ['High-Freq Expert', 'Low-Freq Expert', 'Global Expert']
        colors = ['#FF4136', '#0074D9', '#2ECC40']
        
        for i in range(3):
            plt.plot(t_axis, weights_history[:, i], label=labels[i], color=colors[i], lw=2.5)
            
        plt.xlabel("Flow Time (t)", fontsize=12)
        plt.ylabel("Gating Weight", fontsize=12)
        plt.title(f"Expert Selection Dynamics - Epoch {epoch}", fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 确保目录存在并保存
        import os
        os.makedirs("results/dynamics", exist_ok=True)
        plt.savefig(f"results/dynamics/moe_epoch_{epoch}.png", dpi=300)
        plt.close()