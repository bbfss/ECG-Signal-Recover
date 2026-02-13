# engine/wavemamba_flow_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .base_trainer import BaseTrainer
from config import Config
from utils import calculate_mae, calculate_pcc, calculate_rmse, save_comparison_plot

class WaveMamba_Flow_Trainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, demo_sig):
        """
        model: WaveMambaFlowNet 整合模型
        """
        # 1. 初始化父类 (BaseTrainer 负责处理 self.model 和基础 log)
        super().__init__(model, train_loader, val_loader, demo_sig)
        
        # 2. 导联加权设置 (针对弱势导联 III 和 aVF)
        self.l_weights = torch.ones(12).to(Config.DEVICE).float()
        self.l_weights[2] = 5.0; self.l_weights[5] = 5.0 
        
        # 3. 优化器定义 (分阶段控制)
        # 阶段一优化器：涵盖 UNet 所有组件 (Encoder, Bridge, Decoder, Gate)
        unet_params = [p for n, p in self.model.named_parameters() if "flow_refiner" not in n]
        self.opt_unet = torch.optim.Adam(unet_params, lr=Config.LR)
        
        # 阶段二优化器：仅针对 TransSpeedFlow 精修网络
        self.opt_flow = torch.optim.Adam(self.model.flow_refiner.parameters(), lr=Config.LR)

    def info_nce_loss(self, zt, zc, temperature=0.07):
        """
        3.3.4 节实现：注意力层级的对比对齐损失 (Semantic Handshake)
        """
        zt = F.normalize(zt, dim=1)
        zc = F.normalize(zc, dim=1)
        # 计算批次内时空特征相似度矩阵
        logits = torch.matmul(zt, zc.T) / temperature
        labels = torch.arange(zt.size(0)).to(zt.device)
        return F.cross_entropy(logits, labels)

    def train_epoch(self, epoch):
        # 判定训练阶段
        is_unet_phase = epoch <= Config.UNET_EPOCHS
        
        if is_unet_phase:
            self.model.train()
            desc = f"Epoch {epoch} [Stage 1: Contrastive UNet]"
        else:
            # 阶段二通常冻结 UNet 部分，专注于精修器的病理细节还原
            self.model.eval() 
            self.model.flow_refiner.train()
            desc = f"Epoch {epoch} [Stage 2: TransSpeedFlow]"

        total_loss = 0
        loop = tqdm(self.train_loader, desc=desc)
        
        for x, y in loop:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            
            if is_unet_phase:
                # --- 阶段一：训练 Wavelet-Mamba UNet + C-STAM ---
                # 获取粗重构结果及所有注意力门的对比对 (zt, zc)
                pred_u, pairs = self.model.forward_unet(x)
                
                # a) 加权 MSE 损失
                loss_mse = torch.mean(((pred_u - y)**2) * self.l_weights.view(1, 12, 1))
                
                # b) 注意力层级对比损失 (累加所有 Gate 的对齐误差)
                loss_nce = sum([self.info_nce_loss(zt, zc) for zt, zc in pairs])
                
                # 总损失 (lambda=0.1)
                loss = loss_mse + 0.1 * loss_nce
                
                self.opt_unet.zero_grad(); loss.backward(); self.opt_unet.step()
                loop.set_postfix(u_mse=f"{loss_mse.item():.4f}", nce=f"{loss_nce.item():.4f}")
            
            else:
                # --- 阶段二：训练 Flow 精修网络 (TransSpeedFlow) ---
                with torch.no_grad():
                    pred_u, _ = self.model.forward_unet(x)
                
                # 采样演化时间步 t [0, 1]
                t = torch.rand(x.size(0), device=x.device)
                # 构造条件流路径 (基于流匹配理论)
                xt = (1 - t.view(-1, 1, 1)) * pred_u + t.view(-1, 1, 1) * y
                target_v = y - pred_u # 目标速度场
                
                # 预测瞬时速度
                pred_v = self.model.flow_refiner(t, xt, pred_u)
                
                # a) 速度 MSE 损失
                loss_fm = torch.mean(((pred_v - target_v)**2) * self.l_weights.view(1, 12, 1))
                
                # b) 形态梯度损失 (4.3.3) - 解决均值回归，还原 R 峰斜率
                loss_grad = F.l1_loss(pred_v[:,:,1:] - pred_v[:,:,:-1], 
                                     target_v[:,:,1:] - target_v[:,:,:-1])
                
                # c) 均值偏置损失 (防止基线漂移)
                loss_bias = torch.mean(torch.abs(pred_v.mean(dim=-1) - target_v.mean(dim=-1)))
                
                loss = loss_fm + 4.0 * loss_grad + 0.5 * loss_bias
                
                self.opt_flow.zero_grad(); loss.backward(); self.opt_flow.step()
                loop.set_postfix(fm_mse=f"{loss_fm.item():.4f}", grad=f"{loss_grad.item():.4f}")
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        is_unet_phase = epoch <= Config.UNET_EPOCHS
        
        maes, rmses, pccs = [], [], []
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                
                if is_unet_phase:
                    out, _ = self.model.forward_unet(x)
                else:
                    # 阶段二：执行 30 步 ODE 采样精修 (调焦过程)
                    # 注意：predict 内部处理了 (B, 512, 12) 格式转换
                    x_np = x.permute(0, 2, 1).cpu().numpy()
                    out_np = self.model.predict(x_np, steps=Config.SAMPLE_STEPS)
                    out = torch.from_numpy(out_np).permute(0, 2, 1).to(Config.DEVICE)
                
                maes.append(calculate_mae(y, out))
                rmses.append(calculate_rmse(y, out))
                pccs.append(calculate_pcc(y, out))
                
        return np.mean(maes), np.mean(rmses), np.mean(pccs)

    def visualize(self, epoch):
        self.model.eval()
        is_unet_phase = epoch <= Config.UNET_EPOCHS
        from utils.signal_utils import apply_transient_mask
        
        clean_data = self.demo_sig_np # [512, 12]
        
        with torch.no_grad():
            if is_unet_phase:
                u_recon, _ = self.model.forward_unet(self.demo_sig_tensor)
                recon = u_recon.cpu().numpy()[0]
                label = "UNet_Stage"
            else:
                # 使用级联采样预测 [512, 12]
                recon_np = self.model.predict(self.demo_sig_tensor.permute(0, 2, 1).cpu().numpy(), 
                                            steps=Config.SAMPLE_STEPS)[0]
                recon = recon_np.T # 转为 [12, 512]
                label = "Flow_Stage"
            
        m_trans, mask_t = apply_transient_mask(clean_data, Config.MISSING_RATIO)
        
        save_comparison_plot(
            clean_data=clean_data, 
            masked_input=m_trans, 
            recon_data=recon.T, 
            mask=mask_t, 
            epoch=epoch, 
            model_name=f"{self.model.__class__.__name__}_{label}",
            mode_name="Transient"
        )