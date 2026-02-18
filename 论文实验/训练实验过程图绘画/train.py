import sys
import os
import torch
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt

# --- 1. 环境与路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, "..", ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from config import Config
import utils
from models import (
    AdvancedUNet1D, FlowNetwork, MoEFlowNetwork,
    MaeFE, EKGAN_Generator, EKGAN_Discriminator,
    DeScoD_ScoreNet, ECGRecover
)
from engine import (
    Unet_Flow_Trainer, MAEFETrainer,
    GAN_Trainer, DeScoD_Trainer, ECG_Recover_Trainer, Unet_Moe_Flow_Trainer
)

# ==========================================================
# 可视化组件：支持双阶段 Loss 与 3x2 网格演化图
# ==========================================================
class ECGVisualizer:
    def __init__(self, save_dir="experiment_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.unet_losses = []  # WSSCA Loss
        self.flow_losses = []  # TranspeedFlow Loss

    def add_unet_loss(self, loss):
        self.unet_losses.append(loss)

    def add_flow_loss(self, loss):
        self.flow_losses.append(loss)

    def plot_loss_curve(self, std_model_name):
        """绘制联合 Loss 曲线"""
        plt.figure(figsize=(10, 6))
        if self.unet_losses:
            epochs_unet = range(1, len(self.unet_losses) + 1)
            plt.plot(epochs_unet, self.unet_losses, color='tab:blue', linewidth=2, label='WSSCA Loss')
        if self.flow_losses:
            start_epoch = len(self.unet_losses) + 1
            epochs_flow = range(start_epoch, start_epoch + len(self.flow_losses))
            plt.plot(epochs_flow, self.flow_losses, color='tab:red', linewidth=2, label='TranspeedFlow Loss')
            if self.unet_losses:
                plt.axvline(x=len(self.unet_losses), color='gray', linestyle='--', alpha=0.5)

        plt.xlabel('Total Epochs')
        plt.ylabel('Loss Value')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, f"{std_model_name}_training_losses.png"))
        plt.close()

    @torch.no_grad()
    def plot_flow_evolution(self, flow_model, x_recon, x_gt, epoch, std_model_name):
        """
        核心演化图：3x2 网格显示，自动选择变化最大的导联，同时显示GT和Pred
        """
        flow_model.eval()
        device = next(flow_model.parameters()).device

        # 准备数据: (1, 12, 512)
        xt = x_recon[:1].to(device)
        cond = xt.clone()
        x_gt_np = x_gt[:1].cpu().numpy()

        # --- 1. 执行完整推理并记录历史 ---
        history = []
        # 固定展示 6 个阶段用于 3x2 网格 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        steps_to_show = 6
        time_points_to_show = np.linspace(0, 1, steps_to_show)

        inference_steps = 30
        dt = 1.0 / inference_steps
        curr_xt = xt.clone()

        # 记录需要保存的时间点索引
        plot_indices = [int(inference_steps * tp) for tp in time_points_to_show]
        # 确保最后一步(t=1.0)也被包含
        if inference_steps not in plot_indices: plot_indices[-1] = inference_steps

        for i in range(inference_steps + 1):
            t_val = i / inference_steps
            if i in plot_indices:
                history.append((t_val, curr_xt.cpu().numpy()))

            if i < inference_steps:
                t_tensor = torch.full((1,), t_val, device=device)
                v = flow_model(t_tensor, curr_xt, cond)
                curr_xt = curr_xt + v * dt

        # --- 2. 自动寻找变化最明显的导联 ---
        # 计算 t=0 (初始) 和 t=1 (最终) 之间的绝对差值和
        initial_wave = history[0][1]  # (1, 12, 512)
        final_wave = history[-1][1]   # (1, 12, 512)
        diff_sum = np.sum(np.abs(initial_wave - final_wave), axis=(0, 2)) # (12,)
        best_lead_idx = np.argmax(diff_sum)
        print(f"   >> Visualizer: Auto-selected Lead {best_lead_idx} (max change value: {diff_sum[best_lead_idx]:.4f})")

        # --- 3. 绘制 3x2 网格 ---
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        gt_wave_lead = x_gt_np[0, best_lead_idx]

        # 使用 viridis 颜色图来表示时间进度
        cmap = plt.cm.get_cmap('viridis')

        for i, ax in enumerate(axes):
            t_val, pred_wave_batch = history[i]
            pred_wave_lead = pred_wave_batch[0, best_lead_idx]

            # A. 画 Ground Truth (灰色虚线背景)
            # 仅在第一个图显示图例标签，避免重复
            lbl_gt = 'Ground Truth' if i == 0 else None
            ax.plot(gt_wave_lead, color='grey', linestyle='--', linewidth=1.5, alpha=0.6, label=lbl_gt)

            # B. 画当前预测 (随时间变化的渐变色实线)
            color = cmap(t_val)
            lbl_pred = f'Prediction'

            ax.plot(pred_wave_lead, color=color, linewidth=2, alpha=0.9, label=lbl_pred)

            # C. 设置图表样式
            ax.grid(True, linestyle=':', alpha=0.4)

            # 将 "t=X.X" 小标题写在每个图的下面 (利用 xlabel 并隐藏刻度)
            ax.set_xlabel(f"t = {t_val:.1f}", fontsize=11, fontweight='bold', labelpad=8)
            ax.set_xticks([]) # 隐藏 X 轴刻度值，只保留 label

            # 在第一张图显示图例
            if i == 0:
                ax.legend(loc='upper right', frameon=True, fontsize=9)

        plt.tight_layout()
        # 调整整体标题位置 (可选)
        # fig.suptitle(f"Flow Refinement Evolution (Epoch {epoch}) - Lead {best_lead_idx}", y=1.02, fontsize=14)
        plt.savefig(os.path.join(self.save_dir, f"{std_model_name}_evolution_3x2_ep{epoch}.png"), bbox_inches='tight')
        plt.close()

# ==========================================================
# 映射字典与辅助函数 (保持不变)
# ==========================================================
MODEL_MAP = {
    "unetflow": "Unet_Flow", "maefe": "MaeFE", "ekgan": "EKGAN",
    "descod": "DeScoD", "ecgrecover": "ECGRecover", "unet": "Unet_Baseline",
    "unetmoeflow": "Unet_Moe_Flow",
}
DATASET_MAP = {"ptbxl": "PTBXL", "mitbih": "MITBIH"}

def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed); torch.backends.cudnn.deterministic = True

def get_clean_key(s): return s.lower().replace("-", "").replace("_", "").replace(" ", "")

# ==========================================================
# 工厂逻辑 (保持不变)
# ==========================================================
def get_dataset(data_key):
    std = DATASET_MAP.get(data_key)
    if std == "PTBXL": return utils.get_ptbxl_loaders(), std
    else: raise NotImplementedError
    
def get_trainer_logic(model_key, train_dl, val_dl, demo_sig):
    std = MODEL_MAP.get(model_key); dev = Config.DEVICE; m_d = {}
    if std == "Unet_Moe_Flow":
        m_u = AdvancedUNet1D().to(dev); m_f = MoEFlowNetwork().to(dev)
        t = Unet_Moe_Flow_Trainer(m_u, m_f, train_dl, val_dl, demo_sig)
        m_d = {'unet': m_u, 'moe_flow': m_f}; ep = Config.UNET_EPOCHS + Config.FLOW_EPOCHS
    elif std == "Unet_Flow":
        m_u = AdvancedUNet1D().to(dev); m_f = FlowNetwork().to(dev)
        t = Unet_Flow_Trainer(m_u, m_f, train_dl, val_dl, demo_sig)
        m_d = {'unet': m_u, 'flow': m_f}; ep = Config.UNET_EPOCHS + Config.FLOW_EPOCHS
    elif std == "MaeFE":
        m = MaeFE().to(dev); t = MAEFETrainer(m, train_dl, val_dl, demo_sig)
        m_d = {'model': m}; ep = Config.MAE_EPOCHS
    else: raise ValueError
    return t, m_d, ep, std

# ==========================================================
# 主程序
# ==========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unetflow')
    parser.add_argument('--dataset', type=str, default='ptbxl')
    args = parser.parse_args()
    set_seed(Config.SEED)

    (train_dl, val_dl, demo_sig), std_data_name = get_dataset(get_clean_key(args.dataset))
    trainer, models_dict, total_epochs, std_model_name = get_trainer_logic(
        get_clean_key(args.model), train_dl, val_dl, demo_sig
    )
    viz = ECGVisualizer(save_dir=f"results_{std_model_name}")

    print(f"\n>> 启动: {std_model_name} | Pretrain: {Config.UNET_EPOCHS} | Flow: {Config.FLOW_EPOCHS}")

    best_pcc = -1.0
    for epoch in range(1, total_epochs + 1):
        loss = trainer.train_epoch(epoch)
        mae, rmse, pcc = trainer.validate(epoch)

        if epoch <= Config.UNET_EPOCHS: viz.add_unet_loss(loss)
        else: viz.add_flow_loss(loss)
        
        print(f"Ep {epoch}/{total_epochs} | Loss: {loss:.4f} | PCC: {pcc:.4f}")

        if epoch % 10 == 0 or epoch == total_epochs:
            viz.plot_loss_curve(std_model_name)
            
            if "flow" in args.model.lower():
                x_in, x_gt = next(iter(val_dl))
                f_m = models_dict.get('flow') or models_dict.get('moe_flow')
                u_m = models_dict.get('unet')
                if f_m and u_m:
                    u_m.eval()
                    with torch.no_grad(): x_recon = u_m(x_in.to(Config.DEVICE))
                    # 调用新的 3x2 绘图函数
                    viz.plot_flow_evolution(f_m, x_recon, x_gt, epoch, std_model_name)

            trainer.visualize(epoch)
            torch.save({k: v.state_dict() for k, v in models_dict.items()}, 
                       os.path.join(Config.CHECKPOINT_DIR, f"{std_model_name}_ep{epoch}.pth"))

        if pcc > best_pcc:
            best_pcc = pcc
            torch.save({k: v.state_dict() for k, v in models_dict.items()},
                       os.path.join(Config.CHECKPOINT_DIR, f"{std_model_name}_best.pth"))
            print(f" >> [Best] PCC: {pcc:.4f}")

if __name__ == "__main__":
    main()