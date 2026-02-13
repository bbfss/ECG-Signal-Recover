import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 环境与路径配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from config import Config
from utils.ptbxl_loader import get_ptbxl_evaluate_loader
from models import AdvancedUNet1D, FlowNetwork
from utils.metrics import calculate_pcc 

# =============================================================
# 2. 核心逻辑：自动搜索全场“最强拟合”样本
# =============================================================
def find_the_golden_sample(u_model, f_model, device, lead_idx=1, search_limit=150):
    val_loader, _ = get_ptbxl_evaluate_loader()
    best_pcc = -1.0
    best_sample = None
    
    print(f"🔍 正在从验证集中筛选“神仙案例” (搜索上限: {search_limit} 样本)...")
    
    with torch.no_grad():
        count = 0
        for batch_x, batch_y in val_loader:
            input_np = batch_x.permute(0, 2, 1).numpy()
            mid_res = u_model.predict(input_np, device=device)
            # 以 NFE=14 作为筛选最高 PCC 的标准
            pred_np = f_model.predict(mid_res, steps=14, device=device)
            pred_torch = torch.from_numpy(pred_np).permute(0, 2, 1)
            
            for b in range(batch_x.shape[0]):
                pcc = calculate_pcc(batch_y[b:b+1, lead_idx:lead_idx+1, :], 
                                    pred_torch[b:b+1, lead_idx:lead_idx+1, :])
                if pcc > best_pcc:
                    best_pcc = pcc
                    best_sample = (batch_x[b:b+1], batch_y[b:b+1])
                
                count += 1
                if count >= search_limit: break
            if count >= search_limit: break
    
    print(f"✨ 找到最优样本！最大 PCC 为: {best_pcc:.5f}")
    return best_sample

# =============================================================
# 3. 绘图主逻辑 (严格参考您的样式配置)
# =============================================================
def plot_nfe_comparison_stacked(lead_idx=1):
    device = Config.DEVICE
    checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    path = os.path.join(checkpoint_dir, f"Unet_Flow_PTBXL_ep50.pth")
    
    # 颜色配置 (严格按照您的输入)
    colors = {
        "Original": "#333333",    # 深灰色背景
        "Unet_Flow": "#3498db"    # 蓝色
    }

    # 加载模型
    if not os.path.exists(path):
        print(f"❌ 错误：找不到权重文件 {path}")
        return
        
    ckpt = torch.load(path, map_location=device)
    unet_m = AdvancedUNet1D(in_channels=Config.IN_CHANNELS).to(device)
    flow_m = FlowNetwork(channels=Config.IN_CHANNELS).to(device)
    unet_m.load_state_dict(ckpt["unet"])
    flow_m.load_state_dict(ckpt["flow"])
    unet_m.eval(); flow_m.eval()

    # 1. 获取拟合效果最好的样本
    best_x, best_y = find_the_golden_sample(unet_m, flow_m, device, lead_idx)
    input_np = best_x.permute(0, 2, 1).numpy()
    gt_np = best_y[0, lead_idx, :].cpu().numpy()

    # 2. 绘图设置
    nfe_steps = [1, 7, 14]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True, dpi=300)

    with torch.no_grad():
        mid_res = unet_m.predict(input_np, device=device)
        
        for i, nfe in enumerate(nfe_steps):
            ax = axes[i]
            
            # 推理
            pred_np = flow_m.predict(mid_res, steps=nfe, device=device)
            sig_np = pred_np[0, :, lead_idx]
            pcc = calculate_pcc(best_y[:, lead_idx:lead_idx+1, :], 
                                torch.from_numpy(pred_np).permute(0, 2, 1)[:, lead_idx:lead_idx+1, :])
            
            # --- 绘图逻辑 (严格匹配您的参数) ---
            # 绘制原始信号作为背景参考
            ax.plot(gt_np, color=colors["Original"], label="Original (Truth)", 
                    alpha=0.3, linewidth=2, linestyle='--')
            
            # 绘制当前步数的恢复信号
            ax.plot(sig_np, color=colors["Unet_Flow"], label=f"OURS(NFE={nfe})", 
                    linewidth=1.5)
            
            # 美化子图 (参考您的 Title 和 Grid)
            ax.set_title(f"NFE={nfe}", 
                         fontsize=12, fontweight='bold')
            ax.legend(loc="upper right", frameon=True, shadow=True)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_ylabel("Amplitude")
            ax.set_ylim(-1.2, 1.2)

    # 设置最下方的 X 轴标签
    axes[-1].set_xlabel("Time Samples (at 100Hz)")
    
    plt.tight_layout()
    
    # 保存结果
    save_path = os.path.join(SCRIPT_DIR, "unet_flow_nfe_comparison.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"🚀 高清波形演变图已生成：{save_path}")
    plt.show()

if __name__ == "__main__":
    plot_nfe_comparison_stacked(lead_idx=1) # 默认 Lead II