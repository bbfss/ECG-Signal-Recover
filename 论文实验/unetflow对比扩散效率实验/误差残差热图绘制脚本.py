import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- 1. 环境与路径配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from config import Config
from utils.ptbxl_loader import get_ptbxl_evaluate_loader
from models import AdvancedUNet1D, FlowNetwork, DeScoD_ScoreNet
from models.unet_normal_flow import NormalFlowNetwork

def get_eval_models(device):
    checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    data_name = "PTBXL" 
    def load(m_dict, prefix):
        path = os.path.join(checkpoint_dir, f"{prefix}_{data_name}_best.pth")
        if not os.path.exists(path): path = path.replace("_best.pth", "_ep50.pth")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device)
            for k, obj in m_dict.items(): obj.load_state_dict(ckpt[k]); obj.eval()
            res = list(m_dict.values()); return res[0] if len(res)==1 else tuple(res)
        return None
    return load({"model": DeScoD_ScoreNet().to(device)}, "DeScoD"), \
           load({"unet": AdvancedUNet1D().to(device), "flow": FlowNetwork().to(device)}, "Unet_Flow"), \
           load({"unet": AdvancedUNet1D().to(device), "normal_flow": NormalFlowNetwork().to(device)}, "ECG_Unet_Normal_Flow")

# =============================================================
# 2. 绘图逻辑：误差分布统计图
# =============================================================
def plot_error_violin_comparison(lead_idx=1):
    device = Config.DEVICE
    m_descod, m_ours, m_normal = get_eval_models(device)
    val_loader, _ = get_ptbxl_evaluate_loader()
    
    # 随机取一个样本或多样本汇总
    batch_x, batch_y = next(iter(val_loader))
    input_np = batch_x[0:1].permute(0, 2, 1).numpy()
    gt_np = batch_y[0, lead_idx, :].cpu().numpy()

    nfe_list = [1, 4, 7, 10, 13, 16]
    models = [("DeScoD", m_descod), ("Normal-Flow", m_normal), ("Ours (TranspeedFlow)", m_ours)]
    
    # 构造绘图数据
    all_data = []
    
    print("🧪 正在计算误差分布数据...")
    for name, model_obj in models:
        if model_obj is None: continue
        for nfe in nfe_list:
            with torch.no_grad():
                if "DeScoD" in name:
                    pred = model_obj.predict(input_np, steps=nfe, device=device)
                else:
                    u, f = model_obj
                    pred = f.predict(u.predict(input_np, device=device), steps=nfe, device=device)
            
            # 计算 512 个点的绝对误差
            abs_errors = np.abs(gt_np - pred[0, :, lead_idx])
            
            for err in abs_errors:
                all_results = {
                    "Model": name,
                    "NFE": f"NFE={nfe}",
                    "Absolute Error": err
                }
                all_data.append(all_results)

    df = pd.DataFrame(all_data)

    # --- 开始绘图 ---
    plt.figure(figsize=(14, 8), dpi=200)
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # 使用参考代码的配色
    palette = {"DeScoD": "#ff7f0e", "Normal-Flow": "#2ca02c", "Ours (TranspeedFlow)": "#3498db"}
    
    # 绘制小提琴图
    # split=True 可以左右对比，这里我们直接按 NFE 并列
    sns.violinplot(data=df, x="NFE", y="Absolute Error", hue="Model", 
                   palette=palette, split=False, inner="quart", linewidth=1.2)

    # --- 学术细节美化 ---
    plt.title("Statistical Distribution of Reconstruction Residuals", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Inference Iterations (NFE)", fontsize=12, fontweight='bold')
    plt.ylabel("Absolute Error Magnitude $|x - \hat{x}|$", fontsize=12, fontweight='bold')
    
    plt.grid(True, axis='y', linestyle=':', alpha=0.5)
    plt.ylim(-0.02, 0.4) # 聚焦在误差集中的区域
    
    # 去除多余边框
    sns.despine(offset=10, trim=True)
    
    plt.legend(title="Architectures", title_fontsize='11', loc='upper right', frameon=True, shadow=True)
    
    save_path = os.path.join(SCRIPT_DIR, "error_distribution_violin.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"✅ 成功生成误差分布小提琴图：{save_path}")
    plt.show()

if __name__ == "__main__":
    plot_error_violin_comparison()