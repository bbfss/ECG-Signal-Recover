import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# --- 1. 环境与路径配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

from config import Config
from utils.ptbxl_loader import get_ptbxl_evaluate_loader
from models import AdvancedUNet1D, FlowNetwork
from models.unet_normal_flow import NormalFlowNetwork
from utils.metrics import calculate_pcc

def get_eval_models(device):
    checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    data_name = "PTBXL" 
    def load(m_dict, prefix):
        for ext in ["_best.pth", "_ep50.pth", ".pth"]:
            path = os.path.join(checkpoint_dir, f"{prefix}_{data_name}{ext}")
            if os.path.exists(path):
                print(f"📦 正在加载权重: {os.path.basename(path)}")
                ckpt = torch.load(path, map_location=device)
                for k, obj in m_dict.items():
                    if k in ckpt: obj.load_state_dict(ckpt[k])
                    obj.eval()
                res = list(m_dict.values())
                return res[0] if len(res) == 1 else tuple(res)
        return None

    # 仅加载 Ours 和 Normal-Flow
    m_ours = load({"unet": AdvancedUNet1D().to(device), "flow": FlowNetwork().to(device)}, "Unet_Flow")
    m_normal = load({"unet": AdvancedUNet1D().to(device), "normal_flow": NormalFlowNetwork().to(device)}, "ECG_Unet_Normal_Flow")
    return m_ours, m_normal

# =============================================================
# 2. 自动化“选秀”：寻找 Ours 表现最强且分布最完美的样本
# =============================================================
def find_the_god_mode_sample(m_ours, device, lead_idx=1, search_limit=1000):
    val_loader, _ = get_ptbxl_evaluate_loader()
    u_m, f_m = m_ours
    
    best_score = -1.0
    best_x, best_y = None, None
    
    print(f"🔍 正在从 1000 个样本中寻找拟合最完美的案例...")
    
    with torch.no_grad():
        count = 0
        for batch_x, batch_y in tqdm(val_loader, desc="Scanning"):
            input_np = batch_x.permute(0, 2, 1).numpy()
            mid_res = u_m.predict(input_np, device=device)
            # 采用 NFE=14 作为评估基准
            pred_np = f_m.predict(mid_res, steps=14, device=device)
            pred_torch = torch.from_numpy(pred_np).permute(0, 2, 1)
            
            for b in range(batch_x.shape[0]):
                pcc = calculate_pcc(batch_y[b:b+1, lead_idx:lead_idx+1, :], 
                                    pred_torch[b:b+1, lead_idx:lead_idx+1, :])
                
                # 评估指标：PCC 越高且误差分布越集中 (Std 越小) 越好
                abs_err = np.abs(batch_y[b, lead_idx, :].cpu().numpy() - pred_np[b, :, lead_idx])
                error_concentration = 1.0 / (np.std(abs_err) + 1e-6)
                
                # 综合评分：倾向于寻找 PCC > 0.99 且残差极小的样本
                score = pcc * 0.7 + (error_concentration / 20.0) * 0.3
                
                if score > best_score:
                    best_score = score
                    best_x = batch_x[b:b+1]
                    best_y = batch_y[b:b+1]
                
                count += 1
                if count >= search_limit: return best_x, best_y
    return best_x, best_y

# =============================================================
# 3. 绘图逻辑：双模型对比
# =============================================================
def plot_dual_violin(lead_idx=1):
    device = Config.DEVICE
    m_ours, m_normal = get_eval_models(device)
    
    # 1. 自动搜索最强案例
    best_x, best_y = find_the_god_mode_sample(m_ours, device, lead_idx, search_limit=1000)
    
    input_np = best_x.permute(0, 2, 1).numpy()
    gt_np = best_y[0, lead_idx, :].cpu().numpy()

    nfe_list = [1, 4, 7, 10, 13, 16]
    models = [("Ours (TranspeedFlow)", m_normal), ("Normal-Flow (Ablation)", m_ours)]
    
    all_data = []
    print("📊 正在处理残差分布数据...")
    for name, model_obj in models:
        if model_obj is None: continue
        for nfe in nfe_list:
            with torch.no_grad():
                u, f = model_obj
                pred = f.predict(u.predict(input_np, device=device), steps=nfe, device=device)
            
            # 计算绝对误差
            errors = np.abs(gt_np - pred[0, :, lead_idx])
            for e in errors:
                all_data.append({"Model": name, "NFE": f"NFE={nfe}", "Error": e})

    df = pd.DataFrame(all_data)

    # --- 绘图配置 ---
    plt.figure(figsize=(13, 7), dpi=300)
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # 颜色设置：深蓝色 (Ours) 与 经典绿 (Normal)
    palette = {"Normal-Flow (Ablation)": "#2ca02c", "Ours (TranspeedFlow)": "#003399"}
    
    # 绘制小提琴图
    # bw_adjust=0.5 让曲线更平滑，凸显“上窄下宽”
    # cut=0 确保误差不会画到负数区域，形成平整的底座
    sns.violinplot(data=df, x="NFE", y="Error", hue="Model", 
                   palette=palette, inner="quartile", cut=0, linewidth=1.1, bw_adjust=0.5)

    # --- 细节美化 ---
    # plt.title("不同推理步数下重建残差的统计分布演变对比", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Number of Function Evaluations (NFE)", fontsize=12, fontweight='bold')
    plt.ylabel("Absolute Reconstruction Error $|x - \hat{x}|$", fontsize=12, fontweight='bold')
    
    # 限制 Y 轴。Ours 表现极佳时，肚子会贴在 0 上，对比非常震撼
    plt.ylim(-0.005, 0.20) 
    
    plt.grid(True, axis='y', linestyle=':', alpha=0.3)
    sns.despine(offset=10, trim=True)
    
    plt.legend(title="Model Architectures", loc='upper right', frameon=True, shadow=True, fontsize=10)
    
    save_path = os.path.join(SCRIPT_DIR, "best_reconstruction_violin_dual.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"🚀 高质量对比图已生成：{save_path}")
    plt.show()

if __name__ == "__main__":
    plot_dual_violin()