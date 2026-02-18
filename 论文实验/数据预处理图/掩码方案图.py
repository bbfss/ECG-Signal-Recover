import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import wfdb
from tqdm import tqdm

# --- 1. 环境与路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, "..", ".."))

if root_path not in sys.path:
    sys.path.insert(0, root_path)

from config import Config
from utils.signal_utils import apply_transient_mask, apply_extended_mask, apply_diagonal_mask, filtering

# --- 2. 绘图函数: 6x2 布局 (用于 Transient 和 Extended) ---

def plot_6x2_strategy(original, masked, filename):
    """
    12导联 6x2 布局，每个子图下方均显示 'Time'。
    """
    # 禁用 sharex，以便每个子图都能独立显示横轴标签
    fig, axes = plt.subplots(6, 2, figsize=(16, 16)) 
    # 增加垂直间距 hspace 以容纳每一行的 'Time' 标签
    plt.subplots_adjust(hspace=0.6, wspace=0.15, top=0.96, bottom=0.05)

    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    for i in range(12):
        row = i % 6
        col = i // 6
        ax = axes[row, col]
        
        # 1. 画底层：原始信号（红色）
        ax.plot(original[:, i], color='red', linewidth=0.8, alpha=0.7)
        
        # 2. 处理遮掩部分
        display_masked = masked[:, i].copy()
        display_masked[np.abs(display_masked) < 1e-8] = np.nan
        
        # 3. 画顶层：保留部分（蓝色）
        ax.plot(display_masked, color='blue', linewidth=1.0)
        
        ax.set_ylabel(lead_names[i], rotation=0, labelpad=20, verticalalignment='center', fontweight='bold')
        ax.set_yticks([]) 
        
        # 每一个子图下面都加上 Time
        ax.set_xlabel('Time', fontsize=10)

    plt.savefig(filename, bbox_inches='tight', dpi=120)
    plt.close()

# --- 3. 绘图函数: 对角线合集 (12x5 布局) ---

def plot_diagonal_combined(original, masked_list, filename):
    """
    展示5种对角线掩码方案，每个子图下方均显示 'Time'。
    """
    num_schemes = len(masked_list) 
    # 禁用 sharex
    fig, axes = plt.subplots(12, num_schemes, figsize=(22, 25))
    
    scheme_names = ["C1", "C2", "C3", "C4", "C5"]
    # 增加间距以容纳大量标签
    plt.subplots_adjust(top=0.95, hspace=0.7, wspace=0.2, bottom=0.05)

    for col_idx in range(num_schemes):
        axes[0, col_idx].set_title(scheme_names[col_idx], fontsize=20, pad=15, fontweight='bold')
        
        current_masked_data = masked_list[col_idx]
        for row_idx in range(12):
            ax = axes[row_idx, col_idx]
            
            ax.plot(original[:, row_idx], color='red', linewidth=0.6, alpha=0.5)
            
            display_masked = current_masked_data[:, row_idx].copy()
            display_masked[np.abs(display_masked) < 1e-8] = np.nan
            ax.plot(display_masked, color='blue', linewidth=0.8)
            
            ax.set_yticks([])
            # 每一个信号下面都是 Time
            ax.set_xlabel('Time', fontsize=8)
            
            if col_idx == 0:
                ax.set_ylabel(f"L{row_idx+1}", rotation=0, labelpad=15, fontweight='bold')
            else:
                ax.set_ylabel("") # 清除其他列的 ylabel 避免拥挤

    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

# --- 4. 批量数据获取 ---

def get_20_signals():
    data_root = Config.DATA_PATHS.get("PTBXL")
    csv_path = os.path.join(data_root, "ptbxl_database.csv")
    meta = pd.read_csv(csv_path, index_col='ecg_id')
    col = 'filename_hr' if 'filename_hr' in meta.columns else 'filename_lr'
    
    signals = []
    pbar = tqdm(total=20, desc="[Data] Extracting PTB-XL Signals")
    for _, row in meta.iterrows():
        if len(signals) >= 20: break
        try:
            fp = os.path.join(data_root, row[col])
            sig, _ = wfdb.rdsamp(fp)
            if not np.any(np.isnan(sig)):
                # 预处理：滤波与重采样
                processed = np.array([filtering(sig[:, j], Config.PTBXL_FS) for j in range(12)]).T
                signals.append(processed)
                pbar.update(1)
        except: continue
    pbar.close()
    return signals

# --- 5. 执行主程序 ---

def main():
    output_base = os.path.join(current_dir, "vis_outputs")
    if not os.path.exists(output_base):
        os.makedirs(output_base)

    all_signals = get_20_signals()

    for idx, sig in enumerate(tqdm(all_signals, desc="[Vis] Generating Plots")):
        sample_id = f"sample_{idx+1:02d}"
        
        # 1. 瞬态掩码
        m_trans, _ = apply_transient_mask(sig, Config.MISSING_RATIO)
        plot_6x2_strategy(sig, m_trans, os.path.join(output_base, f"{sample_id}_1_transient.png"))

        # 2. 连续掩码
        m_ext, _ = apply_extended_mask(sig, Config.MISSING_RATIO)
        plot_6x2_strategy(sig, m_ext, os.path.join(output_base, f"{sample_id}_2_extended.png"))

        # 3. 对角线掩码 (5方案合集)
        _, masked_samples = apply_diagonal_mask(sig)
        plot_diagonal_combined(sig, masked_samples, os.path.join(output_base, f"{sample_id}_3_diagonal.png"))

    print(f"\n>> 任务完成！20组演示图（每个信号含Time标签）已存入：\n{output_base}")

if __name__ == "__main__":
    main()