# utils/visualizer.py
import matplotlib.pyplot as plt
import os
import numpy as np
from config import Config
from utils.metrics import calculate_mae, calculate_pcc, calculate_rmse

def save_comparison_plot(clean_data, masked_input, recon_data, mask, epoch, model_name, mode_name):
    """
    统一的 12 导联对比图绘制函数
    :param clean_data: 原始完整信号 [512, 12]
    :param masked_input: 送入模型的带遮掩信号 [512, 12]
    :param recon_data: 模型输出的重建信号 [512, 12]
    :param mask: 掩码矩阵 [512, 12]，0代表被遮掩点
    :param epoch: 当前轮数
    :param model_name: 模型名称（用于文件名）
    :param mode_name: 遮掩模式名称（如 'Transient' 或 'Extended'）
    """
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    fig, axes = plt.subplots(6, 2, figsize=(20, 15), dpi=150)
    fig.suptitle(f"Epoch {epoch} | Model: {model_name} | Mode: {mode_name}", fontsize=20, fontweight='bold', y=0.96)

    for i in range(12):
        ax = axes[i % 6, i // 6]
        
        # --- 使用“老方法”：直接计算该导联全段 512 点的指标 ---
        c_sig = clean_data[:, i] # 原始信号 [512]
        r_sig = recon_data[:, i] # 重建信号 [512]
        
        # 直接调用 metric 里的函数
        # 注意：如果你的 calculate_mae 内部没有处理维度，这里传入 1D 数组即可
        lead_mae = calculate_mae(c_sig, r_sig)
        
        # 2. 处理“带断点的蓝色输入线”：为了视觉上清晰，将掩码为0的点设为 NaN
        input_to_plot = masked_input[:, i].copy()
        input_to_plot[mask[:, i] == 0] = np.nan
        
        # 3. 开始绘图
        # 灰色实线：原始信号 (背景参考)
        ax.plot(clean_data[:, i], color='#BBBBBB', lw=1.5, label='Original', alpha=0.8)
        # 红色虚线：模型重建信号 (实验结果)
        ax.plot(recon_data[:, i], color='red', linestyle='--', lw=1.2, label='Reconstruction')
        # 蓝色虚线：原始保留的输入 (已知部分)
        ax.plot(input_to_plot, color='blue', linestyle='--', lw=1.5, label='Masked Input')
        
        # 4. 细节美化
        ax.set_title(f"{lead_names[i]} (MAE: {lead_mae:.4f})", fontsize=10)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.2)
        if i == 0: 
            ax.legend(loc='upper right', fontsize=8)

    # 5. 保存图片：自动使用 Config 中定义的路径
    save_filename = f"epoch_{epoch}_{mode_name}.png"
    save_path = os.path.join(Config.PLOT_DIR, model_name, save_filename)
    
    # 确保模型子文件夹存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path