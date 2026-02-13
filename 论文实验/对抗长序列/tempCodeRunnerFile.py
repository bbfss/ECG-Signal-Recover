import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. 路径自动识别逻辑 ---
# 获取当前运行脚本所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 设定读取路径和保存路径（固定文件名为你的 CSV 名称）
csv_path = os.path.join(current_dir, 'ECG_Sequence_Length_Experiment.csv')
save_path = os.path.join(current_dir, 'ECG_Scaling_Comparison.svg')

# --- 2. 读取并检查数据 ---
if not os.path.exists(csv_path):
    print(f"错误：未在目录 {current_dir} 中找到文件 'ECG_Sequence_Length_Experiment.csv'")
else:
    df = pd.read_csv(csv_path)

    # 提取不同模型的数据进行对比
    unet_df = df[df['Model'] == 'Standard UNet']
    ours_df = df[df['Model'] == 'Ours (Mamba-Bridge)']

    # --- 3. 绘图设置 ---
    # 创建 1行2列 的子图布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：MAE 误差增长趋势 (反映模型在长序列下的抗性能衰减能力)
    ax1.plot(unet_df['Sequence_Length'], unet_df['MAE'], marker='o', 
             linestyle='--', color='#ff7f0e', label='Standard UNet', linewidth=2)
    ax1.plot(ours_df['Sequence_Length'], ours_df['MAE'], marker='D', 
             linestyle='-', color='#1f77b4', label='Ours (Mamba-Bridge)', linewidth=2.5)
    
    ax1.set_xlabel('Sequence Length (L)', fontsize=12)
    ax1.set_ylabel('MAE ↓', fontsize=12)
    ax1.set_title('Reconstruction Error Trend', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend()

    # 右图：PCC 相关性稳定性 (反映模型捕捉长程节律的稳健性)
    ax2.plot(unet_df['Sequence_Length'], unet_df['PCC'], marker='o', 
             linestyle='--', color='#ff7f0e', label='Standard UNet', linewidth=2)
    ax2.plot(ours_df['Sequence_Length'], ours_df['PCC'], marker='D', 
             linestyle='-', color='#1f77b4', label='Ours (Mamba-Bridge)', linewidth=2.5)
    
    ax2.set_xlabel('Sequence Length (L)', fontsize=12)
    ax2.set_ylabel('PCC ↑', fontsize=12)
    ax2.set_title('Correlation Stability', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()

    # --- 4. 保存为 SVG 矢量图 ---
    plt.tight_layout()
    # 强制指定格式为 svg，适合放入 LaTeX 或 Word 论文
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    
    print(f"成功！可视化图表已保存至：{save_path}")
    plt.show()