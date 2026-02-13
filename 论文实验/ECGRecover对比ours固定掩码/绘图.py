import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 自动处理路径
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
csv_path = os.path.join(current_dir, 'Ultra_Realistic_ECG_Results.csv')
# 创建一个专门存放小图的文件夹，方便 PS 导入
output_folder = os.path.join(current_dir, 'Plot_Components')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. 读取数据
if not os.path.exists(csv_path):
    print(f"错误：找不到文件 {csv_path}")
else:
    df = pd.read_csv(csv_path)
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    metrics = ['MAE', 'RMSE', 'PCC']
    
    # 设置风格：学术简约风
    sns.set_theme(style="ticks", font="serif")
    plt.rcParams['font.family'] = 'serif'
    
    # 定义统一的颜色和点型（对应之前的风格）
    palette = {"ECGRecover (Baseline)": "#E74C3C", "Your Model (with Attention)": "#3498DB"}
    markers = {"ECGRecover (Baseline)": "s", "Your Model (with Attention)": "o"}

    print("开始生成 PS 组装组件...")

    # 3. 嵌套循环：为每个导联和指标生成独立图片
    for lead in leads:
        for metric in metrics:
            # 筛选数据
            subset = df[df['Lead'] == lead].copy()
            
            # 创建固定尺寸的画布
            plt.figure(figsize=(5, 4))
            
            # 绘制点线图
            ax = sns.lineplot(
                data=subset, x='Mask_Type', y=metric, hue='Model',
                style='Model', markers=markers, palette=palette,
                linewidth=2.5, markersize=10, legend=False
            )
            
            # --- 极致清晰的 PS 标注风格 ---
            # 顶部标注导联
            plt.title(f"Lead {lead}", fontsize=16, fontweight='bold', pad=10)
            
            # 左侧标注数值指标
            plt.ylabel(metric, fontsize=14, fontweight='bold')
            
            # 下方标注掩码
            plt.xlabel("Mask Type", fontsize=12)
            
            # 设置坐标轴边框（让它看起来像个独立的盒子，方便 PS 裁切）
            sns.despine(trim=False, offset=5)
            plt.grid(True, linestyle='--', alpha=0.4)
            
            # 针对不同指标调整 Y 轴范围（让趋势更明显）
            if metric == 'PCC':
                plt.ylim(subset[metric].min() - 0.05, 1.0)
            
            # 4. 保存为高分辨率 PNG
            # 文件名：Lead_I_MAE.png 这种格式，你在文件夹里一眼就能认出来
            file_name = f"Component_{lead}_{metric}.png"
            save_path = os.path.join(output_folder, file_name)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
            plt.close() # 及时释放内存

    # 5. 额外生成一个独立的图例组件（Legend），方便你放在 PS 的任何地方
    plt.figure(figsize=(6, 1))
    for model, color in palette.items():
        plt.plot([], [], color=color, marker=markers[model], label=model, linewidth=3, markersize=12)
    plt.axis('off')
    plt.legend(loc='center', ncol=2, frameon=True, fontsize=14, shadow=True)
    plt.savefig(os.path.join(output_folder, "Component_Global_Legend.png"), dpi=300, bbox_inches='tight', transparent=True)

    print(f"✅ 所有组件已生成！请查看文件夹：{output_folder}")