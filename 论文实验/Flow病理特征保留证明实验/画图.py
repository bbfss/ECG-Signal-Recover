import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_individual_comparisons(lead_idx=1):
    """
    lead_idx: 导联索引 (0-11)。建议选 1 (Lead II) 或 6 (V1) 特征最明显。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 定义配置
    models = ["ECGRecover", "UNet_Stage", "Unet_Flow"]
    colors = {
        "Original": "#333333",    # 深灰色，背景参考
        "ECGRecover": "#e74c3c",  # 红色
        "UNet_Stage": "#f39c12",  # 橙色
        "Unet_Flow": "#3498db"    # 蓝色
    }
    
    # 2. 读取数据
    data = {}
    try:
        data["Original"] = pd.read_csv(os.path.join(current_dir, "hierarchy_case_Original.csv")).iloc[:, lead_idx].values
        for m in models:
            path = os.path.join(current_dir, f"hierarchy_case_{m}.csv")
            data[m] = pd.read_csv(path).iloc[:, lead_idx].values
    except Exception as e:
        print(f"❌ 读取文件失败: {e}\n请确保已运行实验代码并生成了 hierarchy_case_*.csv 文件。")
        return

    # 3. 开始绘图
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)
    
    for i, model_name in enumerate(models):
        ax = axes[i]
        
        # 绘制原始信号作为背景参考
        ax.plot(data["Original"], color=colors["Original"], label="Original (Truth)", 
                alpha=0.3, linewidth=2, linestyle='--')
        
        # 绘制当前模型的恢复信号
        ax.plot(data[model_name], color=colors[model_name], label=f"Predicted by {model_name}", 
                linewidth=1.5)
        
        # 美化子图
        ax.set_title(f"Comparison: {model_name} vs. Original", fontsize=12, fontweight='bold')
        ax.legend(loc="upper right", frameon=True, shadow=True)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylabel("Amplitude")

    # 设置最下方的 X 轴标签
    axes[-1].set_xlabel("Time Samples (at 100Hz)")
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(current_dir, "model_comparison_steps.png")
    plt.savefig(output_path, dpi=300)
    print(f"✅ 成功生成对比图：{output_path}")
    plt.show()

if __name__ == "__main__":
    # 运行绘图
    plot_individual_comparisons(lead_idx=1)