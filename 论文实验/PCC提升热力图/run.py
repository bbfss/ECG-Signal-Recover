import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. 加载数据
# 修改为你本地的 CSV 文件路径
# 1. 获取当前脚本文件所在的文件夹绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(current_dir, 'Ultra_Realistic_ECG_Results.csv')
pdf_save_path = os.path.join(current_dir, 'PCC_Improve_Heatmap.svg')
df = pd.read_csv(csv_path)
# 2. 数据准备：计算“改进差值” (Delta)
# 我们要画的是：(Your Model PCC) - (Baseline PCC)
# 这样颜色越深代表你的模型进步越大

# 提取基准模型数据
df_base = df[df['Model'] == 'ECGRecover (Baseline)']
# 提取你的模型数据
df_ours = df[df['Model'] == 'Your Model (with Attention)']

# 将数据从“长表格”转换为“矩阵格式”（透视表）
# 纵轴为导联(Lead)，横轴为任务(Mask_Type)
pivot_base = df_base.pivot(index='Lead', columns='Mask_Type', values='PCC')
pivot_ours = df_ours.pivot(index='Lead', columns='Mask_Type', values='PCC')

# 3. 排序：确保顺序符合论文逻辑 (C1最难 -> C5最易)
mask_order = ['C1', 'C2', 'C3', 'C4', 'C5']
lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# 只保留 12 个导联（排除 Overall 总体平均值，这样热力图对比更细腻）
pivot_base = pivot_base.reindex(index=lead_order, columns=mask_order)
pivot_ours = pivot_ours.reindex(index=lead_order, columns=mask_order)

# 计算差值
delta_pcc = pivot_ours - pivot_base

# 4. 绘制热力图
plt.figure(figsize=(10, 8))

# sns.heatmap 是核心函数
sns.heatmap(
    delta_pcc, 
    annot=True,           # 在格子里显示数值
    fmt=".3f",            # 数值保留 3 位小数
    cmap='YlGnBu',        # 颜色映射：黄-绿-蓝 (学术常用，颜色越深改进越大)
    linewidths=0.5,       # 格子之间的白线间距
    cbar_kws={'label': 'PCC Improvement ($\Delta$)'} # 侧边颜色条的标签
)

# 5. 修饰与保存
plt.title('Lead-wise PCC Improvement (Ours vs. Baseline)', fontsize=14, pad=20)
plt.ylabel('12 Standard ECG Leads', fontsize=12)
plt.xlabel('Masking Tasks (C1: Hardest $\\rightarrow$ C5: Easiest)', fontsize=12)

# 保存为矢量 PDF (投稿专用，无限放大不模糊)
plt.savefig(pdf_save_path, bbox_inches='tight')
plt.show()