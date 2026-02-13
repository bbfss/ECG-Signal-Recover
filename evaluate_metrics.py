import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import Config
# 导入之前定义的加载器
from utils.ptbxl_loader import get_diagonal_masked_ptbxl_loader 
from utils.metrics import calculate_mae, calculate_rmse, calculate_pcc, calculate_prd
from models import ECGRecover

def run_ECGRecover_on_diagonal_masked_ptbxl():
    """
    深度评估：测试 ECGRecover 在 5 种掩码下，12个导联各自的表现及总体表现。
    """
    # =============================================================
    # 【自定义配置区】
    # =============================================================
    TARGET_MODEL_CLASS = ECGRecover
    MODEL_NAME_STR = "ECGRecover"
    WEIGHT_FILE = "ECGRecover_PTBXL_ep50.pth"
    WEIGHT_KEY = "model"
    
    # 12导联名称
    LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    METRICS_MAP = {
        "MAE": calculate_mae,
        "RMSE": calculate_rmse,
        "PCC": calculate_pcc,
        "PRD": calculate_prd
    }
    
    DEVICE = Config.DEVICE
    SAVE_CSV_NAME = f"{MODEL_NAME_STR}_Diagonal_LeadWise_Results.csv"
    # =============================================================

    # --- 1. 初始化模型并加载权重 ---
    print(f"\n[Step 1] Initializing {MODEL_NAME_STR} Model...")
    model = TARGET_MODEL_CLASS().to(DEVICE)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, WEIGHT_FILE)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint[WEIGHT_KEY])
        model.eval()
        print(f" ✅ 权重加载成功: {WEIGHT_FILE}")
    else:
        print(f" ❌ 错误：找不到文件 {checkpoint_path}")
        return

    # --- 2. 获取掩码加载器字典 ---
    print(f"\n[Step 2] Loading 5-fold Diagonal Masked Data...")
    # 注意：get_diagonal_masked_ptbxl_loader 仅返回 loaders 字典
    loaders_dict = get_diagonal_masked_ptbxl_loader()

    # --- 3. 循环测试所有掩码方案 ---
    print("\n[Step 3] Running Lead-wise Inference...")
    final_results = []

    for mask_name, loader in loaders_dict.items():
        # 初始化累计器：包含 'Overall' 和 12个导联索引 (0-11)
        # 结构: { 'Overall': {MAE: [], ...}, 0: {MAE: [], ...}, ... }
        accumulator = { "Overall": {m: [] for m in METRICS_MAP.keys()} }
        for i in range(12):
            accumulator[i] = {m: [] for m in METRICS_MAP.keys()}

        for batch_x, batch_y in tqdm(loader, desc=f"   Testing {mask_name}", leave=False):
            # 维度适配: (B, 12, 512) -> (B, 512, 12)
            input_np = batch_x.permute(0, 2, 1).numpy()
            
            with torch.no_grad():
                pred_np = model.predict(input_np, device=DEVICE)
            
            # 转回 Torch 并保持 (B, 12, 512) 用于计算
            pred_torch = torch.from_numpy(pred_np).permute(0, 2, 1)

            # --- A. 计算 Overall 指标 (整个 12 导联矩阵) ---
            for m_name, m_func in METRICS_MAP.items():
                val_total = m_func(batch_y, pred_torch)
                accumulator["Overall"][m_name].append(val_total)

            # --- B. 计算单个导联指标 ---
            for i in range(12):
                # 切片保持维度为 (B, 1, 512) 以兼容 metrics 函数
                y_true_lead = batch_y[:, i:i+1, :]
                y_pred_lead = pred_torch[:, i:i+1, :]
                
                for m_name, m_func in METRICS_MAP.items():
                    val_lead = m_func(y_true_lead, y_pred_lead)
                    accumulator[i][m_name].append(val_lead)

        # --- C. 汇总该 Mask 模式下的 13 行结果 (12导联 + 1 Overall) ---
        # 1. 记录 Overall 结果
        overall_entry = {"Mask_Type": mask_name, "Lead": "Overall"}
        for m_name in METRICS_MAP.keys():
            overall_entry[m_name] = np.mean(accumulator["Overall"][m_name])
        final_results.append(overall_entry)

        # 2. 记录每个导联的结果
        for i in range(12):
            lead_entry = {"Mask_Type": mask_name, "Lead": LEAD_NAMES[i]}
            for m_name in METRICS_MAP.keys():
                lead_entry[m_name] = np.mean(accumulator[i][m_name])
            final_results.append(lead_entry)

    # --- 4. 结果展示与导出 ---
    df_res = pd.DataFrame(final_results)
    
    # 调整列顺序，让 Mask_Type 和 Lead 在最前
    cols = ["Mask_Type", "Lead"] + list(METRICS_MAP.keys())
    df_res = df_res[cols]

    print("\n" + "="*70)
    print(f"📊 {MODEL_NAME_STR} 导联级对角线掩码评估报告 (前13行示例)")
    print("="*70)
    print(df_res.head(13).to_string(index=False))
    print("="*70)

    # 导出 CSV
    output_path = os.path.join(Config.METRICS_DIR, SAVE_CSV_NAME)
    df_res.to_csv(output_path, index=False)
    print(f"✅ 深度评估完成！结果已保存至: {output_path}")

if __name__ == "__main__":
    run_ECGRecover_on_diagonal_masked_ptbxl()