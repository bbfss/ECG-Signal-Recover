import sys
import os

# --- 1. 环境与路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, "..", ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config
from utils.metrics import calculate_pre, calculate_mae, calculate_pcc
from utils.ptbxl_loader import get_ptbxl_evaluate_loader
from utils.signal_utils import apply_transient_mask, apply_extended_mask, apply_diagonal_mask
from models import ECGRecover, AdvancedUNet1D, FlowNetwork

def run_multi_mask_experiment():
    device = Config.DEVICE
    print(f"🚀 启动全掩码方案对比实验 | 设备: {device}")
    print(f"📂 实验结果将保存至: {current_dir}")

    # --- 2. 模型加载逻辑 ---
    def load_model(model_class, prefix, keys):
        path = f"{Config.CHECKPOINT_DIR}/{prefix}_PTBXL_ep50.pth"
        if not os.path.exists(path):
            print(f"⚠️ 警告: 未找到权重文件 {path}")
            return None
        ckpt = torch.load(path, map_location=device)
        if isinstance(keys, str):
            m = model_class().to(device)
            m.load_state_dict(ckpt[keys])
            return m.eval()
        else:
            models = []
            for k, cls in zip(keys, model_class):
                m = cls().to(device)
                m.load_state_dict(ckpt[k])
                models.append(m.eval())
            return tuple(models)

    m_ecgrecover = load_model(ECGRecover, "ECGRecover", "model")
    m_unet_stage = load_model(AdvancedUNet1D, "Unet_Flow", "unet")
    m_unet, m_flow = load_model([AdvancedUNet1D, FlowNetwork], "Unet_Flow", ["unet", "flow"])

    # --- 3. 实验配置 ---
    mask_schemes = ["Transient", "Extended", "Diagonal"]
    model_names = ["ECGRecover", "UNet_Stage", "Unet_Flow"]
    all_results = []
    
    # 🌟 新的捕获逻辑：寻找 Flow > UNet > ECGRecover 的完美层级案例
    hierarchy_case = {
        "max_gain": -1.0,
        "mask_type": "",
        "signals": None,
        "metrics": {}
    }

    val_loader, _ = get_ptbxl_evaluate_loader()

    # --- 4. 循环遍历掩码方案 ---
    for mask_type in mask_schemes:
        print(f"\n======== 正在评估掩码方案: {mask_type} ========")
        perf = {name: {"PRE": [], "MAE": [], "PCC": []} for name in model_names}

        for batch_orig, _ in tqdm(val_loader, desc=mask_type):
            B = batch_orig.shape[0]
            orig_np = batch_orig.permute(0, 2, 1).numpy()

            masked_inputs = []
            for i in range(B):
                sig = orig_np[i]
                if mask_type == "Transient": m_sig, _ = apply_transient_mask(sig)
                elif mask_type == "Extended": m_sig, _ = apply_extended_mask(sig)
                else: _, m_sigs = apply_diagonal_mask(sig); m_sig = m_sigs[0] 
                masked_inputs.append(m_sig)
            
            input_np = np.stack(masked_inputs)

            with torch.no_grad():
                pred_ecg_rec = m_ecgrecover.predict(input_np, device=device)
                pred_unet_s = m_unet_stage.predict(input_np, device=device)
                mid = m_unet.predict(input_np, device=device)
                pred_flow = m_flow.predict(mid, steps=Config.SAMPLE_STEPS, device=device)

            for i in range(B):
                y_true = batch_orig[i:i+1]
                p_ecg = torch.from_numpy(pred_ecg_rec[i:i+1]).permute(0, 2, 1)
                p_unet = torch.from_numpy(pred_unet_s[i:i+1]).permute(0, 2, 1)
                p_flow = torch.from_numpy(pred_flow[i:i+1]).permute(0, 2, 1)

                pre_ecg = calculate_pre(y_true, p_ecg)
                pre_unet = calculate_pre(y_true, p_unet)
                pre_flow = calculate_pre(y_true, p_flow)

                # 记录平均指标
                perf["ECGRecover"]["PRE"].append(pre_ecg)
                perf["UNet_Stage"]["PRE"].append(pre_unet)
                perf["Unet_Flow"]["PRE"].append(pre_flow)

                # 🌟 核心逻辑：捕获完美层级 (PRE越小越好)
                # 条件：Flow误差 < UNet误差 < ECGRecover误差
                if pre_flow < pre_unet and pre_unet < pre_ecg:
                    # 计算总提升幅度作为筛选标准
                    gain = pre_ecg - pre_flow 
                    if gain > hierarchy_case["max_gain"]:
                        hierarchy_case["max_gain"] = gain
                        hierarchy_case["mask_type"] = mask_type
                        hierarchy_case["metrics"] = {"ECG": pre_ecg, "UNet": pre_unet, "Flow": pre_flow}
                        hierarchy_case["signals"] = {
                            "Original": orig_np[i],
                            "Masked": input_np[i],
                            "ECGRecover": pred_ecg_rec[i],
                            "UNet_Stage": pred_unet_s[i],
                            "Unet_Flow": pred_flow[i]
                        }

        for name in model_names:
            all_results.append({
                "Mask_Type": mask_type,
                "Model": name,
                "PRE (%) ↓": np.mean(perf[name]["PRE"])
            })

    # --- 5. 结果导出 ---
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(current_dir, "multi_mask_results.csv"), index=False)
    
    print("-" * 30)
    if hierarchy_case["signals"]:
        print(f"🎯 成功捕获完美层级案例！(掩码类型: {hierarchy_case['mask_type']})")
        print(f"📈 PRE 表现: ECG({hierarchy_case['metrics']['ECG']:.2f}) > "
              f"UNet({hierarchy_case['metrics']['UNet']:.2f}) > "
              f"Flow({hierarchy_case['metrics']['Flow']:.2f})")
        
        for key, val in hierarchy_case["signals"].items():
            pd.DataFrame(val).to_csv(os.path.join(current_dir, f"hierarchy_case_{key}.csv"), index=False)
        print(f"📸 信号文件已保存为 hierarchy_case_*.csv")
    else:
        print("❓ 未能找到完全符合 Flow < UNet < ECGRecover 的单一样本，请检查模型性能。")

if __name__ == "__main__":
    run_multi_mask_experiment()