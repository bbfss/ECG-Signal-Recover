import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- 1. 路径与环境配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import Config
from utils.metrics import calculate_mae, calculate_rmse, calculate_pcc, calculate_prd, calculate_pre
from utils.ptbxl_loader import get_ptbxl_evaluate_loader
from models import AdvancedUNet1D, FlowNetwork, DeScoD_ScoreNet
from models.unet_normal_flow import NormalFlowNetwork

# --- 2. 自动化配置 ---
METRICS_TO_CALC = {
    "MAE": calculate_mae,
    "RMSE": calculate_rmse,
    "PCC": calculate_pcc,
    "PRD": calculate_prd,
    "PRE": calculate_pre,
}

def get_eval_models(device):
    # 精准定位实验文件夹下的 checkpoints
    checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    data_name = "PTBXL"

    def load_weights(models_to_load, prefix):
        # 优先加载 best，其次 ep50
        for suffix in ["ep50", "best"]:
            file_name = f"{prefix}_{data_name}_{suffix}.pth"
            path = os.path.join(checkpoint_dir, file_name)
            if os.path.exists(path):
                print(f"✅ 成功加载: {file_name}")
                ckpt = torch.load(path, map_location=device)
                for k, m_obj in models_to_load.items():
                    if k in ckpt:
                        m_obj.load_state_dict(ckpt[k])
                        m_obj.eval()
                res = list(models_to_load.values())
                return res[0] if len(res) == 1 else tuple(res)
        print(f"❌ 错误：找不到 {prefix} 的权重文件")
        return None

    # 加载三个核心对比模型
    m_descod = load_weights({"model": DeScoD_ScoreNet().to(device)}, "DeScoD")
    
    m_unet_flow = load_weights({
        "unet": AdvancedUNet1D().to(device),
        "flow": FlowNetwork().to(device)
    }, "Unet_Flow")

    m_normal_flow = load_weights({
        "unet": AdvancedUNet1D().to(device),
        "normal_flow": NormalFlowNetwork().to(device)
    }, "ECG_Unet_Normal_Flow")

    return {
        "DeScoD": m_descod,
        "Unet_Flow": m_unet_flow,
        "Normal_Flow": m_normal_flow
    }

# --- 3. 评估逻辑 ---
def run_deep_evaluation():
    device = Config.DEVICE
    models_dict = get_eval_models(device)
    val_loader, _ = get_ptbxl_evaluate_loader()
    
    all_results = []

    for name, model_obj in models_dict.items():
        if model_obj is None: continue
        
        print(f"\n🔍 正在深度评估模型: {name}")
        perf_accumulator = {m: [] for m in METRICS_TO_CALC.keys()}

        for batch_x, batch_y in tqdm(val_loader, desc=f"推理中"):
            # 统一形状处理 (B, 12, 512) -> (B, 512, 12)
            input_np = batch_x.permute(0, 2, 1).numpy()
            target_torch = batch_y
            
            with torch.no_grad():
                if name == "DeScoD":
                    pred_np = model_obj.predict(input_np, steps=10, device=device)
                else:
                    # 复合模型处理
                    unet_m, flow_m = model_obj
                    mid_res = unet_m.predict(input_np, device=device)
                    # 此处步骤固定为 5 以观察标准性能
                    pred_np = flow_m.predict(mid_res, steps=5, device=device)

            pred_torch = torch.from_numpy(pred_np).permute(0, 2, 1)

            # 计算所有指标
            for m_name, m_func in METRICS_TO_CALC.items():
                val = m_func(target_torch, pred_torch)
                perf_accumulator[m_name].append(val)

        # 汇总
        res_entry = {"Model": name}
        for m_name in METRICS_TO_CALC.keys():
            res_entry[m_name] = np.mean(perf_accumulator[m_name])
        all_results.append(res_entry)

    # 保存结果
    df = pd.DataFrame(all_results)
    save_path = os.path.join(SCRIPT_DIR, "deep_eval_results.csv")
    df.to_csv(save_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"📊 评估结果总结 (结果已存至: deep_eval_results.csv)")
    print(df.to_string(index=False))
    print(f"{'='*60}")

if __name__ == "__main__":
    run_deep_evaluation()