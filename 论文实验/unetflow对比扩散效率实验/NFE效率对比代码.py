import sys
import os

# --- 1. 环境与路径配置 ---
# 锁定当前实验脚本所在的目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 向上翻两级找到项目根目录 (E:\Code\ECG-Signal-Recover)
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# 基础配置与工具
from config import Config
from utils.metrics import calculate_mae, calculate_pcc
from utils.ptbxl_loader import get_ptbxl_evaluate_loader

# 导入所有模型类
from models import AdvancedUNet1D, FlowNetwork, DeScoD_ScoreNet
# 导入你刚刚写的 Normal Flow
from models.unet_normal_flow import NormalFlowNetwork

# =============================================================
# 2. 增强型模型加载逻辑：精准读取本地实验权重
# =============================================================
def get_eval_models(device):
    # 精准定位到当前实验文件夹下的 checkpoints 目录
    checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    data_name = "PTBXL" 

    def load_dict_weights(models_to_load, model_file_prefix):
        # 优先读取 "best" 模型，如果不存在则读取 ep50
        
        file_name = f"{model_file_prefix}_{data_name}_ep50.pth"
        path = os.path.join(checkpoint_dir, file_name)
        
        if not os.path.exists(path):
            file_name = f"{model_file_prefix}_{data_name}_best.pth"
            path = os.path.join(checkpoint_dir, file_name)

        if os.path.exists(path):
            print(f"📦 正在加载权重: {file_name}")
            checkpoint = torch.load(path, map_location=device)
            for key, model_obj in models_to_load.items():
                if key in checkpoint:
                    model_obj.load_state_dict(checkpoint[key])
                    model_obj.eval()
            res = list(models_to_load.values())
            return res[0] if len(res) == 1 else tuple(res)
        else:
            print(f"⚠️ 警告：找不到权重文件 {path}")
            return None

    # 模型 1: 扩散模型基准
    m_descod = load_dict_weights({"model": DeScoD_ScoreNet().to(device)}, "DeScoD")
    
    # 模型 2: 改进版流匹配 (Ours)
    m_unet_flow = load_dict_weights({
        "unet": AdvancedUNet1D().to(device),
        "flow": FlowNetwork().to(device)
    }, "Unet_Flow")

    # 模型 3: 基础版流匹配 (Ablation)
    m_normal_flow = load_dict_weights({
        "unet": AdvancedUNet1D().to(device),
        "normal_flow": NormalFlowNetwork().to(device)
    }, "ECG_Unet_Normal_Flow")

    return {
        "DeScoD": m_descod,
        "Unet_Flow": m_unet_flow,
        "Normal_Flow": m_normal_flow
    }

# =============================================================
# 3. 实验主逻辑：三模型 NFE 对比
# =============================================================
def run_nfe_experiment():
    device = Config.DEVICE
    models_dict = get_eval_models(device)
    
    # 检查核心模型是否加载成功
    valid_models = [k for k, v in models_dict.items() if v is not None]
    if not valid_models:
        print("🛑 没有找到任何可用的模型权重，请检查 checkpoints 文件夹。")
        return

    # 设置测试步数 1-14
    steps_to_test = list(range(1, 15))
    val_loader, _ = get_ptbxl_evaluate_loader()
    
    results = []

    for steps in steps_to_test:
        print(f"\n🚀 测试推理效率 | NFE = {steps}")
        for m_name in valid_models:
            model_obj = models_dict[m_name]
            pcc_vals, latencies = [], []

            # 评估前 20 个 batch 以平衡速度和准确性
            for i, (batch_x, batch_y) in enumerate(tqdm(val_loader, desc=f"   {m_name}")):
                if i > 20: break 
                
                input_np = batch_x.permute(0, 2, 1).numpy()
                target_torch = batch_y
                
                start = time.perf_counter()
                with torch.no_grad():
                    if m_name == "DeScoD":
                        pred_np = model_obj.predict(input_np, steps=steps, device=device)
                    else:
                        # Unet_Flow 和 Normal_Flow 都是复合模型 (UNet + Flow)
                        unet_m, flow_m = model_obj
                        mid_res = unet_m.predict(input_np, device=device)
                        pred_np = flow_m.predict(mid_res, steps=steps, device=device)
                
                end = time.perf_counter()
                latencies.append((end - start) * 1000) # 毫秒
                
                pred_torch = torch.from_numpy(pred_np).permute(0, 2, 1)
                pcc_vals.append(calculate_pcc(target_torch, pred_torch))

            results.append({
                "NFE": steps,
                "Model": m_name,
                "PCC": np.mean(pcc_vals),
                "Latency_ms": np.mean(latencies)
            })

    # 保存数据到脚本同级目录
    df = pd.DataFrame(results)
    save_path = os.path.join(SCRIPT_DIR, "nfe_results.csv")
    df.to_csv(save_path, index=False)
    
    print(f"\n✅ 实验完成！")
    print(f"📊 数据已保存至: {save_path}")
    print(f"💡 接下来你可以运行绘图脚本来查看三曲线对比。")

if __name__ == "__main__":
    run_nfe_experiment()