import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# 基础配置与工具导入
from config import Config
from utils.metrics import calculate_mae, calculate_rmse, calculate_pcc, calculate_prd
from utils.ptbxl_loader import get_ptbxl_evaluate_loader
from utils.ludb_loader import get_ludb_evaluate_loader

from models import *

# =============================================================
# 1. 自动化评估配置 (修改此处即可增加新内容)
# =============================================================

# 【数据集清单】：{ "展示名称": 加载器函数 }
# 每个加载器需遵循 (val_loader, demo_sig) 的返回格式
DATASETS_TO_RUN = {
    "PTBXL": get_ptbxl_evaluate_loader,
    "LUDB": get_ludb_evaluate_loader,
}

# 【指标清单】：{ "列名": metrics.py 中的函数名 }
# 评估时会自动遍历这些函数并记录结果
METRICS_TO_CALC = {
    "MAE": calculate_mae,
    "RMSE": calculate_rmse,
    "PCC": calculate_pcc,
    "PRD": calculate_prd,
}

def get_eval_models(device):
    """
    适配字典格式保存的权重加载逻辑
    """
    # 1. 设置权重存放目录和数据集名称
    # 确保这个路径指向你截图里的那个文件夹
    checkpoint_dir = Config.CHECKPOINT_DIR  
    data_name = "PTBXL" 

    def load_dict_weights(models_to_load, model_file_prefix):
        """
        辅助函数：从字典 checkpoint 中解包并加载权重
        models_to_load: { "key_在字典里的名称": model_实例 }
        """
        file_name = f"{model_file_prefix}_{data_name}_ep50.pth"
        path = os.path.join(checkpoint_dir, file_name)
        
        if os.path.exists(path):
            print(f" ✅ 正在加载权重文件: {file_name}")
            # 加载整个字典
            checkpoint = torch.load(path, map_location=device)
            
            for key, model_obj in models_to_load.items():
                if key in checkpoint:
                    model_obj.load_state_dict(checkpoint[key])
                    model_obj.eval() # 开启评估模式
                    print(f"   👉 子模块 '{key}' 加载成功")
                else:
                    print(f"   ⚠️ 警告：在 {file_name} 中找不到键 '{key}'，请检查保存时的 models_dict")
        else:
            print(f" ❌ 错误：找不到路径 {path}")
        
        # 返回加载后的模型（如果是多个则返回元组）
        res = list(models_to_load.values())
        return res[0] if len(res) == 1 else tuple(res)

    # --- 2. 按照你的文件名截图 逐个加载 ---

    # ECGRecover: 假设你保存时用的 key 是 "model"
    m_ecgrecover = load_dict_weights({"model": ECGRecover().to(device)}, "ECGRecover")

    # MaeFE: 假设你保存时用的 key 是 "model"
    m_maefe = load_dict_weights({"model": MaeFE().to(device)}, "MaeFE")

    # EKGAN: 假设你保存时用的 key 是 "generator"
    m_ekgan = load_dict_weights({"generator": EKGAN_Generator().to(device)}, "EKGAN")

    # DeScoD: 假设你保存时用的 key 是 "model"
    m_descod = load_dict_weights({"model": DeScoD_ScoreNet().to(device)}, "DeScoD")

    # Unet_Flow: 这是组合模型，假设保存时键为 "unet" 和 "flow"
    # 根据你的截图，文件名为 Unet_Flow_PTBXL_best.pth
    m_unet_flow = load_dict_weights({
        "unet": AdvancedUNet1D().to(device),
        "flow": FlowNetwork().to(device)
    }, "Unet_Flow")
    
    m_unet_moe_flow = load_dict_weights({
        "unet": AdvancedUNet1D().to(device),
        "moe_flow": MoEFlowNetwork().to(device)
    }, "Unet_MoE_Flow")

    return {
        "ECGRecover": m_ecgrecover,
        "MaeFE": m_maefe,
        "EKGAN": m_ekgan,
        "DeScoD": m_descod,
        "Unet_Flow": m_unet_flow, # 此时已是 (unet, flow) 元组
        "Unet_MoE_Flow": m_unet_moe_flow,
    }

# =============================================================
# 2. 评估核心逻辑 (通用框架，无需改动)
# =============================================================

def run_evaluation():
    device = Config.DEVICE
    models_dict = get_eval_models(device)
    all_results = []

    # 第一层循环：遍历数据集
    for ds_name, loader_func in DATASETS_TO_RUN.items():
        print(f"\n📊 正在评估数据集: {ds_name}")
        
        # 获取评估加载器
        # 注意：此处的 val_loader 样本数受 Config.EVALUATE_SAMPLE_LIMIT 限制
        val_loader, _ = loader_func() 

        # 第二层循环：遍历模型
        for model_name, model_obj in models_dict.items():
            print(f"  🔍 模型推理中: {model_name}")
            
            # 初始化该模型在此数据集下的各项指标容器
            perf_accumulator = {m_name: [] for m_name in METRICS_TO_CALC.keys()}

            for batch_x, batch_y in tqdm(val_loader, desc=f"    {model_name}"):
                # batch_x/y 形状为 (B, 12, 512)
                # 为了适配 predict，需转为 (B, 512, 12)
                input_np = batch_x.permute(0, 2, 1).numpy()
                
                with torch.no_grad():
                    # --- 分支处理：组合模型与普通模型 ---
                    if model_name in ["Unet_Flow", "Unet_MoE_Flow"]:
                        unet_m, flow_m = model_obj
                        # 第一阶段：UNet 重建
                        mid_res = unet_m.predict(input_np, device=device)
                        # 第二阶段：Flow 细化
                        pred_np = flow_m.predict(mid_res, steps=Config.SAMPLE_STEPS, device=device)
                    elif model_name == "DeScoD":
                        # 扩散模型推理
                        pred_np = model_obj.predict(input_np, steps=Config.DIFF_STEPS, device=device)
                    else:
                        # 通用单体模型预测
                        pred_np = model_obj.predict(input_np, device=device)

                # 将结果转回 (B, 12, 512) 以匹配计算指标时的 batch_y
                pred_torch = torch.from_numpy(pred_np).permute(0, 2, 1)
                
                # --- 第三层循环：动态计算所有指标 ---
                for m_name, m_func in METRICS_TO_CALC.items():
                    # 传入形状均为 (B, 12, 512)
                    val = m_func(batch_y, pred_torch)
                    perf_accumulator[m_name].append(val)

            # 汇总当前模型在当前数据集上的平均表现
            res_entry = {"Dataset": ds_name, "Model": model_name}
            for m_name in METRICS_TO_CALC.keys():
                res_entry[m_name] = np.mean(perf_accumulator[m_name])
            
            all_results.append(res_entry)

    # 3. 结果持久化
    df = pd.DataFrame(all_results)
    df.to_csv(Config.METRICS_CSV, index=False)
    
    print(f"\n✅ 评估完成！结果已存至: {Config.METRICS_CSV}")
    print("-" * 60)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_evaluation()