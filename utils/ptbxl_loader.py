# utils/ptbxl_loader.py
import os
import pandas as pd
import wfdb
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split

from config import Config
from .signal_utils import filtering, apply_transient_mask, apply_extended_mask, ECGDataset

def get_ptbxl_loaders():
    """
    针对 PTB-XL 的数据加载器。
    适配小写输入映射逻辑与 Config.DATA_PATHS 字典。
    """
    # 1. 获取路径
    data_root = Config.DATA_PATHS.get("PTBXL")
    if data_root is None or not os.path.exists(data_root):
        raise FileNotFoundError(f">> 路径不存在，请检查 config.py 中的 DATA_PATHS['PTBXL']: {data_root}")

    csv_path = os.path.join(data_root, "ptbxl_database.csv")
    meta = pd.read_csv(csv_path, index_col='ecg_id')
    
    # 2. 确定采样量
    limit = Config.SAMPLE_LIMIT if Config.SAMPLE_LIMIT is not None else len(meta)
    raw_data, count = [], 0
    # 优先读取 500Hz 版本的路径
    col = 'filename_hr' if 'filename_hr' in meta.columns else 'filename_lr'
    
    # 3. 批量读取原始信号
    pbar = tqdm(total=limit, desc="[Loader] Loading PTB-XL")
    for idx, row in meta.iterrows():
        if count >= limit: break
        try:
            fp = os.path.join(data_root, row[col])
            sig, _ = wfdb.rdsamp(fp)
            if not np.any(np.isnan(sig)): 
                raw_data.append(sig)
                count += 1
                pbar.update(1)
        except: continue
    pbar.close()
    
    # 4. 预处理 (使用补充后的 Config.PTBXL_FS)
    clean_signals = []
    for sig in tqdm(raw_data, desc="[Loader] Preprocessing"):
        # sig 形状 (L, 12) -> 独立对每个导联滤波
        processed = np.array([filtering(sig[:, j], Config.PTBXL_FS) for j in range(Config.IN_CHANNELS)]).T
        clean_signals.append(processed)
    
    # 5. 生成遮掩数据对 (Data Augmentation)
    inputs_list, targets_list = [], []
    for sig in tqdm(clean_signals, desc="[Loader] Generating Masks"):
        # 这里的 sig 是 (L, 12)
        
        # 策略 1: 瞬态掩码 (Transient)
        m_trans, _ = apply_transient_mask(sig, Config.MISSING_RATIO)
        inputs_list.append(m_trans.T)   # 模型期望 (Channels, Length)
        targets_list.append(sig.T)
        
        # 策略 2: 连续掩码 (Extended)
        m_ext, _ = apply_extended_mask(sig, Config.MISSING_RATIO)
        inputs_list.append(m_ext.T)
        targets_list.append(sig.T)
            
# 6. 封装 DataLoader
    X = np.array(inputs_list).astype(np.float32)
    Y = np.array(targets_list).astype(np.float32)
    
    dataset = ECGDataset(X, Y)
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    
    train_ds, val_ds = random_split(
        dataset, [train_len, val_len], 
        generator=torch.Generator().manual_seed(Config.SEED)
    )
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # --- 关键修正：返回符合 BaseTrainer 预期的原始 NumPy 信号 ---
    # 不要在这里做 Tensor 转换，交给 BaseTrainer 去做
    demo_sig = clean_signals[0] # 这是一个形状为 (512, 12) 的 numpy 数组
    
    print(f">> PTB-XL 加载完成: 训练样本 {len(train_ds)}, 验证样本 {len(val_ds)}")
    
    return train_loader, val_loader, demo_sig
def get_ptbxl_evaluate_loader():
    """
    针对 PTB-XL 的评估加载器：
    1. 使用 Config.EVALUATE_SAMPLE_LIMIT 限制最终样本总数。
    2. 不划分训练/验证集，直接返回全量数据的 val_loader。
    3. 返回值: (val_loader, demo_sig)。
    """
    # 1. 获取路径
    data_root = Config.DATA_PATHS.get("PTBXL")
    if data_root is None or not os.path.exists(data_root):
        raise FileNotFoundError(f">> 路径不存在，请检查 config.py: {data_root}")

    csv_path = os.path.join(data_root, "ptbxl_database.csv")
    meta = pd.read_csv(csv_path, index_col='ecg_id')
    
    # 2. 确定最终样本总数限制
    limit = getattr(Config, 'EVALUATE_SAMPLE_LIMIT', len(meta))
    
    inputs_list, targets_list = [], []
    col = 'filename_hr' if 'filename_hr' in meta.columns else 'filename_lr'
    
    # 3. 批量读取并处理信号，直到达到最终样本数限制
    pbar = tqdm(total=limit, desc="[Evaluate] Loading PTB-XL")
    for _, row in meta.iterrows():
        # 严格控制最终生成的样本数量
        if len(inputs_list) >= limit: 
            break
            
        try:
            fp = os.path.join(data_root, row[col])
            sig, _ = wfdb.rdsamp(fp)
            
            if not np.any(np.isnan(sig)): 
                # 信号预处理 (512, 12)
                processed = np.array([filtering(sig[:, j], Config.PTBXL_FS) for j in range(Config.IN_CHANNELS)]).T
                
                # 生成评估用的单一掩码 (1:1 映射，无增强循环)
                masked, _ = apply_transient_mask(processed, Config.MISSING_RATIO)
                
                # 存入列表，转置为模型卷积要求的 (12, 512)
                inputs_list.append(masked.T)
                targets_list.append(processed.T)
                pbar.update(1)
        except: 
            continue
    pbar.close()
    
    # --- 关键返回值 1：demo_sig (512, 12) ---
    # 取第一个原始信号并转置回 (512, 12) 适配 predict 函数
    demo_sig = targets_list[0].T if len(targets_list) > 0 else None

    # 4. 封装数据并转换为 Tensor
    X = np.array(inputs_list).astype(np.float32)
    Y = np.array(targets_list).astype(np.float32)
    dataset = ECGDataset(X, Y)
    
    # 评估模式：shuffle=False, 不进行 9:1 划分
    val_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    print(f">> PTB-XL 评估加载完成: 最终样本总数 {len(dataset)}")
    return val_loader, demo_sig