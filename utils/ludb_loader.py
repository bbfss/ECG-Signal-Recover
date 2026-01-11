import os
import numpy as np
import torch
import wfdb
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

# 导入配置与你写好的信号处理工具
from config import Config
from .signal_utils import filtering, apply_transient_mask, apply_extended_mask, ECGDataset

def load_ludb_data():
    """
    针对 LUDB 的原始数据加载逻辑。
    """
    data_dir = Config.DATA_PATHS.get("LUDB")
    if data_dir is None or not os.path.exists(data_dir):
        data_dir = os.path.join(Config.BASE_DIR, "data", "ludb-1.0.1")
        
    fs = Config.LUDB_FS
        
    raw_data = []
    # LUDB 包含 200 条记录
    for i in tqdm(range(1, 201), desc="[Loader] Loading LUDB"):
        try:
            file_prefix = os.path.join(data_dir, str(i))
            if not os.path.exists(file_prefix + ".dat"):
                continue

            sig, _ = wfdb.rdsamp(file_prefix)

            if not np.any(np.isnan(sig)):
                raw_data.append(sig)
        except Exception:
            continue

    if not raw_data:
        raise RuntimeError(f">> 路径 {data_dir} 下未找到 LUDB 数据文件")

    return raw_data, fs

def get_ludb_evaluate_loader():
    """
    针对 LUDB 的评估加载器：
    1. 使用 Config.EVALUATE_SAMPLE_LIMIT 限制最终样本总数。
    2. 调用你写好的 apply_transient_mask 函数。
    3. 返回值: (val_loader, demo_sig)。
    """
    raw, fs = load_ludb_data()
    
    # 确定最终样本总数限制
    limit = getattr(Config, 'EVALUATE_SAMPLE_LIMIT', len(raw))
    
    inputs_list, targets_list = [], []
    
    pbar = tqdm(total=limit, desc="[Evaluate] Loading LUDB")
    for sig in raw:
        if len(inputs_list) >= limit:
            break
            
        # 滤波预处理 (512, 12)
        processed = np.array([filtering(sig[:, j], fs) for j in range(12)]).T
        
        # --- 调用你写好的遮掩函数 ---
        # 评估模式下 1:1 生成，使用瞬态掩码策略
        masked, _ = apply_transient_mask(processed, Config.MISSING_RATIO)
        
        # 存入列表，转置为模型要求的 (12, 512)
        inputs_list.append(masked.T)
        targets_list.append(processed.T)
        pbar.update(1)
    pbar.close()
    
    # demo_sig 形状为 (512, 12)
    demo_sig = targets_list[0].T if len(targets_list) > 0 else None

    X = np.array(inputs_list).astype(np.float32)
    Y = np.array(targets_list).astype(np.float32)
    dataset = ECGDataset(X, Y)
    
    val_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    print(f">> LUDB 评估加载完成: 最终样本总数 {len(dataset)}")
    return val_loader, demo_sig

def get_ludb_loaders():
    """
    标准的训练/验证加载器 (包含 9:1 划分与数据增强)。
    """
    raw, fs = load_ludb_data()
    clean_signals = []
    for sig in tqdm(raw, desc="[Loader] Preprocessing"):
        processed = np.array([filtering(sig[:, j], fs) for j in range(12)]).T
        clean_signals.append(processed)
    
    demo_sig = clean_signals[0] 

    augment_factor = Config.LUDB_AUGMENT_FACTOR
    
    inputs_list, targets_list = [], []
    for sig in tqdm(clean_signals, desc="[Loader] Generating Masks"):
        for _ in range(augment_factor):
            # 训练模式下调用你写好的两种掩码进行增强
            m_trans, _ = apply_transient_mask(sig, Config.MISSING_RATIO)
            inputs_list.append(m_trans.T)
            targets_list.append(sig.T)
            
            m_ext, _ = apply_extended_mask(sig, Config.MISSING_RATIO)
            inputs_list.append(m_ext.T)
            targets_list.append(sig.T)

    dataset = ECGDataset(np.array(inputs_list).astype(np.float32), np.array(targets_list).astype(np.float32))
    
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    
    train_ds, val_ds = random_split(
        dataset, [train_len, val_len], 
        generator=torch.Generator().manual_seed(Config.SEED)
    )
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, demo_sig