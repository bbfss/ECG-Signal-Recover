# utils/ptbxl_loader.py
import os
import pandas as pd
import wfdb
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split

from config import Config
from .signal_utils import filtering, apply_transient_mask, apply_extended_mask, ECGDataset,apply_diagonal_mask

import ast # 确保在文件顶部添加这个导入
from torch.utils.data import TensorDataset

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

def get_diagonal_masked_ptbxl_loader():
    """
    针对 PTB-XL 数据集的对角线掩码评估加载器（分类返回）。

    核心逻辑：
    1. 从 PTB-XL 读取信号并进行滤波预处理。
    2. 调用 apply_diagonal_mask 生成 5 种不同密度的对角线掩码（12, 6, 4, 3, 2 等分）。
    3. 将不同切割方案的数据分别打包，返回一个包含 5 个 DataLoader 的字典。

    返回:
    - loaders (dict): { 'mask_12': loader, 'mask_6': loader, ... }
                      Key 中的数字代表信号被切成了多少段。
    """
    
    # --- 1. 初始化路径与配置 ---
    data_root = Config.DATA_PATHS.get("PTBXL")
    if data_root is None or not os.path.exists(data_root):
        raise FileNotFoundError(f">> 路径不存在，请检查 config.py: {data_root}")

    # 读取数据库索引文件
    csv_path = os.path.join(data_root, "ptbxl_database.csv")
    meta = pd.read_csv(csv_path, index_col='ecg_id')
    
    # 重要：根据 apply_diagonal_mask 的逻辑，约数 [1, 2, 3, 4, 6] 
    # 对应的“等分份数”分别是 [12, 6, 4, 3, 2]
    group_configs = [12, 6, 4, 3, 2] 
    
    # 初始化数据桶：为每种切割方案准备独立的列表
    storage = {n: {"x": [], "y": []} for n in group_configs}
    
    # 限制读取的原始信号数量
    limit = getattr(Config, 'EVALUATE_SAMPLE_LIMIT', len(meta))
    col = 'filename_hr' if 'filename_hr' in meta.columns else 'filename_lr'

    # --- 2. 批量读取并分发信号 ---
    pbar = tqdm(total=limit, desc="[Evaluate] Categorizing PTB-XL")
    processed_count = 0
    
    for _, row in meta.iterrows():
        if processed_count >= limit: 
            break

        try:
            # 读取生理信号文件
            fp = os.path.join(data_root, row[col])
            sig, _ = wfdb.rdsamp(fp)
            
            # 排除含 NaN 的无效信号
            if not np.any(np.isnan(sig)): 
                # 信号预处理：逐导联滤波并转置回 [512, 12]
                processed = np.array([filtering(sig[:, j], Config.PTBXL_FS) for j in range(Config.IN_CHANNELS)]).T
                
                # 生成 5 种掩码方案
                # original_labels: [sig, sig, sig, sig, sig] (5个相同的原始信号)
                # masked_samples:  [mask12, mask6, mask4, mask3, mask2] (5个不同密度的掩码)
                original_labels, masked_samples = apply_diagonal_mask(processed)
                
                # 根据索引 idx，将数据塞进对应的“份数桶”里
                for idx, n_groups in enumerate(group_configs):
                    # 转置为 (12, 512) 以适配 PyTorch 卷积层输入 [B, C, L]
                    storage[n_groups]["x"].append(masked_samples[idx].T)
                    storage[n_groups]["y"].append(original_labels[idx].T)
                
                processed_count += 1
                pbar.update(1)
        except Exception: 
            # 遇到读取失败的文件直接跳过
            continue
    pbar.close()

    # --- 3. 封装为 DataLoader 字典 ---
    loaders = {}
    for n in group_configs:
        # 转换为 Numpy 数组再转为 Tensor
        X = np.array(storage[n]["x"]).astype(np.float32)
        Y = np.array(storage[n]["y"]).astype(np.float32)
        
        dataset = ECGDataset(X, Y)
        
        # 评估模式下通常不需要打乱顺序 (shuffle=False)
        loaders[f"mask_{n}"] = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    print(f">> PTB-XL 加载完成: 包含方案 {list(loaders.keys())}")
    return loaders


import os
import ast
import pandas as pd
import numpy as np
import wfdb
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_ptbxl_classifier_loaders():
    """
    非 Fold 模式的加载器：
    1. 按照 NORM(44):MI(25):STTC(24):CD(22):HYP(12) 比例归一化抽样。
    2. 使用 Config.SAMPLE_LIMIT 控制总数。
    3. 随机 9:1 划分训练集和验证集。
    """
    # 1. 初始化路径与基础映射
    data_root = Config.DATA_PATHS.get("PTBXL")
    csv_path = os.path.join(data_root, "ptbxl_database.csv")
    meta = pd.read_csv(csv_path, index_col='ecg_id')

    subclass_map = {
        'NORM': 'NORM',
        'IMI': 'MI', 'ASMI': 'MI', 'AMI': 'MI', 'ALMI': 'MI', 'INJAS': 'MI', 'LMI': 'MI', 'INJAL': 'MI', 'IPMI': 'MI',
        'CLBBB': 'CD', 'CRBBB': 'CD', 'ILBBB': 'CD', 'IRBBB': 'CD', 'IVCD': 'CD', 'LAFB': 'CD', 'LPFB': 'CD', 'WPW': 'CD',
        'STTC': 'STTC', 'NST_': 'STTC', 'ISC_': 'STTC', 'ISCA': 'STTC', 'ISCI': 'STTC', 'ISCL': 'STTC',
        'LVH': 'HYP', 'RVH': 'HYP', 'SEV': 'HYP', 'LAO/LAE': 'HYP', 'RAO/RAE': 'HYP'
    }
    label_to_idx = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}

    # 2. 标签解析 (多标签转单标签，优先级：异常 > 正常)
    def aggregate_diagnostic(scp_codes_str):
        scp_dict = ast.literal_eval(scp_codes_str)
        found_superclasses = set()
        for code in scp_dict.keys():
            if code in subclass_map:
                found_superclasses.add(subclass_map[code])
        for category in ['MI', 'STTC', 'CD', 'HYP']:
            if category in found_superclasses: return label_to_idx[category]
        if 'NORM' in found_superclasses: return label_to_idx['NORM']
        return -1

    meta['label'] = meta.scp_codes.apply(aggregate_diagnostic)
    meta = meta[meta.label != -1]

    # 3. 按照你要求的比例执行归一化抽样
    # 原始比例之和为 44+25+24+22+12 = 127
    raw_ratios = {0: 44, 1: 25, 2: 24, 3: 22, 4: 12}
    total_parts = sum(raw_ratios.values())
    
    limit = Config.SAMPLE_LIMIT if Config.SAMPLE_LIMIT is not None else len(meta)
    
    print(f"[Strategy] 正在按比例 ({list(raw_ratios.values())}) 抽样，总数限制: {limit}")
    
    balanced_meta_list = []
    for idx, count_part in raw_ratios.items():
        subset = meta[meta.label == idx]
        # 计算该类别在 limit 中应占的数量
        n_to_draw = int(limit * (count_part / total_parts))
        n_to_draw = min(len(subset), n_to_draw) # 防止超过实际可用数量
        
        if n_to_draw > 0:
            balanced_meta_list.append(subset.sample(n=n_to_draw, random_state=Config.SEED))
    
    # 合并并打乱
    meta_final = pd.concat(balanced_meta_list).sample(frac=1, random_state=Config.SEED)

    # 4. 读取信号逻辑
    signals, labels = [], []
    # 根据频率选择文件名列
    col = 'filename_hr' if Config.PTBXL_FS == 500 else 'filename_lr'
    
    pbar = tqdm(total=len(meta_final), desc="[Loading Signals]")
    for _, row in meta_final.iterrows():
        try:
            fp = os.path.join(data_root, row[col])
            sig, _ = wfdb.rdsamp(fp) # sig shape: (Length, Channels)
            
            if not np.any(np.isnan(sig)):
                from .signal_utils import filtering
                # 截取长度
                sig_trimmed = sig[:Config.SEQ_LEN, :]
                # 预处理/滤波
                processed = np.zeros_like(sig_trimmed)
                for j in range(sig_trimmed.shape[1]):
                    processed[:, j] = filtering(sig_trimmed[:, j], Config.PTBXL_FS)
                
                signals.append(processed.T) # 存为 (C, L)
                labels.append(row['label'])
                pbar.update(1)
        except:
            continue
    pbar.close()

    # 5. 转换为 Tensor 并进行 9:1 随机切分
    X = np.array(signals).astype(np.float32)
    Y = np.array(labels).astype(np.int64)
    
    # 使用 sklearn 进行随机划分
    x_train, x_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.1, random_state=Config.SEED, stratify=Y
    )

    # 6. 封装 Loader
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    print(f">> 加载完毕！")
    print(f">> 总有效样本: {len(Y)}")
    print(f">> 训练集: {len(y_train)} | 验证集: {len(y_val)}")
    print(f">> 最终类别分布: {np.bincount(Y)}")

    return DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True), \
           DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False), \
           x_train[0]