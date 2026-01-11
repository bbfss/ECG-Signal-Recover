# utils/metrics.py
import numpy as np
import torch
from scipy.stats import pearsonr

def _to_numpy(data):
    """辅助函数：统一将输入转为 Numpy 数组，确保计算通用性"""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data

def calculate_mae(real, pred):
    """
    计算平均绝对误差 (Mean Absolute Error)
    """
    real, pred = _to_numpy(real), _to_numpy(pred)
    return np.mean(np.abs(real - pred))

def calculate_rmse(real, pred):
    """
    计算均方根误差 (Root Mean Square Error)
    """
    real, pred = _to_numpy(real), _to_numpy(pred)
    return np.sqrt(np.mean((real - pred) ** 2))

def calculate_pcc(real, pred):
    """
    计算皮尔逊相关系数 (Pearson Correlation Coefficient)
    逐个样本、逐个导联计算后取平均值
    """
    real, pred = _to_numpy(real), _to_numpy(pred)
    batch_size, num_leads, _ = real.shape
    pccs = []
    
    for b in range(batch_size):
        for l in range(num_leads):
            r_sig = real[b, l, :]
            p_sig = pred[b, l, :]
            
            # 鲁棒性检查：如果信号平直（标准差为0），相关系数无意义，填0
            if np.std(r_sig) < 1e-6 or np.std(p_sig) < 1e-6:
                pccs.append(0.0)
                continue
            
            corr, _ = pearsonr(r_sig, p_sig)
            pccs.append(corr if not np.isnan(corr) else 0.0)
            
    return np.mean(pccs)

def calculate_prd(real, pred):
    """
    计算百分比均方根差异 (Percentage Root-mean-square Difference)
    逐个样本、逐个导联计算后取平均值
    """
    real, pred = _to_numpy(real), _to_numpy(pred)
    batch_size, num_leads, _ = real.shape
    prds = []
    
    for b in range(batch_size):
        for l in range(num_leads):
            r_sig = real[b, l, :]
            p_sig = pred[b, l, :]
            
            numerator = np.sum((r_sig - p_sig) ** 2)
            denominator = np.sum(r_sig ** 2)
            
            # 鲁棒性检查：防止除以0
            if denominator < 1e-8:
                prds.append(0.0)
            else:
                prd = np.sqrt(numerator / denominator) * 100
                prds.append(prd)
                
    return np.mean(prds)