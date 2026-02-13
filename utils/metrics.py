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
# utils/metrics.py
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

def calculate_clinical_metrics(y_true, y_probs):
    """
    y_true: 真实标签索引 [N] (例如: [0, 1, 4...])
    y_probs: 模型预测概率矩阵 [N, 5]
    """
    # 1. 确保是 numpy 格式
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    # 2. 从概率矩阵中提取预测索引
    y_pred = np.argmax(y_probs, axis=1)
    
    # 3. 计算指标
    # Macro F1: 对不平衡类别最公平的衡量方式
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Cohen's Kappa: 衡量模型预测与真实值的一致性
    kappa = cohen_kappa_score(y_true, y_pred)
    
    return kappa, f1


from scipy.signal import find_peaks

def _to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data

def calculate_pre(real, pred, fs=100):
    """
    计算峰值相对误差 (Peak Relative Error)
    考虑到了 R 峰定位以及微小的相位偏移容错
    """
    real, pred = _to_numpy(real), _to_numpy(pred)
    batch_size, num_leads, _ = real.shape
    pres = []

    for b in range(batch_size):
        for l in range(num_leads):
            r_sig = real[b, l, :]
            p_sig = pred[b, l, :]

            # 1. 自动寻找真实 R 峰位置
            # 使用心电图典型的间距门限 (100Hz下 0.6s 约为 60 个点)
            peaks, _ = find_peaks(r_sig, distance=60, height=np.mean(r_sig) + np.std(r_sig))
            
            if len(peaks) == 0:
                peaks = [np.argmax(r_sig)]

            # 2. 对比峰值 (增加 3 个点的偏移窗口，取窗口内最大值，防止相位偏移导致误差虚高)
            lead_peak_errors = []
            for p in peaks:
                real_val = r_sig[p]
                
                # 预测信号在相同位置附近的搜索窗口
                win_start, win_end = max(0, p-2), min(len(p_sig), p+3)
                pred_val = np.max(p_sig[win_start:win_end])
                
                err = np.abs(real_val - pred_val) / (np.abs(real_val) + 1e-6)
                lead_peak_errors.append(err)
            
            pres.append(np.mean(lead_peak_errors))

    return np.mean(pres) * 100  # 返回百分比