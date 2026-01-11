# utils / data_utils.py
import numpy as np
import torch
import os
import pandas as pd
import wfdb
from scipy import signal
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from config import Config  # 导入配置类

# 信号的零均值 - 最大绝对值归一化
def normalization(signal_data):
    signal_data = np.nan_to_num(signal_data)
    signal_data = signal_data - np.mean(signal_data)
    max_val = np.max(np.abs(signal_data))
    if max_val < 1e-6: return np.zeros_like(signal_data)
    return (signal_data / max_val).astype(np.float32)

# 带通滤波 + 重采样
def filtering(ecg_signal, fs):
    """
    fs: 传入当前数据集的原始采样频率
    """
    nyquist = 0.5 * fs
    # 动态计算巴特沃斯滤波器的参数
    low = 0.05 / nyquist
    high = min(0.99, 150 / nyquist) 
    
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.lfilter(b, a, np.nan_to_num(ecg_signal))
    
    # 关键点：无论原始 fs 是多少，最终都重采样到 Config.SEQ_LEN，
    # 这样才能保证输入到神经网络的维度是一致的
    resampled = signal.resample(filtered, Config.SEQ_LEN)
    return normalization(resampled)

# 瞬态掩码 (采样点遮掩)
def apply_transient_mask(ecg_signal, missing_ratio=Config.MISSING_RATIO):
    seq_len, num_leads = ecg_signal.shape
    mask = np.ones_like(ecg_signal)
    block_size = 4 
    num_blocks = seq_len // block_size
    num_mask_blocks = int(num_blocks * missing_ratio)
    for l in range(num_leads):
        mask_idx = np.random.choice(num_blocks, num_mask_blocks, replace=False)
        for idx in mask_idx:
            start = idx * block_size
            mask[start : start + block_size, l] = 0
    return (ecg_signal * mask).astype(np.float32), mask

# 连续掩码 (长片段遮掩)
def apply_extended_mask(ecg_signal, missing_ratio=Config.MISSING_RATIO):
    seq_len, num_leads = ecg_signal.shape
    mask = np.ones_like(ecg_signal)
    mask_len = int(seq_len * missing_ratio)
    for l in range(num_leads):
        if seq_len > mask_len:
            start_point = np.random.randint(0, seq_len - mask_len)
            mask[start_point : start_point + mask_len, l] = 0
        else:
            mask[:, l] = 0
    return (ecg_signal * mask).astype(np.float32), mask


class ECGDataset(Dataset):
    def __init__(self, x, y): 
        self.x = torch.from_numpy(x).float() # 输入 (被遮掩的信号)
        self.y = torch.from_numpy(y).float() # 标签 (原始完整信号)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]