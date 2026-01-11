# config.py
import os
import torch

class Config:
    # ================= 1. 路径与多数据集配置 =================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 数据集根目录字典
    DATA_PATHS = {
        "PTBXL": os.path.join(BASE_DIR, "data", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"),
        "LUDB": os.path.join(BASE_DIR, "data", "lobachevsky-university-electrocardiography-database-1.0.1", "data"),
    }
    
    # 结果输出
    OUTPUT_DIR = os.path.join(BASE_DIR, "results")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    PLOT_DIR = os.path.join(OUTPUT_DIR, "vis_plots")
    METRICS_CSV = os.path.join(OUTPUT_DIR, "experiment_metrics.csv")
    # ================= 2. 全局通用训练参数 =================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    LR = 1e-3
    
    # --- 信号通用参数 ---
    PTBXL_FS = 500       # PTB-XL 高增益版本采样率为 500Hz
    LUDB_FS = 500       # LUDB 采样率为 500Hz
    SEQ_LEN = 512        # 训练送入网络的长度
    IN_CHANNELS = 12     # 默认 12 导联
    MISSING_RATIO = 0.3  # 掩码比例
    SAMPLE_LIMIT = 1000    # 建议测试阶段设为 10-100，正式运行设为 None
    EVALUATE_SAMPLE_LIMIT = 100  # 评估时的样本数量限制

    # ================= 3. 算法特有超参数 =================
    
    # --- Unet_Flow ---
    UNET_EPOCHS = 40
    FLOW_EPOCHS = 10
    SAMPLE_STEPS = 30    
    
    # --- EKGAN ---
    PRETRAIN_EPOCHS = 20
    GAN_EPOCHS = 30
    GAN_LR = 2e-4
    LAMBDA_PIXEL = 100   
    
    # --- DeScoD ---
    DIFFUSION_EPOCHS = 50
    SIGMA_MIN = 0.01
    SIGMA_MAX = 1.0
    DIFF_STEPS = 50
    
    # --- MaeFE ---
    PATCH_SIZE = 16
    EMBED_DIM = 256
    MAE_EPOCHS = 50

    # --- ECGRecover ---
    RECOVER_EPOCHS = 50

# 自动化目录创建
for path in [Config.CHECKPOINT_DIR, Config.PLOT_DIR]:
    os.makedirs(path, exist_ok=True)