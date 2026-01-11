import torch
import numpy as np
import random
import os
import argparse
import logging

from config import Config
import utils  # 确保包含 get_ptbxl_loaders 等

from models import (
    AdvancedUNet1D, FlowNetwork, MoEFlowNetwork,  # 确保导入的是 MoE 版本
    MaeFE,
    EKGAN_Generator, EKGAN_Discriminator,
    DeScoD_ScoreNet, 
    ECGRecover,
)
from engine import (
    Unet_Flow_Trainer, MAEFETrainer,
    GAN_Trainer, DeScoD_Trainer, ECG_Recover_Trainer, Unet_Moe_Flow_Trainer
)

# ==========================================================
# 映射字典：处理“小写、无横杠”输入 -> “标准大小写”名称
# ==========================================================
MODEL_MAP = {
    "unetflow": "Unet_Flow",
    "maefe": "MaeFE",
    "ekgan": "EKGAN",
    "descod": "DeScoD",
    "ecgrecover": "ECGRecover",
    "unet": "Unet_Baseline",
    "unetmoeflow": "Unet_Moe_Flow", # 建议增加这个，区分之前的版本
}

DATASET_MAP = {
    "ptbxl": "PTBXL",
    "mitbih": "MITBIH"
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_clean_key(input_str):
    """去除横杠、下划线并转小写"""
    return input_str.lower().replace("-", "").replace("_", "").replace(" ", "")

# ==========================================================
# 工厂函数
# ==========================================================
def get_dataset(data_key):
    std_name = DATASET_MAP.get(data_key)
    if std_name == "PTBXL":
        return utils.get_ptbxl_loaders(), std_name
    elif std_name == "MITBIH":
        # return utils.get_mitbih_loaders(), std_name
        raise NotImplementedError("MITBIH 数据集加载函数尚未在 utils 中实现")
    else:
        raise ValueError(f"不支持的数据集简写: {data_key}")

def get_trainer_logic(model_key, train_dl, val_dl, demo_sig):
    std_name = MODEL_MAP.get(model_key)
    device = Config.DEVICE
    models_dict = {}

    # 修改此处的逻辑
    if std_name == "Unet_Moe_Flow":
        # 1. 初始化基础 UNet
        m_u = AdvancedUNet1D(in_channels=Config.IN_CHANNELS).to(device)
        
        # 2. 初始化专家混合流网络 MoEFlowNetwork
        m_f = MoEFlowNetwork(channels=Config.IN_CHANNELS).to(device) 
        
        # 3. 初始化对应的 Unet_Moe_Flow_Trainer
        trainer = Unet_Moe_Flow_Trainer(m_u, m_f, train_dl, val_dl, demo_sig)
        
        # 4. 这里的 key 会作为保存 checkpoint 时的文件名后缀
        models_dict = {'unet': m_u, 'moe_flow': m_f} 
        
        # 5. 训练总轮数 (UNet预训练 + Flow微调)
        epochs = Config.UNET_EPOCHS + Config.FLOW_EPOCHS
        
    elif std_name == "Unet_Flow":
        m_u = AdvancedUNet1D(in_channels=Config.IN_CHANNELS).to(device)
        m_f = FlowNetwork(channels=Config.IN_CHANNELS).to(device)
        trainer = Unet_Flow_Trainer(m_u, m_f, train_dl, val_dl, demo_sig)
        models_dict = {'unet': m_u, 'flow': m_f}
        epochs = Config.UNET_EPOCHS + Config.FLOW_EPOCHS

    elif std_name == "MaeFE":
        m = MaeFE(seq_len=Config.SEQ_LEN, patch_size=Config.PATCH_SIZE, 
                  in_chans=Config.IN_CHANNELS, embed_dim=Config.EMBED_DIM).to(device)
        trainer = MAEFETrainer(m, train_dl, val_dl, demo_sig)
        models_dict = {'model': m}
        epochs = Config.MAE_EPOCHS

    elif std_name == "EKGAN":
        gen = EKGAN_Generator(in_chan=Config.IN_CHANNELS).to(device)
        disc = EKGAN_Discriminator(in_chan=Config.IN_CHANNELS).to(device)
        trainer = GAN_Trainer(gen, disc, train_dl, val_dl, demo_sig)
        models_dict = {'gen': gen, 'disc': disc}
        epochs = Config.PRETRAIN_EPOCHS + Config.GAN_EPOCHS

    elif std_name == "DeScoD":
        m = DeScoD_ScoreNet(in_channels=Config.IN_CHANNELS).to(device)
        trainer = DeScoD_Trainer(m, train_dl, val_dl, demo_sig)
        models_dict = {'model': m}
        epochs = Config.DIFFUSION_EPOCHS

    elif std_name == "ECGRecover":
        m = ECGRecover(in_channels=Config.IN_CHANNELS).to(device)
        trainer = ECG_Recover_Trainer(m, train_dl, val_dl, demo_sig)
        models_dict = {'model': m}
        epochs = Config.RECOVER_EPOCHS
    else:
        raise ValueError(f"无法识别的模型简写: {model_key}")

    return trainer, models_dict, epochs, std_name

# ==========================================================
# 主程序
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="ECG 训练框架 (支持小写/无横杠输入)")
    
    # 这里的 help 会提示可以输入的选项
    model_choices = list(MODEL_MAP.keys())
    dataset_choices = list(DATASET_MAP.keys())

    parser.add_argument('--model', type=str, default='maefe', 
                        choices=model_choices,
                        help=f"模型选择，可选: {', '.join(model_choices)}")
    
    parser.add_argument('--dataset', type=str, default='ptbxl', 
                        choices=dataset_choices,
                        help=f"数据集选择，可选: {', '.join(dataset_choices)}")

    args = parser.parse_args()
    set_seed(Config.SEED)

    # 转换输入
    m_key = get_clean_key(args.model)
    d_key = get_clean_key(args.dataset)

    # 1. 初始化数据
    (train_dl, val_dl, demo_sig), std_data_name = get_dataset(d_key)

    # 2. 初始化模型
    trainer, models_dict, total_epochs, std_model_name = get_trainer_logic(m_key, train_dl, val_dl, demo_sig)

    # 打印时会自动显示标准名称 (区分大小写)
    print(f"\n{'='*50}")
    print(f">> 启动任务: {std_model_name}")
    print(f">> 使用数据: {std_data_name}")
    print(f">> 训练设备: {Config.DEVICE}")
    print(f"{'='*50}\n")

    best_pcc = -1.0
    for epoch in range(1, total_epochs + 1):
        loss = trainer.train_epoch(epoch)
        mae, rmse, pcc = trainer.validate(epoch)
        
        # 实时打印
        print(f"[{std_model_name}] Epoch {epoch}/{total_epochs} | Loss: {loss:.4f} | MAE: {mae:.4f} | PCC: {pcc:.4f}")

        # 可视化与保存
        if epoch % 10 == 0 or epoch == total_epochs:
            trainer.visualize(epoch)
            save_path = os.path.join(Config.CHECKPOINT_DIR, f"{std_model_name}_{std_data_name}_ep{epoch}.pth")
            torch.save({k: v.state_dict() for k, v in models_dict.items()}, save_path)

        if pcc > best_pcc:
            best_pcc = pcc
            best_path = os.path.join(Config.CHECKPOINT_DIR, f"{std_model_name}_{std_data_name}_best.pth")
            torch.save({k: v.state_dict() for k, v in models_dict.items()}, best_path)
            print(f"   >> [Best] 已保存性能最优模型 (PCC: {pcc:.4f})")

if __name__ == "__main__":
    main()