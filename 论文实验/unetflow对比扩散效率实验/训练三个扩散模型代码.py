import torch
import numpy as np
import random
import os
import sys

# --- 1. 核心路径定位 (确保任意位置运行) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import Config
import utils  

# 导入模型
from models import AdvancedUNet1D, FlowNetwork, DeScoD_ScoreNet
from models.unet_normal_flow import NormalFlowNetwork

# 导入 Trainer
from engine import Unet_Flow_Trainer, DeScoD_Trainer

# 强制保存目录
EXPERIMENT_SAVE_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
os.makedirs(EXPERIMENT_SAVE_DIR, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    set_seed(Config.SEED)
    device = Config.DEVICE
    data_name = "PTBXL" # 默认使用 PTBXL

    print(f"\n{'='*60}")
    print(f"🚀 启动自动化对比实验框架")
    print(f"📍 实验目录: {SCRIPT_DIR}")
    print(f"{'='*60}\n")

    # 1. 预加载数据集 (只需加载一次，节省时间)
    print("📦 正在加载数据集...")
    (train_dl, val_dl, demo_sig) = utils.get_ptbxl_loaders()

    # 2. 定义待训练的模型列表
    # 格式: (模型内部标识, 打印名称, 训练逻辑函数)
    tasks = [
        ("DeScoD", "Diffusion Baseline"),
        ("ECG_Unet_Normal_Flow", "Normal Flow (Ablation)"),
        ("Unet_Flow", "Advanced Flow (Ours)")
    ]

    for std_name, desc in tasks:
        print(f"\n检查任务: {desc} ({std_name})")
        
        # 检查是否已存在
        best_path = os.path.join(EXPERIMENT_SAVE_DIR, f"{std_name}_{data_name}_best.pth")
        if os.path.exists(best_path):
            print(f"✅ [跳过] 该模型已存在训练好的权重，路径: {best_path}")
            continue

        print(f"🔥 开始训练: {desc}...")
        
        # 3. 根据模型名称初始化具体的模型和 Trainer
        if std_name == "DeScoD":
            model = DeScoD_ScoreNet(in_channels=Config.IN_CHANNELS).to(device)
            trainer = DeScoD_Trainer(model, train_dl, val_dl, demo_sig)
            models_dict = {'model': model}
            total_epochs = Config.DIFFUSION_EPOCHS

        elif std_name == "ECG_Unet_Normal_Flow":
            m_u = AdvancedUNet1D(in_channels=Config.IN_CHANNELS).to(device)
            m_f = NormalFlowNetwork(channels=Config.IN_CHANNELS).to(device)
            trainer = Unet_Flow_Trainer(m_u, m_f, train_dl, val_dl, demo_sig)
            models_dict = {'unet': m_u, 'normal_flow': m_f}
            total_epochs = Config.UNET_EPOCHS + Config.FLOW_EPOCHS

        elif std_name == "Unet_Flow":
            m_u = AdvancedUNet1D(in_channels=Config.IN_CHANNELS).to(device)
            m_f = FlowNetwork(channels=Config.IN_CHANNELS).to(device)
            trainer = Unet_Flow_Trainer(m_u, m_f, train_dl, val_dl, demo_sig)
            models_dict = {'unet': m_u, 'flow': m_f}
            total_epochs = Config.UNET_EPOCHS + Config.FLOW_EPOCHS

        # 4. 执行训练循环
        best_pcc = -1.0
        for epoch in range(1, total_epochs + 1):
            loss = trainer.train_epoch(epoch)
            mae, rmse, pcc = trainer.validate(epoch)
            
            # 实时进度条风格打印
            print(f"  > Epoch {epoch}/{total_epochs} | Loss: {loss:.4f} | PCC: {pcc:.4f}")

            # 定期保存
            if epoch % 10 == 0 or epoch == total_epochs:
                trainer.visualize(epoch)
                save_path = os.path.join(EXPERIMENT_SAVE_DIR, f"{std_name}_{data_name}_ep{epoch}.pth")
                torch.save({k: v.state_dict() for k, v in models_dict.items()}, save_path)

            if pcc > best_pcc:
                best_pcc = pcc
                torch.save({k: v.state_dict() for k, v in models_dict.items()}, best_path)
                print(f"    🌟 [New Best] PCC: {pcc:.4f}")

        # 5. 每个模型练完后释放显存，防止 OOM
        del trainer, models_dict
        torch.cuda.empty_cache()
        print(f"✨ {desc} 训练完成。\n")

    print(f"{'='*60}")
    print("🏁 所有对比实验任务已处理完毕！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()