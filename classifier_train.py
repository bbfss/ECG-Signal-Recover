# classifier_train.py
import torch
import numpy as np
import random
import os
from config import Config
import utils  
from utils.metrics import calculate_clinical_metrics 
from models.classifier import ECGClassifier 
from engine.classifier_trainer import ClassifierTrainer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    set_seed(Config.SEED)
    device = Config.DEVICE
    
    epochs = getattr(Config, 'CLASSIFIER_EPOCHS', 50)
    lr = getattr(Config, 'CLASSIFIER_LR', 1e-3)

    print(f"\n{'='*50}\n>> 启动任务: 医生模型训练 (单标签模式)\n{'='*50}\n")

    # 1. 加载数据
    (train_dl, val_dl, demo_sig) = utils.get_ptbxl_classifier_loaders()

    # 2. 初始化模型
    model = ECGClassifier(in_channels=Config.IN_CHANNELS, num_classes=5).to(device)
    model.name = "Classifier_Doctor"

    trainer = ClassifierTrainer(model, train_dl, val_dl, demo_sig)

    # 3. 训练循环
    best_f1 = -1.0
    for epoch in range(1, epochs + 1):
        # --- 训练阶段 ---
        train_result = trainer.train_epoch(epoch)
        
        # --- 修改点 1: 解析 train_loss 和 train_acc ---
        if isinstance(train_result, tuple):
            train_loss, train_acc = train_result
        else:
            train_loss = train_result
            train_acc = 0.0  # 如果 trainer 没返回 acc，默认为 0
        
        # --- 验证阶段 ---
        val_loss, y_true, y_probs = trainer.validate_raw() 
        
        # --- 修改点 2: 计算 Val Accuracy (单标签模式使用 argmax) ---
        y_pred = np.argmax(y_probs, axis=1)
        val_acc = np.mean(y_pred == y_true)
        
        # 获取其他临床指标
        kappa, f1 = calculate_clinical_metrics(y_true, y_probs)
        
        # --- 修改点 3: 更新打印输出 ---
        print(f"[{model.name}] Epoch {epoch}/{epochs}")
        print(f"      Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"      Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"      Macro F1:   {f1:.4f} | Cohen's Kappa: {kappa:.4f}")

        # 保存最优模型 (基于 F1 分数)
        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(Config.CHECKPOINT_DIR, f"{model.name}_best.pth")
            torch.save({'model': model.state_dict(), 'f1': f1}, save_path)
            print(f"   >> [Best] 已保存 (F1: {f1:.4f})")

    print(f"\n[!] 训练结束。最高 F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()