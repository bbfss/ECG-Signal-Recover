# engine/classifier_trainer.py
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class ClassifierTrainer:
    def __init__(self, model, train_loader, val_loader, demo_sig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.demo_sig = demo_sig
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 单标签多分类使用 CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    # 在 engine/classifier_trainer.py 中
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # --- 新增：计算训练准确率 ---
            total_loss += loss.item()
            # 假设是单标签分类，使用 argmax
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct / total
        
        # --- 关键：返回元组 ---
        return avg_loss, avg_acc

    def validate_raw(self):
        self.model.eval()
        all_probs = []
        all_trues = []
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device).long()
                
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # 获取概率 (Softmax)
                probs = torch.softmax(logits, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_trues.append(labels.cpu().numpy())
                
        return (
            total_loss / len(self.val_loader),
            np.concatenate(all_trues),  # 解决维度不等报错
            np.concatenate(all_probs)
        )