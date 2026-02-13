# evaluate_consistency.py
import torch
from config import Config
import utils
from models.classifier import ECGClassifier
from models import AdvancedUNet1D, MoEFlowNetwork # 根据你实际导入名修改

@torch.no_grad()
def run_evaluation():
    device = Config.DEVICE
    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    # 1. 加载“医生”分类器
    doctor = ECGClassifier(12, 5).to(device)
    doctor.load_state_dict(torch.load('results/checkpoints/Classifier_Doctor_best.pth')['model'])
    doctor.eval()
    
    # 2. 加载你的生成模型 (Unet+Flow)
    unet = AdvancedUNet1D(12).to(device)
    flow = MoEFlowNetwork(12).to(device) # 如果是MoE版本
    checkpoint = torch.load('results/checkpoints/Unet_Moe_Flow_PTBXL_best.pth')
    unet.load_state_dict(checkpoint['unet'])
    flow.load_state_dict(checkpoint['moe_flow'])
    unet.eval(); flow.eval()

    # 3. 加载测试数据 (使用评估专用 Loader)
    val_loader, _ = utils.get_ptbxl_classifier_loaders() # 确保返回带标签的数据

    results = {}
    # 这里编写逻辑：分别获取 (1)原始信号 (2)重建信号 在 doctor 下的预测结果
    # ... (调用上述 metrics 和 visualizer 进行绘图) ...
    print("一致性实验评估完成！")

if __name__ == "__main__":
    run_evaluation()