# ECG Signal Reconstruction & Generative Framework

这是一个集成了多种深度学习算法（MAE, GAN, Diffusion, UNet）的 **12 导联 ECG 信号重建与生成** 实验框架，旨在提供高效、可扩展的 ECG 信号处理实验平台。

---

## 📂 项目目录结构

```text

ECG_Project/                # 工程根目录 (你运行命令的地方)
├── data/                   # 【数据仓】存放所有原始和预处理数据
│   └── ptbxl/              # PTB-XL 数据集文件夹
├── models/                 # 【模型库】仅定义神经网络结构
│   ├── __init__.py         
│   ├── maefe.py            # MaeFE 模型定义
│   ├── ekgan.py            # EKGAN 模型定义
│   ├── descod.py           # DeScoD 模型定义
│   └── ...                 # 其他模型
├── engine/                 # 【执行引擎】定义训练和验证逻辑
│   ├── __init__.py         
│   ├── base_trainer.py     # 训练器基类
│   ├── mae_trainer.py      # MaeFE 训练器
│   ├── gan_trainer.py      # GAN 训练器
│   └── ...                 # 其他训练器
├── utils/                  # 【工具箱】存放纯函数插件
│   ├── __init__.py         
│   ├── metrics.py          # 指标计算（如 PCC、MSE 等）
│   ├── ptbxl_loader.py     # PTB-XL 数据集加载与预处理
│   └── visualizer.py       # 12 导联信号可视化逻辑
├── results/                # 【输出结果】运行后自动生成
│   ├── checkpoints/        # 存放模型权重 (.pth)
│   └── vis_plots/          # 存放生成的信号对比效果图
├── config.py               # 【全局配置】管理路径、超参数等
└── train.py                # 【启动脚本】主程序入口（统一启动所有模型）
```

---

## 🚀 快速开始 (Quick Start)

### 1. 进入工程目录

打开终端（PowerShell 或 CMD），运行以下命令进入代码文件夹：

```bash

cd E:\Code\StudyNotes\ECG工程文件
```

### 2. 激活环境

激活已配置好依赖的 Conda 虚拟环境：

```bash

conda activate pytorch
```

### 3. 执行实验指令

通过 `--model` 参数指定算法简写，`--dataset` 指定数据集（当前仅支持 ptbxl）。
提示：模型简写已适配全小写、无横杠输入（如 unetflow）。

#### 算法模型运行指令（直接复制）

- **MaeFE**`python train.py --model maefe --dataset ptbxl`

- **EKGAN**`python train.py --model ekgan --dataset ptbxl`

- **Unet + Flow**`python train.py --model unetflow --dataset ptbxl`

- **DeScoD (扩散模型)**`python train.py --model descod --dataset ptbxl`

- **ECGRecover**`python train.py --model ecgrecover --dataset ptbxl`

---

## 🛠️ 参数说明

|参数名|说明|可选值|
|---|---|---|
|`--model`|指定待运行的算法模型|maefe, ekgan, unetflow, descod, ecgrecover|
|`--dataset`|指定实验使用的数据集|ptbxl|
|`--help`|查看完整的命令行参数说明|-|
---

## 📊 实验输出 (Output)

### 1. 模型权重（results/checkpoints/）

- `{Model}_{Data}_best.pth`：验证集 PCC 指标最高的最优模型权重

- `{Model}_{Data}_epoch_X.pth`：每 10 轮训练备份一次的中间权重文件（X 为轮次编号）

### 2. 可视化结果（results/vis_plots/）

- 每 10 个 Epoch 自动生成 1 张 12 导联信号对比图

- 图中包含 3 类信号：Original（原始 ECG 信号）、Masked（遮掩后的输入信号）、Recon（模型重建/生成的信号）

---

## 📦 依赖安装

运行以下命令安装实验所需依赖包：

```bash

pip install torch wfdb pandas tqdm matplotlib scipy
```

---

## 🔍 补充说明

- 数据集准备：请将 PTB-XL 数据集解压后放入 `data/ptbxl/` 目录下，确保数据文件结构与 `utils/ptbxl_loader.py` 中的加载逻辑匹配

- 参数调整：可在 `config.py` 中修改训练轮次、学习率、 batch size 等超参数

- 问题排查：若运行报错，优先检查环境依赖版本、数据集路径是否正确，或通过 `--help` 确认参数输入格式
> （注：文档部分内容可能由 AI 生成）