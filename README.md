# Sketch Recognition & Generation 系统

本项目实现了 Machine Learning Course Project 的 Task 2 (Unimodal Task)。我们构建了一个完整的系统，包含**笔画序列生成模型 (SketchVAE)** 和 **草图识别模型 (SmallCNN)**，并针对 CPU 训练环境进行了深度工程优化。

## ✨ 我们的核心优化 (Key Features)

我们会给助教展示以下区别于普通实现的工程亮点：

1.  **⚡ 实时流式渲染 (On-the-fly Rendering)**
    *   **传统痛点**：通常需要将数十万张图片解压到磁盘，占用大量空间且 Windows 下 I/O 极慢。
    *   **我们的优化**：实现了 `QuickDrawNpzDataset`，直接读取原始 `.npz` 序列文件，在内存中实时使用 `PIL` 加速渲染为图像。**零磁盘空间占用**，即插即用。

2.  **🚀 动态批处理 (Dynamic Padding)**
    *   **传统痛点**：序列数据通常填充到由 `max_len` 决定的固定长度（如 200），导致大量无效计算。
    *   **我们的优化**：重写了 `collate_fn`，根据每个 Batch 内的最长序列动态调整 Padding。在 CPU 环境下训练速度提升约 **3倍**。

3.  **🔄 端到端评估流水线 (E2E Pipeline)**
    *   提供了一键式脚本 `evaluate_pipeline.py`，自动完成 `Generate (生成) -> Render (渲染) -> Recognize (识别) -> Score (打分)` 的闭环测试，输出定量的生成质量指标。

## 🛠️ 环境依赖

无需安装 torchvision 等复杂依赖，保持轻量：

*   Python 3.8+
*   PyTorch
*   NumPy
*   Pillow (PIL)
*   tqdm

安装命令：
```bash
pip install torch numpy pillow tqdm
```

## 📂 数据准备

请确保 QuickDraw 数据集（`.npz` 格式）放置在 `data/` 目录下（或者根据需要修改 `dataset.py` 中的默认路径）。
推荐结构：
```text
generation/
  ├── data/
  │   ├── QuickDraw_generation/
  │   │   ├── cat.npz
  │   │   ├── airplane.npz
  │   │   └── ...
```

## 🚀 运行指南

### 1. 训练识别模型 (Task A: Recognition)

该模块负责识别草图类别。得益于实时渲染优化，你不需要解压任何图片。

```bash
# Windows (Powershell) 下如遇 OpenMP 报错，请先运行:
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# 训练命令 (示例: 训练飞机、猫、鱼、伞、钟表)
# --max_samples 用于调试时限制数据量，全量训练请去掉该参数
python recognition/train.py --classes "airplane,cat,fish,umbrella,clock" --epochs 5 --batch_size 64
```
*   模型权重将保存在 `recognition/results/` 下。

### 2. 训练生成模型 (Task B: Generation)

该模块负责生成可控的草图序列，使用了 Dynamic Padding 技术加速。

```bash
python generation/train.py --classes "airplane,cat,fish,umbrella,clock" --epochs 10 --batch_size 64
```
*   模型权重将保存在 `generation/results/` 下。

### 3. 端到端评估 (Pipeline Evaluation)

这是我们项目的**杀手锏功能**。它会加载生成模型产出草图，并立即使用识别模型进行打分。

```bash
python evaluate_pipeline.py --n_samples 100
```
该脚本会输出如下格式的报告：
```text
Class: airplane   | Acc: 0.8500 (85/100)
Class: cat        | Acc: 0.7200 (72/100)
...
Overall Generation Accuracy: 0.7900
```

## 📁 核心代码文件说明

*   `generation/dataset.py`: **[优化点]** 包含 `collate_batch` 函数，实现了动态 Padding。
*   `recognition/dataset.py`: **[优化点]** 包含 `stroke3_to_image` 函数，实现了基于 PIL 的高效实时渲染。
*   `evaluate_pipeline.py`: **[优化点]** 系统集成脚本，实现了跨模态评估。
*   `train.py`: 训练入口脚本。

---
*Machine Learning Course Project - Group [你的组号]*
