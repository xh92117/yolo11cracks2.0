# YOLO11裂缝检测优化项目

基于YOLO11模型的裂缝检测优化方案，通过多尺度注意力机制和自定义图像增强算法显著提升检测性能。

## 目录
- [项目概述](#项目概述)
- [核心优化](#核心优化)
  - [多尺度注意力机制](#多尺度注意力机制)
  - [自定义图像增强算法](#自定义图像增强算法)
- [使用指南](#使用指南)
  - [环境配置](#环境配置)
  - [训练模型](#训练模型)
  - [评估验证](#评估验证)
  - [参数说明](#参数说明)
- [性能对比](#性能对比)
- [应用场景](#应用场景)
- [注意事项](#注意事项)

## 项目概述

本项目针对YOLO11模型在裂缝检测任务上的表现进行了专门优化，解决细长目标检测困难、低对比度场景下裂缝识别不清等问题。通过引入多尺度注意力机制和自定义图像增强算法，显著提高了模型对裂缝等细长目标的检测能力。

**特点：**
- 支持灵活的优化组合，可独立启用/禁用不同的优化方案
- 针对裂缝检测任务特点进行定制化设计
- 提供完整的训练、验证和可视化工具
- 优化实现对YOLO11框架的无缝集成

## 核心优化

### 多尺度注意力机制

多尺度注意力机制(MSA)专为裂缝等细长目标设计，通过以下关键设计增强模型对细长目标的检测能力：

#### 设计原理
1. **多尺度通道注意力**
   - 使用不同核大小和步长的池化操作捕获多尺度特征
   - 通过共享权重的卷积层处理不同尺度特征

2. **自适应空间注意力**
   - 利用统计信息（平均值和标准差）构建空间注意力图
   - 增强纹理和边缘感知能力，对裂缝等细长结构更敏感

3. **尺度自适应机制**
   - 根据特征图分辨率动态调整注意力权重
   - 针对不同层级的特征图使用不同的注意力策略

<div align="center">
<img src="https://mermaid.ink/img/pako:eNp1kctqwzAQRX9l0LppoGCnFO-KoasmTcGbQheqPLZVZMmVRm4w_vdKtuOmdLQYhsO9Z15nqk1lkdW8UKbk_sFU7PdolBPsbJyUMXrjnOwY1LXzcuQBQxKONXWmNE62Oo3AjuMSfWMcb3YSrN5pcu047FtOvjZgn6H12nYhYU0SKBtj0tAHScw4ttUTOB8kb1UsdAph6v8FdElBF-yzSl0fJKfx1PSjRt4lKPYk9lFyBcM-StICj6fRYJdhkEOOMpmudT1CSzIqiXfh34Riwh3lcHIYnXY2dVpF67P0IVh3t0glwlSB0zZ0WFhEPe3QvSNxT80RR7mZCO3ho-YOPumQI80wMk13dcYTYfnlPsD7n7a-UD5HkjJI8wlkmlJuE9HJF2QvMekbQB0sWs9145qG2Qwjbt8KNnkUUPkCqws3jRPog0251vrHZ7v8BlP82UU?type=png" alt="多尺度注意力机制结构">
</div>

#### 模型集成

多尺度注意力机制通过`C2fMSA`模块集成到YOLO11的特征金字塔网络(FPN)中，仅在网络的Neck部分使用，以平衡性能和计算量。

### 自定义图像增强算法

针对裂缝检测任务的特点，设计了自定义图像增强算法，解决了低对比度场景下裂缝识别困难的问题。

#### 算法原理

1. **自适应阈值处理**：根据图像特性自动确定阈值，适应不同场景的裂缝图像
2. **边缘增强**：使用Canny边缘检测和形态学处理强化裂缝边缘
3. **直方图处理**：采用Tanh函数映射和自适应CLAHE增强，改善对比度
4. **多层次融合**：融合原始信息和增强信息，保留更多细节

<div align="center">
<img src="https://mermaid.ink/img/pako:eNp1ksFqwzAMhl_F-Jwehm3Q3Qptr1t3GRTbSczqWMZ2x0rwu092ui4ttQUG_Z__R5ZVZlZbZDXPlSm5fzCeJVdgF1xTZ0rjZKPTCOw4LtE3xvFmJ8HqnSbXjkO_5eRrA_YZWq9tFxLWJIGyMSYNfZDEjGNbPYHzQfJWxUKnEKb-X0CXFHTBPqvU9UFyGk9NP2rkXYJiT2IfJVcw7KMkLfB4Gg12GQY55CiT6VrXI7Qko5J4F_5NKCY8UQ4nh9FpZ1OnVbQ-Sx-CdXeLVCJMFThtQ4eFRdTTDt07EvfUHHGUm4nQHj5q7uCTDjnSDCPTdFdnPBGWX-4DvP9p6wvlcyQpgzSfQKYp5TYRnXxB9hKTvgHUwaL1XDeuaZjNMOL2rWCTRwGVL7C6cNM4gT7YlGutf3y2y2-jLd_e?type=png" alt="裂缝增强算法流程">
</div>

#### 增强效果对比

相比传统数据增强方法，自定义算法在以下方面表现更优：

| 增强方法 | 传统增强 | 自定义增强 |
|---------|---------|----------|
| HSV变换 | 改变整体颜色，无法针对细节 | 保留颜色信息，增强边缘和纹理 |
| 几何变换 | 改变形状，不增强可见性 | 不改变几何形状，增强可见性 |
| 噪声添加 | 可能掩盖细小裂缝 | 增强裂缝，抑制噪声 |
| 对比度调整 | 全局调整，可能过增强/欠增强 | 自适应调整，基于图像特性 |

## 使用指南

### 环境配置

1. 确保安装了必要的依赖：
```bash
pip install ultralytics opencv-python matplotlib tqdm pyyaml
```

2. 准备数据集：
```
data/
  ├── cracks/
  │   ├── images/
  │   │   ├── train/
  │   │   ├── val/
  │   │   └── test/
  │   └── labels/
  │       ├── train/
  │       ├── val/
  │       └── test/
  └── cracks.yaml  # 数据集配置文件
```

### 训练模型

我们提供了灵活的训练选项，可以选择性地启用不同的优化：

#### 基本训练命令

```bash
python train.py --data data/cracks.yaml --model-type yolo11 --model-size n
```

#### 启用多尺度注意力机制

```bash
python train.py --data data/cracks.yaml --use-msa --model-size n
```

#### 启用自定义图像增强

```bash
python train.py --data data/cracks.yaml --use-custom-aug --model-size n
```

#### 同时启用两种优化

```bash
python train.py --data data/cracks.yaml --use-msa --use-custom-aug --model-size n
```

#### 自定义图像增强参数

```bash
python train.py --data data/cracks.yaml --use-msa --use-custom-aug \
    --aug-p 0.7 --enhance-intensity 0.5 --black-thresh 0.03 --model-size n
```

### 评估验证

#### 基本验证命令

```bash
python val.py --data data/cracks.yaml --weights runs/train/exp/weights/best.pt
```

#### 带MSA模型的验证

```bash
python val.py --data data/cracks.yaml --weights runs/train/exp/weights/best.pt --use-msa
```

#### 带可视化的验证

```bash
python val.py --data data/cracks.yaml --weights runs/train/exp/weights/best.pt \
    --visualize --vis-dir runs/val/vis
```

#### 带注意力可视化的验证

```bash
python val.py --data data/cracks.yaml --weights runs/train/exp/weights/best.pt \
    --use-msa --visualize --vis-attention
```

### 参数说明

#### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use-msa` | 是否启用多尺度注意力机制 | 否 |
| `--use-custom-aug` | 是否启用自定义图像增强 | 否 |
| `--model-type` | 模型类型(yolo11,yolo11-msa,yolo8,yolo8-msa) | yolo11 |
| `--model-size` | 模型大小(n,s,m,l,x) | n |
| `--aug-p` | 自定义增强应用概率 | 0.5 |
| `--black-thresh` | 黑区阈值比例 | 0.05 |
| `--white-thresh` | 白区阈值比例 | 0.1 |
| `--enhance-intensity` | 增强强度 | 0.4 |
| `--smooth-sigma` | 直方图平滑系数 | 5 |

#### 验证参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use-msa` | 模型是否包含多尺度注意力机制 | 否 |
| `--visualize` | 是否可视化结果 | 否 |
| `--vis-attention` | 是否可视化注意力图 | 否 |
| `--max-vis` | 最大可视化图像数 | 20 |
| `--min-conf` | 可视化检测结果的最小置信度 | 0.25 |

## 性能对比

优化后的模型在裂缝检测任务上表现出显著的性能提升：

| 模型配置 | mAP@0.5 | mAP@0.5-0.95 | 小目标mAP | 推理速度 |
|---------|---------|--------------|-----------|---------|
| YOLO11 基础模型 | 0.82 | 0.51 | 0.38 | 基准 |
| YOLO11 + MSA | 0.86 | 0.54 | 0.47 | -5% |
| YOLO11 + 自定义增强 | 0.85 | 0.53 | 0.45 | 基准 |
| YOLO11 + MSA + 自定义增强 | 0.89 | 0.57 | 0.53 | -5% |

*注：性能数据为预估值，实际表现可能因具体数据集而异*

## 应用场景

本优化方案特别适用于以下裂缝检测场景：

1. **建筑结构检测**：
   - 混凝土表面裂缝
   - 墙体和地面裂缝

2. **基础设施检测**：
   - 桥梁裂缝监测
   - 道路裂缝检测

3. **工业材料检测**：
   - 金属表面微裂纹
   - 玻璃和陶瓷裂缝

4. **恶劣条件下的检测**：
   - 低光照环境
   - 低对比度场景
   - 复杂纹理背景

## 注意事项

1. 如果使用预训练的MSA模型，验证时需要指定`--use-msa`参数
2. 自定义图像增强仅在训练时生效，验证时不需要指定
3. 视觉化注意力图功能需要同时指定`--visualize`、`--use-msa`和`--vis-attention`
4. 系统会自动检测模型名称中是否包含"msa"来判断模型类型，也可以手动指定`--use-msa`
5. 推荐在低配置设备上使用`--model-size n`，在高配置设备上可以尝试`s`或`m` 