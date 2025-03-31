#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练带有多尺度注意力机制和自定义图像增强的YOLO11模型
"""

import os
import argparse
import yaml
import torch
from pathlib import Path
from datetime import datetime
import random
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练YOLO模型')
    
    # 基础训练参数
    parser.add_argument('--data', type=str, default='data/cracks.yaml', help='数据集配置文件路径')
    parser.add_argument('--cfg', type=str, default='ultralytics/cfg/default.yaml', help='模型配置文件路径')
    parser.add_argument('--weights', type=str, default='', help='预训练权重路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批处理大小')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='', help='训练设备，为空时自动选择')
    parser.add_argument('--workers', type=int, default=8, help='数据加载器工作进程数')
    parser.add_argument('--project', type=str, default='runs/train', help='保存结果的项目名称')
    parser.add_argument('--name', type=str, default='exp', help='保存结果的实验名称')
    parser.add_argument('--exist-ok', action='store_true', help='是否覆盖已存在的实验目录')
    
    # 自定义增强开关
    parser.add_argument('--use-custom-aug', action='store_true', help='是否使用自定义图像增强')
    
    # 其他
    parser.add_argument('--verbose', action='store_true', help='详细输出模式')
    parser.add_argument('--seed', type=int, default=0, help='全局随机种子')
    
    return parser.parse_args()


def get_model_name(args):
    """根据参数确定模型名称"""
    base_model = "yolo11" if args.model_type.startswith("yolo11") else "yolo8"
    
    # 如果启用多尺度注意力但模型类型不包含"-msa"后缀，添加它
    if args.use_msa:
        if not base_model.endswith("-msa"):
            base_model = f"{base_model}-msa"
    else:
        # 确保不使用带MSA的模型
        if base_model.endswith("-msa"):
            base_model = base_model.replace("-msa", "")
            
    # 根据模型类型选择正确的配置文件路径
    if base_model.startswith("yolo11"):
        model_path = f"ultralytics/cfg/models/11/{base_model}.yaml"
    else:
        model_path = f"ultralytics/cfg/models/v8/{base_model}.yaml"
            
    return model_path


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
    
    # 加载配置文件
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 如果使用自定义增强，添加自定义增强配置
    if args.use_custom_aug:
        cfg['custom_aug'] = {
            'p': 0.5,
            'black_thresh': 0.05,
            'white_thresh': 0.1,
            'enhance_intensity': 0.4,
            'smooth_sigma': 5
        }
    
    # 选择设备
    device = select_device(args.device)
    
    # 创建模型
    model = YOLO(args.weights) if args.weights else YOLO(args.cfg)
    
    # 训练配置
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name or f"yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'exist_ok': args.exist_ok,
        'pretrained': bool(args.weights),
        'verbose': args.verbose,
        'seed': args.seed,
        'custom_aug': cfg.get('custom_aug', {})
    }
    
    # 开始训练
    model.train(**train_args)


if __name__ == "__main__":
    main() 