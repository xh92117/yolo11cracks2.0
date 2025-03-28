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

from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练带有多尺度注意力机制和自定义图像增强的YOLO11模型')
    
    # 基本训练参数
    parser.add_argument('--data', type=str, default='data/cracks.yaml', help='数据集配置文件路径')
    parser.add_argument('--model-type', type=str, default='yolo11', 
                        choices=['yolo11', 'yolo11-msa', 'yolo8', 'yolo8-msa'], 
                        help='模型类型: yolo11, yolo11-msa, yolo8, yolo8-msa')
    parser.add_argument('--model-size', type=str, default='n', 
                        choices=['n', 's', 'm', 'l', 'x'], 
                        help='模型大小: n, s, m, l, x')
    parser.add_argument('--weights', type=str, default='', help='初始权重文件路径，为空时使用预训练权重')
    parser.add_argument('--batch', type=int, default=16, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='', help='训练设备，为空时自动选择')
    parser.add_argument('--workers', type=int, default=8, help='数据加载器工作进程数')
    parser.add_argument('--project', type=str, default='runs/train', help='保存结果的项目名称')
    parser.add_argument('--name', type=str, default='exp', help='保存结果的实验名称')
    
    # 优化开关
    parser.add_argument('--use-msa', action='store_true', help='是否使用多尺度注意力机制')
    parser.add_argument('--use-custom-aug', action='store_true', help='是否使用自定义图像增强')
    
    # 自定义图像增强参数
    parser.add_argument('--aug-p', type=float, default=0.5, help='自定义增强触发概率')
    parser.add_argument('--black-thresh', type=float, default=0.05, help='黑区阈值比例')
    parser.add_argument('--white-thresh', type=float, default=0.1, help='白区阈值比例')
    parser.add_argument('--enhance-intensity', type=float, default=0.4, help='增强强度')
    parser.add_argument('--smooth-sigma', type=float, default=5, help='直方图平滑系数')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='SGD', 
                        choices=['SGD', 'Adam', 'AdamW', 'RMSProp'], 
                        help='优化器类型')
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01, help='最终学习率 (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD动量')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='权重衰减系数')
    
    # 其他训练参数
    parser.add_argument('--warmup-epochs', type=int, default=3, help='预热轮数')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping轮数')
    parser.add_argument('--cos-lr', action='store_true', help='是否使用余弦学习率')
    parser.add_argument('--resume', action='store_true', help='从最后一个检查点恢复训练')
    parser.add_argument('--amp', action='store_true', help='自动混合精度训练')
    
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
            
    return f"{base_model}{args.model_size}"


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 选择设备
    device = select_device(args.device)
    
    # 动态确定使用的模型类型
    model_name = get_model_name(args)
    
    # 加载模型
    if args.weights and os.path.exists(args.weights):
        model = YOLO(args.weights)
        print(f"加载模型权重：{args.weights}")
    else:
        model = YOLO(model_name)
        print(f"使用模型：{model_name}")
    
    # 准备训练参数
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'patience': args.patience,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': device.type,
        'workers': args.workers,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'cos_lr': args.cos_lr,
        'resume': args.resume,
        'amp': args.amp,
        'verbose': args.verbose,
        'seed': args.seed,
        'project': args.project,
        'name': args.name or f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    # 添加自定义图像增强参数
    if args.use_custom_aug:
        train_args.update({
            'custom_aug': {
                'p': args.aug_p,
                'black_thresh': args.black_thresh,
                'white_thresh': args.white_thresh,
                'enhance_intensity': args.enhance_intensity,
                'smooth_sigma': args.smooth_sigma
            }
        })
        print("启用自定义图像增强算法")
    else:
        print("不使用自定义图像增强算法")
    
    # 打印优化设置状态
    print(f"\n优化设置:")
    print(f"  多尺度注意力机制 (MSA): {'已启用' if args.use_msa else '未启用'}")
    print(f"  自定义图像增强: {'已启用' if args.use_custom_aug else '未启用'}")
    
    # 打印训练配置
    if args.verbose:
        print("\n训练配置:")
        for k, v in train_args.items():
            if k != 'custom_aug':
                print(f"  {k}: {v}")
            else:
                print(f"  {k}:")
                for ak, av in v.items():
                    print(f"    {ak}: {av}")
    
    # 开始训练
    print("\n开始训练...")
    results = model.train(**train_args)
    
    # 保存最佳模型参数
    best_model_path = str(Path(results.save_dir) / 'weights' / 'best.pt')
    
    # 训练结束，打印结果
    print(f"\n训练完成！")
    print(f"最佳模型保存在: {best_model_path}")
    if results.metrics is not None:
        metrics = results.metrics
        print("\n模型指标:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    
    return best_model_path


if __name__ == "__main__":
    main() 