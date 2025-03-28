#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证带有多尺度注意力机制的YOLO11模型性能
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import cv2
import os
import yaml

from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='验证YOLO11裂缝检测模型性能')
    
    # 基本验证参数
    parser.add_argument('--data', type=str, default='data/cracks.yaml', help='数据集配置文件路径')
    parser.add_argument('--weights', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--batch', type=int, default=16, help='批处理大小')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='', help='处理设备')
    parser.add_argument('--half', action='store_true', help='是否使用半精度FP16')
    parser.add_argument('--workers', type=int, default=8, help='数据加载器工作进程数')
    
    # 优化开关
    parser.add_argument('--use-msa', action='store_true', help='模型是否包含多尺度注意力机制')
    
    # 可视化参数
    parser.add_argument('--visualize', action='store_true', help='可视化检测结果')
    parser.add_argument('--vis-dir', type=str, default='runs/val/vis', help='可视化结果保存目录')
    parser.add_argument('--max-vis', type=int, default=20, help='最大可视化图像数')
    parser.add_argument('--min-conf', type=float, default=0.25, help='最小置信度阈值')
    parser.add_argument('--vis-attention', action='store_true', help='可视化注意力图(仅在--use-msa为True时有效)')
    
    # 评估参数
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IoU阈值')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='置信度阈值')
    parser.add_argument('--augment', action='store_true', help='测试时增强')
    parser.add_argument('--verbose', action='store_true', help='详细输出模式')
    parser.add_argument('--save-txt', action='store_true', help='保存结果为.txt文件')
    parser.add_argument('--save-conf', action='store_true', help='保存置信度分数')
    parser.add_argument('--save-json', action='store_true', help='保存JSON格式结果')
    parser.add_argument('--project', type=str, default='runs/val', help='保存结果的项目名称')
    parser.add_argument('--name', type=str, default='exp', help='保存结果的实验名称')
    
    return parser.parse_args()


def check_model_has_msa(model_path):
    """检查模型是否包含多尺度注意力机制"""
    # 首先检查路径是否是配置文件
    if model_path.endswith('.yaml'):
        with open(model_path, 'r') as f:
            config = yaml.safe_load(f)
            return 'msa' in model_path or any('MSA' in str(v) for v in config.values())
    # 如果是权重文件，通过名称判断
    elif model_path.endswith('.pt'):
        return 'msa' in os.path.basename(model_path).lower()
    # 默认情况
    return False


def visualize_results(model, dataset, vis_dir, max_vis=20, min_conf=0.25, vis_attention=False, has_msa=False):
    """可视化检测结果和注意力图"""
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 随机选择图像
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    vis_count = 0
    
    print(f"正在生成可视化结果...")
    for batch in tqdm(dataloader, total=min(max_vis, len(dataloader))):
        if vis_count >= max_vis:
            break
            
        # 获取图像
        img = batch[0]
        img_path = batch[2][0] if len(batch) > 2 else f"img_{vis_count}.jpg"
        filename = Path(img_path).stem
        
        # 检测
        results = model(img, verbose=False)
        result = results[0]
        
        # 决定是否显示注意力图
        show_attention = vis_attention and has_msa
        
        # 可视化检测结果
        fig, axes = plt.subplots(1, 2 if show_attention else 1, figsize=(16, 8))
        if not show_attention:
            axes = [axes]
            
        # 绘制检测结果
        axes[0].imshow(result.plot(conf=min_conf, line_width=2))
        axes[0].set_title("检测结果")
        axes[0].axis('off')
        
        # 尝试可视化注意力图
        if show_attention:
            try:
                # 这里假设模型有一个hooks字典存储了中间特征图
                if hasattr(model.model, 'hooks') and 'msa' in model.model.hooks:
                    attention_maps = model.model.hooks['msa']
                    # 选择注意力图中间维度用于可视化
                    if attention_maps is not None and len(attention_maps) > 0:
                        attn_map = attention_maps[0]  # 假设第一个元素是我们需要的注意力图
                        # 调整尺寸与原图匹配
                        attn_map = cv2.resize(attn_map, (img.shape[3], img.shape[2]))
                        # 绘制热力图
                        axes[1].imshow(img[0].permute(1, 2, 0).cpu().numpy())
                        im = axes[1].imshow(attn_map, cmap='jet', alpha=0.5)
                        fig.colorbar(im, ax=axes[1])
                        axes[1].set_title("注意力热力图")
                        axes[1].axis('off')
                    else:
                        axes[1].text(0.5, 0.5, "注意力图不可用", 
                                   horizontalalignment='center', verticalalignment='center')
                        axes[1].axis('off')
                else:
                    axes[1].text(0.5, 0.5, "模型未提供注意力图hooks", 
                               horizontalalignment='center', verticalalignment='center')
                    axes[1].axis('off')
            except Exception as e:
                print(f"生成注意力图时出错: {e}")
                axes[1].text(0.5, 0.5, f"注意力图错误: {str(e)}", 
                           horizontalalignment='center', verticalalignment='center')
                axes[1].axis('off')
                
        # 保存图像
        fig.tight_layout()
        output_path = vis_dir / f"{filename}_detection.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        
        vis_count += 1
    
    print(f"可视化结果已保存到: {vis_dir}")


def analyze_performance(results, save_path=None, has_msa=False):
    """分析模型性能，特别是对裂缝等细长目标的检测性能"""
    # 创建性能分析报告
    metrics = results.box.metrics.cpu().numpy()
    
    # 基本指标
    report = {
        'precision': metrics.get('metrics/precision(B)', 0),
        'recall': metrics.get('metrics/recall(B)', 0),
        'mAP50': metrics.get('metrics/mAP50(B)', 0),
        'mAP50-95': metrics.get('metrics/mAP50-95(B)', 0)
    }
    
    # 打印报告
    print("\n性能分析报告:")
    print(f"模型类型: {'使用多尺度注意力机制' if has_msa else '标准模型'}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall: {report['recall']:.4f}")
    print(f"mAP@0.5: {report['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {report['mAP50-95']:.4f}")
    
    # 不同尺寸目标的性能
    if hasattr(results, 'by_size'):
        print("\n不同尺寸目标的性能:")
        size_metrics = results.by_size
        for size, metrics in size_metrics.items():
            print(f"{size} 目标: mAP={metrics['map']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    # 保存报告
    if save_path:
        with open(save_path, 'w') as f:
            f.write("性能分析报告:\n")
            f.write(f"模型类型: {'使用多尺度注意力机制' if has_msa else '标准模型'}\n")
            f.write(f"Precision: {report['precision']:.4f}\n")
            f.write(f"Recall: {report['recall']:.4f}\n")
            f.write(f"mAP@0.5: {report['mAP50']:.4f}\n")
            f.write(f"mAP@0.5:0.95: {report['mAP50-95']:.4f}\n")
    
    return report


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 选择设备
    device = select_device(args.device)
    
    # 确定模型是否使用MSA
    has_msa = args.use_msa or check_model_has_msa(args.weights)
    
    # 加载模型
    model = YOLO(args.weights)
    print(f"加载模型：{args.weights}")
    print(f"模型{'包含' if has_msa else '不包含'}多尺度注意力机制")
    
    # 准备验证参数
    val_args = {
        'data': args.data,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': device.type,
        'workers': args.workers,
        'half': args.half,
        'verbose': args.verbose,
        'iou': args.iou_thres,
        'conf': args.conf_thres,
        'augment': args.augment,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'save_json': args.save_json,
        'project': args.project,
        'name': args.name
    }
    
    # 执行验证
    print("\n开始验证...")
    results = model.val(**val_args)
    
    # 分析性能
    analyze_performance(results, 
                       save_path=Path(args.project) / args.name / 'performance_report.txt',
                       has_msa=has_msa)
    
    # 可视化检测结果
    if args.visualize:
        val_dataset = model.predictor.dataset
        visualize_results(
            model=model,
            dataset=val_dataset,
            vis_dir=args.vis_dir,
            max_vis=args.max_vis,
            min_conf=args.min_conf,
            vis_attention=args.vis_attention,
            has_msa=has_msa
        )
    
    print("\n验证完成！")
    return results


if __name__ == "__main__":
    main() 