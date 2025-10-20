#!/usr/bin/env python3
"""
模型训练脚本
功能：使用YOLOv8训练行人和拖拉机检测模型
"""

from ultralytics import YOLO
import torch
import os
import sys
from pathlib import Path

def train_model(
    data_yaml,
    model_size='n',
    epochs=100,
    imgsz=416,
    batch=16,
    project='runs/detect',
    name='tractor_person_detect'
):
    """
    训练YOLOv8模型
    
    Args:
        data_yaml: 数据集配置文件路径
        model_size: 模型大小 (n/s/m/l/x)
        epochs: 训练轮数
        imgsz: 图像尺寸
        batch: 批次大小
        project: 项目保存路径
        name: 实验名称
    """
    print("=" * 60)
    print("YOLOv8 模型训练")
    print("=" * 60)
    
    # 检查数据集文件是否存在
    if not os.path.exists(data_yaml):
        print(f"❌ 错误: 数据集配置文件不存在: {data_yaml}")
        print("请先运行 download_dataset.py 下载数据集")
        sys.exit(1)
    
    # 检查GPU是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    if device == 'cpu':
        print("⚠️  警告: 未检测到GPU，训练将使用CPU，速度会很慢")
        print("建议使用GPU进行训练")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU型号: {gpu_name}")
        print(f"GPU显存: {gpu_memory:.2f} GB")
    
    # 加载预训练模型
    model_name = f'yolov8{model_size}.pt'
    print(f"\n加载预训练模型: {model_name}")
    model = YOLO(model_name)
    
    # 训练参数
    print("\n训练参数:")
    print(f"  数据集: {data_yaml}")
    print(f"  训练轮数: {epochs}")
    print(f"  图像尺寸: {imgsz}x{imgsz}")
    print(f"  批次大小: {batch}")
    print(f"  设备: {device}")
    
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60 + "\n")
    
    try:
        # 开始训练
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            patience=20,
            save=True,
            plots=True,
            
            # 优化器参数
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            
            # 数据增强参数
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            
            # 其他参数
            verbose=True,
            seed=0,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=False,
            close_mosaic=10,
            amp=True,  # 自动混合精度
        )
        
        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)
        
        # 显示训练结果
        save_dir = Path(project) / name
        print(f"\n模型保存位置: {save_dir}")
        print(f"  最佳模型: {save_dir}/weights/best.pt")
        print(f"  最后模型: {save_dir}/weights/last.pt")
        print(f"  训练曲线: {save_dir}/results.png")
        print(f"  混淆矩阵: {save_dir}/confusion_matrix.png")
        
        # 评估模型
        print("\n" + "=" * 60)
        print("模型评估")
        print("=" * 60)
        
        best_model = YOLO(f"{save_dir}/weights/best.pt")
        metrics = best_model.val()
        
        print(f"\nmAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        
        # 各类别精度
        if hasattr(metrics.box, 'maps'):
            print("\n各类别精度:")
            for i, map_value in enumerate(metrics.box.maps):
                print(f"  类别 {i}: {map_value:.4f}")
        
        print("\n" + "=" * 60)
        print("下一步：")
        print("1. 查看训练结果: 打开 results.png")
        print("2. 导出ONNX: 运行 export_onnx.py")
        print("3. 转换Blob: 运行 convert_blob.py")
        print("=" * 60)
        
        return str(save_dir / "weights" / "best.pt")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        print("\n可能的解决方案:")
        print("1. 减小batch size（如果显存不足）")
        print("2. 检查数据集格式是否正确")
        print("3. 确保已安装所有依赖: pip install ultralytics")
        sys.exit(1)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练YOLOv8模型')
    parser.add_argument('--data', type=str, default='datasets/tractor-detection-blxuq-2/data.yaml',
                       help='数据集配置文件路径')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='模型大小 (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=416,
                       help='图像尺寸')
    parser.add_argument('--batch', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--name', type=str, default='tractor_person_detect',
                       help='实验名称')
    
    args = parser.parse_args()
    
    # 开始训练
    model_path = train_model(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name
    )
    
    print(f"\n训练完成的模型: {model_path}")

if __name__ == "__main__":
    main()

