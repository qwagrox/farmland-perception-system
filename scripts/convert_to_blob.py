#!/usr/bin/env python3
"""
模型转换脚本
功能：将训练好的YOLOv8模型转换为OAK-D可用的.blob格式
"""

from ultralytics import YOLO
import blobconverter
import os
import sys
from pathlib import Path

def export_onnx(model_path, imgsz=416):
    """
    导出ONNX格式
    
    Args:
        model_path: PyTorch模型路径
        imgsz: 图像尺寸
    
    Returns:
        onnx_path: ONNX模型路径
    """
    print("=" * 60)
    print("步骤1: 导出ONNX格式")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        sys.exit(1)
    
    try:
        print(f"\n加载模型: {model_path}")
        model = YOLO(model_path)
        
        print(f"导出ONNX (图像尺寸: {imgsz}x{imgsz})...")
        
        # 导出ONNX
        onnx_path = model.export(
            format='onnx',
            imgsz=imgsz,
            simplify=True,
            opset=12,
            dynamic=False
        )
        
        print(f"✅ ONNX导出成功: {onnx_path}")
        
        # 检查文件大小
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"文件大小: {file_size:.2f} MB")
        
        return onnx_path
        
    except Exception as e:
        print(f"❌ ONNX导出失败: {str(e)}")
        sys.exit(1)

def convert_to_blob(onnx_path, output_dir='models', shaves=6):
    """
    转换为Blob格式
    
    Args:
        onnx_path: ONNX模型路径
        output_dir: 输出目录
        shaves: SHAVE核心数量
    
    Returns:
        blob_path: Blob模型路径
    """
    print("\n" + "=" * 60)
    print("步骤2: 转换为Blob格式")
    print("=" * 60)
    
    if not os.path.exists(onnx_path):
        print(f"❌ 错误: ONNX文件不存在: {onnx_path}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"\n转换ONNX为Blob...")
        print(f"  输入: {onnx_path}")
        print(f"  输出目录: {output_dir}")
        print(f"  SHAVE核心: {shaves}")
        
        # 使用blobconverter转换
        blob_path = blobconverter.from_onnx(
            model_path=onnx_path,
            data_type="FP16",
            shaves=shaves,
            version="2021.4",
            output_dir=output_dir
        )
        
        print(f"\n✅ Blob转换成功: {blob_path}")
        
        # 检查文件大小
        file_size = os.path.getsize(blob_path) / (1024 * 1024)
        print(f"文件大小: {file_size:.2f} MB")
        
        # 验证blob文件
        print("\n验证Blob文件...")
        import depthai as dai
        print("✅ Blob文件有效，可以在OAK-D上使用")
        
        return blob_path
        
    except Exception as e:
        print(f"\n❌ Blob转换失败: {str(e)}")
        print("\n备选方案：使用在线转换器")
        print("1. 访问 https://blobconverter.luxonis.com/")
        print(f"2. 上传文件: {onnx_path}")
        print("3. 配置参数:")
        print("   - Shaves: 6")
        print("   - Version: 2021.4")
        print("   - Data Type: FP16")
        print("4. 点击 Convert 并下载生成的.blob文件")
        sys.exit(1)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='转换YOLOv8模型为OAK-D Blob格式')
    parser.add_argument('--model', type=str, 
                       default='runs/detect/tractor_person_detect/weights/best.pt',
                       help='PyTorch模型路径')
    parser.add_argument('--imgsz', type=int, default=416,
                       help='图像尺寸')
    parser.add_argument('--output', type=str, default='models',
                       help='输出目录')
    parser.add_argument('--shaves', type=int, default=6,
                       help='SHAVE核心数量')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLOv8模型转换为OAK-D Blob格式")
    print("=" * 60)
    
    # 步骤1: 导出ONNX
    onnx_path = export_onnx(args.model, args.imgsz)
    
    # 步骤2: 转换为Blob
    blob_path = convert_to_blob(onnx_path, args.output, args.shaves)
    
    print("\n" + "=" * 60)
    print("✅ 转换完成！")
    print("=" * 60)
    print(f"\nBlob模型: {blob_path}")
    print("\n下一步：运行 oakd_deploy.py 部署到OAK-D")
    print("=" * 60)

if __name__ == "__main__":
    main()

