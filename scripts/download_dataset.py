#!/usr/bin/env python3
"""
数据集下载脚本
功能：从Roboflow下载拖拉机检测数据集
"""

from roboflow import Roboflow
import os
import sys

def download_tractor_dataset(api_key, output_dir="./datasets"):
    """
    下载拖拉机检测数据集
    
    Args:
        api_key: Roboflow API密钥
        output_dir: 数据集保存目录
    """
    print("=" * 60)
    print("开始下载拖拉机检测数据集")
    print("=" * 60)
    
    try:
        # 初始化Roboflow
        print("\n1. 连接Roboflow...")
        rf = Roboflow(api_key=api_key)
        
        # 获取项目
        print("2. 获取项目: new-holland/tractor-detection-blxuq")
        project = rf.workspace("new-holland").project("tractor-detection-blxuq")
        
        # 下载数据集（版本2，YOLOv8格式）
        print("3. 下载数据集（YOLOv8格式）...")
        dataset = project.version(2).download("yolov8", location=output_dir)
        
        print(f"\n✅ 数据集下载成功！")
        print(f"📁 保存位置: {dataset.location}")
        
        # 显示数据集信息
        print("\n" + "=" * 60)
        print("数据集信息")
        print("=" * 60)
        
        # 读取data.yaml获取详细信息
        import yaml
        yaml_path = os.path.join(dataset.location, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            print(f"类别数量: {data_config.get('nc', 'N/A')}")
            print(f"类别名称: {data_config.get('names', 'N/A')}")
            print(f"训练集: {data_config.get('train', 'N/A')}")
            print(f"验证集: {data_config.get('val', 'N/A')}")
            print(f"测试集: {data_config.get('test', 'N/A')}")
        
        # 统计图像数量
        train_dir = os.path.join(dataset.location, "train", "images")
        valid_dir = os.path.join(dataset.location, "valid", "images")
        test_dir = os.path.join(dataset.location, "test", "images")
        
        if os.path.exists(train_dir):
            train_count = len([f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))])
            print(f"\n训练图像数量: {train_count}")
        
        if os.path.exists(valid_dir):
            valid_count = len([f for f in os.listdir(valid_dir) if f.endswith(('.jpg', '.png'))])
            print(f"验证图像数量: {valid_count}")
        
        if os.path.exists(test_dir):
            test_count = len([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))])
            print(f"测试图像数量: {test_count}")
        
        print("\n" + "=" * 60)
        print("下一步：运行 train_model.py 开始训练")
        print("=" * 60)
        
        return dataset.location
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("\n请检查：")
        print("1. API密钥是否正确")
        print("2. 网络连接是否正常")
        print("3. 是否已安装roboflow包: pip install roboflow")
        sys.exit(1)

def main():
    """主函数"""
    # 从环境变量或命令行参数获取API密钥
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    
    if not api_key:
        print("请设置ROBOFLOW_API_KEY环境变量或修改脚本中的api_key变量")
        print("\n获取API密钥的步骤：")
        print("1. 访问 https://app.roboflow.com/")
        print("2. 登录账号")
        print("3. 点击右上角头像 → Settings → API Keys")
        print("4. 复制API密钥")
        print("\n设置环境变量：")
        print("export ROBOFLOW_API_KEY='your_api_key_here'")
        
        # 或者直接在这里设置（不推荐，仅用于测试）
        api_key = input("\n请输入您的Roboflow API密钥: ").strip()
        
        if not api_key:
            print("未提供API密钥，退出。")
            sys.exit(1)
    
    # 下载数据集
    dataset_path = download_tractor_dataset(api_key)

if __name__ == "__main__":
    main()

