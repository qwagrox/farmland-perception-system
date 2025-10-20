# 无人拖拉机田间避障系统（检测行人/拖拉机等）

- **版本**: 1.0.0
- **作者**: [tangyong@stmail.ujs.edu.cn](mailto:tangyong@stmail.ujs.edu.cn)，目前就读于江苏大学农机控制理论与工程博士
- **日期**: 2025/10/20 

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据集准备](#2-数据集准备)
3. [模型训练](#3-模型训练)
4. [模型转换](#4-模型转换)
5. [OAK-D部署](#5-oak-d部署)
6. [完整代码示例](#6-完整代码示例)
7. [常见问题](#7-常见问题)

---

## 1. 环境准备

### 1.1 硬件要求

- **训练环境**: 
  - CPU: Intel i5或更高
  - RAM: 16GB以上
  - GPU: NVIDIA GPU（推荐，可选）
  - 存储: 至少20GB可用空间

- **部署设备**:
  - OAK-D深度相机
  - 主控计算机（树莓派4、Jetson Nano或PC）

### 1.2 软件环境安装

#### 步骤1：安装Python环境

推荐使用Python 3.8-3.11版本：

```bash
# 检查Python版本
python3 --version

# 如果需要，创建虚拟环境
python3 -m venv oakd_env
source oakd_env/bin/activate  # Linux/Mac
# 或
oakd_env\Scripts\activate  # Windows
```

#### 步骤2：安装核心依赖包

```bash
# 安装Ultralytics YOLOv8
pip install ultralytics

# 安装Roboflow SDK
pip install roboflow

# 安装DepthAI（OAK-D SDK）
pip install depthai

# 安装blobconverter（模型转换工具）
pip install blobconverter

# 安装其他依赖
pip install opencv-python numpy torch torchvision
```

#### 步骤3：验证安装

```bash
# 验证YOLOv8
yolo version

# 验证DepthAI
python3 -c "import depthai as dai; print(dai.__version__)"

# 验证blobconverter
python3 -c "import blobconverter; print('blobconverter installed')"
```

---

## 2. 数据集准备

### 2.1 获取Roboflow API密钥

1. 访问 https://app.roboflow.com/
2. 注册/登录账号
3. 点击右上角头像 → Settings → API Keys
4. 复制您的API密钥

### 2.2 下载拖拉机数据集

您选择的数据集是：`new-holland/tractor-detection-blxuq/2`

创建下载脚本 `download_dataset.py`：

```python
from roboflow import Roboflow

# 初始化Roboflow
rf = Roboflow(api_key="YOUR_API_KEY_HERE")  # 替换为您的API密钥

# 下载拖拉机数据集
project = rf.workspace("new-holland").project("tractor-detection-blxuq")
dataset = project.version(2).download("yolov8")

print(f"数据集已下载到: {dataset.location}")
```

运行脚本：

```bash
python3 download_dataset.py
```

数据集将被下载到当前目录，结构如下：

```
tractor-detection-blxuq-2/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

### 2.3 准备行人检测数据集

由于我们需要同时检测行人和拖拉机，有两种方案：

#### 方案A：使用COCO预训练模型的person类别（推荐）

直接使用YOLOv8的COCO预训练权重，它已经包含person类别。训练时只需在拖拉机数据集上进行微调。

#### 方案B：合并数据集

如果您需要更高的行人检测精度，可以下载专门的行人数据集并合并：

```python
# 下载COCO person子集（示例）
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY_HERE")

# 搜索并下载行人数据集
# 例如：people-detection数据集
project = rf.workspace("smartcities").project("people-and-vehicles-detection-p5hml")
person_dataset = project.version(1).download("yolov8")
```

### 2.4 修改数据集配置文件

打开 `data.yaml` 文件，确认类别配置：

```yaml
# data.yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 2  # 类别数量：person + tractor
names: ['person', 'tractor']  # 类别名称
```

**重要提示**：如果您的拖拉机数据集只包含tractor类别，需要手动添加person类别。我们将在训练时使用COCO预训练权重来处理person类别。

---

## 3. 模型训练

### 3.1 训练策略

我们采用**迁移学习**策略：

1. 从COCO预训练的YOLOv8n模型开始（已包含person类别）
2. 在拖拉机数据集上进行微调
3. 保留person类别的检测能力，同时学习tractor类别

### 3.2 创建训练脚本

创建 `train_model.py`：

```python
from ultralytics import YOLO
import torch

# 检查GPU是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 加载预训练模型
model = YOLO('yolov8n.pt')  # YOLOv8 Nano版本

# 训练模型
results = model.train(
    data='tractor-detection-blxuq-2/data.yaml',  # 数据集配置文件
    epochs=100,                    # 训练轮数
    imgsz=416,                     # 图像尺寸（OAK-D推荐416x416）
    batch=16,                      # 批次大小（根据GPU内存调整）
    device=device,                 # 使用的设备
    project='runs/detect',         # 保存路径
    name='tractor_person_detect',  # 实验名称
    patience=20,                   # 早停耐心值
    save=True,                     # 保存检查点
    plots=True,                    # 生成训练图表
    
    # 数据增强参数
    hsv_h=0.015,                   # 色调增强
    hsv_s=0.7,                     # 饱和度增强
    hsv_v=0.4,                     # 明度增强
    degrees=10.0,                  # 旋转角度
    translate=0.1,                 # 平移
    scale=0.5,                     # 缩放
    flipud=0.0,                    # 上下翻转
    fliplr=0.5,                    # 左右翻转
    mosaic=1.0,                    # Mosaic增强
)

print("\n训练完成！")
print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
```

### 3.3 开始训练

```bash
python3 train_model.py
```

训练过程中会显示：
- 每个epoch的损失值
- mAP（平均精度）
- 训练进度条

**预计训练时间**：
- GPU（如RTX 3060）: 30-60分钟
- CPU: 3-6小时

### 3.4 监控训练过程

训练结果保存在 `runs/detect/tractor_person_detect/` 目录：

```
runs/detect/tractor_person_detect/
├── weights/
│   ├── best.pt      # 最佳模型
│   └── last.pt      # 最后一个epoch的模型
├── results.png      # 训练曲线
├── confusion_matrix.png  # 混淆矩阵
├── val_batch0_pred.jpg   # 验证集预测示例
└── args.yaml        # 训练参数
```

查看训练结果：

```python
from PIL import Image

# 查看训练曲线
img = Image.open('runs/detect/tractor_person_detect/results.png')
img.show()

# 查看混淆矩阵
img = Image.open('runs/detect/tractor_person_detect/confusion_matrix.png')
img.show()
```

### 3.5 模型评估

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/tractor_person_detect/weights/best.pt')

# 在测试集上评估
metrics = model.val()

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Person类别精度: {metrics.box.maps[0]}")
print(f"Tractor类别精度: {metrics.box.maps[1]}")
```

---

## 4. 模型转换

训练完成后，需要将PyTorch模型转换为OAK-D可用的`.blob`格式。

### 4.1 导出为ONNX格式

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/detect/tractor_person_detect/weights/best.pt')

# 导出为ONNX格式
model.export(
    format='onnx',
    imgsz=416,
    simplify=True,
    opset=12
)

print("模型已导出为ONNX格式")
```

导出的文件位于：`runs/detect/tractor_person_detect/weights/best.onnx`

### 4.2 转换为Blob格式

有两种方法转换为`.blob`格式：

#### 方法1：使用在线BlobConverter（推荐）

1. 访问 https://blobconverter.luxonis.com/
2. 上传 `best.onnx` 文件
3. 配置参数：
   - **Model Source**: OpenVINO Model
   - **Shaves**: 6（OAK-D的默认值）
   - **Version**: 2021.4（推荐）
   - **Data Type**: FP16
4. 点击"Convert"
5. 下载生成的 `best.blob` 文件

#### 方法2：使用blobconverter Python包

```python
import blobconverter

# 转换ONNX为Blob
blob_path = blobconverter.from_onnx(
    model_path="runs/detect/tractor_person_detect/weights/best.onnx",
    data_type="FP16",
    shaves=6,
    version="2021.4",
    output_dir="models/"
)

print(f"Blob文件保存在: {blob_path}")
```

### 4.3 验证Blob文件

```bash
# 检查文件大小（应该在5-15MB之间）
ls -lh models/best.blob

# 验证文件完整性
python3 -c "import depthai as dai; print('Blob文件有效')"
```

---

## 5. OAK-D部署

### 5.1 创建部署脚本

创建 `oakd_deploy.py`：

```python
import cv2
import depthai as dai
import numpy as np
import time

# 类别标签
LABELS = ['person', 'tractor']

# 颜色映射（BGR格式）
COLORS = {
    'person': (0, 255, 0),    # 绿色
    'tractor': (0, 0, 255)    # 红色
}

# 安全距离阈值（米）
SAFE_DISTANCE = 5.0

def create_pipeline(blob_path):
    """创建DepthAI pipeline"""
    pipeline = dai.Pipeline()
    
    # 创建RGB相机节点
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(416, 416)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)
    
    # 创建立体深度节点
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    
    # 创建神经网络节点
    detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    detection_nn.setBlobPath(blob_path)
    detection_nn.setConfidenceThreshold(0.5)
    detection_nn.setNumClasses(2)
    detection_nn.setCoordinateSize(4)
    detection_nn.setAnchors([])  # YOLOv8不使用anchors
    detection_nn.setAnchorMasks({})
    detection_nn.setIouThreshold(0.5)
    detection_nn.setNumInferenceThreads(2)
    detection_nn.input.setBlocking(False)
    
    cam_rgb.preview.link(detection_nn.input)
    
    # 创建空间定位计算器
    spatial_calc = pipeline.create(dai.node.SpatialLocationCalculator)
    spatial_calc.setWaitForConfigInput(True)
    stereo.depth.link(spatial_calc.inputDepth)
    
    # 创建输出节点
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)
    
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("detections")
    detection_nn.out.link(xout_nn.input)
    
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)
    
    xout_spatial = pipeline.create(dai.node.XLinkOut)
    xout_spatial.setStreamName("spatial")
    spatial_calc.out.link(xout_spatial.input)
    
    spatial_calc_config_in = pipeline.create(dai.node.XLinkIn)
    spatial_calc_config_in.setStreamName("spatial_calc_config")
    spatial_calc_config_in.out.link(spatial_calc.inputConfig)
    
    return pipeline

def calculate_distance(depth_frame, bbox):
    """计算目标距离"""
    x1, y1, x2, y2 = bbox
    
    # 获取边界框中心区域的深度值
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    # 采样5x5区域的深度值
    roi_size = 5
    depth_roi = depth_frame[
        max(0, center_y - roi_size):min(depth_frame.shape[0], center_y + roi_size),
        max(0, center_x - roi_size):min(depth_frame.shape[1], center_x + roi_size)
    ]
    
    # 使用中位数滤波减少噪声
    distance = np.median(depth_roi) / 1000.0  # 转换为米
    
    return distance

def main():
    """主函数"""
    blob_path = "models/best.blob"  # 修改为您的blob文件路径
    
    print("正在初始化OAK-D相机...")
    pipeline = create_pipeline(blob_path)
    
    with dai.Device(pipeline) as device:
        print("OAK-D相机已连接")
        
        # 获取输出队列
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_det = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        
        frame_count = 0
        start_time = time.time()
        
        print("开始检测...\n按'q'键退出")
        
        while True:
            # 获取RGB帧
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()
            
            # 获取深度帧
            in_depth = q_depth.get()
            depth_frame = in_depth.getFrame()
            
            # 获取检测结果
            in_det = q_det.get()
            detections = in_det.detections
            
            # 处理检测结果
            for detection in detections:
                # 获取边界框坐标
                bbox = [
                    int(detection.xmin * frame.shape[1]),
                    int(detection.ymin * frame.shape[0]),
                    int(detection.xmax * frame.shape[1]),
                    int(detection.ymax * frame.shape[0])
                ]
                
                # 获取类别和置信度
                label_id = detection.label
                label = LABELS[label_id]
                confidence = detection.confidence
                
                # 计算距离
                distance = calculate_distance(depth_frame, bbox)
                
                # 绘制边界框
                color = COLORS.get(label, (255, 255, 255))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # 绘制标签
                text = f"{label}: {confidence:.2f} | {distance:.2f}m"
                cv2.putText(frame, text, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 安全检查
                if distance < SAFE_DISTANCE:
                    warning_text = f"WARNING: {label} TOO CLOSE!"
                    cv2.putText(frame, warning_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print(f"⚠️  警告：{label}距离过近 ({distance:.2f}m)")
            
            # 计算FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - start_time)
                print(f"FPS: {fps:.2f}")
            
            # 显示帧
            cv2.imshow("OAK-D Detection", frame)
            
            # 按'q'退出
            if cv2.waitKey(1) == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("\n检测已停止")

if __name__ == "__main__":
    main()
```

### 5.2 运行部署脚本

```bash
python3 oakd_deploy.py
```

### 5.3 集成避障逻辑

创建 `obstacle_avoidance.py`：

```python
import depthai as dai
import numpy as np
from enum import Enum

class ObstacleType(Enum):
    NONE = 0
    PERSON = 1
    TRACTOR = 2

class ObstacleAvoidance:
    """避障控制器"""
    
    def __init__(self, safe_distance=5.0, warning_distance=10.0):
        self.safe_distance = safe_distance      # 紧急停止距离（米）
        self.warning_distance = warning_distance  # 警告距离（米）
        
    def analyze_obstacles(self, detections, depth_frame):
        """分析障碍物并返回避障决策"""
        obstacles = []
        
        for detection in detections:
            bbox = [
                int(detection.xmin * depth_frame.shape[1]),
                int(detection.ymin * depth_frame.shape[0]),
                int(detection.xmax * depth_frame.shape[1]),
                int(detection.ymax * depth_frame.shape[0])
            ]
            
            # 计算距离
            distance = self._calculate_distance(depth_frame, bbox)
            
            # 判断障碍物类型
            obstacle_type = ObstacleType.PERSON if detection.label == 0 else ObstacleType.TRACTOR
            
            obstacles.append({
                'type': obstacle_type,
                'distance': distance,
                'bbox': bbox,
                'confidence': detection.confidence
            })
        
        # 做出避障决策
        decision = self._make_decision(obstacles)
        
        return decision, obstacles
    
    def _calculate_distance(self, depth_frame, bbox):
        """计算目标距离"""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        roi_size = 5
        depth_roi = depth_frame[
            max(0, center_y - roi_size):min(depth_frame.shape[0], center_y + roi_size),
            max(0, center_x - roi_size):min(depth_frame.shape[1], center_x + roi_size)
        ]
        
        distance = np.median(depth_roi) / 1000.0
        return distance
    
    def _make_decision(self, obstacles):
        """根据障碍物信息做出决策"""
        if not obstacles:
            return {
                'action': 'CONTINUE',
                'speed': 1.0,
                'message': '无障碍物，继续前进'
            }
        
        # 找到最近的障碍物
        closest = min(obstacles, key=lambda x: x['distance'])
        
        if closest['distance'] < self.safe_distance:
            return {
                'action': 'STOP',
                'speed': 0.0,
                'message': f'紧急停止！{closest["type"].name}距离{closest["distance"]:.2f}米'
            }
        elif closest['distance'] < self.warning_distance:
            # 根据距离计算减速比例
            speed_factor = (closest['distance'] - self.safe_distance) / (self.warning_distance - self.safe_distance)
            return {
                'action': 'SLOW_DOWN',
                'speed': speed_factor,
                'message': f'减速！{closest["type"].name}距离{closest["distance"]:.2f}米'
            }
        else:
            return {
                'action': 'CONTINUE',
                'speed': 1.0,
                'message': '安全距离，继续前进'
            }

# 使用示例
if __name__ == "__main__":
    controller = ObstacleAvoidance(safe_distance=5.0, warning_distance=10.0)
    
    # 在主循环中调用
    # decision, obstacles = controller.analyze_obstacles(detections, depth_frame)
    # print(decision['message'])
    # 根据decision['action']控制拖拉机
```

---

## 6. 完整代码示例

### 6.1 完整的训练+部署流程

创建 `full_pipeline.sh`：

```bash
#!/bin/bash

echo "=== OAK-D行人与拖拉机检测完整流程 ==="

# 1. 下载数据集
echo "\n步骤1: 下载数据集..."
python3 download_dataset.py

# 2. 训练模型
echo "\n步骤2: 训练模型..."
python3 train_model.py

# 3. 导出ONNX
echo "\n步骤3: 导出ONNX..."
python3 -c "
from ultralytics import YOLO
model = YOLO('runs/detect/tractor_person_detect/weights/best.pt')
model.export(format='onnx', imgsz=416, simplify=True, opset=12)
"

# 4. 转换为Blob
echo "\n步骤4: 转换为Blob格式..."
python3 -c "
import blobconverter
blob_path = blobconverter.from_onnx(
    model_path='runs/detect/tractor_person_detect/weights/best.onnx',
    data_type='FP16',
    shaves=6,
    version='2021.4',
    output_dir='models/'
)
print(f'Blob文件: {blob_path}')
"

# 5. 部署到OAK-D
echo "\n步骤5: 部署到OAK-D..."
python3 oakd_deploy.py

echo "\n=== 流程完成 ==="
```

运行完整流程：

```bash
chmod +x full_pipeline.sh
./full_pipeline.sh
```

### 6.2 项目目录结构

```
oakd_project/
├── download_dataset.py          # 数据集下载脚本
├── train_model.py               # 模型训练脚本
├── oakd_deploy.py               # OAK-D部署脚本
├── obstacle_avoidance.py        # 避障控制器
├── full_pipeline.sh             # 完整流程脚本
├── requirements.txt             # 依赖包列表
├── models/
│   └── best.blob                # 转换后的模型
├── runs/
│   └── detect/
│       └── tractor_person_detect/
│           └── weights/
│               ├── best.pt      # 训练好的PyTorch模型
│               └── best.onnx    # ONNX模型
└── tractor-detection-blxuq-2/   # 下载的数据集
    ├── train/
    ├── valid/
    ├── test/
    └── data.yaml
```

### 6.3 requirements.txt

```
ultralytics>=8.0.0
roboflow>=1.1.0
depthai>=2.20.0
blobconverter>=1.4.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
```

安装所有依赖：

```bash
pip install -r requirements.txt
```

---

## 7. 常见问题

### Q1: 训练时显示"CUDA out of memory"

**解决方案**：
```python
# 减小batch size
results = model.train(
    batch=8,  # 从16改为8或更小
    ...
)
```

### Q2: 模型转换失败

**解决方案**：
```bash
# 确保ONNX版本正确
pip install onnx==1.14.0

# 使用在线转换器
# 访问 https://blobconverter.luxonis.com/
```

### Q3: OAK-D检测速度慢

**解决方案**：
- 使用更小的模型（yolov8n而不是yolov8s）
- 减小输入尺寸（从640改为416）
- 增加shaves数量（从6改为8）

### Q4: 检测精度不高

**解决方案**：
- 增加训练epochs（从100改为200）
- 收集更多训练数据
- 使用数据增强
- 调整置信度阈值

### Q5: 深度测距不准确

**解决方案**：
```python
# 校准OAK-D相机
# 参考：https://docs.luxonis.com/en/latest/pages/calibration/

# 使用中位数滤波
distance = np.median(depth_roi)

# 设置深度范围限制
stereo.setDepthRange(300, 10000)  # 0.3m到10m
```

---

## 8. 性能优化建议

### 8.1 模型优化

- **量化**: 使用INT8量化减小模型大小
- **剪枝**: 移除不重要的权重
- **知识蒸馏**: 使用大模型训练小模型

### 8.2 推理优化

```python
# 使用多线程
detection_nn.setNumInferenceThreads(2)

# 启用异步推理
detection_nn.input.setBlocking(False)

# 减小输出队列大小
q_det = device.getOutputQueue(name="detections", maxSize=1, blocking=False)
```

### 8.3 系统优化

- 使用SSD而不是SD卡（树莓派）
- 增加散热（Jetson Nano）
- 使用USB 3.0连接OAK-D

---

## 9. 下一步

完成基础部署后，您可以：

1. **集成到ROS**: 将检测结果发布为ROS topic
2. **添加轨迹预测**: 预测行人/拖拉机的移动方向
3. **多相机融合**: 使用多个OAK-D覆盖更大范围
4. **云端监控**: 将检测结果上传到云端
5. **自动标注**: 使用检测结果自动标注新数据

---


