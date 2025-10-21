# MID-360与OAK-D融合的负障碍物检测完整实现方案

**基于AMFA算法和视觉深度检测算法的传感器融合系统**

---

## 目录

1. [系统架构设计](#1-系统架构设计)
2. [硬件配置与安装](#2-硬件配置与安装)
3. [传感器标定](#3-传感器标定)
4. [MID-360 AMFA算法实现](#4-mid-360-amfa算法实现)
5. [OAK-D深度检测实现](#5-oak-d深度检测实现)
6. [传感器融合算法](#6-传感器融合算法)
7. [完整代码实现](#7-完整代码实现)
8. [性能优化](#8-性能优化)
9. [实际部署指南](#9-实际部署指南)

---

## 1. 系统架构设计

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    无人拖拉机避障系统                          │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────▼────────┐                    ┌────────▼────────┐
│  MID-360雷达   │                    │   OAK-D相机     │
│  (向下15°)     │                    │   (水平0°)      │
└───────┬────────┘                    └────────┬────────┘
        │                                       │
        │ 点云数据                              │ RGB+深度
        │ 10Hz, 200k点/秒                       │ 30Hz
        │                                       │
┌───────▼────────┐                    ┌────────▼────────┐
│ AMFA检测器     │                    │ 深度检测器      │
│ (虚拟扫描线)   │                    │ (RANSAC+梯度)   │
└───────┬────────┘                    └────────┬────────┘
        │                                       │
        │ 负障碍物                              │ 负障碍物
        │ 3.7-50m                               │ 0.5-10m
        │                                       │
        └───────────────────┬───────────────────┘
                            │
                    ┌───────▼────────┐
                    │  传感器融合     │
                    │  (卡尔曼滤波)   │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  决策模块       │
                    │  (优先级分配)   │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  控制输出       │
                    │  (避障指令)     │
                    └────────────────┘
```

### 1.2 传感器分工

| 传感器 | 主要任务 | 检测范围 | 优势 | 劣势 |
|--------|---------|---------|------|------|
| **MID-360** | 负障碍物检测（主力）<br>远距离覆盖 | 3.7-50m<br>360° | 远距离<br>全方位<br>不受光照影响 | 近距离盲区<br>点云密度低 |
| **OAK-D** | 近距离负障碍物（辅助）<br>正障碍物检测（主力） | 0.5-10m<br>前方72° | 高精度<br>高密度<br>RGB+深度 | 受光照影响<br>范围有限 |

### 1.3 融合策略

**距离分区融合**：

```
距离范围          主传感器      辅助传感器      融合方式
─────────────────────────────────────────────────────────
0.0 - 0.5m       无            无             盲区
0.5 - 3.7m       OAK-D         无             OAK-D单独
3.7 - 10m        MID-360       OAK-D          加权融合
10m - 50m        MID-360       无             MID-360单独
```

---

## 2. 硬件配置与安装

### 2.1 硬件清单

| 设备 | 型号 | 数量 | 价格 | 用途 |
|------|------|------|------|------|
| 激光雷达 | Livox MID-360 | 1 | ¥4,999 | 负障碍物检测（远距离） |
| 深度相机 | OAK-D | 1 | $199 | 负障碍物检测（近距离）<br>正障碍物检测 |
| 计算平台 | Jetson Xavier NX | 1 | ¥2,999 | 数据处理 |
| 安装支架 | 可调角度支架 | 2 | ¥200 | 传感器安装 |

**总成本**：约 ¥8,500

### 2.2 安装位置

**拖拉机顶部布局**：

```
                拖拉机前进方向 →
                
        ┌─────────────────────────┐
        │                         │
        │      MID-360            │  ← 向下倾斜15°
        │         ●               │     安装高度: 2.0m
        │                         │
        │                         │
        │      OAK-D              │  ← 水平安装0°
        │         ▣               │     安装高度: 1.5m
        │                         │
        └─────────────────────────┘
        
        侧视图：
        
                MID-360
                   ●
                    ╲ 15°
                     ╲
        ─────────────●──────────  ← OAK-D (水平)
                      ╲
                       ╲
        ════════════════════════  ← 地面
```

### 2.3 安装参数

**MID-360**：
- 安装高度：2.0米（相对地面）
- 倾斜角度：向下15度
- 安装位置：拖拉机顶部中心
- 朝向：前方

**OAK-D**：
- 安装高度：1.5米（相对地面）
- 倾斜角度：水平0度
- 安装位置：拖拉机前部中心
- 朝向：前方

**相对位置**：
- OAK-D在MID-360下方0.5米
- 两者在同一垂直平面上
- 水平偏移：0米（对齐）

---

## 3. 传感器标定

### 3.1 坐标系定义

**车体坐标系（基准）**：
- 原点：拖拉机后轴中心
- X轴：前进方向
- Y轴：左侧方向
- Z轴：向上方向

**MID-360坐标系**：
```python
# 相对车体坐标系的变换
T_lidar = {
    'translation': [2.5, 0.0, 2.0],  # 前2.5m, 左0m, 上2.0m
    'rotation': [15, 0, 0],  # 俯仰15度, 横滚0, 偏航0
}
```

**OAK-D坐标系**：
```python
# 相对车体坐标系的变换
T_camera = {
    'translation': [3.0, 0.0, 1.5],  # 前3.0m, 左0m, 上1.5m
    'rotation': [0, 0, 0],  # 俯仰0度, 横滚0, 偏航0
}
```

### 3.2 外参标定

**方法1：手动测量**（精度±5cm）

```python
import numpy as np

def get_transform_matrix(translation, rotation_deg):
    """
    计算变换矩阵
    
    Args:
        translation: [x, y, z] 平移（米）
        rotation_deg: [pitch, roll, yaw] 旋转（度）
    
    Returns:
        T: 4x4变换矩阵
    """
    # 旋转矩阵（欧拉角 -> 旋转矩阵）
    pitch, roll, yaw = np.radians(rotation_deg)
    
    # 俯仰（绕Y轴）
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # 横滚（绕X轴）
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # 偏航（绕Z轴）
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # 组合旋转
    R = Rz @ Ry @ Rx
    
    # 构建4x4变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    
    return T

# MID-360到车体的变换
T_lidar_to_vehicle = get_transform_matrix(
    [2.5, 0.0, 2.0],
    [15, 0, 0]
)

# OAK-D到车体的变换
T_camera_to_vehicle = get_transform_matrix(
    [3.0, 0.0, 1.5],
    [0, 0, 0]
)

# MID-360到OAK-D的变换
T_lidar_to_camera = np.linalg.inv(T_camera_to_vehicle) @ T_lidar_to_vehicle
```

**方法2：自动标定**（精度±1cm）

使用标定板自动标定（推荐）：

```python
# 使用棋盘格标定板
# 1. 在视野内放置标定板
# 2. 同时采集MID-360点云和OAK-D图像
# 3. 检测标定板角点
# 4. 使用PnP算法计算变换矩阵

# 参考：https://github.com/koide3/direct_visual_lidar_calibration
```

### 3.3 时间同步

**硬件同步**（推荐）：

```python
# 使用PPS信号同步
# MID-360和OAK-D都支持PPS输入

# 或使用软件时间戳
import time

class TimestampSynchronizer:
    """时间戳同步器"""
    
    def __init__(self):
        self.lidar_offset = 0.0  # 雷达时间偏移
        self.camera_offset = 0.0  # 相机时间偏移
    
    def sync_timestamp(self, lidar_ts, camera_ts):
        """同步时间戳"""
        lidar_ts_sync = lidar_ts + self.lidar_offset
        camera_ts_sync = camera_ts + self.camera_offset
        return lidar_ts_sync, camera_ts_sync
```

---

## 4. MID-360 AMFA算法实现

### 4.1 数据接收

**使用Livox SDK**：

```python
import numpy as np
from livox import LivoxLidar

class MID360DataReceiver:
    """MID-360数据接收器"""
    
    def __init__(self):
        self.lidar = LivoxLidar()
        self.point_buffer = []
        
    def start(self):
        """启动数据接收"""
        self.lidar.connect()
        self.lidar.start_sampling()
        self.lidar.register_callback(self.point_callback)
    
    def point_callback(self, points):
        """点云回调"""
        # points: Nx3数组 (x, y, z)
        self.point_buffer.extend(points)
    
    def get_frame(self):
        """获取一帧点云"""
        # MID-360是100ms一帧
        if len(self.point_buffer) > 0:
            frame = np.array(self.point_buffer)
            self.point_buffer = []
            return frame
        return None
```

### 4.2 AMFA检测器集成

```python
from amfa_mid360_detector import AMFANegativeObstacleDetector

class MID360NegativeObstacleDetector:
    """MID-360负障碍物检测器"""
    
    def __init__(self, lidar_height=2.0, tilt_angle=15):
        # AMFA检测器
        self.amfa_detector = AMFANegativeObstacleDetector(
            lidar_height=lidar_height,
            tilt_angle=tilt_angle
        )
        
        # 数据接收器
        self.data_receiver = MID360DataReceiver()
        
        # 变换矩阵（雷达坐标系 -> 车体坐标系）
        self.T_lidar_to_vehicle = get_transform_matrix(
            [2.5, 0.0, 2.0],
            [15, 0, 0]
        )
    
    def start(self):
        """启动检测"""
        self.data_receiver.start()
    
    def detect(self):
        """检测负障碍物"""
        # 获取点云
        points_lidar = self.data_receiver.get_frame()
        
        if points_lidar is None:
            return []
        
        # 转换到车体坐标系
        points_vehicle = self.transform_points(
            points_lidar,
            self.T_lidar_to_vehicle
        )
        
        # AMFA检测
        obstacles = self.amfa_detector.detect(points_vehicle)
        
        return obstacles
    
    def transform_points(self, points, T):
        """变换点云"""
        # 转换为齐次坐标
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        
        # 应用变换
        points_transformed = (T @ points_homo.T).T
        
        # 返回3D坐标
        return points_transformed[:, :3]
```

---

## 5. OAK-D深度检测实现

### 5.1 数据接收

**使用DepthAI SDK**：

```python
import depthai as dai
import cv2

class OAKDDataReceiver:
    """OAK-D数据接收器"""
    
    def __init__(self):
        self.pipeline = self.create_pipeline()
        self.device = None
        
    def create_pipeline(self):
        """创建DepthAI pipeline"""
        pipeline = dai.Pipeline()
        
        # 立体深度节点
        stereo = pipeline.createStereoDepth()
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(640, 480)
        
        # RGB相机
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setIspScale(1, 3)  # 640x360
        
        # 输出
        xout_depth = pipeline.createXLinkOut()
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)
        
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.isp.link(xout_rgb.input)
        
        return pipeline
    
    def start(self):
        """启动设备"""
        self.device = dai.Device(self.pipeline)
        self.q_depth = self.device.getOutputQueue("depth", maxSize=4, blocking=False)
        self.q_rgb = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)
    
    def get_frame(self):
        """获取一帧数据"""
        depth_frame = self.q_depth.get()
        rgb_frame = self.q_rgb.get()
        
        depth = depth_frame.getFrame()
        rgb = rgb_frame.getCvFrame()
        
        return rgb, depth
```

### 5.2 深度检测器

```python
class OAKDNegativeObstacleDetector:
    """OAK-D负障碍物检测器"""
    
    def __init__(self, camera_height=1.5):
        self.camera_height = camera_height
        
        # 数据接收器
        self.data_receiver = OAKDDataReceiver()
        
        # 相机内参（OAK-D默认）
        self.fx = 640  # 焦距x
        self.fy = 640  # 焦距y
        self.cx = 320  # 主点x
        self.cy = 240  # 主点y
        
        # 变换矩阵
        self.T_camera_to_vehicle = get_transform_matrix(
            [3.0, 0.0, 1.5],
            [0, 0, 0]
        )
        
        # 检测参数
        self.min_depth = 0.15  # 最小深度（米）
        self.max_depth = 10.0  # 最大深度（米）
        self.ground_threshold = 0.05  # 地面阈值
    
    def start(self):
        """启动检测"""
        self.data_receiver.start()
    
    def detect(self):
        """检测负障碍物"""
        # 获取RGB和深度图
        rgb, depth = self.data_receiver.get_frame()
        
        # 转换深度图为点云
        points_camera = self.depth_to_pointcloud(depth)
        
        # 转换到车体坐标系
        points_vehicle = self.transform_points(
            points_camera,
            self.T_camera_to_vehicle
        )
        
        # 检测负障碍物
        obstacles = self.detect_negative_obstacles(points_vehicle)
        
        return obstacles
    
    def depth_to_pointcloud(self, depth):
        """深度图转点云"""
        h, w = depth.shape
        points = []
        
        for v in range(h):
            for u in range(w):
                z = depth[v, u] / 1000.0  # mm -> m
                
                if z < 0.3 or z > 10.0:
                    continue
                
                # 反投影
                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy
                
                points.append([x, y, z])
        
        return np.array(points)
    
    def detect_negative_obstacles(self, points):
        """检测负障碍物（RANSAC + 梯度）"""
        if len(points) < 100:
            return []
        
        # 地面分割
        ground_mask = self.segment_ground(points)
        
        # 找到低于地面的点
        negative_mask = points[:, 2] < -self.min_depth
        negative_mask = negative_mask & ~ground_mask
        
        if np.sum(negative_mask) < 10:
            return []
        
        # 聚类
        negative_points = points[negative_mask]
        obstacles = self.cluster_obstacles(negative_points)
        
        return obstacles
    
    def segment_ground(self, points):
        """地面分割（简化版RANSAC）"""
        # 与MID-360检测器相同的逻辑
        # ...（省略，参考前面的代码）
        pass
    
    def cluster_obstacles(self, points):
        """聚类负障碍物"""
        from sklearn.cluster import DBSCAN
        
        # DBSCAN聚类
        clustering = DBSCAN(eps=0.3, min_samples=5).fit(points)
        labels = clustering.labels_
        
        obstacles = []
        for label in set(labels):
            if label == -1:
                continue
            
            cluster_points = points[labels == label]
            
            # 计算特征
            center = np.mean(cluster_points, axis=0)
            depth = np.abs(np.min(cluster_points[:, 2]))
            width = np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0])
            
            obstacle = {
                'center': center,
                'distance': np.linalg.norm(center[:2]),
                'width': width,
                'depth': depth,
                'confidence': 0.8,
                'source': 'oakd'
            }
            
            obstacles.append(obstacle)
        
        return obstacles
    
    def transform_points(self, points, T):
        """变换点云"""
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        points_transformed = (T @ points_homo.T).T
        return points_transformed[:, :3]
```

---

## 6. 传感器融合算法

### 6.1 融合策略

**多层融合架构**：

```
层级1：数据层融合（点云融合）
    ↓
层级2：特征层融合（障碍物融合）
    ↓
层级3：决策层融合（优先级分配）
```

### 6.2 卡尔曼滤波融合

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanFilter:
    """卡尔曼滤波器（用于障碍物跟踪）"""
    
    def __init__(self, dt=0.1):
        self.dt = dt
        
        # 状态向量 [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)
        
        # 状态协方差
        self.P = np.eye(6) * 1.0
        
        # 状态转移矩阵
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        
        # 过程噪声
        self.Q = np.eye(6) * 0.1
        
        # 观测矩阵
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])
        
        # 观测噪声
        self.R = np.eye(3) * 0.5
    
    def predict(self):
        """预测步骤"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z, R=None):
        """更新步骤"""
        if R is not None:
            self.R = R
        
        # 创新
        y = z - self.H @ self.x
        
        # 创新协方差
        S = self.H @ self.P @ self.H.T + self.R
        
        # 卡尔曼增益
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        self.x = self.x + K @ y
        
        # 更新协方差
        self.P = (np.eye(6) - K @ self.H) @ self.P
    
    def get_state(self):
        """获取状态"""
        return self.x[:3]  # 返回位置

class SensorFusion:
    """传感器融合器"""
    
    def __init__(self):
        # 跟踪的障碍物列表
        self.tracked_obstacles = []
        
        # 融合参数
        self.association_threshold = 1.0  # 关联阈值（米）
        self.max_age = 5  # 最大年龄（帧数）
        
    def fuse(self, lidar_obstacles, camera_obstacles):
        """
        融合MID-360和OAK-D的检测结果
        
        Args:
            lidar_obstacles: MID-360检测的障碍物
            camera_obstacles: OAK-D检测的障碍物
        
        Returns:
            fused_obstacles: 融合后的障碍物
        """
        # 步骤1：预测所有跟踪的障碍物
        for track in self.tracked_obstacles:
            track['kf'].predict()
            track['age'] += 1
        
        # 步骤2：数据关联
        all_detections = []
        
        # 添加雷达检测
        for obs in lidar_obstacles:
            all_detections.append({
                'position': obs.center,
                'width': obs.width,
                'depth': obs.depth,
                'confidence': obs.confidence,
                'source': 'lidar',
                'R': np.eye(3) * 0.3  # 雷达观测噪声较小
            })
        
        # 添加相机检测
        for obs in camera_obstacles:
            all_detections.append({
                'position': obs['center'],
                'width': obs['width'],
                'depth': obs['depth'],
                'confidence': obs['confidence'],
                'source': 'camera',
                'R': np.eye(3) * 0.5  # 相机观测噪声较大
            })
        
        # 步骤3：匈牙利算法关联
        matched_pairs, unmatched_detections, unmatched_tracks = self.associate(
            all_detections,
            self.tracked_obstacles
        )
        
        # 步骤4：更新匹配的跟踪
        for det_idx, track_idx in matched_pairs:
            detection = all_detections[det_idx]
            track = self.tracked_obstacles[track_idx]
            
            # 卡尔曼更新
            track['kf'].update(detection['position'], detection['R'])
            
            # 更新属性
            track['width'] = 0.7 * track['width'] + 0.3 * detection['width']
            track['depth'] = 0.7 * track['depth'] + 0.3 * detection['depth']
            track['confidence'] = min(1.0, track['confidence'] + 0.1)
            track['age'] = 0
            track['source'] = detection['source']
        
        # 步骤5：创建新跟踪
        for det_idx in unmatched_detections:
            detection = all_detections[det_idx]
            
            kf = KalmanFilter()
            kf.x[:3] = detection['position']
            
            self.tracked_obstacles.append({
                'kf': kf,
                'width': detection['width'],
                'depth': detection['depth'],
                'confidence': detection['confidence'],
                'age': 0,
                'source': detection['source']
            })
        
        # 步骤6：删除老旧跟踪
        self.tracked_obstacles = [
            track for track in self.tracked_obstacles
            if track['age'] < self.max_age
        ]
        
        # 步骤7：输出融合结果
        fused_obstacles = []
        for track in self.tracked_obstacles:
            if track['confidence'] > 0.5:
                fused_obstacles.append({
                    'center': track['kf'].get_state(),
                    'distance': np.linalg.norm(track['kf'].get_state()[:2]),
                    'width': track['width'],
                    'depth': track['depth'],
                    'confidence': track['confidence'],
                    'source': track['source']
                })
        
        return fused_obstacles
    
    def associate(self, detections, tracks):
        """
        数据关联（匈牙利算法）
        
        Returns:
            matched_pairs: [(det_idx, track_idx), ...]
            unmatched_detections: [det_idx, ...]
            unmatched_tracks: [track_idx, ...]
        """
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        # 计算代价矩阵（欧氏距离）
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                dist = np.linalg.norm(
                    det['position'] - track['kf'].get_state()
                )
                cost_matrix[i, j] = dist
        
        # 匈牙利算法
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 过滤距离过大的匹配
        matched_pairs = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.association_threshold:
                matched_pairs.append((i, j))
        
        # 未匹配的检测和跟踪
        matched_det_indices = set([i for i, j in matched_pairs])
        matched_track_indices = set([j for i, j in matched_pairs])
        
        unmatched_detections = [
            i for i in range(len(detections))
            if i not in matched_det_indices
        ]
        
        unmatched_tracks = [
            j for j in range(len(tracks))
            if j not in matched_track_indices
        ]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
```

### 6.3 决策融合

```python
class DecisionFusion:
    """决策融合器"""
    
    def __init__(self):
        # 安全距离配置
        self.emergency_distance = 3.0  # 紧急停止距离
        self.warning_distance = 8.0  # 警告距离
        self.safe_distance = 15.0  # 安全距离
    
    def make_decision(self, fused_obstacles):
        """
        根据融合后的障碍物做出决策
        
        Args:
            fused_obstacles: 融合后的障碍物列表
        
        Returns:
            decision: 决策结果
        """
        if len(fused_obstacles) == 0:
            return {
                'action': 'continue',
                'speed_factor': 1.0,
                'steering_adjustment': 0.0,
                'warning_level': 'safe'
            }
        
        # 找到最近的障碍物
        closest_obstacle = min(
            fused_obstacles,
            key=lambda obs: obs['distance']
        )
        
        distance = closest_obstacle['distance']
        confidence = closest_obstacle['confidence']
        
        # 根据距离和置信度做决策
        if distance < self.emergency_distance and confidence > 0.7:
            # 紧急停止
            return {
                'action': 'emergency_stop',
                'speed_factor': 0.0,
                'steering_adjustment': 0.0,
                'warning_level': 'emergency',
                'obstacle': closest_obstacle
            }
        
        elif distance < self.warning_distance and confidence > 0.6:
            # 减速并尝试绕行
            speed_factor = (distance - self.emergency_distance) / \
                          (self.warning_distance - self.emergency_distance)
            
            # 计算绕行方向
            obstacle_y = closest_obstacle['center'][1]
            steering_adjustment = -np.sign(obstacle_y) * 15  # 度
            
            return {
                'action': 'slow_and_avoid',
                'speed_factor': max(0.3, speed_factor),
                'steering_adjustment': steering_adjustment,
                'warning_level': 'warning',
                'obstacle': closest_obstacle
            }
        
        elif distance < self.safe_distance:
            # 轻微减速
            speed_factor = 0.7
            
            return {
                'action': 'slow_down',
                'speed_factor': speed_factor,
                'steering_adjustment': 0.0,
                'warning_level': 'caution',
                'obstacle': closest_obstacle
            }
        
        else:
            # 继续前进
            return {
                'action': 'continue',
                'speed_factor': 1.0,
                'steering_adjustment': 0.0,
                'warning_level': 'safe'
            }
```

---

## 7. 完整代码实现

### 7.1 主系统类

```python
import threading
import time

class NegativeObstacleDetectionSystem:
    """负障碍物检测系统（MID-360 + OAK-D融合）"""
    
    def __init__(self):
        # MID-360检测器
        self.lidar_detector = MID360NegativeObstacleDetector(
            lidar_height=2.0,
            tilt_angle=15
        )
        
        # OAK-D检测器
        self.camera_detector = OAKDNegativeObstacleDetector(
            camera_height=1.5
        )
        
        # 传感器融合器
        self.sensor_fusion = SensorFusion()
        
        # 决策融合器
        self.decision_fusion = DecisionFusion()
        
        # 运行状态
        self.running = False
        
        # 结果缓存
        self.latest_decision = None
        self.latest_obstacles = []
        
        # 性能统计
        self.fps = 0
        self.processing_time = 0
    
    def start(self):
        """启动系统"""
        print("[System] 启动负障碍物检测系统...")
        
        # 启动传感器
        self.lidar_detector.start()
        self.camera_detector.start()
        
        # 启动主循环
        self.running = True
        self.main_thread = threading.Thread(target=self.main_loop)
        self.main_thread.start()
        
        print("[System] 系统启动完成")
    
    def stop(self):
        """停止系统"""
        print("[System] 停止系统...")
        self.running = False
        self.main_thread.join()
        print("[System] 系统已停止")
    
    def main_loop(self):
        """主循环"""
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            loop_start = time.time()
            
            # 步骤1：MID-360检测
            lidar_obstacles = self.lidar_detector.detect()
            
            # 步骤2：OAK-D检测
            camera_obstacles = self.camera_detector.detect()
            
            # 步骤3：传感器融合
            fused_obstacles = self.sensor_fusion.fuse(
                lidar_obstacles,
                camera_obstacles
            )
            
            # 步骤4：决策融合
            decision = self.decision_fusion.make_decision(fused_obstacles)
            
            # 更新结果
            self.latest_decision = decision
            self.latest_obstacles = fused_obstacles
            
            # 性能统计
            loop_time = time.time() - loop_start
            self.processing_time = loop_time
            
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                self.fps = frame_count / elapsed
                
                print(f"[System] FPS: {self.fps:.1f}, "
                      f"处理时间: {loop_time*1000:.1f}ms, "
                      f"障碍物数: {len(fused_obstacles)}, "
                      f"决策: {decision['action']}")
            
            # 控制帧率（最大10Hz）
            sleep_time = max(0, 0.1 - loop_time)
            time.sleep(sleep_time)
    
    def get_decision(self):
        """获取最新决策"""
        return self.latest_decision
    
    def get_obstacles(self):
        """获取最新障碍物列表"""
        return self.latest_obstacles
```

### 7.2 使用示例

```python
def main():
    """主函数"""
    
    print("=" * 60)
    print("MID-360 + OAK-D 负障碍物检测系统")
    print("=" * 60)
    
    # 创建系统
    system = NegativeObstacleDetectionSystem()
    
    # 启动系统
    system.start()
    
    try:
        # 运行60秒
        for i in range(600):
            time.sleep(0.1)
            
            # 获取决策
            decision = system.get_decision()
            
            if decision and decision['warning_level'] != 'safe':
                print(f"\n[警告] {decision['warning_level'].upper()}")
                print(f"  动作: {decision['action']}")
                print(f"  速度因子: {decision['speed_factor']:.2f}")
                
                if 'obstacle' in decision:
                    obs = decision['obstacle']
                    print(f"  障碍物距离: {obs['distance']:.2f}m")
                    print(f"  障碍物深度: {obs['depth']:.2f}m")
                    print(f"  置信度: {obs['confidence']:.2f}")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        # 停止系统
        system.stop()
        print("系统已安全退出")

if __name__ == "__main__":
    main()
```

---

## 8. 性能优化

### 8.1 多线程优化

```python
import queue
from concurrent.futures import ThreadPoolExecutor

class OptimizedDetectionSystem(NegativeObstacleDetectionSystem):
    """优化的检测系统（多线程）"""
    
    def __init__(self):
        super().__init__()
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # 结果队列
        self.lidar_queue = queue.Queue(maxsize=2)
        self.camera_queue = queue.Queue(maxsize=2)
    
    def main_loop(self):
        """优化的主循环"""
        while self.running:
            # 并行检测
            lidar_future = self.executor.submit(self.lidar_detector.detect)
            camera_future = self.executor.submit(self.camera_detector.detect)
            
            # 等待结果
            lidar_obstacles = lidar_future.result()
            camera_obstacles = camera_future.result()
            
            # 融合和决策（串行）
            fused_obstacles = self.sensor_fusion.fuse(
                lidar_obstacles,
                camera_obstacles
            )
            
            decision = self.decision_fusion.make_decision(fused_obstacles)
            
            # 更新结果
            self.latest_decision = decision
            self.latest_obstacles = fused_obstacles
```

### 8.2 ROI处理

```python
def apply_roi(points, roi_config):
    """
    应用ROI（感兴趣区域）
    
    只处理拖拉机前方和侧方的点云，忽略后方
    """
    # ROI配置
    x_min, x_max = roi_config['x']  # 例如：[0, 50]
    y_min, y_max = roi_config['y']  # 例如：[-5, 5]
    z_min, z_max = roi_config['z']  # 例如：[-2, 2]
    
    # 过滤
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    
    return points[mask]
```

### 8.3 降采样

```python
def voxel_downsample(points, voxel_size=0.1):
    """
    体素降采样
    
    减少点云数量，提高处理速度
    """
    import open3d as o3d
    
    # 转换为Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 体素降采样
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # 转换回numpy
    points_down = np.asarray(pcd_down.points)
    
    return points_down
```

---

## 9. 实际部署指南

### 9.1 硬件连接

**接线图**：

```
Jetson Xavier NX
    │
    ├─ USB 3.0 ──→ OAK-D
    │
    ├─ Ethernet ──→ MID-360
    │
    └─ CAN Bus ──→ 拖拉机控制器
```

### 9.2 软件部署

**步骤1：安装依赖**

```bash
# 系统依赖
sudo apt-get update
sudo apt-get install -y python3-pip cmake

# Python依赖
pip3 install numpy opencv-python scikit-learn

# DepthAI（OAK-D）
pip3 install depthai

# Livox SDK（MID-360）
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2
mkdir build && cd build
cmake .. && make
sudo make install

# Open3D（可选，用于可视化）
pip3 install open3d
```

**步骤2：配置权限**

```bash
# USB权限（OAK-D）
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# 网络权限（MID-360）
sudo setcap cap_net_raw+ep $(which python3)
```

**步骤3：运行系统**

```bash
# 启动检测系统
python3 negative_obstacle_detection_system.py

# 或作为服务运行
sudo systemctl start negative-obstacle-detection
```

### 9.3 性能基准

**预期性能**（Jetson Xavier NX）：

| 指标 | 数值 |
|------|------|
| 总帧率 | 8-10 Hz |
| MID-360处理时间 | 80-100ms |
| OAK-D处理时间 | 40-60ms |
| 融合时间 | 10-20ms |
| 总延迟 | <150ms |
| CPU占用 | 40-60% |
| 内存占用 | 1.5-2.0GB |
| 功耗 | 15-20W |

### 9.4 调试工具

**可视化工具**：

```python
import open3d as o3d
import matplotlib.pyplot as plt

class Visualizer:
    """可视化工具"""
    
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
    def update(self, points, obstacles):
        """更新可视化"""
        # 显示点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 显示障碍物（红色球体）
        for obs in obstacles:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
            sphere.translate(obs['center'])
            sphere.paint_uniform_color([1, 0, 0])
            self.vis.add_geometry(sphere)
        
        self.vis.add_geometry(pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
```

---

## 10. 总结

### 10.1 系统优势

**完整的传感器融合方案**：

✅ **MID-360**：远距离、全方位、高鲁棒性  
✅ **OAK-D**：近距离、高精度、RGB+深度  
✅ **AMFA算法**：基于物理模型、自适应、低误检  
✅ **卡尔曼融合**：平滑跟踪、降低噪声  
✅ **决策融合**：智能决策、分级响应

### 10.2 覆盖范围

| 距离范围 | 传感器 | 检测能力 |
|---------|--------|---------|
| 0.0-0.5m | 盲区 | 无 |
| 0.5-3.7m | OAK-D | 高精度 |
| 3.7-10m | MID-360 + OAK-D | 双重保障 |
| 10-50m | MID-360 | 远距离预警 |

### 10.3 实施路线

**第一阶段**（2周）：
- 硬件安装和标定
- 单传感器测试

**第二阶段**（2周）：
- 融合算法实现
- 参数调优

**第三阶段**（1周）：
- 实地测试
- 性能验证

**总时间**：5周

