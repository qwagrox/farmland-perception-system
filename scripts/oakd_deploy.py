#!/usr/bin/env python3
"""
OAK-D部署脚本
功能：在OAK-D相机上运行行人和拖拉机检测
"""

import cv2
import depthai as dai
import numpy as np
import time
import argparse
from pathlib import Path

# 类别配置
LABELS = ['person', 'tractor']
COLORS = {
    'person': (0, 255, 0),    # 绿色
    'tractor': (0, 0, 255)    # 红色
}

# 安全距离配置
SAFE_DISTANCE = 5.0      # 紧急停止距离（米）
WARNING_DISTANCE = 10.0  # 警告距离（米）

class OAKDDetector:
    """OAK-D检测器类"""
    
    def __init__(self, blob_path, conf_threshold=0.5, iou_threshold=0.5):
        """
        初始化检测器
        
        Args:
            blob_path: Blob模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
        """
        self.blob_path = blob_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.pipeline = None
        self.device = None
        
    def create_pipeline(self):
        """创建DepthAI pipeline"""
        print("创建DepthAI pipeline...")
        
        pipeline = dai.Pipeline()
        
        # RGB相机
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(416, 416)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(30)
        
        # 单目相机（用于立体深度）
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        # 立体深度
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        # 神经网络
        detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
        detection_nn.setBlobPath(self.blob_path)
        detection_nn.setConfidenceThreshold(self.conf_threshold)
        detection_nn.setNumClasses(len(LABELS))
        detection_nn.setCoordinateSize(4)
        detection_nn.setAnchors([])
        detection_nn.setAnchorMasks({})
        detection_nn.setIouThreshold(self.iou_threshold)
        detection_nn.setNumInferenceThreads(2)
        detection_nn.input.setBlocking(False)
        
        cam_rgb.preview.link(detection_nn.input)
        
        # 输出
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)
        
        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("detections")
        detection_nn.out.link(xout_nn.input)
        
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)
        
        self.pipeline = pipeline
        print("✅ Pipeline创建成功")
        
        return pipeline
    
    def calculate_distance(self, depth_frame, bbox):
        """
        计算目标距离
        
        Args:
            depth_frame: 深度帧
            bbox: 边界框 [x1, y1, x2, y2]
        
        Returns:
            distance: 距离（米）
        """
        x1, y1, x2, y2 = bbox
        
        # 调整坐标到深度帧尺寸
        h, w = depth_frame.shape
        x1 = int(x1 * w / 416)
        y1 = int(y1 * h / 416)
        x2 = int(x2 * w / 416)
        y2 = int(y2 * h / 416)
        
        # 获取中心区域
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 采样区域
        roi_size = 5
        y_min = max(0, center_y - roi_size)
        y_max = min(h, center_y + roi_size)
        x_min = max(0, center_x - roi_size)
        x_max = min(w, center_x + roi_size)
        
        depth_roi = depth_frame[y_min:y_max, x_min:x_max]
        
        # 过滤无效深度值
        valid_depths = depth_roi[depth_roi > 0]
        
        if len(valid_depths) == 0:
            return -1
        
        # 使用中位数
        distance = np.median(valid_depths) / 1000.0  # 转换为米
        
        return distance
    
    def run(self, show_fps=True, save_video=False, output_path='output.avi'):
        """
        运行检测
        
        Args:
            show_fps: 是否显示FPS
            save_video: 是否保存视频
            output_path: 视频保存路径
        """
        if self.pipeline is None:
            self.create_pipeline()
        
        print("\n正在连接OAK-D相机...")
        
        try:
            with dai.Device(self.pipeline) as device:
                print("✅ OAK-D相机已连接")
                
                # 获取输出队列
                q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                q_det = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
                q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
                
                # 视频写入器
                video_writer = None
                if save_video:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (416, 416))
                
                frame_count = 0
                start_time = time.time()
                fps = 0
                
                print("\n开始检测...")
                print("按 'q' 退出, 'p' 暂停, 's' 截图")
                print("=" * 60)
                
                paused = False
                
                while True:
                    if not paused:
                        # 获取帧
                        in_rgb = q_rgb.get()
                        in_depth = q_depth.get()
                        in_det = q_det.get()
                        
                        frame = in_rgb.getCvFrame()
                        depth_frame = in_depth.getFrame()
                        detections = in_det.detections
                        
                        # 处理检测结果
                        closest_distance = float('inf')
                        closest_label = None
                        
                        for detection in detections:
                            # 边界框
                            bbox = [
                                int(detection.xmin * 416),
                                int(detection.ymin * 416),
                                int(detection.xmax * 416),
                                int(detection.ymax * 416)
                            ]
                            
                            # 类别和置信度
                            label_id = detection.label
                            if label_id >= len(LABELS):
                                continue
                            
                            label = LABELS[label_id]
                            confidence = detection.confidence
                            
                            # 计算距离
                            distance = self.calculate_distance(depth_frame, bbox)
                            
                            if distance < 0:
                                distance_text = "N/A"
                            else:
                                distance_text = f"{distance:.2f}m"
                                
                                # 记录最近的目标
                                if distance < closest_distance:
                                    closest_distance = distance
                                    closest_label = label
                            
                            # 绘制边界框
                            color = COLORS.get(label, (255, 255, 255))
                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                            
                            # 绘制标签
                            text = f"{label}: {confidence:.2f} | {distance_text}"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(frame, (bbox[0], bbox[1] - text_size[1] - 10),
                                        (bbox[0] + text_size[0], bbox[1]), color, -1)
                            cv2.putText(frame, text, (bbox[0], bbox[1] - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        # 安全警告
                        if closest_distance < SAFE_DISTANCE:
                            warning_text = f"STOP! {closest_label} at {closest_distance:.2f}m"
                            cv2.putText(frame, warning_text, (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            print(f"🛑 紧急停止: {closest_label} 距离 {closest_distance:.2f}m")
                        elif closest_distance < WARNING_DISTANCE:
                            warning_text = f"SLOW! {closest_label} at {closest_distance:.2f}m"
                            cv2.putText(frame, warning_text, (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                            print(f"⚠️  减速: {closest_label} 距离 {closest_distance:.2f}m")
                        
                        # 显示FPS
                        if show_fps:
                            frame_count += 1
                            if frame_count % 30 == 0:
                                fps = frame_count / (time.time() - start_time)
                            
                            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # 保存视频
                        if video_writer is not None:
                            video_writer.write(frame)
                        
                        # 显示
                        cv2.imshow("OAK-D Detection", frame)
                    
                    # 键盘控制
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        paused = not paused
                        print("暂停" if paused else "继续")
                    elif key == ord('s'):
                        screenshot_path = f"screenshot_{int(time.time())}.jpg"
                        cv2.imwrite(screenshot_path, frame)
                        print(f"截图已保存: {screenshot_path}")
                
                # 清理
                if video_writer is not None:
                    video_writer.release()
                
                cv2.destroyAllWindows()
                print("\n检测已停止")
                
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}")
            print("\n请检查:")
            print("1. OAK-D相机是否正确连接")
            print("2. Blob文件是否存在且有效")
            print("3. 是否已安装depthai: pip install depthai")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='OAK-D行人和拖拉机检测')
    parser.add_argument('--blob', type=str, default='models/best.blob',
                       help='Blob模型路径')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IOU阈值')
    parser.add_argument('--save', action='store_true',
                       help='保存检测视频')
    parser.add_argument('--output', type=str, default='output.avi',
                       help='输出视频路径')
    
    args = parser.parse_args()
    
    # 检查blob文件
    if not Path(args.blob).exists():
        print(f"❌ 错误: Blob文件不存在: {args.blob}")
        print("请先运行 convert_to_blob.py 转换模型")
        return
    
    print("=" * 60)
    print("OAK-D 行人与拖拉机检测系统")
    print("=" * 60)
    print(f"模型: {args.blob}")
    print(f"置信度阈值: {args.conf}")
    print(f"IOU阈值: {args.iou}")
    print(f"安全距离: {SAFE_DISTANCE}m")
    print(f"警告距离: {WARNING_DISTANCE}m")
    print("=" * 60)
    
    # 创建检测器
    detector = OAKDDetector(
        blob_path=args.blob,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # 运行检测
    detector.run(show_fps=True, save_video=args.save, output_path=args.output)

if __name__ == "__main__":
    main()

