import threading
import cv2
import numpy as np
import os
import time
import queue
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO


class YoloSegNode(Node):
    def __init__(self):
        super().__init__('yolo_seg_node')

        # --- 1. 参数声明与路径处理 ---
        self.declare_parameter('input_topic', '/ascamera/camera_publisher/rgb0/image')
        self.declare_parameter('depth_topic', '/ascamera/camera_publisher/depth0/image_raw')
        self.declare_parameter('output_topic', '/yolo_result')

        try:
            package_share_dir = get_package_share_directory('yolo_seg_ros2')
            default_model_path = os.path.join(package_share_dir, 'YOLO_26n_crack.pt')
        except Exception:
            default_model_path = 'YOLO_26n_crack.pt'

        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('process_fps', 5.0)
        self.declare_parameter('smooth_win', 5)  # 平滑窗口

        # 获取参数值
        self.model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.imgsz = self.get_parameter('imgsz').value
        self.process_fps = self.get_parameter('process_fps').value

        # --- 2. 核心组件初始化 ---
        self.bridge = CvBridge()
        self.get_logger().info(f'Loading model: {self.model_path}')
        self.model = YOLO(self.model_path)

        # 量化线程与队列
        self.task_q = queue.Queue(maxsize=2)
        self.latest_metrics = {"area_px": 0, "length_px": 0, "avg_width_px": 0.0, "max_width_px": 0.0}
        self.hist_metrics = {k: deque(maxlen=self.get_parameter('smooth_win').value) for k in
                             self.latest_metrics.keys()}

        self.stop_event = threading.Event()
        self.quant_thread = threading.Thread(target=self.quant_worker, daemon=True)
        self.quant_thread.start()

        # 状态控制
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_msg = None
        self.processing = False
        self.prev_time = time.time()

        # --- 3. ROS 订阅与发布 ---
        sub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(Image, self.get_parameter('input_topic').value, self.image_callback, sub_qos)
        self.publisher = self.create_publisher(Image, self.get_parameter('output_topic').value, 10)

        # 定时器触发推理
        timer_period = 1.0 / max(self.process_fps, 0.1)
        self.timer = self.create_timer(timer_period, self.process_latest_frame)

        self.get_logger().info('YOLO Quant Node with Auto-Reset initialized.')

    # ===========================
    # 量化计算工作线程
    # ===========================
    def quant_worker(self):
        """后台处理耗时的骨架化和宽度计算"""
        while not self.stop_event.is_set():
            try:
                mask01 = self.task_q.get(timeout=0.2)

                # 计算当前帧指标
                metrics = self.calculate_metrics(mask01)

                # 更新平滑队列并计算均值
                with self.frame_lock:
                    for k, v in metrics.items():
                        self.hist_metrics[k].append(v)
                        self.latest_metrics[k] = float(np.mean(self.hist_metrics[k]))

                self.task_q.task_done()
            except queue.Empty:
                continue

    def calculate_metrics(self, mask01):
        """核心量化算法"""
        area_px = int(mask01.sum())
        # 如果 Mask 为全黑（没检测到），直接返回全 0
        if area_px <= 0:
            return {"area_px": 0, "length_px": 0, "avg_width_px": 0.0, "max_width_px": 0.0}

        # 1. 最大宽度: 距离变换 (Distance Transform)
        dist = cv2.distanceTransform(mask01, cv2.DIST_L2, 3)
        max_width_px = 2.0 * float(dist.max())

        # 2. 长度: 骨架化 (Thinning)
        length_px = 0
        try:
            # 优先使用 OpenCV 扩展库的快速算法
            if hasattr(cv2, 'ximgproc'):
                skeleton = cv2.ximgproc.thinning((mask01 * 255).astype(np.uint8))
                length_px = int((skeleton > 0).sum())
            else:
                # 兜底方案：使用简单的轮廓周长一半作为近似长度
                contours, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    length_px += cv2.arcLength(cnt, True) / 2.0
        except Exception:
            length_px = area_px // 5  # 极简兜底

        # 3. 平均宽度: 面积 / 长度
        avg_width_px = float(area_px) / float(length_px + 1e-6)

        return {
            "area_px": area_px,
            "length_px": int(length_px),
            "avg_width_px": avg_width_px,
            "max_width_px": max_width_px
        }

    # ===========================
    # 图像回调
    # ===========================
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_frame = cv_image
                self.latest_msg = msg
        except Exception as e:
            self.get_logger().error(f'Image Convert Error: {e}')

    def process_latest_frame(self):
        if self.processing or self.latest_frame is None:
            return

        with self.frame_lock:
            frame = self.latest_frame.copy()
            msg = self.latest_msg
            metrics_display = self.latest_metrics.copy()
            self.latest_frame = None

        self.processing = True
        try:
            results = self.model.predict(source=frame, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False)

            # --- 关键修改：处理检测结果并强制刷新量化线程 ---
            h, w = frame.shape[:2]
            if results[0].masks is not None and len(results[0].masks) > 0:
                # 合并所有实例的 Mask
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                masks_data = results[0].masks.data.cpu().numpy()
                for m in masks_data:
                    m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    combined_mask = np.maximum(combined_mask, (m_resized > 0.5).astype(np.uint8))

                # 发送到任务队列
                if not self.task_q.full():
                    self.task_q.put_nowait(combined_mask)
            else:
                # 【修复核心】：如果没有检测到，发送空 Mask 强制重置数值
                empty_mask = np.zeros((h, w), dtype=np.uint8)
                if not self.task_q.full():
                    self.task_q.put_nowait(empty_mask)

            # --- 绘制 UI ---
            annotated_frame = results[0].plot()
            curr_time = time.time()
            fps = 1.0 / (curr_time - self.prev_time + 1e-9)
            self.prev_time = curr_time

            self.draw_metrics_overlay(annotated_frame, fps, metrics_display)

            # --- 发布图像 ---
            out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            out_msg.header = msg.header
            self.publisher.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Inference/Quant Error: {e}')
        finally:
            self.processing = False

    def draw_metrics_overlay(self, img, fps, m):
        """在图像左上角绘制透明背景的量化看板"""
        # 背景板
        cv2.rectangle(img, (10, 10), (260, 180), (0, 0, 0), -1)
        cv2.addWeighted(img, 0.7, img, 0.3, 0, img)  # 半透明效果

        color = (0, 255, 0)  # 亮绿色
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 写入数值 (如果数值太小，显示为 0)
        cv2.putText(img, f"FPS: {fps:.1f}", (20, 35), font, 0.6, color, 2)
        cv2.putText(img, f"Area: {m['area_px'] if m['area_px'] > 1 else 0} px", (20, 65), font, 0.6, color, 1)
        cv2.putText(img, f"Length: {m['length_px'] if m['length_px'] > 1 else 0} px", (20, 95), font, 0.6, color, 1)
        cv2.putText(img, f"Avg Width: {m['avg_width_px']:.2f} px", (20, 125), font, 0.6, color, 1)
        cv2.putText(img, f"Max Width: {m['max_width_px']:.2f} px", (20, 155), font, 0.6, color, 1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloSegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_event.set()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()