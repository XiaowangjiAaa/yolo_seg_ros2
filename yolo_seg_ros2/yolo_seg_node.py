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

        # --- 1. 参数声明 ---
        self.declare_parameter('input_topic', '/ascamera/camera_publisher/rgb0/image')
        self.declare_parameter('depth_topic', '/ascamera/camera_publisher/depth0/image_raw')
        self.declare_parameter('output_topic', '/yolo_result')

        # 路径处理
        try:
            package_share_dir = get_package_share_directory('yolo_seg_ros2')
            default_model_path = os.path.join(package_share_dir, 'YOLO_26n_crack.pt')
        except Exception:
            default_model_path = 'YOLO_26n_crack.pt'
        self.declare_parameter('model_path', default_model_path)

        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('process_fps', 5.0)

        # 量化相关参数
        self.declare_parameter('smooth_win', 5)  # 平滑窗口大小

        # --- 2. 变量初始化 ---
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.imgsz = self.get_parameter('imgsz').get_parameter_value().integer_value
        self.process_fps = self.get_parameter('process_fps').get_parameter_value().double_value

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        # 队列与多线程量化
        self.task_q = queue.Queue(maxsize=3)
        self.latest_metrics = {"area_px": 0, "length_px": 0, "avg_width_px": 0.0, "max_width_px": 0.0}
        self.hist_metrics = {k: deque(maxlen=self.get_parameter('smooth_win').value) for k in
                             self.latest_metrics.keys()}

        self.stop_event = threading.Event()
        self.quant_thread = threading.Thread(target=self.quant_worker, daemon=True)
        self.quant_thread.start()

        # 图像处理相关状态
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_depth_frame = None
        self.latest_msg = None
        self.processing = False
        self.prev_time = time.time()

        # --- 3. ROS 订阅与发布 ---
        sub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, self.get_parameter('input_topic').value, self.image_callback, sub_qos)
        self.create_subscription(Image, self.get_parameter('depth_topic').value, self.depth_callback, sub_qos)
        self.publisher = self.create_publisher(Image, self.get_parameter('output_topic').value, 10)

        # 定时推理
        timer_period = 1.0 / max(self.process_fps, 0.1)
        self.timer = self.create_timer(timer_period, self.process_latest_frame)
        self.get_logger().info('YOLO Quantification Node Started.')

    # ===========================
    # 量化核心算法 (Worker Thread)
    # ===========================
    def quant_worker(self):
        while not self.stop_event.is_set():
            try:
                mask01 = self.task_q.get(timeout=0.1)
                metrics = self.calculate_metrics(mask01)

                # 更新平滑队列
                for k, v in metrics.items():
                    self.hist_metrics[k].append(v)

                # 计算平滑均值
                with self.frame_lock:
                    self.latest_metrics = {k: float(np.mean(v)) for k, v in self.hist_metrics.items()}

                self.task_q.task_done()
            except queue.Empty:
                continue

    def calculate_metrics(self, mask01):
        area_px = int(mask01.sum())
        if area_px <= 0:
            return {"area_px": 0, "length_px": 0, "avg_width_px": 0.0, "max_width_px": 0.0}

        # 最大宽度 (距离变换)
        dist = cv2.distanceTransform(mask01, cv2.DIST_L2, 3)
        max_width_px = 2.0 * float(dist.max())

        # 长度 (骨架化 - 简化版使用 cv2.ximgproc)
        try:
            skeleton = cv2.ximgproc.thinning((mask01 * 255).astype(np.uint8))
            length_px = int((skeleton > 0).sum())
        except Exception:
            length_px = area_px // 10  # 兜底逻辑

        avg_width_px = float(area_px) / float(length_px + 1e-9)
        return {
            "area_px": area_px,
            "length_px": length_px,
            "avg_width_px": avg_width_px,
            "max_width_px": max_width_px
        }

    # ===========================
    # 图像回调与处理
    # ===========================
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_frame = cv_image
                self.latest_msg = msg
        except Exception as e:
            self.get_logger().error(f'RGB Error: {e}')

    def depth_callback(self, msg):
        try:
            self.latest_depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth Error: {e}')

    def process_latest_frame(self):
        if self.processing or self.latest_frame is None:
            return

        with self.frame_lock:
            frame = self.latest_frame.copy()
            msg = self.latest_msg
            metrics_view = self.latest_metrics.copy()
            self.latest_frame = None

        self.processing = True
        try:
            results = self.model.predict(source=frame, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False)

            # 1. 提取合并 Mask 并发送到量化线程
            if results[0].masks is not None:
                combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                masks = results[0].masks.data.cpu().numpy()
                for m in masks:
                    m_resized = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    combined_mask = np.maximum(combined_mask, (m_resized > 0.5).astype(np.uint8))

                if not self.task_q.full():
                    self.task_q.put_nowait(combined_mask)

            # 2. 绘制 UI
            annotated_frame = results[0].plot()
            curr_time = time.time()
            fps = 1.0 / (curr_time - self.prev_time + 1e-9)
            self.prev_time = curr_time

            self.draw_ui(annotated_frame, fps, metrics_view)

            # 3. 发布
            out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            out_msg.header = msg.header
            self.publisher.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Process Error: {e}')
        finally:
            self.processing = False

    def draw_ui(self, img, fps, m):
        c = (0, 255, 0)  # 绿色
        cv2.putText(img, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        cv2.putText(img, f"Area: {m['area_px']} px", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        cv2.putText(img, f"Length: {m['length_px']} px", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        cv2.putText(img, f"AvgWidth: {m['avg_width_px']:.2f} px", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        cv2.putText(img, f"MaxWidth: {m['max_width_px']:.2f} px", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)


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