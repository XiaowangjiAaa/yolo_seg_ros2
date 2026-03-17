import threading
import cv2
import numpy as np
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# 导入 ROS 2 包路径查找工具
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO


class YoloSegNode(Node):
    def __init__(self):
        super().__init__('yolo_seg_node')

        self.declare_parameter('input_topic', '/ascamera/camera_publisher/rgb0/image')
        self.declare_parameter('depth_topic', '/ascamera/camera_publisher/depth0/image_raw')
        self.declare_parameter('output_topic', '/yolo_result')

        # --- 相对路径处理逻辑 ---
        try:
            # 获取包安装后的 share 路径
            package_share_dir = get_package_share_directory('yolo_seg_ros2')
            default_model_path = os.path.join(package_share_dir, 'YOLO_26n_crack.pt')
        except Exception:
            # 如果没找到包（例如直接运行脚本），尝试回退到本地
            default_model_path = 'YOLO_26n_crack.pt'

        self.declare_parameter('model_path', default_model_path)
        # ----------------------

        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('imgsz', 320)  # 已修正拼写错误
        self.declare_parameter('process_fps', 5.0)
        self.declare_parameter('skip_frames', 0)
        self.declare_parameter('max_det', 10)

        # 深度测距参数
        self.declare_parameter('depth_patch_radius', 2)  # 中心点周围 patch 半径
        self.declare_parameter('min_depth_mm', 100)  # 过滤过近无效值
        self.declare_parameter('max_depth_mm', 10000)  # 过滤过远无效值

        # 获取参数值
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.imgsz = self.get_parameter('imgsz').get_parameter_value().integer_value
        self.process_fps = self.get_parameter('process_fps').get_parameter_value().double_value
        self.skip_frames = self.get_parameter('skip_frames').get_parameter_value().integer_value
        self.max_det = self.get_parameter('max_det').get_parameter_value().integer_value

        self.depth_patch_radius = self.get_parameter('depth_patch_radius').get_parameter_value().integer_value
        self.min_depth_mm = self.get_parameter('min_depth_mm').get_parameter_value().integer_value
        self.max_depth_mm = self.get_parameter('max_depth_mm').get_parameter_value().integer_value

        self.bridge = CvBridge()

        self.get_logger().info(f'Loading segmentation model: {self.model_path}')
        # 确保路径存在
        if not os.path.exists(self.model_path):
            self.get_logger().error(f'Model file NOT FOUND at: {self.model_path}')

        self.model = YOLO(self.model_path)

        # RGB / Depth 输入 QoS
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 输出 QoS
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Image, self.input_topic, self.image_callback, sub_qos
        )

        self.depth_subscription = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, sub_qos
        )

        self.publisher = self.create_publisher(
            Image, self.output_topic, pub_qos
        )

        self.latest_msg = None
        self.latest_frame = None
        self.latest_depth_msg = None
        self.latest_depth_frame = None

        self.frame_lock = threading.Lock()
        self.processing = False
        self.received_count = 0
        self.processed_count = 0

        timer_period = 1.0 / max(self.process_fps, 0.1)
        self.timer = self.create_timer(timer_period, self.process_latest_frame)

        self.get_logger().info('YOLO Seg Depth Node initialized.')

    def image_callback(self, msg: Image):
        self.received_count += 1
        if self.skip_frames > 0 and (self.received_count % (self.skip_frames + 1) != 1):
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_msg = msg
                self.latest_frame = cv_image
        except Exception as e:
            self.get_logger().error(f'RGB Callback error: {e}')

    def depth_callback(self, msg: Image):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.frame_lock:
                self.latest_depth_msg = msg
                self.latest_depth_frame = depth_image
        except Exception as e:
            self.get_logger().error(f'Depth Callback error: {e}')

    def get_depth_median_mm(self, depth, cx, cy, k=2):
        h, w = depth.shape[:2]
        x1, x2 = max(0, cx - k), min(w, cx + k + 1)
        y1, y2 = max(0, cy - k), min(h, cy + k + 1)
        patch = depth[y1:y2, x1:x2]
        valid = patch[(patch > self.min_depth_mm) & (patch < self.max_depth_mm)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def draw_distance_label(self, image, x1, y1, text):
        text_x, text_y = max(5, x1), max(20, y1 - 10)
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    def process_latest_frame(self):
        if self.processing:
            return

        with self.frame_lock:
            if self.latest_frame is None or self.latest_depth_frame is None:
                return
            frame = self.latest_frame.copy()
            depth = self.latest_depth_frame.copy()
            msg = self.latest_msg
            self.latest_frame = None

        self.processing = True
        try:
            results = self.model.predict(
                source=frame, conf=self.conf_threshold, imgsz=self.imgsz,
                max_det=self.max_det, verbose=False
            )

            annotated_frame = results[0].plot()

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cx, cy = (xyxy[0] + xyxy[2]) // 2, (xyxy[1] + xyxy[3]) // 2

                    if 0 <= cx < depth.shape[1] and 0 <= cy < depth.shape[0]:
                        d_mm = self.get_depth_median_mm(depth, cx, cy, k=self.depth_patch_radius)
                        label = f'{d_mm / 1000.0:.2f} m' if d_mm else 'N/A'
                        self.draw_distance_label(annotated_frame, xyxy[0], xyxy[1], label)

            out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            out_msg.header = msg.header
            self.publisher.publish(out_msg)
            self.processed_count += 1
        except Exception as e:
            self.get_logger().error(f'Process frame error: {e}')
        finally:
            self.processing = False


def main(args=None):
    rclpy.init(args=args)
    node = YoloSegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()