import threading
import time
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from ultralytics import YOLO


class YoloSegNode(Node):
    def __init__(self):
        super().__init__('yolo_seg_ncnn_node')

        self.declare_parameter('input_topic', '/ascamera/camera_publisher/rgb0/image')
        self.declare_parameter('output_topic', '/yolo_result')
        self.declare_parameter('model_path', '/home/ubuntu/ros2_ws/src/yolo_seg_ros2/yolo26n-seg_ncnn_model')
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('process_fps', 5.0)
        self.declare_parameter('draw_masks', True)
        self.declare_parameter('max_det', 10)

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.imgsz = self.get_parameter('imgsz').get_parameter_value().integer_value
        self.process_fps = self.get_parameter('process_fps').get_parameter_value().double_value
        self.draw_masks = self.get_parameter('draw_masks').get_parameter_value().bool_value
        self.max_det = self.get_parameter('max_det').get_parameter_value().integer_value

        self.bridge = CvBridge()

        self.get_logger().info(f'Loading NCNN model from: {self.model_path}')
        self.model = YOLO(self.model_path)

        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            sub_qos
        )

        self.publisher = self.create_publisher(
            Image,
            self.output_topic,
            pub_qos
        )

        self.frame_lock = threading.Lock()
        self.latest_msg = None
        self.latest_frame = None

        self.last_annotated_frame = None
        self.last_infer_time = 0.0
        self.processing = False

        timer_period = 1.0 / max(self.process_fps, 0.1)
        self.timer = self.create_timer(timer_period, self.process_latest_frame)

        self.get_logger().info(f'Subscribing: {self.input_topic}')
        self.get_logger().info(f'Publishing:  {self.output_topic}')
        self.get_logger().info(f'process_fps: {self.process_fps}')
        self.get_logger().info(f'imgsz:       {self.imgsz}')
        self.get_logger().info('NCNN node started.')

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_msg = msg
                self.latest_frame = frame
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
        except Exception as e:
            self.get_logger().error(f'image_callback error: {e}')

    def process_latest_frame(self):
        if self.processing:
            return

        with self.frame_lock:
            if self.latest_frame is None or self.latest_msg is None:
                return
            frame = self.latest_frame.copy()
            msg = self.latest_msg

        self.processing = True
        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                max_det=self.max_det,
                verbose=False
            )

            result = results[0]

            if self.draw_masks:
                annotated = result.plot()
            else:
                annotated = frame.copy()
                if result.boxes is not None:
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
                    confs = result.boxes.conf.cpu().numpy()
                    clss = result.boxes.cls.cpu().numpy().astype(int)

                    for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, clss):
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'{cls_id}:{conf:.2f}'
                        cv2.putText(
                            annotated,
                            label,
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )

            self.last_annotated_frame = annotated
            self.last_infer_time = time.time()

            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out_msg.header = msg.header
            self.publisher.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'process_latest_frame error: {e}')
        finally:
            self.processing = False


def main(args=None):
    rclpy.init(args=args)
    node = YoloSegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down yolo_seg_ncnn_node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
# class YoloSegNode(Node):
#     def __init__(self):
#         super().__init__('yolo_seg_node')
#
#         self.declare_parameter('input_topic', '/ascamera/camera_publisher/rgb0/image')
#         self.declare_parameter('depth_topic', '/ascamera/camera_publisher/depth0/image_raw')
#         self.declare_parameter('output_topic', '/yolo_result')
#         self.declare_parameter('model_path', '/home/ubuntu/yolov8n-seg.pt')
#         self.declare_parameter('conf_threshold', 0.25)
#         self.declare_parameter('imgsz', 320)
#         self.declare_parameter('process_fps', 5.0)
#         self.declare_parameter('skip_frames', 0)
#         self.declare_parameter('max_det', 10)
#
#         # 深度测距参数
#         self.declare_parameter('depth_patch_radius', 2)   # 中心点周围 patch 半径，2 表示 5x5
#         self.declare_parameter('min_depth_mm', 100)       # 过滤过近无效值
#         self.declare_parameter('max_depth_mm', 10000)     # 过滤过远无效值
#
#         self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
#         self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
#         self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
#         self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
#         self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
#         self.imgsz = self.get_parameter('imgsz').get_parameter_value().integer_value
#         self.process_fps = self.get_parameter('process_fps').get_parameter_value().double_value
#         self.skip_frames = self.get_parameter('skip_frames').get_parameter_value().integer_value
#         self.max_det = self.get_parameter('max_det').get_parameter_value().integer_value
#
#         self.depth_patch_radius = self.get_parameter('depth_patch_radius').get_parameter_value().integer_value
#         self.min_depth_mm = self.get_parameter('min_depth_mm').get_parameter_value().integer_value
#         self.max_depth_mm = self.get_parameter('max_depth_mm').get_parameter_value().integer_value
#
#         self.bridge = CvBridge()
#
#         self.get_logger().info(f'Loading segmentation model: {self.model_path}')
#         self.model = YOLO(self.model_path)
#
#         # RGB / Depth 输入用低延迟 QoS
#         sub_qos = QoSProfile(
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             history=HistoryPolicy.KEEP_LAST,
#             depth=1
#         )
#
#         # 输出给浏览器/显示端用 RELIABLE
#         pub_qos = QoSProfile(
#             reliability=ReliabilityPolicy.RELIABLE,
#             history=HistoryPolicy.KEEP_LAST,
#             depth=1
#         )
#
#         self.subscription = self.create_subscription(
#             Image,
#             self.input_topic,
#             self.image_callback,
#             sub_qos
#         )
#
#         self.depth_subscription = self.create_subscription(
#             Image,
#             self.depth_topic,
#             self.depth_callback,
#             sub_qos
#         )
#
#         self.publisher = self.create_publisher(
#             Image,
#             self.output_topic,
#             pub_qos
#         )
#
#         self.latest_msg = None
#         self.latest_frame = None
#         self.latest_depth_msg = None
#         self.latest_depth_frame = None
#
#         self.frame_lock = threading.Lock()
#         self.processing = False
#         self.received_count = 0
#         self.processed_count = 0
#
#         timer_period = 1.0 / max(self.process_fps, 0.1)
#         self.timer = self.create_timer(timer_period, self.process_latest_frame)
#
#         self.get_logger().info('YOLO Seg Depth Node has started.')
#         self.get_logger().info(f'Subscribing RGB topic: {self.input_topic}')
#         self.get_logger().info(f'Subscribing Depth topic: {self.depth_topic}')
#         self.get_logger().info(f'Publishing result topic: {self.output_topic}')
#         self.get_logger().info(f'Confidence threshold: {self.conf_threshold}')
#         self.get_logger().info(f'imgsz: {self.imgsz}')
#         self.get_logger().info(f'process_fps: {self.process_fps}')
#         self.get_logger().info(f'skip_frames: {self.skip_frames}')
#         self.get_logger().info(f'max_det: {self.max_det}')
#         self.get_logger().info(f'depth_patch_radius: {self.depth_patch_radius}')
#         self.get_logger().info(f'min_depth_mm: {self.min_depth_mm}')
#         self.get_logger().info(f'max_depth_mm: {self.max_depth_mm}')
#
#     def image_callback(self, msg: Image):
#         self.received_count += 1
#
#         if self.skip_frames > 0 and (self.received_count % (self.skip_frames + 1) != 1):
#             return
#
#         try:
#             cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#             with self.frame_lock:
#                 self.latest_msg = msg
#                 self.latest_frame = cv_image
#         except CvBridgeError as e:
#             self.get_logger().error(f'RGB CvBridge error: {e}')
#         except Exception as e:
#             self.get_logger().error(f'Unexpected error in image_callback: {e}')
#
#     def depth_callback(self, msg: Image):
#         try:
#             # 保持原始深度格式，当前已知是 16UC1
#             depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
#             with self.frame_lock:
#                 self.latest_depth_msg = msg
#                 self.latest_depth_frame = depth_image
#         except CvBridgeError as e:
#             self.get_logger().error(f'Depth CvBridge error: {e}')
#         except Exception as e:
#             self.get_logger().error(f'Unexpected error in depth_callback: {e}')
#
#     def get_depth_median_mm(self, depth, cx, cy, k=2):
#         """
#         从深度图中取中心点附近小区域的有效深度中位数
#         depth: 16UC1, 单位 mm
#         """
#         h, w = depth.shape[:2]
#
#         x1 = max(0, cx - k)
#         x2 = min(w, cx + k + 1)
#         y1 = max(0, cy - k)
#         y2 = min(h, cy + k + 1)
#
#         patch = depth[y1:y2, x1:x2]
#
#         # 过滤无效值和异常值
#         valid = patch[(patch > self.min_depth_mm) & (patch < self.max_depth_mm)]
#
#         if valid.size == 0:
#             return None
#
#         return float(np.median(valid))
#
#     def draw_distance_label(self, image, x1, y1, text):
#         text_x = max(5, x1)
#         text_y = max(20, y1 - 10)
#
#         # 先画黑底再画绿字，更清楚
#         cv2.putText(
#             image,
#             text,
#             (text_x, text_y),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (0, 0, 0),
#             4,
#             cv2.LINE_AA
#         )
#         cv2.putText(
#             image,
#             text,
#             (text_x, text_y),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (0, 255, 0),
#             2,
#             cv2.LINE_AA
#         )
#
#     def process_latest_frame(self):
#         if self.processing:
#             return
#
#         with self.frame_lock:
#             if self.latest_frame is None or self.latest_msg is None:
#                 return
#             if self.latest_depth_frame is None:
#                 return
#
#             frame = self.latest_frame.copy()
#             depth = self.latest_depth_frame.copy()
#             msg = self.latest_msg
#
#             # 清空 RGB 缓存，保留最新深度缓存供下次继续用也可以；
#             # 这里 RGB 清空避免重复处理旧帧
#             self.latest_frame = None
#             self.latest_msg = None
#
#         self.processing = True
#         try:
#             results = self.model.predict(
#                 source=frame,
#                 conf=self.conf_threshold,
#                 imgsz=self.imgsz,
#                 max_det=self.max_det,
#                 verbose=False
#             )
#
#             annotated_frame = results[0].plot()
#
#             # 读取检测框并测距
#             if results[0].boxes is not None and len(results[0].boxes) > 0:
#                 for box in results[0].boxes:
#                     xyxy = box.xyxy[0].cpu().numpy().astype(int)
#                     x1, y1, x2, y2 = xyxy.tolist()
#
#                     cx = (x1 + x2) // 2
#                     cy = (y1 + y2) // 2
#
#                     if 0 <= cx < depth.shape[1] and 0 <= cy < depth.shape[0]:
#                         depth_mm = self.get_depth_median_mm(
#                             depth,
#                             cx,
#                             cy,
#                             k=self.depth_patch_radius
#                         )
#
#                         if depth_mm is not None:
#                             distance_m = depth_mm / 1000.0
#                             label_text = f'{distance_m:.2f} m'
#                         else:
#                             label_text = 'N/A'
#
#                         # 画中心点
#                         #cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 255), -1)
#
#                         # 画距离文字
#                         self.draw_distance_label(annotated_frame, x1, y1, label_text)
#
#             out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
#             out_msg.header = msg.header
#             self.publisher.publish(out_msg)
#
#             self.processed_count += 1
#
#             if self.processed_count % 10 == 0:
#                 num_masks = 0
#                 if results[0].masks is not None:
#                     num_masks = len(results[0].masks)
#
#                 self.get_logger().info(
#                     f'received={self.received_count}, processed={self.processed_count}, segment_objects={num_masks}'
#                 )
#
#         except Exception as e:
#             self.get_logger().error(f'Unexpected error in process_latest_frame: {e}')
#         finally:
#             self.processing = False
#
#
# def main(args=None):
#     rclpy.init(args=args)
#     node = YoloSegNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('Shutting down yolo_seg_node...')
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()
#
#
# if __name__ == '__main__':
#     main()