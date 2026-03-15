import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from ultralytics import YOLO


class YoloSegNode(Node):
    def __init__(self):
        super().__init__('yolo_seg_node')

        self.declare_parameter('input_topic', '/ascamera/camera_publisher/rgb0/image')
        self.declare_parameter('output_topic', '/yolo_result')

        # 改成 NCNN 导出目录，而不是 .pt 文件
        self.declare_parameter('model_path', '/home/ubuntu/yolov8n-seg_ncnn_model')
        self.declare_parameter('is_ncnn', True)

        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('process_fps', 5.0)
        self.declare_parameter('skip_frames', 0)
        self.declare_parameter('max_det', 10)

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.is_ncnn = self.get_parameter('is_ncnn').get_parameter_value().bool_value

        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self.imgsz = self.get_parameter('imgsz').get_parameter_value().integer_value
        self.process_fps = self.get_parameter('process_fps').get_parameter_value().double_value
        self.skip_frames = self.get_parameter('skip_frames').get_parameter_value().integer_value
        self.max_det = self.get_parameter('max_det').get_parameter_value().integer_value

        self.bridge = CvBridge()

        backend_name = 'NCNN' if self.is_ncnn else 'PyTorch'
        self.get_logger().info(f'Loading {backend_name} segmentation model: {self.model_path}')
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

        self.latest_msg = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.processing = False
        self.received_count = 0
        self.processed_count = 0

        timer_period = 1.0 / max(self.process_fps, 0.1)
        self.timer = self.create_timer(timer_period, self.process_latest_frame)

        self.get_logger().info('YOLO Seg Node has started.')
        self.get_logger().info(f'Subscribing image topic: {self.input_topic}')
        self.get_logger().info(f'Publishing result topic: {self.output_topic}')
        self.get_logger().info(f'Backend: {backend_name}')
        self.get_logger().info(f'Confidence threshold: {self.conf_threshold}')
        self.get_logger().info(f'imgsz: {self.imgsz}')
        self.get_logger().info(f'process_fps: {self.process_fps}')
        self.get_logger().info(f'skip_frames: {self.skip_frames}')
        self.get_logger().info(f'max_det: {self.max_det}')

    def image_callback(self, msg: Image):
        self.received_count += 1

        if self.skip_frames > 0 and (self.received_count % (self.skip_frames + 1) != 1):
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_msg = msg
                self.latest_frame = cv_image
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
        except Exception as e:
            self.get_logger().error(f'Unexpected error in image_callback: {e}')

    def process_latest_frame(self):
        if self.processing:
            return

        with self.frame_lock:
            if self.latest_frame is None or self.latest_msg is None:
                return
            frame = self.latest_frame.copy()
            msg = self.latest_msg
            self.latest_frame = None
            self.latest_msg = None

        self.processing = True
        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                max_det=self.max_det,
                verbose=False
            )

            annotated_frame = results[0].plot()

            out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            out_msg.header = msg.header
            self.publisher.publish(out_msg)

            self.processed_count += 1

            if self.processed_count % 10 == 0:
                num_masks = 0
                if results[0].masks is not None:
                    num_masks = len(results[0].masks)

                self.get_logger().info(
                    f'received={self.received_count}, processed={self.processed_count}, segment_objects={num_masks}'
                )

        except Exception as e:
            self.get_logger().error(f'Unexpected error in process_latest_frame: {e}')
        finally:
            self.processing = False


def main(args=None):
    rclpy.init(args=args)
    node = YoloSegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down yolo_seg_node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()