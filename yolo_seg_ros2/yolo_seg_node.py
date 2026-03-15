import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from ultralytics import YOLO
import cv2


class YoloSegNode(Node):
    def __init__(self):
        super().__init__('yolo_seg_node')

        self.declare_parameter('input_topic', '/ascamera/camera_publisher/rgb0/image')
        self.declare_parameter('output_topic', '/yolo_result')
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('conf_threshold', 0.4)

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value

        self.bridge = CvBridge()

        self.get_logger().info(f'Loading YOLO model from: {self.model_path}')
        self.model = YOLO(self.model_path)

        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(
            Image,
            self.output_topic,
            10
        )

        self.frame_count = 0

        self.get_logger().info('YOLO Seg Node has started.')
        self.get_logger().info(f'Subscribing image topic: {self.input_topic}')
        self.get_logger().info(f'Publishing result topic: {self.output_topic}')
        self.get_logger().info(f'Confidence threshold: {self.conf_threshold}')

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            results = self.model.predict(
                source=cv_image,
                conf=self.conf_threshold,
                verbose=False
            )

            annotated_frame = results[0].plot()

            out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            out_msg.header = msg.header
            self.publisher.publish(out_msg)

            self.frame_count += 1
            if self.frame_count % 10 == 0:
                num_boxes = 0
                if results[0].boxes is not None:
                    num_boxes = len(results[0].boxes)
                self.get_logger().info(
                    f'Processed {self.frame_count} frames, detections: {num_boxes}'
                )

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
        except Exception as e:
            self.get_logger().error(f'Unexpected error in image_callback: {e}')


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