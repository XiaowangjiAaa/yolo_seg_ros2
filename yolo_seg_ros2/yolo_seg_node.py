import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class YoloSegNode(Node):
    def __init__(self):
        super().__init__('yolo_seg_node')

        # 参数：输入和输出话题都可以后面改
        self.declare_parameter('input_topic', '/camera/rgb/image_raw')
        self.declare_parameter('output_topic', '/yolo_result')

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.bridge = CvBridge()

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

    def image_callback(self, msg: Image):
        try:
            # ROS Image -> OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 这里先不做 YOLO，只做透传
            # 后面你可以在这里加 YOLO 推理和绘框
            result_image = cv_image

            # OpenCV -> ROS Image
            out_msg = self.bridge.cv2_to_imgmsg(result_image, encoding='bgr8')
            out_msg.header = msg.header

            self.publisher.publish(out_msg)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                h, w = result_image.shape[:2]
                self.get_logger().info(
                    f'Received and republished {self.frame_count} frames, image size: {w}x{h}'
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