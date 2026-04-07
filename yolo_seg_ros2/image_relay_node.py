import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np


class DepthVisualizer(Node):
    def __init__(self):
        super().__init__('depth_visualizer')

        self.declare_parameter('input_topic', '/depth_relay')
        self.declare_parameter('output_topic', '/depth_relay_vis')
        self.declare_parameter('max_depth_mm', 5000.0)  # 5米内做显示映射

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.max_depth_mm = float(self.get_parameter('max_depth_mm').value)

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image,
            input_topic,
            self.callback,
            10
        )
        self.pub = self.create_publisher(
            Image,
            output_topic,
            10
        )

        self.get_logger().info(f'Depth visualize: {input_topic} -> {output_topic}')

    def callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            depth = np.array(depth, dtype=np.float32)

            # 无效值处理
            depth[depth <= 0] = np.nan

            # 限制显示范围
            depth = np.clip(depth, 0, self.max_depth_mm)

            # 归一化到 0~255
            vis = (depth / self.max_depth_mm) * 255.0
            vis = np.nan_to_num(vis, nan=0.0).astype(np.uint8)

            out_msg = self.bridge.cv2_to_imgmsg(vis, encoding='mono8')
            out_msg.header = msg.header
            self.pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Failed to visualize depth: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = DepthVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()