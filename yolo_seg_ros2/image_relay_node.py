import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class ImageRelayNode(Node):
    def __init__(self):
        super().__init__('image_relay_node')

        # topics
        self.rgb_input = '/ascamera/camera_publisher/rgb0/image'
        self.depth_input = '/ascamera/camera_publisher/depth0/image_raw'

        self.rgb_output = '/rgb_relay'
        self.depth_output = '/depth_relay'

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # RGB
        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_input,
            self.rgb_callback,
            qos
        )
        self.rgb_pub = self.create_publisher(
            Image,
            self.rgb_output,
            qos
        )

        # Depth
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_input,
            self.depth_callback,
            qos
        )
        self.depth_pub = self.create_publisher(
            Image,
            self.depth_output,
            qos
        )

        self.get_logger().info(f'Relaying:\n  RGB: {self.rgb_input} -> {self.rgb_output}\n  DEPTH: {self.depth_input} -> {self.depth_output}')

    def rgb_callback(self, msg):
        self.rgb_pub.publish(msg)

    def depth_callback(self, msg):
        self.depth_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImageRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()