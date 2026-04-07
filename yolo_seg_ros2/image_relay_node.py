import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class ImageRelayNode(Node):
    def __init__(self):
        super().__init__('image_relay_node')

        self.declare_parameter('rgb_input', '/ascamera/camera_publisher/rgb0/image')
        self.declare_parameter('depth_input', '/ascamera/camera_publisher/depth0/image_raw')
        self.declare_parameter('rgb_output', '/rgb_relay')
        self.declare_parameter('depth_output', '/depth_relay')

        rgb_input = self.get_parameter('rgb_input').value
        depth_input = self.get_parameter('depth_input').value
        rgb_output = self.get_parameter('rgb_output').value
        depth_output = self.get_parameter('depth_output').value

        # 订阅摄像头：按传感器常见方式
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 对外发布：更通用，兼容很多下游工具
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.rgb_pub = self.create_publisher(Image, rgb_output, pub_qos)
        self.depth_pub = self.create_publisher(Image, depth_output, pub_qos)

        self.rgb_sub = self.create_subscription(
            Image,
            rgb_input,
            self.rgb_callback,
            sub_qos
        )

        self.depth_sub = self.create_subscription(
            Image,
            depth_input,
            self.depth_callback,
            sub_qos
        )

        self.get_logger().info(
            f"Relaying:\n"
            f"  RGB: {rgb_input} -> {rgb_output}\n"
            f"  DEPTH: {depth_input} -> {depth_output}"
        )

    def rgb_callback(self, msg: Image):
        self.rgb_pub.publish(msg)

    def depth_callback(self, msg: Image):
        self.depth_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImageRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down image_relay_node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()