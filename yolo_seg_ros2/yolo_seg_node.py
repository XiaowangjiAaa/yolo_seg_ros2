import rclpy
from rclpy.node import Node


class YoloSegNode(Node):
    def __init__(self):
        super().__init__('yolo_seg_node')
        self.get_logger().info('YOLO Seg Node has started.')


def main(args=None):
    rclpy.init(args=args)
    node = YoloSegNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()