from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_seg_ros2',
            executable='yolo_seg_node',
            name='yolo_seg_node',
            output='screen'
        )
    ])