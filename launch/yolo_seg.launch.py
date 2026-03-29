from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_seg_ros2',
            executable='image_relay_node',   # ✅ 改这里
            name='image_relay_node',         # ✅ 名字也建议改
            output='screen',
            parameters=[
                {'input_topic': '/ascamera/camera_publisher/rgb0/image'},
                {'output_topic': '/yolo_result'},
            ]
        )
    ])