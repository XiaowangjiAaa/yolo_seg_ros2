# from launch import LaunchDescription
# from launch_ros.actions import Node
#
#
# def generate_launch_description():
#     return LaunchDescription([
#         Node(
#             package='yolo_seg_ros2',
#             executable='yolo_seg_node',
#             name='yolo_seg_node',
#             output='screen',
#             parameters=[
#                 {'input_topic': '/camera/rgb/image_raw'},
#                 {'output_topic': '/yolo_result'},
#             ]
#         )
#     ])

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolo_seg_ros2',
            executable='yolo_seg_node',
            name='yolo_seg_node',
            output='screen',
            parameters=[
                {'input_topic': '/ascamera/camera_publisher/rgb0/image'},
                {'output_topic': '/yolo_result'},
            ]
        )
    ])