"""The whole-body controller plus a fixed target frame to reach.

    ros2 launch stretch_wbc_ros wbc.launch.py x:=0.6 y:=0.4 z:=0.8
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    x, y, z = (LaunchConfiguration(k) for k in ("x", "y", "z"))
    return LaunchDescription(
        [
            DeclareLaunchArgument("x", default_value="0.6"),
            DeclareLaunchArgument("y", default_value="0.4"),
            DeclareLaunchArgument("z", default_value="0.8"),
            Node(package="stretch_wbc_ros", executable="wbc_node", output="screen"),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                arguments=["--x", x, "--y", y, "--z", z, "--frame-id", "odom",
                           "--child-frame-id", "target"],
            ),
        ]
    )
