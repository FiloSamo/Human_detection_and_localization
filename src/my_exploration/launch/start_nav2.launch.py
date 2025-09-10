import os

from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch.actions import TimerAction, RegisterEventHandler
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit

def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time', default='True')
    rviz = LaunchConfiguration('rviz', default='True')
    slam = LaunchConfiguration('slam', default='True')

    slam_nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('turtlebot3_navigation2'),
                                      'launch'),'/navigation2.launch.py']),
                                      launch_arguments={'use_sim_time': use_sim_time,
                                                        'rviz': rviz,
                                                        'slam' : slam}.items()
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    ld.add_action(slam_nav)

    return ld