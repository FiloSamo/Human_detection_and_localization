import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

# Create a function to describe what you want to launch
def generate_launch_description():
    # Find package directories
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    
    # Define parameters
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose_turtlebot = LaunchConfiguration('x_pose_turtlebot', default='-2.0')
    y_pose_turtlebot = LaunchConfiguration('y_pose_turtlebot', default='-0.5')
    x_pose_1 = LaunchConfiguration('x_pose_dog', default='1.0')
    y_pose_1 = LaunchConfiguration('y_pose_dog', default='0.5')
    x_pose_2 = LaunchConfiguration('x_pose_cow', default='0.0')
    y_pose_2 = LaunchConfiguration('y_pose_cow', default='2.2')
    x_pose_3 = LaunchConfiguration('x_pose_horse', default='-0.4')
    y_pose_3 = LaunchConfiguration('y_pose_horse', default='-1.2') #-1
    
    urdf_path = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'models',
        'human_male_tdf',
        'model.sdf'
    )

    world = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'worlds',
        'turtlebot3_world.world'
    )
    
    # Launch commands
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose_turtlebot,
            'y_pose': y_pose_turtlebot
        }.items()
    )

    human_1 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'human1',
            '-file', urdf_path,
            '-x', x_pose_1,
            '-y', y_pose_1,
            '-z', '0.01'
        ],
        output='screen',
    )
    human_2 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'human2',
            '-file', urdf_path,
            '-x', x_pose_2,
            '-y', y_pose_2,
            '-z', '0.01'
        ],
        output='screen',
    )
    human_3 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'human3',
            '-file', urdf_path,
            '-x', x_pose_3,
            '-y', y_pose_3,
            '-z', '0.01'
        ],
        output='screen',
    )
    
    # Create launch description
    ld = LaunchDescription()

    # Add actions to the launch description
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)
    ld.add_action(human_1)
    ld.add_action(human_2)  # Add cow spawn
    ld.add_action(human_3)  # Add horse spawn

    return ld



