# Human Detection and Localization 👨‍🦱📷🤖

## Overview

This is a university project for robot perception. The task was to use sensors to localize a target in an unknown environment.  
We implemented a **human localization system** where a TurtleBot3 Burger autonomously explores the environment and detects the presence of humans.  
No prior information about the environment or the number of humans is provided.  

To achieve this, we designed a ROS2 package called **`my_exploration`**.

## Features
- **Human detection** using YOLOv11  
- **Localization** with a custom Extended Kalman Filter implementation  
- **Visualization**: publishes an image showing the explored area and detected human positions on the map  

## Key Component: `project_yolo.py`

The `project_yolo.py` file is the main Python script for human detection and localization.  
It uses a YOLO-based neural network to detect and segment humans from the robot’s camera feed during navigation.  

Main functions:
- Load a YOLO model (e.g. `yolo11n-seg.pt`)
- Process incoming images from the robot camera
- Perform object detection and segmentation
- Permorm localization using Extended Kalman Filter
- Publish detection results to ROS2 topics for use in navigation and mapping

### Usage
You can run `project_yolo.py` either as a standalone ROS2 node or as part of the provided launch files, which handle parameters and topic remappings automatically.

## Installation
1. Clone the repository into your ROS2 workspace `src` folder:
   ```bash
   cd ~/ros2/turtlebot3_ws/src
   git clone <repo_url>
   ```
   
2. Install dependencies (YOLO, OpenCV, etc.) as required by `project_yolo.py`:
   ```bash
   pip install -r requirements.txt
   ```

3. Build the workspace:
   ```bash
   cd ~/ros2/turtlebot3_ws
   colcon build
   source install/setup.bash
   ```

## Launching

![vis](https://github.com/user-attachments/assets/9e8a916a-44fd-464d-998a-cfab0301873e)

First of all, source the environment for every new terminal:
```bash
source install/local_setup.bash
```

Start the Simulation
```bash
export ROS_DISTRO=humble
export TURTLEBOT3_MODEL=waffle
ros2 launch turtlebot3_gazebo world_human.launch.py
```

Start Nav2 stack:
```bash
ros2 launch my_exploration start_nav2.launch.py
```

Start Detection:
```bash
ros2 run my_exploration project_yolo
```

Start the exploration (we use the explore_lite package, you can use other type of exploration, like the teleop_key):
```bash
ros2 run explore_lite explore.launch.py
```

This setup launches the navigation stack and the YOLO-based detection node, enabling the robot to explore and localize humans.

Data & Resources

- data/ → sample maps and YOLO model weights
- my_map/ → map files for simulation
- resource/ → ROS2 resource files

## Contributors

- **Filippo Samorì**  
- **Vittorio Caputo**  
- **Matteo Bonucci**  

MSc students in Automation Engineering at the University of Bologna.

## Acknowledgements

This project was made possible thanks to the following open-source projects and resources:

- **[Explore Lite](https://github.com/hrnr/m-explore)** – ROS2 exploration package used for autonomous frontier-based exploration.
- **[TurtleBot3 Simulation](https://github.com/ROBOTIS-GIT/turtlebot3_simulations)** – Official simulation environments for TurtleBot3 robots.
- **[TurtleBot3](https://github.com/ROBOTIS-GIT/turtlebot3)** – Official TurtleBot3 ROS2 packages for robot configuration and navigation.
- **[YOLO](https://github.com/ultralytics/ultralytics)** – Real-time object detection neural network used for human detection.
  
## License

This project is licensed under the MIT License.

## Contact

For questions or contributions, feel free to reach out:
📧 filippo.samori@studio.unibo.it
