# my_exploration ROS2 Package

## Overview

`my_exploration` is a ROS2 package designed for autonomous robot exploration and mapping. It integrates custom navigation, mapping, and object detection capabilities, making it suitable for research and development with TurtleBot3 and simulated environments.

## Features
- Autonomous exploration and navigation
- Map management and storage
- Object detection using YOLOv5/YOLOv8
- Launch files for easy simulation and navigation startup
- Integration with custom and standard ROS2 nodes

## Key Component: `project_yolo.py`

`project_yolo.py` is the main Python script for object detection in the exploration pipeline. It leverages a YOLO-based neural network to detect and segment objects from camera images during robot navigation. The script is designed to:
- Load a YOLO model (e.g., `yolo11n-seg.pt`)
- Process incoming images from the robot's camera
- Perform object detection and segmentation
- Publish detection results to ROS2 topics for further use in navigation or mapping

### Usage
You can run `project_yolo.py` as part of your ROS2 launch sequence or as a standalone node. It is typically launched via the provided launch files, which set up the necessary parameters and topic remappings.

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
To start navigation and object detection:
```bash
ros2 launch my_exploration start_nav2.launch.py
```
This will launch the navigation stack and the YOLO-based detection node.

## Data & Resources
- `data/` contains sample maps and YOLO model weights
- `my_map/` contains map files for simulation
- `resource/` contains ROS2 resource files

## Testing
Unit tests for code style and compliance are provided in the `test/` directory:
- `test_copyright.py`
- `test_flake8.py`
- `test_pep257.py`

Run tests with:
```bash
pytest src/my_exploration/test/
```

## License
This project is licensed under the MIT License.

## Contact
For questions or contributions, please open an issue or pull request on the repository.
