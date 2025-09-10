import rclpy
from rclpy.node import Node
import message_filters

from cv_bridge import CvBridge
import numpy as np
import cv2
import os
from rclpy.wait_for_message import wait_for_message
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import String, Bool
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Pose
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

import matplotlib.pyplot as plt


# from threading import Thread
# from tf2_ros import Duration
# from action_msgs.msg import GoalStatus
# from nav2_msgs.action import NavigateToPose
# from geometry_msgs.msg import PoseStamped
# from rclpy.action import ActionClient

from ultralytics import YOLO

from PyKDL import Frame, Vector, Rotation, dot

def find_set(img, color):
    
    set_ = set()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j,0] == color[0] and img[i, j,1] == color[1] and img[i, j,2] == color[2]:
                set_.add((i, j))
    return set_

def kdl_from_frame(tf):
        # Converts the frame from the Ros2 msg.transform into a Pykdl Frame
        position = Vector(tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z)
        orientation = Rotation.Quaternion(tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w)
        return Frame(orientation, position)
    
def kdl_from_pose( pose):
    # Converts the frame from the Ros2 msg.transform into a Pykdl Frame
    position = Vector(pose.position.x, pose.position.y, pose.position.z)
    orientation = Rotation.Quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
    return Frame(orientation, position)

camera_info_topic = '/rgbd_camera/camera_info'
camera_topic = '/rgbd_camera/image_raw'
depth_topic = '/rgbd_camera/depth/image_raw'

NN_image_topic = '/neuralnet/image_raw'
NN_pose_topic = '/neuralnet/pose'

map_topic = '/map_img'

class Project(Node):

    def __init__(self):
        super().__init__("NeuralNet")

        ## Variable initialization 
     
        self.map_res = 0.05
        origin= [2.94, 2.57, 0]
        self.true_poses = [[-0.4 ,-1.2], [0.0 , 2.2], [1.0 ,0.5]]
        self.mapArea = 7836 # computed offline

        self.noise_var = np.array([[0.01, 0], [0, 40]])

        self.origin =  [origin[0] / self.map_res, origin[1] / self.map_res]
        
        # Load the map
        self.map = cv2.imread(self.create_path("my_map.jpg"))
        self.clean_map = self.map.copy()

        self.coverage_set = set()

        # obtain the obstacles in the map
        self.obstacles = self.find_obstacles(self.map)
        
        self.cam_K = None
        self.cam_D = None

        self.scale_factor = 0.3

        self.t_map = None
        self.t_base = None

        self.net = None
        self.out_img = None
        self.depth_img = None

        # NN parameters
        self.modelpath = self.create_path("yolo11n-seg.pt")
        # Initialization of the Neural Network
        self.net = self.init_network(self.modelpath)

        # Bridge between Ros2 and Opencv
        self.bridge = CvBridge()

        # Image topic subscription
        
        # Synchronization for the Image RGB and Imgae depth topics
        self.image_sub = message_filters.Subscriber(self, Image, camera_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
         
        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.depth_sub], 10) # time sinchronization
        self.ts.registerCallback(self.img_callback)

        ## Alternative without synchronization
        #self.img_sub = self.create_subscription(Image, camera_topic, self.img_callback, 10)
        #self.depth_sub = self.create_subscription(Image, depth_topic , self.depth_callback, 10)

        self.img_pub = self.create_publisher(Image, NN_image_topic, 10)
        self.pose_pub = self.create_publisher(Pose, NN_pose_topic, 10)
        self.map_pub = self.create_publisher(Image, map_topic, 10)
                
        # timers

        self.timer = self.create_timer(0.5, self.timer_callback)

        # tf2 implementation to read frames' transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Obtaining the camera model
        ret, msg = wait_for_message(CameraInfo, self, camera_info_topic) 
        self.cam_K = np.array(msg.k).reshape(3,3) # Intrinsic parameters
        self.cam_D = np.array((msg.d)) # Distortion parameters

        self. t_poses = []
        self.t_pose_covs = []
        self.robot_pose = None

        self.get_logger().info("Initialization finished")

    def img_callback(self, msg_img, msg_depth):
        '''
            Callback from the time synchronizer that obtain messages from the rgb image topic and the
            depth image topic. The two messages have the same time stamp.
            After obtaining the two images, the function compute the target position and update the map.
        '''

        # Obtain the images time stamp to synchronize the robot position with the images
        self.time_stamp = msg_img.header.stamp # it is used to obtain the transformation between the map and the camera frame at the correct time

        try:
            # Homogeneous transformation
            # map frame to publish the position of the tables
            self.t_map = kdl_from_frame(self.tf_buffer.lookup_transform('map', 'camera_rgb_optical_frame', self.time_stamp))
        except Exception as e:
            self.t_map = None
            return
        
        # convert the images to opencv format
        self.img = self.bridge.imgmsg_to_cv2(msg_img, desired_encoding="bgr8") 
        self.depth_img = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding=msg_depth.encoding)

        # Resize the images
        self.scaled_img = cv2.resize(self.img, (0, 0), fx = self.scale_factor, fy = self.scale_factor)
        self.depth_img = cv2.resize(self.depth_img, (0, 0), fx = self.scale_factor, fy = self.scale_factor)

        # Obtain the camera intrinsic parameters
        if self.cam_K is None:
            ret, msg =  wait_for_message(CameraInfo, self, camera_info_topic)
            self.cam_K = np.array(msg.k).reshape(3,3)
            self.cam_D = np.array((msg.d))

        # Initialization of the Neural Network if it is not initialized
        if self.net is None:
            self.net = self.init_network(self.modelWeights, self.textGraph)

        # Detection of the target object
        if self.t_map is not None: # if the transformation between the map and the camera frame is available
            # detection
            indices, centers, out_msg , boxes, masks= self.NNdetector(self.net, self.scaled_img)

            self.out_img = out_msg # setting the output image with the bounding boxes and masks

            # if some objects are detected           
            if not (indices == []) and self.depth_img is not None:

                _ , _ , theta = self.t_map.M.GetRPY() # orientation of the camera frame in the map frame
                theta = theta + np.pi/2 # correction in the evaluation of theta
                
                # robot pose in the map frame
                self.robot_pose = np.array([self.t_map.p.x(), self.t_map.p.y(), theta])

                for i in range(len(indices)):

                    if indices[i] in [0]: # if the detected object is the target object (human)

                        u =  centers[i][0] // self.scale_factor # x coordinate of the center of the object
                        v =  centers[i][1] // self.scale_factor # y coordinate of the center of the object

                        # Compute the 3D coordinates of the object in the camera frame
                        X, Y, Z = self.get_3d_point(self.depth_img, boxes[i], masks[i], u , v )
                        
                        print(f"3D Coordinates: X={X}, Y={Y}, Z={Z}")

                        # Compute the measurements for the Extended Kalman Filter
                        measurement = np.array([Z, u - self.cam_K[0, 2]])
                        measurement = np.reshape(measurement, (2,1)) # reshape the measurement in vector form

                        X, Y = self.compute_world_point([X, Y, Z], self.robot_pose) # Compute the world coordinates of the target point
                        print(f"Number of targets {len(self.t_poses)}")

                        found = False
                        for i in range(len(self.t_poses)):
                            t_pose = self.t_poses[i]
                            t_pose_cov = self.t_pose_covs[i]
                            if np.linalg.norm(t_pose - np.array([[X], [Y]])) < 1.0:
                                 # Update the target position with the Extended Kalman Filter
                                t_pose , t_pose_cov = self.ExtendedKalmanFilter(self.robot_pose, t_pose, t_pose_cov , measurement, self.cam_K, self.noise_var)
                                self.t_poses[i] = t_pose
                                self.t_pose_covs[i] = t_pose_cov
                                found = True

                                # # check if there are multiple instances of the same target
                                # for j in range(len(self.t_poses)):
                                #     pose = self.t_poses[j]
                                #     if np.linalg.norm(pose - t_pose) < 0.1 and i != j:
                                #         del self.t_poses[i]
                                #         del self.t_pose_covs[i]
                                #         break
                                    
                        if found == False:
                            t_pose = np.array([X, Y])
                            t_pose = np.reshape(t_pose, (2,1))
                            self.t_poses.append(t_pose)
                            t_pose_cov = self.compute_covariance(self.robot_pose, measurement, self.cam_K, self.noise_var)
                            self.t_pose_covs.append(t_pose_cov)

                        print(f"Map Coordinates: X={X}, Y={Y}")
                        
                        print(f"estimated poses: {self.t_poses}")
                        print(f"estimated poses covariance: {self.t_pose_covs}")
                        print(f"Estimated coverage: {len(self.coverage_set) / self.mapArea}")

    def timer_callback(self):
        """
            Timer callback to publish the messages
        """
        if self.out_img is not None:
            self.publish_img(self.out_img)

        if self.map is not None:
            try:
                # Homogeneous transformation
                # map frame to publish the position of the tables
                self.t_map = kdl_from_frame(self.tf_buffer.lookup_transform('map', 'camera_rgb_optical_frame', rclpy.time.Time()))
            except Exception as e:
                self.t_map = None
                return

            if self.t_map is not None:
                # Convert the robot pose from the camera frame to the map frame
                _ , _ , theta = self.t_map.M.GetRPY() # orientation of the camera frame in the map frame
                theta = theta + np.pi/2 # correction in the evaluation of theta
                    
                # robot pose in the map frame
                self.robot_pose = np.array([self.t_map.p.x(), self.t_map.p.y(), theta])
        
                # Update the visualization of the target position in the map
                self.update_map(self.t_poses, self.robot_pose)
                self.publish_map(self.map)
        
    # Methods for publishing

    def publish_img(self, img):
        """  Publish the image with the bounding boxes and masks """
        msg = self.bridge.cv2_to_imgmsg(img, encoding='passthrough')
        self.img_pub.publish(msg)

    def publish_map(self, img):
        """  Publish the map image """
        msg = self.bridge.cv2_to_imgmsg(img, encoding='passthrough')
        self.map_pub.publish(msg)

    # Methods for the implementation of the target detection

        # Method for obtain the path for the neural network parameters
    def create_path(self, path):
        """ 
            Create the path for the files needed for the project
        """

        out_path = os.path.join(get_package_share_directory('my_exploration'),
                                      'data', path)
        return out_path

    def get_3d_point(self, depth_img, box, mask, u, v):
        """Computes 3D coordinates from image coordinates.

            Args:
                depth_img: depth image
                box: bounding box of the object (left, top, right, bottom)
                mask: mask of the object (same size of the depth image with 255 for the object and 0 for the background)
                x: x coordinate of the object
                y: y coordinate of the object

            Returns:
                X, Y, Z: 3D coordinates of the object
        
        """
        # Obtain the camera intrinsic parameters if they are not available
        if self.cam_K is None:
            ret, info_msg = wait_for_message(CameraInfo, self, camera_info_topic)
            self.cam_K = np.array(info_msg.k).reshape(3, 3)
            self.cam_D = np.array(list(info_msg.d))

        # extract the bounding box and the mask
        left = box[0]
        top = box[1]
        right = box[2]
        bottom = box[3]

        box_depth = depth_img[top:bottom, left:right] # depth values of the bounding box
        mask_depth = mask[top:bottom, left:right] # mask of the bounding box
        depth_list = []

        # Compute the depth value of the object inside the bounding box using the mask
        for i in range(right - left):
            for j in range(bottom - top):
                if mask_depth[j, i] == 255:
                    if not (box_depth[j,i] == float('inf') or box_depth[j,i] == float('-inf')) :
                        depth_list.append(box_depth[j,i])

        depth_value = np.median(depth_list) # taking the mediana depth value

        # center = [int((right-left)/2) , int((bottom - top) /2) ]
        # depth_value = depth_img[center[1], center[0]] # depth value of the object

        if depth_value == float('inf') or depth_value == float('-inf'): # if the depth value is not valid
            return 0.0 , 0.0 , 0.0

        # Compute the 3D coordinates of the object
        X = depth_value * (u - self.cam_K[0, 2]) / self.cam_K[0, 0]
        Y = depth_value * (v - self.cam_K[1, 2]) / self.cam_K[1, 1]
        Z = depth_value

        return X, Y, Z
    
    def compute_world_point(self, point , robot_pose):
        """
            Compute the world coordinates of the target point

            Args:
                point: 2D coordinates of the target point
                robot_pose: robot pose in the map frame (x, y, theta)

            Returns:
                X, Y: world coordinates of the target point

        """
        x = point[0]
        y = point[1]
        z = point[2]
        
        beta = np.arcsin(-x/z) # angle between the x axis and the line connecting the robot and the target point
        theta = robot_pose[2] # orientation of the robot in the map frame

        # Compute the world coordinates of the target point
        X = robot_pose[0] + z * np.cos(theta + beta) 
        Y = robot_pose[1] + z * np.sin(theta + beta)

        return X, Y
    
    def compute_measurements(self, t_pose, robot_pose, cam_K):

        """
            Compute the expected measurements for the Extended Kalman Filter

            Args:
                t_pose: target pose in the map frame (x, y)
                robot_pose: robot pose in the map frame (x, y, theta)
                cam_K: camera intrinsic parameters

            Returns:
                measurement: expected measurements for the EKF (z, x)
        
        """
        # Compute the expected measurements

        z = np.sqrt((t_pose[0] - robot_pose[0])**2 + (t_pose[1] - robot_pose[1])**2)
        u = - np.sin(np.arctan2(t_pose[1] - robot_pose[1], t_pose[0] - robot_pose[0]) - robot_pose[2]) * cam_K[0, 0]

        measurement = np.array([z, u])
        measurement = np.reshape(measurement, (2,1))
    
        return measurement
    
    def compute_covariance(self, robot_pose, measurement, cam_K, noise_cov):

        """
            Compute the covariance matrix of the estimated target position the first time is detected

            Args:
                robot_pose: robot pose in the map frame (x, y, theta)
                measurement: measurement of the target position (z, x)
                cam_K: camera intrinsic parameters
                noise_cov: noise covariance matrix

            Returns:
                covariance: covariance matrix of the estimated target position 
        """
        measurement = measurement.squeeze()

        z = measurement[0]
        u = measurement[1]
        
        robot_pose = robot_pose.squeeze()
        theta = robot_pose[2]

        R = noise_cov

        beta = np.arcsin(-u/cam_K[0, 0]) # angle between the x axis and the line connecting the robot and the target point

        # Compute the Jacobian matrix of the target position with respect to the target measurements
        dXdz = np.cos(theta + beta)
        dYdz = np.sin(theta + beta)
        
        dXdx = -z * np.sin(theta + beta)* 1/(np.sqrt(1 -(u/cam_K[0, 0])**2))  -1/cam_K[0, 0]
        dYdx = z * np.cos(theta + beta)* 1/(np.sqrt(1 -(u/cam_K[0, 0])**2)) * -1/cam_K[0, 0]

        G = np.array([[dXdz, dXdx], [dYdz, dYdx]])

        covariance = G @ R @ G.T
    
        return covariance

    def ExtendedKalmanFilter(self, robot_pose, t_pose, t_pose_cov, measurement, cam_K, noise_cov):

        """
            Implementation of the Extended Kalman Filter for the target position estimation

            Args:
                robot_pose: robot pose in the map frame (x, y, theta)
                t_pose: target pose in the map frame (x, y)
                t_pose_cov: covariance matrix of the target position
                measurement: measurement of the target position (z, x)
                cam_K: camera intrinsic parameters
                noise_cov: noise covariance matrix

            Returns:
                t_pose: estimated target position
                t_pose_cov: covariance matrix of the estimated target position
        """

        # Compute the expected measurements from the previous target position and the sensor model
        estimated_measurement = self.compute_measurements(t_pose, robot_pose, cam_K)

        print(f"Estimated measurement : {estimated_measurement}")
        print(f"Measurement : {measurement}")

        # Compute the innovation (difference between the measurement and the expected measurement)
        innovation = measurement - estimated_measurement
        innovation = np.reshape(innovation, (2,1))

        print(f"innovation : {innovation}")

        t_pose = t_pose.squeeze()
        robot_pose = robot_pose.squeeze()

        # Compute the Jacobian matrix of the expected measurements with respect to the target position

        r = np.sqrt((t_pose[0] - robot_pose[0])**2 + (t_pose[1] - robot_pose[1])**2)

        dZdX = (t_pose[0] - robot_pose[0])/ r
        dZdY = (t_pose[1] - robot_pose[1])/ r

        dudX = - np.cos(np.arctan2(t_pose[1] - robot_pose[1], t_pose[0] - robot_pose[0]) - robot_pose[2])*(-t_pose[1] + robot_pose[1])/(r**2) * cam_K[0, 0]

        dudY = - np.cos(np.arctan2(t_pose[1] - robot_pose[1], t_pose[0] - robot_pose[0]) - robot_pose[2])*(t_pose[0] - robot_pose[0])/(r**2) * cam_K[0, 0]

        H = np.array([[dZdX, dZdY] , [dudX, dudY]])  # Jacobian matrix 
        R = noise_cov # Noise covariance matrix

        # Compute the Kalman Gain
        K = t_pose_cov @ H.T @ np.linalg.inv(H @ t_pose_cov @ H.T + R)

        t_pose = np.reshape(t_pose, (2,1))
    
        t_pose = t_pose + K @ innovation   # Update the target position
        t_pose_cov = t_pose_cov - K @ H @ t_pose_cov # Update the covariance matrix

        return t_pose, t_pose_cov

    def update_map(self,poses,robot_pose):
        """
            Update the map with the target position

            Args:
                poses: list of target (x, y) positions

        """

        
        self.map = self.clean_map.copy()
        obstacles_set = self.obstacles

        theta = -robot_pose[2]
        robot_pose = [ self.origin[0] + robot_pose[0] / self.map_res, self.origin[1] - robot_pose[1] / self.map_res]

        map = self.clean_map.copy()
        
        depth_max = 10 // self.map_res
        beta = 1.029

        triangle = np.array([
            [robot_pose[0] , robot_pose[1]],
            [robot_pose[0] + depth_max * np.cos(theta + beta/2), robot_pose[1] + depth_max * np.sin(theta + beta/2)],
            [robot_pose[0] + depth_max * np.cos(theta - beta/2), robot_pose[1] + depth_max * np.sin(theta - beta/2)]], dtype=np.int32)

        cv2.fillPoly(map, [triangle], (255, 0, 0))

        fov = find_set(map, (255, 0, 0))
        real_fov = fov.copy()
        intersection = fov.intersection(obstacles_set)

        for element in intersection:
            x, y = element
            map[x, y] = [0, 0, 255]

            obs = np.array([float(x), float(y)])
            
            vector_dist = [x - robot_pose[1], y - robot_pose[0]]
            # normalize the vector
            vector_dist = vector_dist / np.linalg.norm(vector_dist)

            point = obs

            while True:
                point += vector_dist
                step_x = int(np.ceil(point[0]))
                step_y = int(np.ceil(point[1]))

                if (step_x, step_y) in fov:
                    if (step_x, step_y) in real_fov:
                        real_fov.remove((step_x, step_y))
                else:
                    break

        # draw the fov
        
        fov_map = np.zeros([map.shape[0], map.shape[1]], dtype=np.uint8)

        for element in real_fov:
            x, y = element
            fov_map[x, y] = 255

        fov_map = cv2.erode(fov_map, np.ones((3, 3)), iterations=1)
        fov_map = cv2.dilate(fov_map, np.ones((3, 3)), iterations=1)

        fov_map = cv2.cvtColor(fov_map, cv2.COLOR_GRAY2BGR)
        real_fov = find_set(fov_map, (255,255,255))

        # update the coverage set
        self.coverage_set = self.coverage_set.union(real_fov)

        for element in real_fov:
            x, y = element
            self.map[x, y] = [255, 0, 0]

        # draw the estimated targets
        for pose in self.true_poses:
            x = pose[0] 
            y = pose[1] 

            X = self.origin[0] + x // self.map_res
            Y = self.origin[1] - y  // self.map_res

            cv2.circle(self.map, (int(X), int(Y)), 2, (0, 255, 0), -1)

        # draw real targets
        for pose in poses:
            x = pose[0] 
            y = pose[1] 

            X = self.origin[0] + x // self.map_res
            Y = self.origin[1] - y  // self.map_res

            cv2.circle(self.map, (int(X), int(Y)), 2, (0, 0, 255), -1)
        
        # draw the robot pose
        cv2.circle(self.map, (int(robot_pose[0]), int(robot_pose[1])), 2, (0, 0, 120), -1)

    def find_obstacles(self, map):
        # binary thresholding
        map = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)  # Converte in scala di grigi
        cv2.threshold(map, 240, 255, cv2.THRESH_BINARY, map)
        map = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)  # Converti di nuovo in BGR per il colore

        obstacles_set = find_set(map, (0, 0, 0))

        return obstacles_set

            
    ###                   ####
    #     NEURAL NETWORK     #
    ###                   ####

    def init_network(self,path):
        """
            Initialize the Neural Network

            Args:
               path: path of the neural network parameters
        
        """

        net = YOLO( path )
        return net
    
    def NNdetector(self, net, img):
        """
            Detection of the target object using the Neural Network"

            Args:
                net: Neural Network
                img: input image

            Returns:
                classIds: class Ids of the detected objects
                centers: centers of the detected objects
                img: image with the bounding boxes and masks
                newboxes: bounding boxes of the detected objects
                newmasks: masks of the detected objects
        """

        confThreshold = 0.7  # Confidence threshold
        font = cv2.FONT_HERSHEY_COMPLEX

        out = net(img)

        cols = img.shape[0]
        rows = img.shape[1]

        classIds = []
        newboxes = []
        newmasks = []
        centers = []

        if out[0].masks is not None:
            for result in out:
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks.data  # Masks object for segmentation masks outputs
                labels = boxes.cls.cpu().numpy()
                names = result.names
                confidences = boxes.conf.cpu().numpy()

                
            for i in range(len(boxes)):
                if confidences[i] > confThreshold:
                    bbox = boxes[i].xyxy.cpu().numpy()
                    left = int(bbox[0,0])
                    top = int(bbox[0,1])
                    right = int(bbox[0,2])
                    bottom = int(bbox[0,3])
                    newboxes.append([left, top, right, bottom])   
                    mmask = masks[i].cpu().numpy()
                    mmask = cv2.resize(mmask, (rows, cols))
                    _, mmask = cv2.threshold(mmask, 0.5, 255, cv2.THRESH_BINARY)
                    newmasks.append(mmask)
                    classIds.append(labels[i]) 
                    center = [(left + right) / 2 , (top + bottom) / 2]
                    centers.append(center)
                    size_ratio = (bottom-top)/rows 

            for i in range(len(newboxes)):
                box = newboxes[i]
                mask = newmasks[i]
                left = int(box[0])
                top = int(box[1])
                right = int(box[2])
                bottom = int(box[3])

                if size_ratio >= 0.0:
                    label = names[classIds[i]]
                    cv2.putText(img, label, (left-20, top-10), font, 1, (255, 0, 0))
                    cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 1)
                    cv2.circle(img, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)

                    # Get mask coordinates
                    contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for cnt in contours:
                        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 1)

        return classIds, centers, img, newboxes, newmasks

def main(args=None):
    rclpy.init(args=args)

    node = Project()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()