import rclpy
from rclpy.node import Node
import copy
from geometry_msgs.msg import (
    TwistWithCovariance,
    PoseWithCovarianceStamped,
    PoseWithCovariance,
    Pose,
    PoseStamped,
    Twist,
    Transform,
    TransformStamped,
    Quaternion,
)
from std_msgs import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# from autoware_auto_vehicle_msgs.msg import VelocityReport
# from carla_msgs.msg import CarlaEgoVehicleInfo,CarlaEgoVehicleStatus
from rclpy.qos import QoSReliabilityPolicy, QoSProfile, QoSHistoryPolicy,DurabilityPolicy
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

class DistanceVerifier(Node):
    def __init__(self):
        super().__init__("distance_verifier")
        # self.declare_parameter("angle", rclpy.Parameter.Type.DOUBLE)

        self.yabloc_path_subscription = self.create_subscription(
            PoseStamped,
            "/localization/validation/path/pf",
            self.yabloc_path_listener_callback,
            QoSProfile(
                depth=10,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),

        )
        self.carla_path_subscription = self.create_subscription(
            PoseStamped,
            "/groundtruth_pose",
            self.carla_path_listener_callback,
            1,
        )

        self.image_publisher = self.create_publisher(
            Image,
            "/difference_report",
            QoSProfile(
                depth=10,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.ade_publisher = self.create_publisher(
            Float32,
            "/ade_value",
            QoSProfile(
                depth=10,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.fde_publisher = self.create_publisher(
            Float32,
            "/fde_value",
            QoSProfile(
                depth=10,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self.groundtruth = None
        self.prediction = None
        self.difference = None
        self.image = None
        self.bridge = CvBridge()
        self.differences = []
        self.clock = 0
        self.time = []

        self.groundtruths = []
        self.predictions = []
        self.ade = None
        self.fde = None

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def yabloc_path_listener_callback(self, msg):
        self.get_logger().info("catching prediction path position from yabloc")
        self.prediction = msg.pose.position
        self.update_path_position()
    def carla_path_listener_callback(self,msg):
        self.get_logger().info("catching ground truth path position form rosbag")
        self.groundtruth = msg.pose.position
        self.update_path_position()

    def timer_callback(self):
        if self.image is not None:
            self.image_publisher.publish(self.image)
        if self.ade is not None:
            self.ade_publisher.publish(self.ade)
        if self.fde is not None:
            self.fde_publisher.publish(self.fde)
        

    def calculate_difference(self):
        x_diff = self.prediction.x - self.groundtruth.x
        y_diff = self.prediction.y - self.groundtruth.y
        z_diff = self.prediction.z - self.groundtruth.z
        return (x_diff**2 + y_diff**2 + z_diff**2)**(1/2)
    
    def generate_report_image(self):
        fig, ax = plt.subplot()
        ax.plot(self.time,self.differences)
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        cols, rows = fig.canvas.get_width_height()
        image = np.frombuffer(buf,dtype=np.int8).reshape(rows,cols,3)
        self.image = self.bridge.cv2_to_imgmsg(image,"bgr8")
        
    def update_path_position(self):
        if self.groundtruth is None or self.prediction is None:
            return
        diff = self.calculate_difference()
        if self.difference == diff: 
            return
        else:
            self.difference = diff
            self.differences.append(self.difference)
            self.groundtruths.append(self.groundtruth)
            self.predictions.append(self.prediction)
            self.clock+=1
            self.time.append(self.clock)
            self.calculate_ade()
            self.calculate_fde()
    
    def calculate_ade(self):
        ade_value = Float32()
        ade_value.data = np.mean(cdist(self.predictions,self.groundtruths))
        self.ade = ade_value
    def calculate_fde(self):
        fde_value = Float32()
        fde_value.data = np.linalg.norm(self.predictions[-1,:] - self.groundtruths[-1,:])
        self.fde = fde_value


def main():
    rclpy.init()
    distance_verifier = DistanceVerifier()

    rclpy.spin(distance_verifier)
    distance_verifier.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
