import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
# from autoware_auto_vehicle_msgs.msg import VelocityReport
# from carla_msgs.msg import CarlaEgoVehicleInfo,CarlaEgoVehicleStatus
from rclpy.qos import  QoSProfile, DurabilityPolicy
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import cv2

class DistanceVerifier(Node):
    def __init__(self):
        super().__init__("distance_verifier")
        # self.declare_parameter("angle", rclpy.Parameter.Type.DOUBLE)

        self.yabloc_path_subscription = self.create_subscription(
            Path,
            "/localization/validation/path/pf",
            self.yabloc_path_listener_callback,
            1,
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
            String,
            "/ade_value",
            QoSProfile(
                depth=10,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.fde_publisher = self.create_publisher(
            String,
            "/fde_value",
            QoSProfile(
                depth=10,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.difference_publisher = self.create_publisher(
            String,
            "/difference_value",
            QoSProfile(
                depth=10,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self.groundtruth = None
        self.checkpoint = None
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
        self.difference_value = None

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def yabloc_path_listener_callback(self, msg):
        self.prediction = msg.poses[-1].pose.position
    def carla_path_listener_callback(self,msg):
        self.groundtruth = msg.pose.position
        self.update_path_position()

    def timer_callback(self):
        if self.image is not None:
            self.image_publisher.publish(self.image)
        if self.ade is not None:
            self.ade_publisher.publish(self.ade)
        if self.fde is not None:
            self.fde_publisher.publish(self.fde)
        if self.difference_value is not None:
            self.difference_publisher.publish(self.difference_value)
        if self.checkpoint is not None and self.checkpoint==self.groundtruth:
            self.save_image("result.png")
        else:
            self.checkpoint = self.groundtruth
        

    def calculate_difference(self):
        x_diff = self.prediction.x - self.groundtruth.x
        y_diff = self.prediction.y - self.groundtruth.y
        z_diff = self.prediction.z - self.groundtruth.z
        diff_value = (x_diff**2 + y_diff**2 + z_diff**2)**(1/2)
        return diff_value
    
    def save_image(self,name = None):
        if self.image  is None:
            self.get_logger().info("Can't save image, since there appears to be no image.")
            return
        image = self.bridge.imgmsg_to_cv2(self.image,"bgr8")
        if name is not None:
            cv2.imwrite(name,image)
        else:
            cv2.imwrite("distance_result.png",image)
        self.get_logger().info("======== Image saved!!!!! ========")
    
    def generate_report_image(self):
        fig, ax = plt.subplots()
        ax.plot(self.time,self.differences)
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        self.image = self.bridge.cv2_to_imgmsg(cv2.cvtColor(image,cv2.COLOR_RGBA2BGR),"bgr8")
        plt.close()
    
    def calculate_ade(self):
        ade_value = String()
        # ade_value.data = "=== ADE ===\n" + str(self.difference)
        ade_value.data = "=== ADE ===\n" + str(np.mean(cdist(self.predictions,self.groundtruths)))
        self.ade =  ade_value
    def calculate_fde(self):
        fde_value = String()
        fde_value.data = "=== FDE ===\n" + str(np.linalg.norm(self.predictions[-1,:] - self.groundtruths[-1,:]))
        self.fde = fde_value    
    def generate_difference_value(self):
        difference_value = String()
        difference_value.data = "=== Distance Difference ===\n" + str(self.difference)
        self.difference_value = difference_value

    def update_path_position(self):
        if self.groundtruth is None or self.prediction is None:
            return
        diff = self.calculate_difference()
        if self.difference == diff: 
            return
        else:
            self.difference = diff
            self.generate_difference_value()
            self.differences.append(self.difference)
            self.groundtruths.append([self.groundtruth.x, self.groundtruth.y, self.groundtruth.z])
            self.predictions.append([self.prediction.x, self.prediction.y, self.prediction.z])
            self.clock+=1
            self.time.append(self.clock)
            self.generate_report_image()
            self.calculate_ade()
            # self.calculate_fde()
    


def main():
    rclpy.init()
    distance_verifier = DistanceVerifier()

    rclpy.spin(distance_verifier)
    distance_verifier.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
