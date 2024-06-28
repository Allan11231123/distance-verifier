import rclpy
from rclpy.node import Node
import copy
from geometry_msgs.msg import (
    TwistWithCovariance,
    PoseWithCovarianceStamped,
    PoseWithCovariance,
    Pose,
    Twist,
    Transform,
    TransformStamped,
    Quaternion,
)

from autoware_auto_vehicle_msgs.msg import VelocityReport
from carla_msgs.msg import CarlaEgoVehicleInfo,CarlaEgoVehicleStatus
from rclpy.qos import QoSReliabilityPolicy, QoSProfile, QoSHistoryPolicy,DurabilityPolicy
import math


class DistanceVerifier(Node):
    def __init__(self):
        super().__init__("distance_verifier")
        # self.declare_parameter("angle", rclpy.Parameter.Type.DOUBLE)

        self.yabloc_path_subscription = self.create_subscription(
            CarlaEgoVehicleInfo,
            "/localization/validation/path/pf",
            self.yabloc_path_listener_callback,
            QoSProfile(
                depth=10,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),

        )
        self.carla_path_subscription = self.create_subscription(
            CarlaEgoVehicleStatus,
            "/groundtruth_pose",
            self.carla_path_listener_callback,
            1,
        )

        # self.init_publisher = self.create_publisher(
        #     VelocityReport,
        #     "/vehicle/status/velocity_status",
        #     QoSProfile(
        #         depth=10,
        #         durability=DurabilityPolicy.TRANSIENT_LOCAL,
        #     ),
        # )

        self.groundtruth = None
        self.prediction = None
        self.steer = 0.0
        self.stamped_time = None
        self.out_report = None
        

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def yabloc_path_listener_callback(self, msg):
        self.get_logger().info("catching prediction path from yabloc")
        self.prediction = msg
    def carla_path_listener_callback(self,msg):
        self.get_logger().info("catching ground truth path form rosbag")
        self.groundtruth = msg

    def timer_callback(self):
        if self.out_report is not None:
            self.init_publisher.publish(self.out_report)

    def update_vehicle_report(self):
        if self.vehicle_status is None or self.vehicle_info is None:
            return
        veh_report = VelocityReport()
        veh_report.longitudinal_velocity = self.vehicle_status.velocity
        veh_report.lateral_velocity = 0.0
        veh_report.heading_rate = self.compute_heading()
        self.out_report = veh_report

    def compute_heading(self):
        time_s = self.vehicle_status.header.stamp.sec
        time_ns = self.vehicle_status.header.stamp.nanosec
        time = time_s + time_ns*(10**(-9))
        if self.stamped_time is None:
            self.stamped_time = time
            return 0.0

        wheel0_info = self.vehicle_info.wheels[0]
        max_angle = wheel0_info.max_steer_angle
        steering = self.vehicle_status.control.steer
        rad = math.radians((max_angle/2)*steering)
        
        result = (rad-self.steer)/(time-self.stamped_time)
        self.steer = rad
        self.stamped_time = time
        return result

def main():
    rclpy.init()
    distance_verifier = DistanceVerifier()

    rclpy.spin(distance_verifier)
    distance_verifier.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
