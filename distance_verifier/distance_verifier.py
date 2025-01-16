import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, DurabilityPolicy
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import math
import bisect
import tf_transformations
import cv2
from std_msgs.msg import String
import yaml

class DistanceVerifier(Node):
    def __init__(self):
        super().__init__("ade_metric")

        groundtruth_path = config["groundtruth_path"]
        errorfile_path = config["errorfile_path"]

        # Subscribe to Pose data
        self.yabloc_path_subscription = self.create_subscription(
            PoseStamped,
            "/localization/pf/pose",
            self.yabloc_path_listener_callback,
            1,
        )

        # Publisher for three types of error images
        self.lateral_publisher = self.create_publisher(
            Image,
            "/lateral_report",
            QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.longitudinal_publisher = self.create_publisher(
            Image,
            "/longitudinal_report",
            QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.distance_publisher = self.create_publisher(
            Image,
            "/distance_report",
            QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

        # ADE text publisher
        self.ade_publisher = self.create_publisher(
            String,
            "/ade_value",
            QoSProfile(
                depth=10,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self.bridge = CvBridge()

        # Load Ground Truth
        self.ground_truth_data = self.load_ground_truth_data(groundtruth_path)
        
        # Create a CSV file to store the errors
        self.csv_file = open(errorfile_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['sec', 'nsec', 'lateral', 'longitudinal', 'distance'])
        
        
        self.laterals = []
        self.longitudinals = []
        self.distances = []
        self.time = []
        self.clock = 0

    def load_ground_truth_data(self, filename):
        ground_truth = []
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                sec = int(row['sec'])
                nanosec = int(row['nsec'])
                x = float(row['x'])
                y = float(row['y'])
                ground_truth.append((sec, nanosec, x, y))
        return ground_truth

    def interpolate_ground_truth(self, sec, nanosec):
        target_time = sec + nanosec * 1e-9
        times = [gt[0] + gt[1] * 1e-9 for gt in self.ground_truth_data]
        idx = bisect.bisect_left(times, target_time)

        if idx == 0 or idx >= len(times):
            return None

        t1, t2 = times[idx - 1], times[idx]
        x1, y1 = self.ground_truth_data[idx - 1][2:4]
        x2, y2 = self.ground_truth_data[idx][2:4]

        ratio = (target_time - t1) / (t2 - t1)
        x_interp = x1 + ratio * (x2 - x1)
        y_interp = y1 + ratio * (y2 - y1)

        return x_interp, y_interp

    def yabloc_path_listener_callback(self, msg):

        # Predicted position and orientation
        pred_x = msg.pose.position.x
        pred_y = msg.pose.position.y
        orientation_q = msg.pose.orientation
        qx, qy, qz, qw = orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
        _, _, yaw_p = tf_transformations.euler_from_quaternion([qx, qy, qz, qw])
        
        # Vehicle direction
        u_x = math.cos(yaw_p)
        u_y = math.sin(yaw_p)

        ground_truth_pos = self.interpolate_ground_truth(msg.header.stamp.sec, msg.header.stamp.nanosec)
        if ground_truth_pos:
            gt_x, gt_y = ground_truth_pos

            # Calculate errors
            lateral_error = self.calculate_lateral_error(pred_x, pred_y, gt_x, gt_y, u_x, u_y)
            longitudinal_error = self.calculate_longitudinal_error(pred_x, pred_y, gt_x, gt_y, u_x, u_y)
            distance_error = self.calculate_distance_error(pred_x, pred_y, gt_x, gt_y)

            # Update errors
            self.update_errors(msg.header.stamp.sec, msg.header.stamp.nanosec,
                               lateral_error, longitudinal_error, distance_error)
    
    def calculate_lateral_error(self, pred_x, pred_y, gt_x, gt_y, u_x, u_y):
        x_diff = gt_x - pred_x
        y_diff = gt_y - pred_y
        lateral_error = x_diff * (-u_y) + y_diff * u_x
        return lateral_error

    def calculate_longitudinal_error(self, pred_x, pred_y, gt_x, gt_y, u_x, u_y):
        x_diff = gt_x - pred_x
        y_diff = gt_y - pred_y
        longitudinal_error = x_diff * u_x + y_diff * u_y
        return longitudinal_error

    def calculate_distance_error(self, pred_x, pred_y, gt_x, gt_y):
        x_diff = gt_x - pred_x
        y_diff = gt_y - pred_y
        distance_error = math.sqrt(x_diff**2 + y_diff**2)
        return distance_error

    def update_errors(self, sec, nanosec, lateral_error, longitudinal_error, distance_error):
        # Update errors
        self.laterals.append(lateral_error)
        self.longitudinals.append(longitudinal_error)
        self.distances.append(distance_error)

        # Write to CSV
        self.csv_writer.writerow([sec, nanosec, lateral_error, longitudinal_error, distance_error])

        # Update time
        self.clock += 1
        self.time.append(self.clock)

        # Generate report images
        self.generate_lateral_report_image()
        self.generate_longitudinal_report_image()
        self.generate_distance_report_image()

        # Publish ADE value
        difference_value = String()
        difference_value.data = (
            f"=== Errors (Avg) ===\n"
            f"Lateral: {np.mean(self.laterals):.4f}\n"
            f"Longitudinal: {np.mean(self.longitudinals):.4f}\n"
            f"Distance: {np.mean(self.distances):.4f}"
        )
        self.ade_publisher.publish(difference_value)

    def generate_lateral_report_image(self):
        fig, ax = plt.subplots()
        ax.plot(self.time, self.laterals, label="Lateral Error", color='blue')
        ax.set_title("Lateral Error Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Lateral Error (m)")
        ax.legend()
        fig.canvas.draw()

        image = np.array(fig.canvas.renderer.buffer_rgba())
        img_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(image, cv2.COLOR_RGBA2BGR), "bgr8")
        self.lateral_publisher.publish(img_msg)
        plt.close(fig)

    def generate_longitudinal_report_image(self):
        fig, ax = plt.subplots()
        ax.plot(self.time, self.longitudinals, label="Longitudinal Error", color='green')
        ax.set_title("Longitudinal Error Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Longitudinal Error (m)")
        ax.legend()
        fig.canvas.draw()

        image = np.array(fig.canvas.renderer.buffer_rgba())
        img_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(image, cv2.COLOR_RGBA2BGR), "bgr8")
        self.longitudinal_publisher.publish(img_msg)
        plt.close(fig)

    def generate_distance_report_image(self):
        fig, ax = plt.subplots()
        ax.plot(self.time, self.distances, label="Distance Error", color='red')
        ax.set_title("Distance Error Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Distance Error (m)")
        ax.legend()
        fig.canvas.draw()

        image = np.array(fig.canvas.renderer.buffer_rgba())
        img_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(image, cv2.COLOR_RGBA2BGR), "bgr8")
        self.distance_publisher.publish(img_msg)
        plt.close(fig)


def main():
    rclpy.init()
    distance_verifier = DistanceVerifier()
    rclpy.spin(distance_verifier)
    distance_verifier.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
