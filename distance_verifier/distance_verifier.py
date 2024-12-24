import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, DurabilityPolicy
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import bisect
import tf_transformations
import cv2
from std_msgs.msg import String

class DistanceVerifier(Node):
    def __init__(self):
        super().__init__("distance_verifier")

        self.declare_parameter('file_name', "daytime")

        self.yabloc_path_subscription = self.create_subscription(
            PoseStamped,
            "/localization/pf/pose",
            self.yabloc_path_listener_callback,
            1,
        )

        self.lateral_publisher = self.create_publisher(
            Image,
            "/lateral_report",
            QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

        self.ade_publisher = self.create_publisher(
            String,
            "/ade_value",
            QoSProfile(
                depth=10,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self.bridge = CvBridge()

        groundtruth_file_name = self.get_parameter('file_name').value
        groundtruth_file_path = "config/" + groundtruth_file_name + "_ground_truth.csv"
        self.ground_truth_data = self.load_ground_truth_data(groundtruth_file_path)
        self.csv_file = open('lateral_error.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['sec', 'nanosec', 'lateral'])

        
        self.laterals = []
        self.time = []
        self.clock = 0
        self.laterals_image = None

    def load_ground_truth_data(self, filename):
        ground_truth = []
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                sec = int(row['sec'])
                nanosec = int(row['nanosec'])
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
        pred_x = msg.pose.position.x
        pred_y = msg.pose.position.y
        orientation_q = msg.pose.orientation
        qx, qy, qz, qw = orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
        _, _, yaw_p = tf_transformations.euler_from_quaternion([qx, qy, qz, qw])
        u_x = math.cos(yaw_p)
        u_y = math.sin(yaw_p)

        ground_truth_pos = self.interpolate_ground_truth(msg.header.stamp.sec, msg.header.stamp.nanosec)
        if ground_truth_pos:
            gt_x, gt_y = ground_truth_pos
            lateral_error = self.calculate_lateral_error(pred_x, pred_y, gt_x, gt_y, u_x, u_y)
            self.update_lateral_error(msg.header.stamp.sec, msg.header.stamp.nanosec, lateral_error)

    def calculate_lateral_error(self, pred_x, pred_y, gt_x, gt_y, u_x, u_y):
        x_diff = gt_x - pred_x
        y_diff = gt_y - pred_y
        lateral_error = x_diff * (-u_y) + y_diff * u_x
        return lateral_error

    def update_lateral_error(self, sec, nanosec, lateral_error):
        self.laterals.append(lateral_error)
        self.csv_writer.writerow([sec, nanosec, lateral_error])
        self.clock += 1
        self.time.append(self.clock)
        self.generate_lateral_report_image()
        difference_value = String()
        difference_value.data = "=== Lateral ADE ===\n" + str(np.mean(self.laterals))
        self.ade_publisher.publish(difference_value)


    def generate_lateral_report_image(self):
        fig, ax = plt.subplots()
        ax.plot(self.time, self.laterals, label="Lateral Error")
        ax.set_title("Lateral Error Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Lateral Error (m)")
        ax.legend()
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        self.laterals_image = self.bridge.cv2_to_imgmsg(cv2.cvtColor(image, cv2.COLOR_RGBA2BGR), "bgr8")
        self.lateral_publisher.publish(self.laterals_image)
        plt.close()

def main():
    rclpy.init()
    distance_verifier = DistanceVerifier()
    rclpy.spin(distance_verifier)
    distance_verifier.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
