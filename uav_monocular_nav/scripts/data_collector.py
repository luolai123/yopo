#!/usr/bin/env python3
"""
ROS node for collecting images, auto-labeled masks, and UAV state for training.
"""
import json
import os
import random
from typing import Tuple

import cv2
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker


class ObstacleProjector:
    def __init__(self, obstacles, camera_info):
        self.obstacles = obstacles
        self.camera_info = camera_info

    def ray_hits_obstacle(self, ray_cam):
        for ob in self.obstacles:
            x, y, r = ob["x"], ob["y"], ob["radius"]
            if abs(ray_cam[0] - x) < r and abs(ray_cam[1] - y) < r:
                return True
        return False

    def build_mask(self, height, width):
        fx = self.camera_info["fx"]
        fy = self.camera_info["fy"]
        cx = self.camera_info["cx"]
        cy = self.camera_info["cy"]
        xs = np.linspace(0, width - 1, width)
        ys = np.linspace(0, height - 1, height)
        xv, yv = np.meshgrid(xs, ys)
        x_cam = (xv - cx) / fx
        y_cam = (yv - cy) / fy
        dirs = np.stack([x_cam, y_cam, np.ones_like(x_cam)], axis=-1)
        mask = np.ones((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                if self.ray_hits_obstacle(dirs[i, j]):
                    mask[i, j] = 0
        return mask


class DataCollector:
    def __init__(self):
        self.bridge = CvBridge()
        self.camera_info = self.load_camera_info(rospy.get_param("~camera_info"))
        obstacles_path = rospy.get_param("~obstacles")
        with open(obstacles_path, "r") as f:
            self.obstacles = json.load(f)
        self.projector = ObstacleProjector(self.obstacles, self.camera_info)
        self.save_dir = rospy.get_param("~save_dir", os.path.expanduser("~/uav_data"))
        self.num_samples = rospy.get_param("~num_samples", 500)
        self.sample_rate = rospy.get_param("~sample_rate", 1.0)
        xy_limits = rospy.get_param("~xy_limits", [-3.0, 3.0])
        z_limits = rospy.get_param("~z_limits", [0.8, 1.5])
        self.goal_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (float(xy_limits[0]), float(xy_limits[1])),
            (float(z_limits[0]), float(z_limits[1])),
        )
        os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "states"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "goals"), exist_ok=True)

        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_cb, queue_size=1)
        self.state_sub = rospy.Subscriber("/uav/state", Odometry, self.state_cb, queue_size=1)
        self.goal_pub = rospy.Publisher("/uav/goal_point", PointStamped, queue_size=1)
        self.marker_pub = rospy.Publisher("/uav_monocular_nav/obstacles", Marker, queue_size=1)

        self.latest_image = None
        self.latest_state = None
        self.current_goal = None
        self.saved_samples = 0
        rospy.Timer(rospy.Duration(1.0 / self.sample_rate), self.timer_cb)
        self.publish_new_goal()

    def load_camera_info(self, path):
        data = yaml.safe_load(open(path, "r"))
        return {
            "fx": data["camera_matrix"]["data"][0],
            "fy": data["camera_matrix"]["data"][4],
            "cx": data["camera_matrix"]["data"][2],
            "cy": data["camera_matrix"]["data"][5],
            "width": data.get("image_width", 640),
            "height": data.get("image_height", 480),
        }

    def image_cb(self, msg):
        self.latest_image = msg

    def state_cb(self, msg):
        self.latest_state = msg

    def timer_cb(self, _):
        if self.latest_image is None or self.latest_state is None:
            return
        if self.saved_samples >= self.num_samples:
            rospy.loginfo_once("Reached requested dataset size; stopping collection.")
            return
        if self.current_goal is None:
            self.publish_new_goal()
            return
        stamp = rospy.Time.now().to_sec()
        img = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding="bgr8")
        mask = self.projector.build_mask(img.shape[0], img.shape[1])
        goal = {"goal": [self.current_goal.point.x, self.current_goal.point.y, self.current_goal.point.z]}
        basename = f"{int(stamp)}.png"
        cv2.imwrite(os.path.join(self.save_dir, "images", basename), img)
        cv2.imwrite(os.path.join(self.save_dir, "masks", basename), mask)
        state_dict = {
            "position": [self.latest_state.pose.pose.position.x, self.latest_state.pose.pose.position.y, self.latest_state.pose.pose.position.z],
            "velocity": [self.latest_state.twist.twist.linear.x, self.latest_state.twist.twist.linear.y, self.latest_state.twist.twist.linear.z],
            "acceleration": [0, 0, 0],
        }
        json.dump(state_dict, open(os.path.join(self.save_dir, "states", basename.replace(".png", ".json")), "w"))
        json.dump(goal, open(os.path.join(self.save_dir, "goals", basename.replace(".png", ".json")), "w"))
        self.publish_obstacle_markers()
        self.saved_samples += 1
        rospy.loginfo(f"Saved sample {self.saved_samples}/{self.num_samples}: {basename}")
        self.publish_new_goal()

    def publish_new_goal(self):
        msg = PointStamped()
        msg.header.frame_id = "map"
        msg.point.x = random.uniform(*self.goal_bounds[0])
        msg.point.y = random.uniform(*self.goal_bounds[0])
        msg.point.z = random.uniform(*self.goal_bounds[1])
        self.current_goal = msg
        self.goal_pub.publish(msg)
        rospy.loginfo(f"Publishing new goal: ({msg.point.x:.2f}, {msg.point.y:.2f}, {msg.point.z:.2f})")

    def publish_obstacle_markers(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 2.0
        marker.color.r = 1.0
        marker.color.a = 0.5
        for i, ob in enumerate(self.obstacles):
            marker.id = i
            marker.pose.position.x = ob["x"]
            marker.pose.position.y = ob["y"]
            marker.pose.position.z = ob.get("height", 2.0) / 2.0
            self.marker_pub.publish(marker)


if __name__ == "__main__":
    rospy.init_node("data_collector")
    node = DataCollector()
    rospy.spin()
