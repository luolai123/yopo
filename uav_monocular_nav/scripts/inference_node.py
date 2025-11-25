#!/usr/bin/env python3
"""Online inference node using segmentation and planner networks."""
import os

import rospy
import torch
import torchvision.transforms as T
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker

from training.models import SegmentationNet, PlannerNet, terminal_state_from_net
from training.quintic import solve_quintic


class InferenceNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seg_path = rospy.get_param("~segmentation_model")
        planner_path = rospy.get_param("~planner_model")
        self.camera_info = self._load_camera_info(rospy.get_param("~camera_info"))

        self.seg_model = SegmentationNet().to(self.device)
        if os.path.exists(seg_path):
            self.seg_model.load_state_dict(torch.load(seg_path, map_location=self.device))
        self.seg_model.eval()

        self.planner_model = PlannerNet().to(self.device)
        if os.path.exists(planner_path):
            self.planner_model.load_state_dict(torch.load(planner_path, map_location=self.device))
        self.planner_model.eval()

        self.transform = T.Compose([T.ToTensor()])

        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_cb, queue_size=1)
        self.state_sub = rospy.Subscriber("/uav/state", Odometry, self.state_cb, queue_size=1)
        self.traj_pub = rospy.Publisher("/uav_monocular_nav/trajectory", Marker, queue_size=1)
        self.goal_pub = rospy.Publisher("/uav/goal_point", PointStamped, queue_size=1)

        self.latest_image = None
        self.latest_state = None

    def _load_camera_info(self, path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return {
            "fx": data["camera_matrix"]["data"][0],
            "fy": data["camera_matrix"]["data"][4],
            "cx": data["camera_matrix"]["data"][2],
            "cy": data["camera_matrix"]["data"][5],
        }

    def image_cb(self, msg):
        self.latest_image = msg
        self.try_inference()

    def state_cb(self, msg):
        self.latest_state = msg

    def try_inference(self):
        if self.latest_image is None or self.latest_state is None:
            return
        cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding="bgr8")
        tensor_img = self.transform(cv_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _ = self.seg_model(tensor_img)
            delta_p_raw, y_v, y_a = self.planner_model(tensor_img)

        p0 = torch.tensor([
            self.latest_state.pose.pose.position.x,
            self.latest_state.pose.pose.position.y,
            self.latest_state.pose.pose.position.z,
        ], device=self.device).unsqueeze(0)
        v0 = torch.tensor([
            self.latest_state.twist.twist.linear.x,
            self.latest_state.twist.twist.linear.y,
            self.latest_state.twist.twist.linear.z,
        ], device=self.device).unsqueeze(0)
        a0 = torch.zeros_like(v0)

        p_T, v_T, a_T, T_pred = terminal_state_from_net(delta_p_raw, y_v, y_a, p0)
        traj = solve_quintic(p0, v0, a0, p_T, v_T, a_T, T_pred)
        self.publish_marker(traj["pos"][0].cpu().numpy())
        idx = min(5, traj["pos"].shape[-1] - 1)
        self.publish_goal(traj["pos"][0, :, idx].cpu().numpy())

    def publish_marker(self, positions):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.g = 1.0
        marker.color.a = 0.8
        marker.points = []
        for p in positions.T:
            pt = Point()
            pt.x, pt.y, pt.z = p.tolist()
            marker.points.append(pt)
        self.traj_pub.publish(marker)

    def publish_goal(self, point):
        msg = PointStamped()
        msg.header.frame_id = "map"
        msg.point.x, msg.point.y, msg.point.z = point.tolist()
        self.goal_pub.publish(msg)


def main():
    rospy.init_node("inference_node")
    node = InferenceNode()
    rospy.spin()


if __name__ == "__main__":
    main()
