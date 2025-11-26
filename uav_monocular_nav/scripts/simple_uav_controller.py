#!/usr/bin/env python3
"""Simple P controller that follows published goal points."""
import rospy
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker


class SimpleUavController:
    def __init__(self):
        state_topic = rospy.get_param("~state_topic", "/odom")
        self.goal_sub = rospy.Subscriber("/uav/goal_point", PointStamped, self.goal_cb, queue_size=1)
        self.state_sub = rospy.Subscriber(state_topic, Odometry, self.state_cb, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.pose_pub = rospy.Publisher("/uav/pose_marker", Marker, queue_size=1)
        self.goal = None
        self.state = None
        self.kp = rospy.get_param("~kp", 0.5)
        rospy.Timer(rospy.Duration(0.05), self.control_cb)

    def goal_cb(self, msg):
        self.goal = msg.point

    def state_cb(self, msg):
        self.state = msg.pose.pose

    def control_cb(self, _):
        if self.goal is None or self.state is None:
            return
        cmd = Twist()
        cmd.linear.x = self.kp * (self.goal.x - self.state.position.x)
        cmd.linear.y = self.kp * (self.goal.y - self.state.position.y)
        cmd.linear.z = self.kp * (self.goal.z - self.state.position.z)
        self.cmd_pub.publish(cmd)
        self.publish_pose_marker()

    def publish_pose_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.scale.x = marker.scale.y = marker.scale.z = 0.3
        marker.color.b = 1.0
        marker.color.a = 0.8
        marker.pose = self.state
        self.pose_pub.publish(marker)


def main():
    rospy.init_node("simple_uav_controller")
    SimpleUavController()
    rospy.spin()


if __name__ == "__main__":
    main()
