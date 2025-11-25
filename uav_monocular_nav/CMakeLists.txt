cmake_minimum_required(VERSION 3.0.2)
project(uav_monocular_nav)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  message_generation
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/data_collector.py
  scripts/inference_node.py
  scripts/simple_uav_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
