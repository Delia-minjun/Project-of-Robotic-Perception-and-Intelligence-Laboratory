#source
cd 2025_ws

source /opt/ros/humble/setup.bash
source ~/2025_ws/install/setup.bash

#arm
ros2 launch interbotix_xsarm_control xsarm_control.launch.py \
    robot_model:=px100 \
    use_rviz:=true


#pan-tilt
ros2 launch pan_tilt_bringup panTilt_bringup.launch.py
ros2 topic pub /pan_tilt_cmd_deg pan_tilt_msgs/msg/PanTiltCmdDeg "{yaw: 30.0, pitch: 30.0, speed: 5}"

#hardware
ros2 launch iqr_tb4_bringup bringup.launch.py

#keyboard
ros2 run teleop_twist_keyboard teleop_twist_keyboard

#test-files
python3 /home/tony/2025_ws/src/arm_controller_demo.py
python3 /home/tony/2025_ws/src/test_xmj.py
python3 /home/tony/2025_ws/src/gripper.py
python3 /home/tony/2025_ws/src/ttt.py
python3 /home/tony/2025_ws/src/arm_listener.py
python3 /home/tony/2025_ws/src/chech_joint_state.py
python3 /home/tony/2025_ws/src/check_aruco.py
#navigation

cd src/navigator_2025/launch
ros2 launch navigator_launch.py

ros2 lifecycle set /map_server configure
ros2 lifecycle set /map_server activate

ros2 launch turtlebot4_viz view_robot.launch.py
#camera
ros2 run realsense2_camera realsense2_camera_node
ros2 launch realsense2_camera rs_camera.launch.py

#aruco
ros2 launch ros2_aruco aruco_recognition.launch.py

#查看所有正在运行的ROS2节点
ros2 node list
#查看所有话题
ros2 topic list