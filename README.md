<h1 align="center"> Autonomous Navigation and Grasping System for Mobile Robots</h1>

<p align="center">
  <img src="https://img.shields.io/badge/OS-Ubuntu%20%2F%20Linux-E95420?style=flat-square&logo=ubuntu&logoColor=white" alt="OS" />
  <img src="https://img.shields.io/badge/Framework-ROS%202-22314E?style=flat-square&logo=ros&logoColor=white" alt="ROS2" />
  <img src="https://img.shields.io/badge/Hardware-Mobile%20Manipulator-blue?style=flat-square" alt="Hardware" />
  <img src="https://img.shields.io/badge/Task-Pick%20&%20Place-brightgreen?style=flat-square" alt="Task" />
</p>

## 📖 Project Overview
This project was developed as the final project for the **Robotic Perception and Intelligence** course. The objective was to engineer a fully autonomous software system for a mobile manipulator to complete a comprehensive arena challenge. 

The full system pipeline includes:
- **Autonomous Navigation**: Utilizing SLAM and `Nav2` to navigate across a mapped arena with both fixed and randomly placed obstacles.
- **Visual Detection**: Utilizing a RealSense camera to detect traffic lights and stop signs during navigation.
- **Pick & Place (My Focus)**: Autonomously locating target objects (via ArUco markers or markerless detection), grasping them with a 4-DOF arm, and placing them in designated target areas.

## 🛠️ Hardware & Software Stack
The project was deployed on a custom mobile manipulator platform.
- **Mobile Base**: Differential drive chassis (`iqr_tb4_ros`).
- **Manipulator**: 4-DOF Robotic Arm (`interbotix_ros_manipulators`).
- **Vision & Head**: RealSense RGB-D Camera mounted on a controllable Pan-Tilt mechanism.
- **Software Framework**: Ubuntu/Linux running **ROS 2**. Packages used include `slam_toolbox`, `nav2`, `realsense-ros`, and custom planners.

---

## My Contribution: 4-DOF Arm Control & Manipulation

> **Note**: The full integrated workspace (including SLAM, Nav2, and Vision nodes) was deployed directly on the robot's onboard PC. **This repository specifically showcases my core contribution to the team: The Robot Arm Control and Pick & Place logic.**

I was responsible for the manipulation architecture of the robot. The codes within this repository handle:

1. **Kinematics & Control**: Controlling the 4-DOF Interbotix arm to execute smooth and precise trajectories.
2. **Dynamic Pick & Place**: 
   - Grasping target objects from fixed coordinates.
   - Grasping target objects from random positions within a specific area (integrating spatial coordinates from the vision team).
3. **Integration Setup**: Ensuring the arm safely returns to its designated resting configuration during navigation to maintain the robot's center of gravity and prevent visual obstruction.

## Base Framework & Acknowledgments
The foundation of the entire project's ROS 2 workspace (bringing up the chassis, camera, and basic drivers) is based on the open-source laboratory template. You can find the base architectural framework here:

👉 [SUSTech-EE211-Robotic-Perception-and-Intelligence-Laboratory](https://github.com/Gralerfics/SUSTech-EE211-Robotic-Perception-and-Intelligence-Laboratory.git)
