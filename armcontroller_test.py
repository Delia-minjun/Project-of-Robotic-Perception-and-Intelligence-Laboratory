import roboticstoolbox as rtb
import rclpy
from rclpy.node import Node
from interbotix_xs_msgs.msg import JointSingleCommand, JointGroupCommand
from sensor_msgs.msg import JointState
import numpy as np
import time
import math
from spatialmath import SE3
from geometry_msgs.msg import Pose, PoseStamped
from pan_tilt_msgs.msg import PanTiltCmdDeg
import tf_transformations
from std_msgs.msg import String
from tf2_ros import TransformListener, Buffer, TransformException
from scipy.spatial.transform import Rotation as scipyR

class ArmController(Node):
    def __init__(self):
        super().__init__("ArmController")
        
        # ==================== 配置参数 ====================
        # 只需要修改这一行来切换模式：
        self.MODE = "VISION"  # 可选值: "FIXED" - 固定位姿模式, "VISION" - 视觉引导模式
        # ================================================
        
        # 创建发布器（共用）
        self.arm_cmd_pub = self.create_publisher(
            JointSingleCommand, "/px100/commands/joint_single", 10)
        self.arm_group_pub = self.create_publisher(
            JointGroupCommand, "/px100/commands/joint_group", 10)
        
        # 创建订阅器（共用）
        self.joint_states_sub = self.create_subscription(
            JointState, "/px100/joint_states", self.joint_states_cb, 10)
        
        # 机器人模型（共用）
        self.robot = rtb.models.px100()
        self.num_joints = 4
        
        # 关节控制参数（共用）
        self.joint_pos = []
        self._has_joint_states = False
        
        # 夹爪控制参数（共用）
        self.gripper_opened = 1.5
        self.gripper_closed = 0.7
        
        # 关节限制（共用）
        self.joint_lower_limits = [-1.5, -0.4, -1.6, -1.8]
        self.joint_upper_limits = [1.5, 0.9, 1.7, 1.8]
        
        # 固定位姿模式参数
        self.target_joint_positions = [0.0, 0.7, -0.6, 0.7]
        self.handup_joint_positions = [0.0, 0.5, -1.0, 0.5]
        self.home_joint_positions = [0.0, 0.0, 0.0, 0.0]
        
        # 状态控制变量
        self.control_state = "INIT"
        self.state_start_time = time.time()
        
        # 添加原始代码中的变量
        self.cnt = 0
        self.thred = 0.1
        self.moving_time = 2.0
        
        # 初始化关节命令
        self.arm_group_command = JointGroupCommand()
        self.arm_group_command.name = "arm"
        self.arm_group_command.cmd = self.target_joint_positions
        
        # 夹爪控制命令
        self.gripper_command = JointSingleCommand()
        self.gripper_command.name = "gripper"
        
        # 如果使用视觉模式，初始化视觉相关组件
        if self.MODE == "VISION":
            self._init_vision_components()
        
        # 创建定时器
        self.control_timer = self.create_timer(0.1, self.timer_cb)
        
        self.get_logger().info(f"控制器初始化完成 - 模式: {self.MODE}")
    
    def _init_vision_components(self):
        """初始化视觉模式专用组件"""
        # 云台发布器
        self.pan_tilt_pub = self.create_publisher(
            PanTiltCmdDeg, "/pan_tilt_cmd_deg", 10)
        
        # 可视化发布器
        self.aruco_target_pose_pub = self.create_publisher(
            PoseStamped, "/aruco_target_pose", 10)
        self.block_center_pose_pub = self.create_publisher(
            PoseStamped, "/block_center_pose", 10)
        self.arm_target_pose_pub = self.create_publisher(
            PoseStamped, "/arm_target_pose", 10)
        self.transformed_pose_pub = self.create_publisher(  # 新增：发布转换后的坐标
            PoseStamped, "/transformed_target_pose", 10)
        
        # ArUco订阅器
        self.aruco_pose_sub = self.create_subscription(
            PoseStamped, "/aruco_pose", self.aruco_pose_cb, 10)
        
        # 用户命令订阅
        self.command_sub = self.create_subscription(
            String, "/arm_commands", self.command_cb, 10)
        
        # TF系统 - 使用相同的buffer和listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 视觉模式专用参数
        self.current_aruco_pose = None
        self.block_length = 0.05
        self.action_matrix = None
        self.valid_times = 0
        self.is_solved = False
        self.allow_execute_trigger = True
        self.shoulder_offset = -0.05
        
        # 整合的TF变换参数
        self.base_frame = 'px100/base_link'  # 机械臂基座坐标系
        self.camera_frame = 'camera_color_optical_frame'  # 摄像头坐标系
        self.last_transform_time = time.time()
        self.transform_interval = 0.1  # TF变换查询间隔（秒）
        
        # 逆运动学初始猜测
        self.initial_guesses = [[0.0] * self.num_joints] * 3
        self.initial_guesses[1][0] = np.deg2rad(-30)
        self.initial_guesses[2][0] = np.deg2rad(30)
    
    # ==================== 共用方法 ====================
    
    def joint_states_cb(self, msg):
        """关节状态回调函数"""
        self.joint_pos = msg.position
        if len(self.joint_pos) >= self.num_joints:
            self._has_joint_states = True
    
    def go_home(self):
        """回到初始位置"""
        command = JointGroupCommand()
        command.name = "arm"
        command.cmd = self.home_joint_positions
        self.arm_group_pub.publish(command)
        time.sleep(1.0)
        return True
    
    def go_handup(self):
        """抬起手臂"""
        command = JointGroupCommand()
        command.name = "arm"
        command.cmd = self.handup_joint_positions
        self.arm_group_pub.publish(command)
        time.sleep(1.0)
        return True
    
    def gripper(self, effort, duration=1.0):
        """控制夹爪"""
        command = JointSingleCommand()
        command.name = "gripper"
        command.cmd = effort
        self.arm_cmd_pub.publish(command)
        time.sleep(duration)
        return True
    
    def control_gripper(self, effort, state_name):
        """控制夹爪（原代码中的方法）"""
        command = JointSingleCommand()
        command.name = "gripper"
        command.cmd = effort
        self.arm_cmd_pub.publish(command)
        self.get_logger().info(f"控制夹爪: {state_name} (力度: {effort})")
        return True
    
    def send_joint_command(self, joint_angles):
        """发送关节角度命令"""
        if len(joint_angles) != self.num_joints:
            self.get_logger().error(f"关节角度数量错误: 期望{self.num_joints}, 得到{len(joint_angles)}")
            return False
        
        command = JointGroupCommand()
        command.name = "arm"
        command.cmd = joint_angles
        self.arm_group_pub.publish(command)
        return True
    
    def check_joint_error(self, target_joints, threshold=0.15):
        """检查是否到达目标关节位置"""
        if not self._has_joint_states:
            return False
        
        current_pos = self.joint_pos[:self.num_joints]
        error = np.abs(np.array(current_pos) - np.array(target_joints))
        return np.all(error < threshold)
    
    def control_with_kinematics(self, tx, ty, tz):
        """通过逆运动学控制"""
        tpos = SE3(tx, ty, tz)
        ik_sol = self.robot.ikine_LM(tpos, end=self.robot[11])
        if ik_sol.success:
            self.get_logger().info(f"逆解有效: {ik_sol.q}")
            return ik_sol.q.tolist()
        else:
            self.get_logger().error("逆解无效!")
            return None
    
    # ==================== 视觉模式专用方法 ====================
    
    def compute_T_0c(self):
        """计算机械臂基座标系到摄像头坐标系的齐次变换矩阵的逆矩阵"""
        try:
            trans = self.tf_buffer.lookup_transform(
                'px100/base_link', 
                'camera_color_optical_frame',
                rclpy.time.Time()
            )
            
            translation = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            
            rotation = [
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ]
            
            R = tf_transformations.quaternion_matrix(rotation)
            matrix = np.eye(4)
            matrix[:3, :3] = R[:3, :3]
            matrix[:3, 3] = translation
            
            return np.linalg.inv(matrix)
        
        except Exception as e:
            self.get_logger().error(f"获取TF变换失败: {e}")
            return np.eye(4)
    
    def compute_T_ca(self):
        """计算从摄像头坐标系到ArUco标记坐标系的齐次变换矩阵"""
        if self.current_aruco_pose is None:
            return np.eye(4)
        
        quat_ca = [
            self.current_aruco_pose.orientation.x,
            self.current_aruco_pose.orientation.y,
            self.current_aruco_pose.orientation.z,
            self.current_aruco_pose.orientation.w
        ]
        
        R_ca = tf_transformations.quaternion_matrix(quat_ca)
        t_ca = np.array([
            self.current_aruco_pose.position.x,
            self.current_aruco_pose.position.y,
            self.current_aruco_pose.position.z
        ])
        
        T_ca = np.eye(4)
        T_ca[:3, :3] = R_ca[:3, :3]
        T_ca[:3, 3] = t_ca
        
        # 修正变换：从表面中心到物体中心
        T_ca[0:3, 3] -= self.block_length / 2.0 * T_ca[0:3, 2]
        
        return T_ca
    
    def matrix_to_pose(self, matrix):
        """将4x4齐次变换矩阵转换为Pose消息"""
        pose = Pose()
        pose.position.x = matrix[0, 3]
        pose.position.y = matrix[1, 3]
        pose.position.z = matrix[2, 3]
        
        quaternion = tf_transformations.quaternion_from_matrix(matrix)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
        
        return pose
    
    def matrix_control(self, target_matrix, execute=True, custom_joints=None, delay=0.5):
        """通过逆运动学控制机械臂到达目标位姿"""
        # 提取目标位置和姿态
        target_pos = target_matrix[:3, 3]
        target_rot = target_matrix[:3, :3]
        
        # 创建SE3位姿
        target_pose = SE3.Rt(target_rot, target_pos)
        
        # 尝试逆运动学求解
        solution_found = False
        solution_valid = False
        solution_joints = None
        
        for guess in self.initial_guesses:
            ik_sol = self.robot.ikine_LM(target_pose, end=self.robot[11], q0=guess)
            
            if ik_sol.success:
                solution_joints = ik_sol.q.tolist()
                solution_found = True
                
                # 检查关节限制
                valid = True
                for i, (q, lower, upper) in enumerate(zip(
                    solution_joints, self.joint_lower_limits, self.joint_upper_limits
                )):
                    if q < lower or q > upper:
                        self.get_logger().warn(f"关节{i}角度{q:.3f}超出限制[{lower:.3f}, {upper:.3f}]")
                        valid = False
                        break
                
                if valid:
                    solution_valid = True
                    break
        
        if custom_joints is not None:
            solution_joints = custom_joints
            for i in range(self.num_joints):
                if solution_joints[i] is None and solution_valid:
                    solution_joints[i] = solution_joints[i]
        
        # 执行控制
        reached = False
        if execute and solution_joints is not None:
            self.send_joint_command(solution_joints)
            time.sleep(delay)
            
            # 检查是否到达
            if self.check_joint_error(solution_joints, threshold=0.1):
                reached = True
        
        return solution_joints, solution_found, solution_valid, reached
    
    # ==================== 整合的TF变换方法 ====================
    
    def get_target_position_in_arm_frame(self):
        """
        整合的方法：将ArUco检测到的目标点从摄像头坐标系转换到机械臂基座坐标系
        使用改进的TF变换方法
        """
        if self.current_aruco_pose is None:
            self.get_logger().info("等待ArUco位姿数据...")
            return None
        
        current_time = time.time()
        if current_time - self.last_transform_time < self.transform_interval:
            # 避免频繁查询TF变换
            return None
        
        self.last_transform_time = current_time
        
        try:
            # 查询从摄像头到机械臂基座的TF变换
            t = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time())
            
            # 提取位置和姿态信息
            camera_frame_pos_msg = t.transform.translation
            camera_frame_rot_msg = t.transform.rotation
            
            # 转换为numpy数组
            camera_frame_quat = np.array([
                camera_frame_rot_msg.x,
                camera_frame_rot_msg.y,
                camera_frame_rot_msg.z,
                camera_frame_rot_msg.w
            ])
            
            camera_frame_pos = np.array([
                camera_frame_pos_msg.x,
                camera_frame_pos_msg.y,
                camera_frame_pos_msg.z
            ])
            
            # 从四元数获取旋转矩阵
            R = scipyR.from_quat(camera_frame_quat).as_matrix()
            
            # 构建齐次变换矩阵
            T = np.eye(4)  # 初始化齐次变换矩阵
            T[0:3, 0:3] = R
            T[0:3, 3] = camera_frame_pos
            
            # 获取ArUco在摄像头坐标系中的位置
            aruco_pos_wrt_camera = np.array([
                self.current_aruco_pose.position.x,
                self.current_aruco_pose.position.y,
                self.current_aruco_pose.position.z
            ])
            
            # 转换为齐次坐标
            aruco_pos_wrt_camera_tilde = np.concatenate((aruco_pos_wrt_camera, [1]))
            
            # 计算在机械臂基座坐标系中的位置
            target_pos_wrt_arm_frame = T @ aruco_pos_wrt_camera_tilde
            
            # 从齐次坐标提取三维位置
            target_pos_3d = target_pos_wrt_arm_frame[:3]
            
            # 发布转换后的坐标用于可视化
            transformed_pose = PoseStamped()
            transformed_pose.header.stamp = self.get_clock().now().to_msg()
            transformed_pose.header.frame_id = self.base_frame
            transformed_pose.pose.position.x = target_pos_3d[0]
            transformed_pose.pose.position.y = target_pos_3d[1]
            transformed_pose.pose.position.z = target_pos_3d[2]
            transformed_pose.pose.orientation = self.current_aruco_pose.orientation
            
            self.transformed_pose_pub.publish(transformed_pose)
            
            self.get_logger().info(
                f"坐标变换结果 - 摄像头系: {aruco_pos_wrt_camera}, "
                f"机械臂系: {target_pos_3d}"
            )
            
            return target_pos_3d
            
        except TransformException as ex:
            self.get_logger().warn(f"TF变换查询失败: {ex}")
            return None
        except Exception as e:
            self.get_logger().error(f"坐标变换出错: {e}")
            return None
    
    def compute_integrated_transform_matrix(self):
        """
        使用整合的方法计算从机械臂基座到目标物体的完整变换矩阵
        """
        if self.current_aruco_pose is None:
            return np.eye(4)
        
        # 获取目标点在机械臂基座坐标系中的位置
        target_pos_in_base = self.get_target_position_in_arm_frame()
        
        if target_pos_in_base is None:
            return np.eye(4)
        
        # 创建完整的变换矩阵
        T_0a = np.eye(4)
        T_0a[0:3, 3] = target_pos_in_base
        
        # 使用ArUco的姿态信息（已经在基座坐标系中）
        quat_0a = [
            self.current_aruco_pose.orientation.x,
            self.current_aruco_pose.orientation.y,
            self.current_aruco_pose.orientation.z,
            self.current_aruco_pose.orientation.w
        ]
        
        R_0a = scipyR.from_quat(quat_0a).as_matrix()
        T_0a[0:3, 0:3] = R_0a
        
        # 修正变换：从表面中心到物体中心
        T_0a[0:3, 3] -= self.block_length / 2.0 * T_0a[0:3, 2]
        
        return T_0a
    
    def pan_tilt(self, pitch=0.0, yaw=0.0, speed=10):
        """控制云台"""
        cmd = PanTiltCmdDeg()
        cmd.pitch = pitch
        cmd.yaw = yaw
        cmd.speed = speed
        self.pan_tilt_pub.publish(cmd)
        time.sleep(0.5)
        return True
    
    def aruco_pose_cb(self, msg):
        """ArUco位姿回调函数"""
        self.current_aruco_pose = msg.pose
        
        # 实时显示转换结果（可选）
        if self.MODE == "VISION":
            transformed_pos = self.get_target_position_in_arm_frame()
            if transformed_pos is not None:
                self.get_logger().debug(f"实时转换: {transformed_pos}")
    
    def command_cb(self, msg):
        """用户命令回调"""
        if msg.data == 'start':
            self.control_state = 'DETECT'
            self.get_logger().info("收到开始命令")
        elif msg.data == 'stop':
            self.control_state = 'INIT'
            self.get_logger().info("收到停止命令")
    
    # ==================== 控制主循环 ====================
    
    def timer_cb(self):
        """主控制循环 - 根据模式选择不同的控制逻辑"""
        if self.MODE == "FIXED":
            self._fixed_mode_control()
        elif self.MODE == "VISION":
            self._vision_mode_control()
    
    def _fixed_mode_control(self):
        """固定位姿模式控制逻辑 - 完全不变"""
        current_time = time.time()
        elapsed = current_time - self.state_start_time
        
        # 根据控制状态执行相应操作
        if self.control_state == "INIT":
            # 注意：固定位姿模式通常不需要云台控制，这里注释掉
            # self.pan_tilt(pitch=15.0, yaw=0.0, speed=10)
            
            if self._has_joint_states:
                self.get_logger().info('\n收到关节状态，开始移动')
                self.get_logger().info(f'当前关节位置: {self.joint_pos[:4]}')
                self.control_state = "MOVE_TO_TARGET"
                self.state_start_time = current_time
                # 张开夹爪
                self.control_gripper(self.gripper_opened, "initial_opened")
        
        elif self.control_state == "MOVE_TO_TARGET":
            # 发送机械臂位置命令
            command = JointGroupCommand()
            command.name = "arm"
            command.cmd = self.target_joint_positions
            self.arm_group_pub.publish(command)
            
            # 检查是否到达目标位置
            if self._has_joint_states and len(self.joint_pos) >= 4:
                current_pos = self.joint_pos[:4]
                error = np.abs(np.array(current_pos) - np.array(self.target_joint_positions))
                
                if np.all(error < 0.15):
                    self.get_logger().info('\n到达目标位置!')
                    self.get_logger().info(f'当前关节: {[f"{p:.3f}" for p in current_pos]}')
                    self.get_logger().info(f'目标关节: {[f"{p:.3f}" for p in self.target_joint_positions]}')
                    self.get_logger().info('夹爪保持张开3秒...')
                    
                    self.control_state = "KEEP_OPEN"
                    self.state_start_time = current_time
        
        elif self.control_state == "KEEP_OPEN":
            # 保持夹爪张开3秒
            keep_duration = 3.0
            
            if elapsed < keep_duration:
                # 持续保持夹爪张开
                self.control_gripper(self.gripper_opened, "keeping_open")
                
                if int(elapsed) != int(elapsed - 0.1):
                    remaining = keep_duration - elapsed
                    self.get_logger().info(f'夹爪张开保持中... 剩余时间: {remaining:.1f}秒')
            else:
                self.get_logger().info('夹爪张开保持完成，开始闭合夹爪')
                self.control_state = "CLOSE_GRIPPER"
                self.state_start_time = current_time
        
        elif self.control_state == "CLOSE_GRIPPER":
            # 缓慢闭合夹爪
            close_duration = 2.0  # 2秒缓慢闭合
            
            if elapsed < close_duration:
                # 分阶段闭合
                progress = elapsed / close_duration
                close_effort = self.gripper_opened - (self.gripper_opened - self.gripper_closed) * progress
                self.control_gripper(close_effort, "closing")
                
                if int(elapsed * 2) != int((elapsed - 0.1) * 2):
                    self.get_logger().info(f'缓慢闭合夹爪... 进度: {progress*100:.0f}%')
            else:
                # 完全闭合并保持
                self.control_gripper(self.gripper_closed, "closed")
                self.get_logger().info('夹爪闭合完成，保持夹持状态')
                self.control_state = "HOLD_GRASP"
                self.state_start_time = current_time
        
        elif self.control_state == "HOLD_GRASP":
            # 持续保持夹持状态，不松开
            # 每5秒发送一次保持命令，防止掉电
            if int(elapsed) % 5 == 0 and int(elapsed) != int(elapsed - 0.1):
                self.control_gripper(self.gripper_closed, "holding")
                self.get_logger().info(f'保持夹持状态中... 已持续: {elapsed:.1f}秒')
        
        elif self.control_state == "LIFT_UP":
            # 抬起手臂
            self.go_handup()
            self.get_logger().info("手臂抬起")
            self.control_state = "GO_HOME"
            time.sleep(1.0)
        
        elif self.control_state == "GO_HOME":
            # 回到初始位置
            self.go_home()
            self.get_logger().info("回到初始位置")
            self.control_state = "RELEASE"
            time.sleep(1.0)
        
        elif self.control_state == "RELEASE":
            # 释放物体
            self.control_gripper(self.gripper_opened, "releasing")
            self.get_logger().info("夹爪张开，物体释放")
            self.control_state = "WAIT"
            self.state_start_time = current_time
        
        elif self.control_state == "WAIT":
            if elapsed > 3.0:  # 等待3秒
                self.get_logger().info("等待结束，重新开始")
                self.control_state = "INIT"
    
    def _vision_mode_control(self):
        """视觉引导模式控制逻辑 - 使用整合的TF变换方法"""
        current_time = time.time()
        
        if len(self.joint_pos) >= self.num_joints:
            if self.control_state == 'INIT':
                self.get_logger().info('初始化云台和机械臂...')
                self.pan_tilt(pitch=15.0, yaw=0.0, speed=10)
                
                if self.go_home() and self.gripper(self.gripper_opened, 0.5):
                    self.get_logger().info('准备就绪.')
                    self.control_state = 'DETECT'
                    
                    # 重置控制变量
                    self.valid_times = 0
                    self.is_solved = False
                    self.allow_execute_trigger = True
                    
                    time.sleep(1.0)
            
            elif self.control_state == 'DETECT':
                if self.current_aruco_pose:
                    self.get_logger().info('检测到ArUco标记，进行坐标转换...')
                    
                    # 使用整合的方法计算变换矩阵
                    self.action_matrix = self.compute_integrated_transform_matrix()
                    
                    if np.allclose(self.action_matrix, np.eye(4)):
                        self.get_logger().warn('变换矩阵计算失败，等待有效数据...')
                        return
                    
                    # 微调位置（保持原有逻辑）
                    dx, dy = self.action_matrix[0, 3], self.action_matrix[1, 3]
                    d = math.sqrt(dx * dx + dy * dy)
                    if d > 0.001:
                        self.action_matrix[0, 3] -= dx * 0.02 / d
                        self.action_matrix[1, 3] -= dy * 0.02 / d
                    
                    self.action_matrix[2, 3] += 0.005
                    
                    # 调整末端方向
                    if d > 0.001:
                        nx, ny = dx / d, dy / d
                        self.action_matrix[:3, :3] = np.array([
                            [nx, -ny, 0],
                            [ny, nx, 0],
                            [0, 0, 1]
                        ])
                    
                    # 测试逆解
                    _, solution_found, solution_valid, _ = self.matrix_control(
                        self.action_matrix, execute=False
                    )
                    
                    self.get_logger().info(f'逆解找到: {solution_found}; 逆解有效: {solution_valid}.')
                    self.is_solved = solution_valid
                    
                    # 如果解有效且允许执行，进入下一状态
                    if self.is_solved and self.allow_execute_trigger:
                        self.valid_times += 1
                        if self.valid_times > 3:  # 连续3次检测有效
                            self.valid_times = 0
                            self.allow_execute_trigger = False
                            self.control_state = 'TWIST_WAIST'
                else:
                    self.get_logger().info('等待检测ArUco标记...', throttle_duration_sec=2.0)
            
            elif self.control_state == 'TWIST_WAIST':
                self.get_logger().info('调整腰部关节...')
                self.gripper(self.gripper_opened, 1.0)
                
                _, solution_found, solution_valid, reached = self.matrix_control(
                    self.action_matrix,
                    custom_joints=[None, self.shoulder_offset, -1.4, 0.0],
                    delay=0.5
                )
                
                if reached:
                    self.get_logger().info('调整完成.')
                    self.control_state = 'GRASP'
            
            elif self.control_state == 'GRASP':
                self.get_logger().info('执行抓取...')
                _, solution_found, solution_valid, reached = self.matrix_control(
                    self.action_matrix,
                    delay=1.0
                )
                
                if reached:
                    self.get_logger().info('抓取到位.')
                    self.gripper(self.gripper_closed, 1.0)
                    self.control_state = 'HAND_UP'
            
            elif self.control_state == 'HAND_UP':
                self.get_logger().info('抬起手臂...')
                if self.go_handup():
                    self.get_logger().info('抬起完成.')
                    self.control_state = 'HOLD'
            
            elif self.control_state == 'HOLD':
                self.get_logger().info('保持位置...')
                if self.go_home():
                    self.get_logger().info('回到初始位置.')
                    time.sleep(2.0)
                    self.control_state = 'INIT'
            
            elif self.control_state == 'RELEASE':
                self.get_logger().info('释放物体...')
                self.go_home()
                self.gripper(self.gripper_opened, 1.0)
                self.get_logger().info('释放完成.')
                self.control_state = 'INIT'
            
            else:
                self.get_logger().warn('未知状态.')
                self.control_state = 'INIT'


def main():
    rclpy.init(args=None)
    
    try:
        controller = ArmController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("\n程序被中断")
    except Exception as e:
        print(f"程序出错: {e}")
    finally:
        controller.destroy_node()
        rclpy.shutdown()
        print("程序结束")


if __name__ == '__main__':
    main()