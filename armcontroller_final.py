import roboticstoolbox as rtb
import rclpy
from rclpy.node import Node
from interbotix_xs_msgs.msg import JointSingleCommand, JointGroupCommand
from sensor_msgs.msg import JointState
import numpy as np
import time
import math
from spatialmath import SE3
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from pan_tilt_msgs.msg import PanTiltCmdDeg
import tf_transformations
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import transforms3d

class ArmController(Node):
    def __init__(self):
        super().__init__("ArmController")
        
        # ==================== 配置参数 ====================
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
        self.target_joint_positions = [0.0, 0.7, -0.7, 0.3]
        self.handup_joint_positions = [0.8, 0.0, 0.0, 0.5]
        self.home_joint_positions = [0.8, 0.3, 0.5, 0.3]
        
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
        
        # 视觉模式专用变量
        if self.MODE == "VISION":
            self._init_vision_components()
        
        # 创建控制定时器
        self.control_timer = self.create_timer(0.1, self.timer_cb)
        
        self.get_logger().info(f"控制器初始化完成 - 模式: {self.MODE}")
    
    def _init_vision_components(self):
        """初始化视觉模式专用组件"""
        # TF监听器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # ArUco订阅器
        self.aruco_pose_sub = self.create_subscription(
            PoseArray, "/aruco_poses", self.aruco_poses_cb, 10)
        
        # 用户命令订阅
        self.command_sub = self.create_subscription(
            String, "/arm_commands", self.command_cb, 10)
        
        # 视觉模式专用参数
        self.current_aruco_pose = None
        self.aruco_position_camera = None  # 相机坐标系下的ArUco位置
        self.aruco_position_base = None    # 机械臂基座坐标系下的ArUco位置
        self.block_length = 0.05
        self.valid_times = 0
        self.is_solved = False
        self.allow_execute_trigger = True
        
        # 逆运动学初始猜测
        self.initial_guesses = [[0.0] * self.num_joints] * 3
        self.initial_guesses[1][0] = np.deg2rad(-30)
        self.initial_guesses[2][0] = np.deg2rad(30)
        
        # TF变换关系（根据你的标定结果）
        self.camera_to_ee_translation = np.array([0.19, -0.11, -0.278])  # 平移
        self.camera_to_ee_rotation = np.array([0.0, 0.0, 0.0, 1.0])      # 旋转（四元数）
        
        # 创建TF变换矩阵
        self.T_camera_to_ee = self._create_transform_matrix(
            self.camera_to_ee_translation, self.camera_to_ee_rotation)
        
        # 标志位：是否启用FIXED模式作为后备
        self.use_fixed_mode_backup = True
        
        self.get_logger().info("视觉组件初始化完成，等待TF变换...")
    
    def _create_transform_matrix(self, translation, rotation_quat):
        """创建4x4变换矩阵"""
        # 四元数转旋转矩阵
        R = tf_transformations.quaternion_matrix(rotation_quat)
        
        # 创建变换矩阵
        T = np.eye(4)
        T[:3, :3] = R[:3, :3]
        T[:3, 3] = translation
        
        return T
    
    def transform_pose_camera_to_base(self, pose_camera):
        """
        将位姿从相机坐标系变换到机械臂基座坐标系
        使用: pose_base = T_base_to_camera * pose_camera
        但我们有: T_camera_to_ee 和 T_ee_to_base
        所以: T_camera_to_base = T_ee_to_base * T_camera_to_ee
        """
        try:
            # 1. 获取相机到机械臂末端的变换（已标定）
            T_camera_to_ee = self.T_camera_to_ee
            
            # 2. 获取机械臂末端到基座的变换（通过正向运动学）
            if len(self.joint_pos) >= self.num_joints:
                # 获取当前关节位置
                current_joints = self.joint_pos[:self.num_joints]
                
                # 计算正向运动学得到末端位姿
                T_ee = self.robot.fkine(current_joints, end=self.robot[11])
                T_ee_to_base = T_ee.A  # 转换为numpy数组
                
                # 3. 组合变换：T_camera_to_base = T_ee_to_base * T_camera_to_ee
                T_camera_to_base = np.dot(T_ee_to_base, T_camera_to_ee)
                
                # 4. 将ArUco位姿从相机坐标系变换到基座坐标系
                # 创建ArUco在相机坐标系中的齐次坐标
                aruco_pos_camera = np.array([
                    pose_camera.position.x,
                    pose_camera.position.y,
                    pose_camera.position.z,
                    1.0
                ])
                
                # 变换到基座坐标系
                aruco_pos_base_homogeneous = np.dot(T_camera_to_base, aruco_pos_camera)
                aruco_pos_base = aruco_pos_base_homogeneous[:3]
                
                self.get_logger().info(f"ArUco位置变换: 相机系={[pose_camera.position.x, pose_camera.position.y, pose_camera.position.z]} -> 基座系={aruco_pos_base}")
                return aruco_pos_base
            else:
                self.get_logger().warn("没有关节状态，无法计算正向运动学")
                return None
                
        except Exception as e:
            self.get_logger().error(f"坐标变换失败: {e}")
            return None
    
    def get_tf_transform(self, target_frame, source_frame):
        """获取TF变换"""
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, now)
            return transform
        except Exception as e:
            self.get_logger().warn(f"无法获取TF变换 {source_frame} -> {target_frame}: {e}")
            return None
    
    def transform_using_tf(self, pose_camera):
        """使用TF系统进行坐标变换"""
        try:
            # 尝试通过TF系统获取变换
            transform = self.get_tf_transform(
                "px100/base_link", "camera_color_optical_frame")
            
            if transform:
                # 提取变换信息
                trans = transform.transform.translation
                rot = transform.transform.rotation
                
                # 创建变换矩阵 T_camera_to_base
                T = tf_transformations.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
                T[:3, 3] = [trans.x, trans.y, trans.z]
                
                # 变换ArUco位置
                aruco_pos_camera = np.array([
                    pose_camera.position.x,
                    pose_camera.position.y,
                    pose_camera.position.z,
                    1.0
                ])
                
                aruco_pos_base_homogeneous = np.dot(T, aruco_pos_camera)
                aruco_pos_base = aruco_pos_base_homogeneous[:3]
                
                self.get_logger().info(f"TF变换: 相机系 -> 基座系 = {aruco_pos_base}")
                return aruco_pos_base
            else:
                self.get_logger().warn("TF变换不可用，使用标定参数")
                return self.transform_pose_camera_to_base(pose_camera)
                
        except Exception as e:
            self.get_logger().error(f"TF变换失败: {e}")
            return None
    
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
    
    def calculate_ik(self, tx, ty, tz):
        """使用逆运动学计算目标关节位置"""
        try:
            tpos = SE3(tx, ty, tz)
            
            # robot[11] is the index of ee_gripper_link in the urdf
            ik_sol = self.robot.ikine_LM(tpos, end=self.robot[11])
            
            if ik_sol.success:
                self.get_logger().info(f"逆运动学求解成功!")
                self.get_logger().info(f"关节角度解: {ik_sol.q}")
                
                # 验证正运动学
                fk_val = self.robot.fkine(ik_sol.q, end=self.robot[11])
                self.get_logger().info(f"计算得到的末端位置: {fk_val.t}")
                self.get_logger().info(f"目标末端位置: [{tx}, {ty}, {tz}]")
                
                return ik_sol.q.tolist()
            else:
                self.get_logger().error("逆运动学求解失败!")
                return None
                
        except Exception as e:
            self.get_logger().error(f"逆运动学计算时发生错误: {e}")
            return None
    
    # ==================== 视觉模式专用方法 ====================
    
    def aruco_poses_cb(self, msg):
        """ArUco位姿数组回调函数"""
        if len(msg.poses) > 0:
            # 取第一个检测到的ArUco标记
            pose_camera = msg.poses[0]
            self.current_aruco_pose = pose_camera
            
            # 提取相机坐标系下的位置
            tx_cam = pose_camera.position.x
            ty_cam = pose_camera.position.y
            tz_cam = pose_camera.position.z
            
            self.aruco_position_camera = [tx_cam, ty_cam, tz_cam]
            
            self.get_logger().info(
                f"检测到ArUco标记，相机坐标系位置: "
                f"tx={tx_cam:.3f}, ty={ty_cam:.3f}, tz={tz_cam:.3f}",
                throttle_duration_sec=1.0
            )
            
            # 尝试进行坐标变换
            try:
                # 方法1：使用TF系统
                aruco_base_tf = self.transform_using_tf(pose_camera)
                
                # 方法2：使用标定参数
                if aruco_base_tf is None:
                    aruco_base_tf = self.transform_pose_camera_to_base(pose_camera)
                
                if aruco_base_tf is not None:
                    self.aruco_position_base = aruco_base_tf
                    self.get_logger().info(
                        f"坐标变换完成: 基座坐标系位置: "
                        f"tx={aruco_base_tf[0]:.3f}, ty={aruco_base_tf[1]:.3f}, tz={aruco_base_tf[2]:.3f}"
                    )
                else:
                    self.get_logger().warn("坐标变换失败，等待TF关系建立")
                    self.aruco_position_base = None
                    
            except Exception as e:
                self.get_logger().error(f"坐标变换过程中出错: {e}")
                self.aruco_position_base = None
        else:
            self.current_aruco_pose = None
            self.aruco_position_camera = None
            self.aruco_position_base = None

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
        """固定位姿模式控制逻辑 - 保持不变"""
        current_time = time.time()
        elapsed = current_time - self.state_start_time
        
        # 根据控制状态执行相应操作
        if self.control_state == "INIT":
            if self._has_joint_states:
                self.get_logger().info('\n收到关节状态，开始移动')
                self.get_logger().info(f'当前关节位置: {self.joint_pos[:4]}')
                self.control_state = "LIFT_UP"
                self.state_start_time = current_time
                # 张开夹爪
                self.control_gripper(self.gripper_opened, "initial_opened")

        elif self.control_state == "LIFT_UP":
            # 发送机械臂位置命令
            command = JointGroupCommand()
            command.name = "arm"
            command.cmd = self.handup_joint_positions
            self.arm_group_pub.publish(command)

            if self._has_joint_states and len(self.joint_pos) >= 4:
                current_pos = self.joint_pos[:4]
                error = np.abs(np.array(current_pos) - np.array(self.handup_joint_positions))

                if np.all(error < 0.15):
                    self.get_logger().info('\n到达目标位置!')
                    self.get_logger().info(f'当前关节: {[f"{p:.3f}" for p in current_pos]}')
                    self.get_logger().info(f'目标关节: {[f"{p:.3f}" for p in self.handup_joint_positions]}')

                    self.control_state = "MOVE_TO_TARGET"
                    self.state_start_time = current_time
            
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
        
        elif self.control_state == "LIFT_UP_2":
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
        """视觉引导模式控制逻辑 - 使用TF变换后的坐标"""
        current_time = time.time()
        
        if len(self.joint_pos) >= self.num_joints:
            if self.control_state == 'INIT':
                self.get_logger().info('视觉模式初始化...')
                
                # 回到初始位置
                if self.go_home():
                    self.get_logger().info('回到初始位置完成')
                
                # 张开夹爪
                if self.gripper(self.gripper_opened, 0.5):
                    self.get_logger().info('夹爪张开完成')
                
                # 重置控制变量
                self.aruco_position_camera = None
                self.aruco_position_base = None
                self.valid_times = 0
                self.is_solved = False
                self.allow_execute_trigger = True
                
                time.sleep(1.0)
                self.control_state = 'DETECT'
            
            elif self.control_state == 'DETECT':
                # 检查是否有变换后的ArUco位置
                if self.aruco_position_base is not None:
                    self.get_logger().info('检测到ArUco标记位置（已变换到基座坐标系），开始逆运动学计算...')
                    
                    # 使用变换后的位置作为目标位置
                    tx, ty, tz = self.aruco_position_base
                    
                    # 添加抓取偏移（如果需要）
                    grasp_offset = 0.02  # 2cm
                    tx_offset = tx - grasp_offset if tx > 0 else tx + grasp_offset
                    
                    self.get_logger().info(f"原始目标: [{tx:.3f}, {ty:.3f}, {tz:.3f}]")
                    self.get_logger().info(f"带偏移目标: [{tx_offset:.3f}, {ty:.3f}, {tz:.3f}]")
                    
                    # 计算逆运动学
                    target_joints = self.calculate_ik(tx_offset, ty, tz)
                    
                    if target_joints is not None:
                        self.get_logger().info('逆运动学计算成功，准备移动')
                        self.target_joint_positions = target_joints
                        self.control_state = "MOVE_TO_TARGET"
                        self.state_start_time = current_time
                        
                        # 张开夹爪
                        self.control_gripper(self.gripper_opened, "initial_opened")
                        # 重置ArUco位置，避免重复使用
                        self.aruco_position_base = None
                    else:
                        self.get_logger().warn('逆运动学计算失败，切换到FIXED模式控制流程')
                        
                        if self.use_fixed_mode_backup:
                            # 切换到FIXED模式的控制流程
                            self.get_logger().info('开始执行FIXED模式控制流程...')
                            
                            # 重置状态，准备开始FIXED模式流程
                            self.control_state = "INIT_FIXED_BACKUP"
                            self.state_start_time = current_time
                            
                            # 重置ArUco位置，避免重复使用
                            self.aruco_position_base = None
                        else:
                            self.get_logger().error('逆运动学计算失败，等待新的ArUco检测...')
                            self.aruco_position_base = None
                elif self.aruco_position_camera is not None:
                    self.get_logger().info('检测到ArUco，等待坐标变换完成...')
                else:
                    self.get_logger().info('等待检测ArUco标记...', throttle_duration_sec=2.0)
            
            elif self.control_state == "INIT_FIXED_BACKUP":
                # FIXED模式备份流程 - 初始化
                self.get_logger().info('FIXED模式备份流程初始化...')
                
                # 确保在初始位置
                if self.go_home():
                    self.get_logger().info('回到初始位置完成')
                
                # 张开夹爪
                if self.gripper(self.gripper_opened, 0.5):
                    self.get_logger().info('夹爪张开完成')
                
                time.sleep(1.0)
                self.control_state = "LIFT_UP_FIXED"
                self.state_start_time = current_time

            elif self.control_state == "LIFT_UP_FIXED":
                # FIXED模式备份流程 - 移动到目标位置
                # 发送机械臂位置命令（使用FIXED模式的目标位置）
                command = JointGroupCommand()
                command.name = "arm"
                command.cmd = self.handup_joint_positions  # 使用FIXED模式的目标位置
                self.arm_group_pub.publish(command)
                
                # 检查是否到达目标位置
                if self._has_joint_states and len(self.joint_pos) >= 4:
                    current_pos = self.joint_pos[:4]
                    error = np.abs(np.array(current_pos) - np.array(self.handup_joint_positions))
                    
                    if np.all(error < 0.15):
                        self.get_logger().info('\n到达FIXED模式目标位置!')
                        self.get_logger().info(f'当前关节: {[f"{p:.3f}" for p in current_pos]}')
                        self.get_logger().info(f'目标关节: {[f"{p:.3f}" for p in self.handup_joint_positions]}')
                        
                        self.control_state = "MOVE_TO_TARGET_FIXED"
                        self.state_start_time = current_time

            elif self.control_state == "MOVE_TO_TARGET_FIXED":
                # FIXED模式备份流程 - 移动到目标位置
                # 发送机械臂位置命令（使用FIXED模式的目标位置）
                command = JointGroupCommand()
                command.name = "arm"
                command.cmd = self.target_joint_positions  # 使用FIXED模式的目标位置
                self.arm_group_pub.publish(command)
                
                # 检查是否到达目标位置
                if self._has_joint_states and len(self.joint_pos) >= 4:
                    current_pos = self.joint_pos[:4]
                    error = np.abs(np.array(current_pos) - np.array(self.target_joint_positions))
                    
                    if np.all(error < 0.15):
                        self.get_logger().info('\n到达FIXED模式目标位置!')
                        self.get_logger().info(f'当前关节: {[f"{p:.3f}" for p in current_pos]}')
                        self.get_logger().info(f'目标关节: {[f"{p:.3f}" for p in self.target_joint_positions]}')
                        self.get_logger().info('夹爪保持张开3秒...')
                        
                        self.control_state = "KEEP_OPEN_FIXED"
                        self.state_start_time = current_time
            
            elif self.control_state == "KEEP_OPEN_FIXED":
                # FIXED模式备份流程 - 保持夹爪张开3秒
                elapsed = current_time - self.state_start_time
                keep_duration = 3.0
                
                if elapsed < keep_duration:
                    # 持续保持夹爪张开
                    self.control_gripper(self.gripper_opened, "keeping_open")
                    
                    if int(elapsed) != int(elapsed - 0.1):
                        remaining = keep_duration - elapsed
                        self.get_logger().info(f'夹爪张开保持中... 剩余时间: {remaining:.1f}秒')
                else:
                    self.get_logger().info('夹爪张开保持完成，开始闭合夹爪')
                    self.control_state = "CLOSE_GRIPPER_FIXED"
                    self.state_start_time = current_time
            
            elif self.control_state == "CLOSE_GRIPPER_FIXED":
                # FIXED模式备份流程 - 缓慢闭合夹爪
                elapsed = current_time - self.state_start_time
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
                    self.control_state = "HOLD_GRASP_FIXED"
                    self.state_start_time = current_time
            elif self.control_state == "HOLD_GRASP_FIXED":
                # FIXED模式备份流程 - 持续保持夹持状态
                elapsed = current_time - self.state_start_time
                
                # 保持5秒后抬起手臂
                if elapsed > 5.0:
                    self.get_logger().info('夹持完成，抬起手臂')
                    self.control_state = "LIFT_UP_FIXED_2"
                    self.state_start_time = current_time
                
                # 每5秒发送一次保持命令，防止掉电
                if int(elapsed) % 5 == 0 and int(elapsed) != int(elapsed - 0.1):
                    self.control_gripper(self.gripper_closed, "holding")
                    self.get_logger().info(f'保持夹持状态中... 已持续: {elapsed:.1f}秒')
            
            elif self.control_state == "LIFT_UP_FIXED_2":
                # FIXED模式备份流程 - 抬起手臂
                self.go_handup()
                self.get_logger().info("手臂抬起")
                self.control_state = "GO_HOME_FIXED"
                time.sleep(1.0)
            
            elif self.control_state == "GO_HOME_FIXED":
                # FIXED模式备份流程 - 回到初始位置
                self.go_home()
                self.get_logger().info("回到初始位置")
                self.control_state = "RELEASE_FIXED"
                time.sleep(1.0)
            
            elif self.control_state == "RELEASE_FIXED":
                # 每5秒发送一次保持命令，防止掉电
                if int(elapsed) % 5 == 0 and int(elapsed) != int(elapsed - 0.1):
                    self.control_gripper(self.gripper_closed, "holding")
                    self.get_logger().info(f'保持夹持状态中... 已持续: {elapsed:.1f}秒')
                # FIXED模式备份流程 - 释放物体
                # self.control_gripper(self.gripper_opened, "releasing")
                # self.get_logger().info("夹爪张开，物体释放")
                # self.control_state = "WAIT_FIXED"
                # self.state_start_time = current_time
            
            elif self.control_state == "WAIT_FIXED":
                elapsed = current_time - self.state_start_time
                if elapsed > 3.0:  # 等待3秒
                    self.get_logger().info("FIXED模式备份流程完成，重新开始视觉检测")
                    self.control_state = "INIT"  # 回到INIT状态，重新开始
            
            # 以下状态使用代码①的控制逻辑（视觉模式成功时的流程）
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
                elapsed = current_time - self.state_start_time
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
                elapsed = current_time - self.state_start_time
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
                elapsed = current_time - self.state_start_time
                
                # 保持5秒后抬起手臂
                if elapsed > 5.0:
                    self.get_logger().info('夹持完成，抬起手臂')
                    self.control_state = "LIFT_UP"
                    self.state_start_time = current_time
                
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
                elapsed = current_time - self.state_start_time
                if elapsed > 3.0:  # 等待3秒
                    self.get_logger().info("等待结束，重新开始检测")
                    self.control_state = "INIT"
            
            else:
                self.get_logger().warn('未知状态，重置为INIT')
                self.control_state = "INIT"


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