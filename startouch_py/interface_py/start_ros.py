import time
import rclpy
import threading
from rclpy.node import Node
from sensor_msgs.msg import JointState
from startouchclass import SingleArm  # 导入你的主臂控制类

class MainArmJointPublisher(Node):
    def __init__(self):
        super().__init__('main_arm_joint_publisher')

        # 1. 定义关节名称
        self.joint_names = ["joint1", "joint2", "joint3","joint4","joint5","joint6","gripper"]

        # 2. 初始化主臂控制器
        self.arm_controller_right = SingleArm(can_interface_="can0", enable_fd_=False)
        self.arm_follower_right = SingleArm(can_interface_="can1", enable_fd_=False)
        self.arm_controller_left = SingleArm(can_interface_="can2", enable_fd_=False)
        self.arm_follower_left = SingleArm(can_interface_="can3", enable_fd_=False)
        self.get_logger().info("4臂控制器初始化完成")

        # 3. 第一次获取机械臂状态，确保连接正常
        self.positions_controller_right = self.arm_controller_right.get_joint_positions()
        self.positions_follower_right = self.arm_follower_right.get_joint_positions()
        self.positions_controller_left = self.arm_controller_left.get_joint_positions()
        self.positions_follower_left = self.arm_follower_left.get_joint_positions()
        self.get_logger().info("4臂初始状态获取完成")

        # 4. 创建关节状态发布者
        self.joint_pub_controller_right = self.create_publisher(JointState, '/right_arm/joint_states_target', 2)
        self.joint_pub_follower_right = self.create_publisher(JointState, '/right_arm/joint_states_now', 2)
        self.joint_pub_controller_left = self.create_publisher(JointState, '/left_arm/joint_states_target', 2)
        self.joint_pub_follower_left = self.create_publisher(JointState, '/left_arm/joint_states_now', 2)
        self.get_logger().info("4话题初始化完成")

        # 5. 设置发布频率
        self.timer = self.create_timer(0.01, self.publish_joint_states)
        self.get_logger().info("定时器初始化完成，开始发布关节状态")

        # 6. 初始化并启动重力补偿线程
        self.right_thread_running = True    # 线程运行标志（用于控制线程退出）
        self.right_gravity_control_thread = threading.Thread(
            target=self.right_thread,       # 线程执行的函数
            daemon=True)                    # 设为守护线程：主程序退出时线程自动关闭
        self.right_gravity_control_thread.start()   # 启动线程
        self.get_logger().info("右双臂独立线程已启动")

        self.left_thread_running = True     # 线程运行标志（用于控制线程退出）
        self.left_gravity_control_thread = threading.Thread(
            target=self.left_thread,        # 线程执行的函数
            daemon=True)                    # 设为守护线程：主程序退出时线程自动关闭
        self.left_gravity_control_thread.start()    # 启动线程
        self.get_logger().info("左双臂独立线程已启动")

        self.last_freq_time = time.time()

    def publish_joint_states(self):
        try:
            # 1. 计算并打印发布频率
            time_bias = time.time() - self.last_freq_time
            freq = 1.0 / time_bias if time_bias > 0 else float('inf')
            self.get_logger().info(f"frequency: {freq:.2f} Hz")
            self.last_freq_time = time.time()

            # 3. 检查关节数量是否匹配（避免发布错误数据）
            if len(self.positions_controller_right) != len(self.joint_names):
                self.get_logger().warn(f"关节数量不匹配：读取到{len(self.positions_controller_right)}个关节，预期{len(self.joint_names)}个")
                return
            if len(self.positions_follower_right) != len(self.joint_names):
                self.get_logger().warn(f"关节数量不匹配：读取到{len(self.positions_follower_right)}个关节，预期{len(self.joint_names)}个")
                return
            if len(self.positions_controller_left) != len(self.joint_names):
                self.get_logger().warn(f"关节数量不匹配：读取到{len(self.positions_controller_left)}个关节，预期{len(self.joint_names)}个")
                return
            if len(self.positions_follower_left) != len(self.joint_names):
                self.get_logger().warn(f"关节数量不匹配：读取到{len(self.positions_follower_left)}个关节，预期{len(self.joint_names)}个")
                return
            
            # 4. 构建JointState消息
            joint_msg_controller_right = JointState()
            joint_msg_controller_right.header.stamp = self.get_clock().now().to_msg()  # 时间戳
            joint_msg_controller_right.name = self.joint_names  # 关节名称列表
            joint_msg_controller_right.position = self.positions_controller_right  # 关节位置（弧度或角度，取决于硬件）

            joint_msg_follower_right = JointState()
            joint_msg_follower_right.header.stamp = self.get_clock().now().to_msg()  # 时间戳
            joint_msg_follower_right.name = self.joint_names  # 关节名称列表
            joint_msg_follower_right.position = self.positions_follower_right  # 关节位置（弧度或角度，取决于硬件）

            joint_msg_controller_left = JointState()
            joint_msg_controller_left.header.stamp = self.get_clock().now().to_msg()  # 时间戳
            joint_msg_controller_left.name = self.joint_names  # 关节名称列表
            joint_msg_controller_left.position = self.positions_controller_left  # 关节位置（弧度或角度，取决于硬件）

            joint_msg_follower_left = JointState()
            joint_msg_follower_left.header.stamp = self.get_clock().now().to_msg()  # 时间戳
            joint_msg_follower_left.name = self.joint_names  # 关节名称列表
            joint_msg_follower_left.position = self.positions_follower_left  # 关节位置（弧度或角度，取决于硬件）
            
            # 5. 发布消息
            self.joint_pub_controller_right.publish(joint_msg_controller_right)
            self.joint_pub_follower_right.publish(joint_msg_follower_right)
            self.joint_pub_controller_left.publish(joint_msg_controller_left)
            self.joint_pub_follower_left.publish(joint_msg_follower_left)
            
        except Exception as e:
            self.get_logger().error(f"发布关节状态失败: {str(e)}")

    def right_thread(self):
        """右臂独立线程函数：包含单独的 while 循环"""
        # 线程循环：只要 running 标志为 True，就持续执行
        while self.right_thread_running:
            try:
                self.positions_controller_right = self.arm_controller_right.get_joint_positions()
                self.positions_follower_right = self.arm_follower_right.get_joint_positions()
                self.arm_controller_right.gravity_compensation()
                self.arm_follower_right.set_joint_raw(positions = self.positions_controller_right, velocities = [0, 0, 0, 0, 0, 0])

            except Exception as e:
                self.get_logger().error(f"右臂独立线程执行失败: {str(e)}")
                time.sleep(0.5)  # 避免过快循环导致日志刷屏
    
    def left_thread(self):
        """左臂独立线程函数：包含单独的 while 循环"""
        # 线程循环：只要 running 标志为 True，就持续执行
        while self.left_thread_running:
            try:
                self.positions_controller_left = self.arm_controller_left.get_joint_positions()
                self.positions_follower_left = self.arm_follower_left.get_joint_positions()
                self.arm_controller_left.gravity_compensation()
                self.arm_follower_left.set_joint_raw(positions = self.positions_controller_left, velocities = [0, 0, 0, 0, 0, 0])

            except Exception as e:
                self.get_logger().error(f"左臂独立线程执行失败: {str(e)}")
                time.sleep(0.5)  # 避免过快循环导致日志刷屏

def main(args=None):
    # 初始化ROS 2
    rclpy.init(args=args)
    
    # 创建并运行节点
    node = MainArmJointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("用户中断，退出程序")
    finally:
        # 清理资源
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()