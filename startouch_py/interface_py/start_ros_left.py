import time
import rclpy
import threading
from rclpy.node import Node
from sensor_msgs.msg import JointState
from startouchclass import SingleArm  # 导入你的主臂控制类

class MainArmJointPublisher(Node):
    def __init__(self):
        super().__init__('main_arm_joint_publisher_left')

        # 1. 定义关节名称
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

        # 2. 初始化主臂控制器
        self.arm_controller = SingleArm(can_interface_="can2", enable_fd_=False)
        self.arm_follower = SingleArm(can_interface_="can3", enable_fd_=False)
        print("左臂控制器初始化完成")

        # 3. 第一次获取机械臂状态，确保连接正常
        self.positions_controller = self.arm_controller.get_joint_positions()
        self.positions_follower = self.arm_follower.get_joint_positions()
        print("左臂初始状态获取完成")

        # 4. 创建关节状态发布者
        self.joint_pub_controller = self.create_publisher(JointState, '/left_arm/joint_states_target', 2)
        self.joint_pub_follower = self.create_publisher(JointState, '/left_arm/joint_states_now', 2)
        print("话题初始化完成")

        # 5. 初始化并启动重力补偿线程
        self.gravity_thread = threading.Thread(target=self.gravity, daemon=True)
        self.gravity_thread.start() # 启动线程
        print("重力补偿独立线程已启动")

        self.control_thread = threading.Thread(target=self.control, daemon=True)
        self.control_thread.start() # 启动线程
        print("控制独立线程已启动")

        self.publish_thread = threading.Thread(target=self.publish_joint_states, daemon=True)
        self.publish_thread.start() # 启动线程
        print("发布关节线程已启动")

    def publish_joint_states(self):
        try:
            last_freq_time = time.time()
            while True:
                # 1. 计算并打印发布频率
                time_bias = time.time() - last_freq_time
                freq = 1.0 / time_bias if time_bias > 0 else float('inf')
                print(f"frequency: {int(freq):3d} Hz")
                last_freq_time = time.time()

                # 3. 检查关节数量是否匹配（避免发布错误数据）
                if len(self.positions_controller) != len(self.joint_names):
                    print(f"关节数量不匹配：读取到{len(self.positions_controller)}个关节，预期{len(self.joint_names)}个")
                    return
                if len(self.positions_follower) != len(self.joint_names):
                    print(f"关节数量不匹配：读取到{len(self.positions_follower)}个关节，预期{len(self.joint_names)}个")
                    return
                
                # 4. 构建JointState消息
                joint_msg_controller = JointState()
                joint_msg_controller.header.stamp = self.get_clock().now().to_msg()  # 时间戳
                joint_msg_controller.name = self.joint_names  # 关节名称列表
                joint_msg_controller.position = [float(x) for x in self.positions_controller]  # 关节位置（弧度或角度，取决于硬件）

                joint_msg_follower = JointState()
                joint_msg_follower.header.stamp = self.get_clock().now().to_msg()  # 时间戳
                joint_msg_follower.name = self.joint_names  # 关节名称列表
                joint_msg_follower.position = [float(x) for x in self.positions_follower]  # 关节位置（弧度或角度，取决于硬件）
                
                # 5. 发布消息
                self.joint_pub_controller.publish(joint_msg_controller)
                self.joint_pub_follower.publish(joint_msg_follower)
                time.sleep(0.01)
        except Exception as e:
            print(f"发布关节状态失败: {str(e)}")

    def gravity(self):
        while True:
            try:
                self.arm_controller.gravity_compensation()
                time.sleep(0.01)
            except Exception as e:
                print(f"左重力补偿独立线程执行失败: {str(e)}")
                time.sleep(0.1)  # 避免过快循环导致日志刷屏

    def control(self):
        while True:
            try:
                self.positions_controller = self.arm_controller.get_joint_positions()
                self.positions_follower = self.arm_follower.get_joint_positions()
                self.arm_follower.set_joint_raw(positions = self.positions_controller, velocities = [0, 0, 0, 0, 0, 0])
                time.sleep(0.01)
            except Exception as e:
                print(f"左臂控制独立线程执行失败: {str(e)}")
                time.sleep(0.1)  # 避免过快循环导致日志刷屏

def main(args=None):
    # 初始化ROS 2
    rclpy.init(args=args)
    
    # 创建并运行节点
    node = MainArmJointPublisher()
    
    try:
        rclpy.spin(node)
    finally:
        # 清理资源
        node.arm_controller.arm.cleanup()
        node.arm_follower.arm.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()