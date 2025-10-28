import threading
from startouchclass import SingleArm


arm_controller = SingleArm(can_interface_ = "can0",enable_fd_ = False)
arm_follower = SingleArm(can_interface_ = "can1",enable_fd_ = False)

def gravity_thread():
    while True:
        arm_controller.gravity_compensation()

threading.Thread(target=gravity_thread, daemon=True).start()

try:
    while True:
        temp_positions = arm_controller.get_joint_positions()
        arm_follower.set_joint_raw(positions = temp_positions, velocities = [0, 0, 0, 0, 0, 0])
finally:
    print("程序被中断，主动清理资源")
    arm_controller.arm.cleanup()
    arm_follower.arm.cleanup()