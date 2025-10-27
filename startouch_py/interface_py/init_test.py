from startouchclass import SingleArm
import time

arm_controller = SingleArm(can_interface_ = "can0", enable_fd_=False)

arm_controller1 = SingleArm(can_interface_ = "can1", enable_fd_=False)