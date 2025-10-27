from startouchclass import SingleArm

arm_controller = SingleArm(can_interface_ = "can0",enable_fd_ = False)
arm_follower = SingleArm(can_interface_ = "can1",enable_fd_ = False)

while True:
    temp_positions = arm_controller.get_joint_positions()
    arm_controller.gravity_compensation()
    # temp_velocities = arm_follower.get_joint_velocities()
    arm_follower.set_joint_raw(positions = temp_positions, velocities = [0, 0, 0, 0, 0, 0])