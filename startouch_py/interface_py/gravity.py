from startouchclass import SingleArm
import time
# 创建机械臂连接  连接接口为"can0"
arm_controller = SingleArm(can_interface_ = "can0")

try:
    while True:
        print("XXXXXXXXXXX")
        arm_controller.gravity_compensation()
        time.sleep(0.01)  # 控制频率
except KeyboardInterrupt:
    print("Stopped by user.")

#回到home点
# arm_controller.go_home()
# 结束机械臂控制，删除指定机械臂对象
arm_controller.cleanup()
