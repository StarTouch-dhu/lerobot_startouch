import time
from startouchclass import SingleArm

# 初始化机械臂
arm = SingleArm(can_interface_="can1", enable_fd_=False)


# 5 个关键点（单位：弧度）
# points = [
#     [-1.602388, 1.118296, 2.209316, -1.066796, 0.0, 0.0],
#     [-1.602388, 1.118296, 2.209316, 0.042535 , 0.0, 0.0],
#     [-1.602388, 1.118296, 2.209316, -1.843862, 0.0, 0.0],
#     [-1.602388, 1.118296, 2.209316, -1.066796, 0.0, 0.0],
#     [-1.473831, 0.819600, 1.180095, -1.365873, 0.0, 0.0],
#     [-1.473831, 0.0, 0.0, 0.0, 0.0, 0.0]
# ]
points = [
    [1.473831, 0.0, 0.0, 0.0, 0.0, 0.0]
]

# 每个点的执行时间（单位：秒）
tf = 2

# 依次执行每个点
for i, q in enumerate(points):
    print(f"[Step {i+1}] Moving to: {q}")
    arm.set_joint(q, tf=tf)
    time.sleep(100)  # 稍微多等一点，确保运动完成

print("[Done] All points executed.")
