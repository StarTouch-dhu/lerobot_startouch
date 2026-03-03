# startouch_inference_async_ros

异步 ROS 推理脚本（双臂 Startouch + LeRobot policy）。

## 脚本

- `inference_startouch_dual_async_ros.py`

## 运行

在项目根目录执行：

```bash
python src/lerobot/scripts/startouch_inference_async_ros/inference_startouch_dual_async_ros.py
```

## 异步结构

- 主线程：采集机器人观测 + 发布最新动作到 `/master/joint`
- 推理线程：消费“最新观测队列”并执行模型推理
- 队列策略：`maxsize=1`，始终只保留最新观测，避免动作滞后
