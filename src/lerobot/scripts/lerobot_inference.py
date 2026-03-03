
import numpy as np
import time
from pathlib import Path
from pprint import pformat

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors

from lerobot.processor.rename_processor import rename_stats

from lerobot.utils.control_utils import predict_action

from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)

from lerobot.robots.startouch_arm.config_startouch_arm import StartouchArmConfig
from lerobot.robots.startouch_arm.startouch_arm import StartouchArm
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class RecordConfig:
    def __init__(self):
        policy_path = "/home/dhu/lerobot/outputs/train/startouch1101/checkpoints/070000/pretrained_model"
        self.policy = PreTrainedConfig.from_pretrained(policy_path)
        self.policy.pretrained_path = policy_path
        self.display_data = True
        self.play_sounds = True
        self.fps = 30
        self.single_task = "pick up the blocks"

class JointStateBridge(Node):
    def __init__(self, pub_topic="/master/joint"):
        super().__init__("inference_bridge")
        self.pub = self.create_publisher(JointState, pub_topic, 10)

    def publish_action(self, action: np.ndarray, names: list[str]):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = [float(x) for x in action.tolist()]
        self.pub.publish(msg)


def inference_loop(robot, bridge, fps, policy, preprocessor, postprocessor, single_task, dataset, display_data=True):
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    last_action = None

    while rclpy.ok():
        t0 = time.perf_counter()

        obs = robot.get_observation()
        # 将 robot obs 映射为 policy 需要的 key
        if "left_shoulder_pan" not in obs:  # 简单确认已填充
            continue

        state_names = [k for k, v in robot.observation_features.items() if v is float or v == float]
        state_vec = np.array([obs[n] for n in state_names], dtype=np.float32)

        # 三路图像
        observation_frame = {
            "observation.state": state_vec,
            "observation.images.laptop": obs["laptop"],
            "observation.images.right":  obs["right"],
            "observation.images.left":   obs["left"],
        }

        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=single_task,
            robot_type="startouch_arm",
        )[0]

        # 简单平滑：0.9*last_action + 0.1*action_values
        if last_action is None:
            smoothed_action = action_values
        else:
            smoothed_action = 0.9 * last_action + 0.1 * action_values
        last_action = smoothed_action

        action_names = dataset.meta.features["action"]["names"]
        if display_data:
            print("[DEBUG] action:", dict(zip(action_names, smoothed_action.tolist())))

        bridge.publish_action(np.asarray(smoothed_action.tolist(), dtype=float), action_names)

        busy_wait(1 / fps - (time.perf_counter() - t0))


def record(cfg: RecordConfig):
    init_logging()
    print(pformat(cfg.__dict__))
    
    # robot_cfg = StartouchArmConfig()
    # robot = StartouchArm(robot_cfg, mode="inference")
    # robot.connect()

    # dataset 仅用来拿 meta/stats
    dataset = LeRobotDataset(
        "tyl/pick_blocks",
        root=Path("/home/dhu/.cache/huggingface/lerobot/tyl/pick_blocks"),
    )

    policy = make_policy(cfg.policy, ds_meta=dataset.meta)
    print("[INFO] Policy loaded from:", cfg.policy.pretrained_path)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, {}),
        preprocessor_overrides={"device_processor": {"device": cfg.policy.device}},
    )

    bridge = JointStateBridge()

    log_say("Start inference", cfg.play_sounds)
    try:
        inference_loop(robot, bridge, cfg.fps, policy, preprocessor, postprocessor, cfg.single_task, dataset, cfg.display_data)
    except KeyboardInterrupt:
        log_say("Keyboard interrupt, stopping...", cfg.play_sounds)
    finally:
        log_say("Stop inference", cfg.play_sounds, blocking=True)
        # 建议统一在这里关
        if rclpy.ok():
            rclpy.shutdown()
        robot.disconnect()


def main():
    cfg = RecordConfig()
    record(cfg)


if __name__ == "__main__":
    main()
