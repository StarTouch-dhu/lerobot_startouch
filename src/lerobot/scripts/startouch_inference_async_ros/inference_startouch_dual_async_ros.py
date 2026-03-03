import threading
import time
from pathlib import Path
from pprint import pformat

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots.startouch_arm.config_startouch_arm import StartouchArmConfig
from lerobot.robots.startouch_arm.startouch_arm import StartouchArm
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say


class JointStateBridge(Node):
    def __init__(self, pub_topic: str = "/master/joint"):
        super().__init__("inference_bridge_async")
        self.pub = self.create_publisher(JointState, pub_topic, 10)

    def publish_action(self, action: np.ndarray, names: list[str]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = [float(x) for x in action.tolist()]
        self.pub.publish(msg)


class AsyncInferenceConfig:
    def __init__(self):
        policy_path = "/home/lft/workspace/python_project/VLA/lerobot_tyl/outputs/ACT/train/startouch1031/checkpoints/100000/pretrained_model"
        self.policy = PreTrainedConfig.from_pretrained(policy_path)
        self.policy.pretrained_path = policy_path

        self.dataset_repo_id = "wyd/pick_blocks"
        self.dataset_root = Path("/home/lft/workspace/python_project/VLA/data_v31/startouch_test1")

        self.fps = 30
        self.single_task = "pick up the blocks"
        self.display_data = False
        self.play_sounds = True

        self.metrics_print_interval_s = 1.0
        self.monitor_sleep_s = 0.01


def _build_observation_frame(obs: dict, state_names: list[str]) -> dict[str, np.ndarray]:
    state_vec = np.array([obs[name] for name in state_names], dtype=np.float32)
    return {
        "observation.state": state_vec,
        "observation.images.laptop": obs["laptop"],
        "observation.images.right": obs["right"],
        "observation.images.left": obs["left"],
    }


def _observation_worker(
    stop_event: threading.Event,
    robot: StartouchArm,
    state_names: list[str],
    obs_lock: threading.Lock,
    obs_store: dict,
    err_lock: threading.Lock,
    err_store: dict,
    metrics_print_interval_s: float,
):
    required_obs_keys = ["laptop", "right", "left", state_names[0]]

    obs_count = 0
    obs_window_t0 = time.perf_counter()

    while not stop_event.is_set():
        try:
            obs = robot.get_observation()
            if any(k not in obs for k in required_obs_keys):
                time.sleep(0.005)
                continue

            with obs_lock:
                obs_store["seq"] += 1
                obs_store["obs"] = obs
                obs_count += 1

            now = time.perf_counter()
            dt = now - obs_window_t0
            if dt >= metrics_print_interval_s:
                obs_hz = obs_count / dt
                print(f"[METRICS] obs_hz={obs_hz:.2f}")
                obs_count = 0
                obs_window_t0 = now

        except Exception as e:
            with err_lock:
                err_store["error"] = RuntimeError(f"Observation worker failed: {e}")
            stop_event.set()
            break


def _inference_worker(
    stop_event: threading.Event,
    obs_lock: threading.Lock,
    obs_store: dict,
    action_lock: threading.Lock,
    action_store: dict,
    err_lock: threading.Lock,
    err_store: dict,
    policy,
    preprocessor,
    postprocessor,
    action_names: list[str],
    state_names: list[str],
    single_task: str,
    display_data: bool,
    metrics_print_interval_s: float,
):
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    infer_count = 0
    infer_window_t0 = time.perf_counter()
    last_seq = 0

    while not stop_event.is_set():
        try:
            with obs_lock:
                current_seq = obs_store["seq"]
                obs = obs_store["obs"]

            if obs is None or current_seq == last_seq:
                time.sleep(0.001)
                continue

            observation_frame = _build_observation_frame(obs, state_names)
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

            action_arr = np.asarray(action_values.tolist(), dtype=float)
            if display_data:
                print("[DEBUG][async] action:", dict(zip(action_names, action_arr.tolist())))

            with action_lock:
                action_store["values"] = action_arr
                infer_count += 1

            last_seq = current_seq

            now = time.perf_counter()
            dt = now - infer_window_t0
            if dt >= metrics_print_interval_s:
                infer_hz = infer_count / dt
                print(f"[METRICS] infer_hz={infer_hz:.2f}")
                infer_count = 0
                infer_window_t0 = now

        except Exception as e:
            with err_lock:
                err_store["error"] = RuntimeError(f"Inference worker failed: {e}")
            stop_event.set()
            break


def _publish_worker(
    stop_event: threading.Event,
    bridge: JointStateBridge,
    action_lock: threading.Lock,
    action_store: dict,
    action_names: list[str],
    fps: int,
    metrics_print_interval_s: float,
):
    publish_loop_count = 0
    publish_count = 0
    publish_window_t0 = time.perf_counter()

    while rclpy.ok() and not stop_event.is_set():
        t0 = time.perf_counter()
        publish_loop_count += 1

        with action_lock:
            latest_action = action_store["values"]

        if latest_action is not None:
            bridge.publish_action(latest_action, action_names)
            publish_count += 1

        now = time.perf_counter()
        dt = now - publish_window_t0
        if dt >= metrics_print_interval_s:
            publish_loop_hz = publish_loop_count / dt
            publish_hz = publish_count / dt
            print(f"[METRICS] publish_loop_hz={publish_loop_hz:.2f}, publish_hz={publish_hz:.2f}")
            publish_loop_count = 0
            publish_count = 0
            publish_window_t0 = now

        busy_wait(1 / fps - (time.perf_counter() - t0))


def run_async_inference(cfg: AsyncInferenceConfig) -> None:
    init_logging()
    print(pformat(cfg.__dict__))

    robot = StartouchArm(StartouchArmConfig(), mode="inference")
    robot.connect()

    dataset = LeRobotDataset(cfg.dataset_repo_id, root=cfg.dataset_root)
    action_names = dataset.meta.features["action"]["names"]
    state_names = dataset.meta.features["observation.state"]["names"]

    safe_device = get_safe_torch_device(cfg.policy.device)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, {}),
        preprocessor_overrides={"device_processor": {"device": safe_device.type}},
    )

    bridge = JointStateBridge()

    stop_event = threading.Event()

    obs_lock = threading.Lock()
    obs_store = {"seq": 0, "obs": None}

    action_lock = threading.Lock()
    action_store = {"values": None}

    err_lock = threading.Lock()
    err_store = {"error": None}

    obs_thread = threading.Thread(
        target=_observation_worker,
        args=(
            stop_event,
            robot,
            state_names,
            obs_lock,
            obs_store,
            err_lock,
            err_store,
            cfg.metrics_print_interval_s,
        ),
        daemon=True,
    )

    infer_thread = threading.Thread(
        target=_inference_worker,
        args=(
            stop_event,
            obs_lock,
            obs_store,
            action_lock,
            action_store,
            err_lock,
            err_store,
            policy,
            preprocessor,
            postprocessor,
            action_names,
            state_names,
            cfg.single_task,
            cfg.display_data,
            cfg.metrics_print_interval_s,
        ),
        daemon=True,
    )

    publish_thread = threading.Thread(
        target=_publish_worker,
        args=(
            stop_event,
            bridge,
            action_lock,
            action_store,
            action_names,
            cfg.fps,
            cfg.metrics_print_interval_s,
        ),
        daemon=True,
    )

    log_say("Start async inference", cfg.play_sounds)

    obs_thread.start()
    infer_thread.start()
    publish_thread.start()

    try:
        while rclpy.ok() and not stop_event.is_set():
            with err_lock:
                worker_error = err_store["error"]
            if worker_error is not None:
                raise worker_error
            time.sleep(cfg.monitor_sleep_s)

    except KeyboardInterrupt:
        log_say("Keyboard interrupt, stopping...", cfg.play_sounds)
    finally:
        stop_event.set()

        obs_thread.join(timeout=1.0)
        infer_thread.join(timeout=1.0)
        publish_thread.join(timeout=1.0)

        log_say("Stop async inference", cfg.play_sounds, blocking=True)

        if rclpy.ok():
            rclpy.shutdown()
        robot.disconnect()


def main() -> None:
    run_async_inference(AsyncInferenceConfig())


if __name__ == "__main__":
    main()
