#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import rclpy
import logging

from functools import cached_property
from typing import Any
from sensor_msgs.msg import JointState
from lerobot.cameras.utils import make_cameras_from_configs

from rclpy.callback_groups import ReentrantCallbackGroup

import numpy as np
import torch
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot

from .config_startouch_arm import StartouchArmConfig

logger = logging.getLogger(__name__)


class StartouchArm(Robot):
    config_class = StartouchArmConfig
    name = "startouch_arm"

    def __init__(self, config: StartouchArmConfig, mode="teleop"):
        super().__init__(config)
        self.config = config
        self.mode = mode
        self.robot_type = "startouch_arm"

        self.latest_state_map = None  # inference 模式下，来自 /puppet/joint 的 {name: value}

        # arms
        self.leader_arms = dict(config.leader_arms)
        self.follower_arms = dict(config.follower_arms)

        # 关节名称映射
        self.joint_name_mapping = {
            "left_arm": {
                "joint1": "left_shoulder_pan",
                "joint2": "left_shoulder_lift",
                "joint3": "left_elbow_flex",
                "joint4": "left_wrist_1",
                "joint5": "left_wrist_2",
                "joint6": "left_wrist_3",
                "gripper": "left_gripper"
                
            },
            "right_arm": {
                "joint1": "right_shoulder_pan",
                "joint2": "right_shoulder_lift",
                "joint3": "right_elbow_flex",
                "joint4": "right_wrist_1",
                "joint5": "right_wrist_2",
                "joint6": "right_wrist_3",
                "gripper": "right_gripper"
            }
        }

        # cameras
        self.cameras = make_cameras_from_configs(self.config.cameras)

        # ros2
        self.node = None
        self.subs = {}
        self.latest_joint_state = {}
        self.latest_action_state = {}
        self.callback_group = ReentrantCallbackGroup()
        self._is_connected = False

    ## ======================================== ##
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            # 关节写成 float，不要包 dtype/shape/names
            "left_shoulder_pan": float,
            "left_shoulder_lift": float,
            "left_elbow_flex": float,
            "left_wrist_1": float,
            "left_wrist_2": float,
            "left_wrist_3": float,
            "left_gripper": float,
            "right_shoulder_pan": float,
            "right_shoulder_lift": float,
            "right_elbow_flex": float,
            "right_wrist_1": float,
            "right_wrist_2": float,
            "right_wrist_3": float,
            "right_gripper": float,
            # 摄像头直接写 tuple
            "laptop": (480, 640, 3),
            "right": (480, 640, 3),
            "left": (480, 640, 3),
        }

    ## ======================================== ##
    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "left_shoulder_pan": float,
            "left_shoulder_lift": float,
            "left_elbow_flex": float,
            "left_wrist_1": float,
            "left_wrist_2": float,
            "left_wrist_3": float,
            "left_gripper": float,
            "right_shoulder_pan": float,
            "right_shoulder_lift": float,
            "right_elbow_flex": float,
            "right_wrist_1": float,
            "right_wrist_2": float,
            "right_wrist_3": float,
            "right_gripper": float,
        }

    ## ======================================== ##
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            rclpy.spin_once(self.node, timeout_sec=0.5)
            obs = {}

            if self.mode == "teleop":
                states = []
                for arm, joints in self.follower_arms.items():
                    v = self.latest_joint_state.get(arm)
                    if v is None or v.numel() != len(joints):
                        logging.warning(f"No joint state received for {arm}, using zeros.")
                        v = torch.zeros(len(joints), dtype=torch.float32)
                    states.append(v)
                state_array = torch.cat(states, dim=0).numpy()

                obs_schema = self.observation_features
                state_names = [k for k, v in obs_schema.items() if v is float or v == float]
                if not state_names:
                    state_names = [j for _, joints in self.follower_arms.items() for j in joints]

                if len(state_names) != len(state_array):
                    logging.error(f"State names length ({len(state_names)}) != state array length ({len(state_array)}).")

                for name, val in zip(state_names, state_array):
                    obs[name] = float(val)
            
            elif self.mode == "inference":
                obs_schema = self.observation_features
                state_names = [k for k, v in obs_schema.items() if v is float or v == float]
                if not state_names:
                    state_names = [j for _, joints in self.follower_arms.items() for j in joints]

                values = []
                for name in state_names:
                    if self.latest_state_map and name in self.latest_state_map:
                        values.append(self.latest_state_map[name])
                    else:
                        logging.warning(f"No value for joint '{name}' from /puppet/joint, using 0.0")
                        values.append(0.0)

                # 展开为独立键（和你原来的风格保持一致）
                for name, val in zip(state_names, values):
                    obs[name] = float(val)


            for cam_key, cam in self.cameras.items():
                try:
                    img = cam.read()
                    if img is None:
                        logging.warning(f"No image received from camera {cam_key}, using zeros.")
                        img = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
                    obs[cam_key] = img if isinstance(img, np.ndarray) else img.numpy()
                except Exception as e:
                    logging.error(f"Error reading camera {cam_key}: {e}")
                    obs[cam_key] = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)

            return obs

        except Exception as e:
            logging.error(f"Error in get_observation: {e}")
            return {}

    ## ======================================== ##
    def get_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        try:
            rclpy.spin_once(self.node, timeout_sec=0.5)
            action = {}

            # 找到 feature 定义里的关节名字
            act_schema = self.action_features
            if isinstance(act_schema.get("action"), dict) and "names" in act_schema["action"]:
                # 数据集式 schema：{"action": {"dtype":..., "shape":..., "names":[...]}}
                action_names = act_schema["action"]["names"]
            else:
                action_names = [k for k, v in act_schema.items() if v is float or v == float]
                if not action_names:
                    # 最后兜底：沿 leader_arms 顺序
                    action_names = [j for _, joints in self.leader_arms.items() for j in joints]

            # 拼接当前动作向量
            values = []
            for arm, joints in self.leader_arms.items():
                v = self.latest_action_state.get(arm)
                if v is None or v.numel() != len(joints):
                    logging.warning(f"No action received for {arm}, using zeros.")
                    v = torch.zeros(len(joints), dtype=torch.float32)
                values.append(v)
            action_array = torch.cat(values, dim=0).numpy()  # shape=(14,)

            if len(action_names) != len(action_array):
                logging.error(
                    f"Action names length ({len(action_names)}) != action array length ({len(action_array)})"
                )

            # 展开成独立键
            for name, val in zip(action_names, action_array):
                action[name] = float(val)

            return action

        except Exception as e:
            logging.error(f"Error in get_action: {e}")
            return {}

    ## ======================================== ##
    def send_action(self, action):
        logging.debug(f"send_action called with {action}, but ignored (controlled by external ROS2 nodes).")
        return action

    ## ======================================== ##
    def _joint_cb(self, msg: JointState, arm: str):
        try:
            mapping = self.joint_name_mapping.get(arm, {})
            joint_order = self.follower_arms[arm]
            positions = [0.0] * len(joint_order)
            for i, name in enumerate(msg.name):
                mapped_name = mapping.get(name, name)
                if mapped_name in joint_order:
                    idx = joint_order.index(mapped_name)
                    positions[idx] = msg.position[i]

            pos = torch.as_tensor(positions, dtype=torch.float32)
            self.latest_joint_state[arm] = pos.clone()
            # logging.info(f"Received joint state for {arm}: {pos.tolist()}, mapped from {msg.name}")
        except Exception as e:
            logging.error(f"Error processing joint state for {arm}: {e}")

    ## ======================================== ##
    def _action_cb(self, msg: JointState, arm: str):
        try:
            mapping = self.joint_name_mapping.get(arm, {})
            joint_order = self.leader_arms[arm]
            positions = [0.0] * len(joint_order)
            for i, name in enumerate(msg.name):
                mapped_name = mapping.get(name, name)
                if mapped_name in joint_order:
                    idx = joint_order.index(mapped_name)
                    positions[idx] = msg.position[i]
            pos = torch.as_tensor(positions, dtype=torch.float32)
            self.latest_action_state[arm] = pos.clone()
            logging.debug(f"Received action for {arm}: {pos.tolist()}, mapped from {msg.name}")
        except Exception as e:
            logging.error(f"Error processing action for {arm}: {e}")

    ## ======================================== ##
    def _inference_cb(self, msg: JointState):
        try:
            if msg.name and msg.position:
                # 把 /puppet/joint 的名字→位置 做成 dict
                self.latest_state_map = {n: float(p) for n, p in zip(msg.name, msg.position)}
        except Exception as e:
            logging.error(f"Error in _inference_cb: {e}")
            
    ## ======================================== ##
    @property
    def is_calibrated(self) -> bool:
        return True
        
    ## ======================================== ##
    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    ## ======================================== ##
    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node("startouch_arm")
        if self.mode == "teleop":
            for arm in self.leader_arms.keys():
                topic = f"/{arm}/joint_states_target"
                self.subs[f"{arm}_action"] = self.node.create_subscription(
                    JointState, topic, lambda msg, arm=arm: self._action_cb(msg, arm), 3,
                    callback_group=self.callback_group,
                )
                self.latest_action_state[arm] = None

            for arm in self.follower_arms.keys():
                topic = f"/{arm}/joint_states_now"
                self.subs[arm] = self.node.create_subscription(
                    JointState, topic, lambda msg, arm=arm: self._joint_cb(msg, arm), 3,
                    callback_group=self.callback_group,
                )
                self.latest_joint_state[arm] = None

        elif self.mode == "inference":
            self.node.create_subscription(JointState, "/puppet/joint", self._inference_cb, 3)

        self._is_connected = True

        for cam in self.cameras.values():
            cam.connect()
        
        # -------- Warmup：等待 ROS2 回调至少收到一条消息 --------
        import time
        start_t = time.time()
        while time.time() - start_t < 2.0:  # 最多等 2 秒
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.mode == "teleop":
                ok_js = all(v is not None for v in self.latest_joint_state.values())
                ok_act = all(v is not None for v in self.latest_action_state.values())
                if ok_js and ok_act:
                    break
            elif self.mode == "inference":
                if self.latest_state_map:
                    break

        if self.mode == "teleop":
            if any(v is None for v in self.latest_joint_state.values()):
                logging.warning("Some joint states are still None after warmup.")
            if any(v is None for v in self.latest_action_state.values()):
                logging.warning("Some action states are still None after warmup.")
        elif self.mode == "inference":
            if not self.latest_state_map:
                logging.warning("No /puppet/joint received after warmup.")

    ## ======================================== ##
    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        if self.node is not None:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self._is_connected = False

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    ## ======================================== ##
    def calibrate(self) -> None:
        print("StartouchArm.calibrate() not implemented, skipping.")

    ## ======================================== ##
    def configure(self) -> None:
        print("StartouchArm.configure() not implemented, skipping.")



