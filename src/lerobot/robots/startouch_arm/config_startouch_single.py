# from dataclasses import dataclass, field
# from lerobot.robots.robot import RobotConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras import CameraConfig

# from .config_startouch_arm import StartouchArmConfig
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

@RobotConfig.register_subclass("startouch_arm_single")
@dataclass
class StartouchArmConfig_single(RobotConfig):
    type: str = "startouch_arm_single"  # 必须与 StartouchArm.robot_type 对齐

    leader_arms: dict[str, list[str]] = field(
        default_factory=lambda: {
            "left_arm": [
                "left_shoulder_pan",
                "left_shoulder_lift",
                "left_elbow_flex",
                "left_wrist_1",
                "left_wrist_2",
                "left_wrist_3",
                "left_gripper",
            ],
        }
    )
    follower_arms: dict[str, list[str]] = field(
        default_factory=lambda: {
            "left_arm": [
                "left_shoulder_pan",
                "left_shoulder_lift",
                "left_elbow_flex",
                "left_wrist_1",
                "left_wrist_2",
                "left_wrist_3",
                "left_gripper",
            ],
        }
    )
    cameras: dict[str, OpenCVCameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480),
            "left": OpenCVCameraConfig(index_or_path=2, fps=30, width=640, height=480),
        }
    )
