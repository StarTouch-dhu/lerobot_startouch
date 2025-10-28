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


from typing import Any







from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot

from .config_startouch_arm import StartouchArmConfig

logger = logging.getLogger(__name__)


class StartouchArm(Robot):
    config_class = StartouchArmConfig
    name = "startouch_arm"

    def __init__(self, config: StartouchArmConfig):
        super().__init__(config)
        self.config = config






    ## ======================================== ##

    ## ======================================== ##

    ## ======================================== ##

    ## ======================================== ##

    ## ======================================== ##

    ## ======================================== ##

    ## ======================================== ##

    ## ======================================== ##

    ## ======================================== ##

    ## ======================================== ##

    ## ======================================== ##

    ## ======================================== ##
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        

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





