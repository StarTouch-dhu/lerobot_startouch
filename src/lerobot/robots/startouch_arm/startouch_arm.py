import logging


from ..robot import Robot

from .config_startouch_arm import StartouchArmConfig

logger = logging.getLogger(__name__)


class StartouchArm(Robot):
    config_class = StartouchArmConfig
    name = "startouch_arm"

    def __init__(self, config: StartouchArmConfig):
        super().__init__(config)
        self.config = config











