from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiBinary
from gymnasium.core import RenderFrame
import numpy as np


class MotorEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.action_space = Dict(
            {
                "speed": Box(-10000, 20000, dtype=np.float32),
                "torque": Box(-400, 400, dtype=np.float32),
                "controls": MultiBinary(4),
                # controls:
                #   speed_send, torque_send, measure, abort
            }
        )
        self.observation_space = Box(
            np.array([0, 0, 0]),
            np.array([200, 200, 200]),
            dtype=np.float32,
        )
        self.metadata = {"render_modes": ["human", "text"]}
        self.render_mode = "text"

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        return super().step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return super().render()
