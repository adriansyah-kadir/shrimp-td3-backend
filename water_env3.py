from typing import Any, SupportsFloat
import numpy as np
from gymnasium import Env, spaces


class WaterEnv(Env):
    def __init__(self, *_, **__) -> None:
        super().__init__()
        self.observation_space = spaces.Box(0, 1, (6,))
        self.action_space = spaces.Box(-1, 1, (6,))

        self.min_space = np.array([10, 0, 0, 5, 1, 0], np.float32)
        self.max_space = np.array([50, 50, 100, 10, 20, 10], np.float32)
        self.range_space = self.max_space - self.min_space
        self.target_space = np.array(
            [
                28,  # temp
                26,  # sal (ppt)
                28,  # turb (NTU)
                7,  # pH
                5,  # do
                0.001,  # ammonia
            ],
            dtype=np.float32,
        )
        self.state = self.observation_space.sample()
        self.steps = 0
        self.global_steps = 0
        self.volume = 10_000  # liters
        self.change_volume = 200  # volume affected by each fill/drain action

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        target = (self.target_space - self.min_space) / self.range_space
        next_state = self.state + action * 0.1
        distance = next_state - target
        state_error = np.mean(np.abs(distance))
        direction_error = np.mean(np.maximum(0, action * distance))
        action_cost = np.mean(np.abs(action))
        reward = -(state_error * 2) - action_cost - direction_error

        self.state = np.clip(next_state, 0.0, 1.0)
        self.steps += 1

        terminating = state_error <= 0.02
        truncating = self.steps % 5000 == 0

        return (
            self.state.copy(),
            float(reward),
            bool(terminating),
            bool(truncating),
            {},
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.steps = 0
        self.state = self.observation_space.sample()
        return self.state, {}

    def close(self):
        return super().close()
