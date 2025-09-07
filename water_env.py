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
        target = self.target_space
        action_weights = np.ones(self.action_space.shape)  # action cost weights
        change_per_step = abs(self.range_space) * 0.05  # action change per step
        update_per_step = np.array([-5, -1, 1, 0, -1, 0.01], np.float32)  # natural
        state = self.state * self.range_space + self.min_space  # unnormalize state
        next_state = state + update_per_step * 0.05  # apply natural update
        next_state = next_state + change_per_step * action  # apply action control
        error = np.mean(np.abs(next_state - target))
        action_cost = np.sum(np.abs(action) * action_weights)  # total action usage
        reward = -error - 0.05 * action_cost
        self.state = np.clip((next_state - self.min_space) / self.range_space, 0.0, 1.0)

        self.steps += 1
        self.global_steps += 1

        # terminating = np.any(
        #     (self.state > self.observation_space.high)  # pyright: ignore
        #     | (self.state < self.observation_space.low)  # pyright: ignore
        # )

        max_steps = min(5000, max(100, (self.global_steps // 1000) * 100))
        truncating = self.steps >= max_steps

        terminating = truncating

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
