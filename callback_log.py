from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


np.set_printoptions(formatter={"float": lambda x: "{0:0.6f}".format(x)})


class ConsoleLoggerCallback(BaseCallback):
    def __init__(self, verbose=1, log_freq=10):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.n_calls = 0
        self.min_space = np.array([10, 0, 0, 5, 1, 0], np.float32)
        self.max_space = np.array([50, 50, 100, 10, 20, 10], np.float32)
        self.range_space = self.max_space - self.min_space

    def _on_step(self) -> bool:
        self.n_calls += 1
        env = self.locals["env"]
        if self.n_calls % self.log_freq == 0:
            reward = self.locals["rewards"]
            action = self.locals["actions"]
            obs = self.locals["new_obs"] * self.range_space + self.min_space
            print(
                f"[Step {self.n_calls}]\nReward: {reward}\nAction: {action}\nState Norm: {obs}\nState: {self.locals['new_obs']}"
            )
        return True
