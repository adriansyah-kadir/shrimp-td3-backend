from water_env3 import WaterEnv
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env
from callback_log import ConsoleLoggerCallback
import numpy as np

# Add custom callback
log_callback = ConsoleLoggerCallback(log_freq=100)

env = WaterEnv()
n_actions = env.action_space.shape[0]  # pyright: ignore
check_env(env)

action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
)

model = TD3(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="runs",
    action_noise=action_noise,
)

# model = TD3.load("td3_water.zip", env)

model.learn(
    total_timesteps=int(1e5),
    progress_bar=True,
    callback=log_callback,
    reset_num_timesteps=True,
)

model.save("td3_water2")

env.close()
