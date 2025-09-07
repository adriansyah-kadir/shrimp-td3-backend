from stable_baselines3 import TD3
from water_env import WaterEnv

env = WaterEnv()
model = TD3.load("td3_water2.zip")

state = ([*env.target_space[:-1], 1] - env.min_space) / env.range_space
print(state)

# while state[5] < 1:
#     state[5] *= 1.1
action, _ = model.predict(state)
print(action.tolist(), (state * env.range_space + env.min_space).tolist(), sep="\n")
