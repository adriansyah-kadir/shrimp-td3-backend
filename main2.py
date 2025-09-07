from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from water_env2 import WaterEnv

# Create the environment
env = WaterEnv(max_steps=200, action_cost_weight=0.01)

# Add action noise for exploration (important for TD3)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
)

# Initialize TD3 agent
# Better TD3 hyperparameters for your environment
model = TD3(
    "MlpPolicy",
    env,
    action_noise=NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions)
    ),  # More exploration
    verbose=1,
    learning_rate=3e-4,  # Lower learning rate
    buffer_size=100000,
    batch_size=128,  # Smaller batch size
    tau=0.001,  # Slower target updates
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    policy_delay=2,
    target_policy_noise=0.1,  # Less noise
    target_noise_clip=0.2,
)

# Train the agent
print("Training TD3 agent...")
model.learn(total_timesteps=200000)

# Test the trained agent
print("Testing trained agent...")
obs, _ = env.reset()
for i in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 20 == 0:
        print(f"Step {i}, Reward: {reward:.3f}")
        env.render()
    if terminated or truncated:
        obs, _ = env.reset()

model.save("td3_water3")
