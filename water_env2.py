import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Any


class WaterEnv(gym.Env):
    """
    Pond Water Quality Simulation Environment

    State Space (6 parameters):
    - Temperature (°C): 0-40
    - Salinity (ppt): 0-35
    - Turbidity (NTU): 0-100
    - pH: 4-10
    - Dissolved Oxygen (mg/L): 0-20
    - Ammonia NH3 (mg/L): 0-10

    Action Space (6 actions):
    - Each action corresponds to a state parameter
    - Actions range from -1 (decrease) to 1 (increase)
    """

    def __init__(self, max_steps: int = 1000, action_cost_weight: float = 0.1):
        super(WaterEnv, self).__init__()

        self.max_steps = max_steps
        self.current_step = 0
        self.action_cost_weight = action_cost_weight  # Weight for action cost penalty

        # Define state bounds [min, max] for each parameter
        self.state_bounds = {
            "temperature": [0, 40],  # °C
            "salinity": [0, 35],  # ppt (parts per thousand)
            "turbidity": [0, 100],  # NTU (Nephelometric Turbidity Units)
            "pH": [4, 10],  # pH scale
            "dissolved_oxygen": [0, 20],  # mg/L
            "ammonia": [0, 10],  # mg/L NH3
        }

        # Optimal target values for healthy pond (for reward calculation)
        self.optimal_values = {
            "temperature": 28.0,  # °C - optimal for most pond life
            "salinity": 26,  # ppt - low salinity freshwater
            "turbidity": 28.0,  # NTU - clear water with minimal particles
            "pH": 7,  # slightly alkaline for biological processes
            "dissolved_oxygen": 5,  # mg/L - high oxygen for fish health
            "ammonia": 0.01,  # mg/L - minimal ammonia
        }

        # Define observation space (normalized to [0, 1])
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Define action space (6 actions, each from -1 to 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Action scaling factors (how much each action affects the parameter)
        self.action_scales = {
            "temperature": 2.0,  # ±2°C per action
            "salinity": 1.0,  # ±1 ppt per action
            "turbidity": 5.0,  # ±5 NTU per action
            "pH": 0.5,  # ±0.5 pH units per action
            "dissolved_oxygen": 2.0,  # ±2 mg/L per action
            "ammonia": 0.5,  # ±0.5 mg/L per action
        }

        self.state = None
        self.last_actions = None

        self.reset()

    def reset(
        self,
        seed=None,
        options=None,
    ):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        self.current_step = 0

        # Initialize state with random values within bounds
        if seed is not None:
            np.random.seed(seed)

        self.state = {
            "temperature": np.random.uniform(15, 35),
            "salinity": np.random.uniform(0, 10),
            "turbidity": np.random.uniform(5, 50),
            "pH": np.random.uniform(6, 9),
            "dissolved_oxygen": np.random.uniform(4, 15),
            "ammonia": np.random.uniform(0, 3),
        }
        self.last_actions = None
        return self._get_normalized_state(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        self.current_step += 1

        # Store actions for reward calculation
        self.last_actions = action.copy()

        # Apply direct actions to states
        self._apply_actions(action)

        # Apply natural interactions between states
        self._apply_natural_interactions()

        # Clip states to valid bounds
        self._clip_states()

        # Calculate reward (includes action cost)
        reward = self._calculate_reward()

        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps

        return (
            self._get_normalized_state(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _apply_actions(self, actions: np.ndarray):
        """Apply direct actions to each state parameter"""
        state_keys = list(self.state_bounds.keys())

        for i, action in enumerate(actions):
            param = state_keys[i]
            # Scale action and apply to state
            change = action * self.action_scales[param]
            self.state[param] += change

    def _apply_natural_interactions(self):
        """Apply natural interactions between water quality parameters"""
        dt = 0.1  # time step for simulation

        # Temperature effects
        temp = self.state["temperature"]

        # Higher temperature decreases DO (oxygen solubility decreases)
        if temp > 25:
            self.state["dissolved_oxygen"] -= (temp - 25) * 0.02 * dt

        # Higher temperature increases ammonia toxicity and bacterial activity
        if temp > 20:
            self.state["ammonia"] += (temp - 20) * 0.001 * dt

        # pH effects
        pH = self.state["pH"]

        # Extreme pH affects ammonia toxicity (NH3 vs NH4+ equilibrium)
        if pH > 8:
            # Higher pH increases toxic NH3 form
            self.state["ammonia"] += (pH - 8) * 0.005 * dt
        elif pH < 6:
            # Lower pH reduces DO through biological stress
            self.state["dissolved_oxygen"] -= (6 - pH) * 0.01 * dt

        # Dissolved Oxygen effects
        DO = self.state["dissolved_oxygen"]

        # Low DO increases ammonia (reduced nitrification)
        if DO < 4:
            self.state["ammonia"] += (4 - DO) * 0.01 * dt

        # Very low DO increases turbidity (algae die-off, sediment resuspension)
        if DO < 2:
            self.state["turbidity"] += (2 - DO) * 2.0 * dt

        # Ammonia effects
        NH3 = self.state["ammonia"]

        # High ammonia reduces DO (bacterial oxygen consumption for nitrification)
        if NH3 > 1:
            self.state["dissolved_oxygen"] -= (NH3 - 1) * 0.1 * dt

        # High ammonia can increase turbidity (algal blooms from nutrient loading)
        if NH3 > 2:
            self.state["turbidity"] += (NH3 - 2) * 1.0 * dt

        # Turbidity effects
        turbidity = self.state["turbidity"]

        # High turbidity reduces DO (blocks light for photosynthesis)
        if turbidity > 30:
            self.state["dissolved_oxygen"] -= (turbidity - 30) * 0.005 * dt

        # High turbidity can slightly buffer pH changes
        if turbidity > 50:
            pH_target = 7.5
            self.state["pH"] += (pH_target - self.state["pH"]) * 0.01 * dt

        # Salinity effects
        salinity = self.state["salinity"]

        # High salinity reduces DO solubility
        if salinity > 10:
            self.state["dissolved_oxygen"] -= (salinity - 10) * 0.01 * dt

        # High salinity affects pH buffering capacity
        if salinity > 15:
            self.state["pH"] += 0.005 * dt  # Slight pH increase

        # Natural stabilization processes
        self._apply_natural_stabilization(dt)

    def _apply_natural_stabilization(self, dt: float):
        """Apply natural stabilization processes that occur in pond ecosystems"""

        # Temperature stabilization (thermal inertia)
        temp_target = 22  # Ambient temperature
        self.state["temperature"] += (
            (temp_target - self.state["temperature"]) * 0.001 * dt
        )

        # pH buffering (natural alkalinity)
        pH_target = 7.2
        self.state["pH"] += (pH_target - self.state["pH"]) * 0.005 * dt

        # DO replenishment (atmospheric exchange and photosynthesis)
        DO_target = 8.0
        if self.state["dissolved_oxygen"] < DO_target:
            self.state["dissolved_oxygen"] += (
                (DO_target - self.state["dissolved_oxygen"]) * 0.01 * dt
            )

        # Ammonia natural reduction (nitrification)
        if self.state["ammonia"] > 0.1 and self.state["dissolved_oxygen"] > 2:
            self.state["ammonia"] *= 0.99  # 1% reduction per step

        # Turbidity settling
        if self.state["turbidity"] > 5:
            self.state["turbidity"] *= 0.995  # 0.5% reduction per step

    def _clip_states(self):
        """Ensure all states remain within valid bounds"""
        for param, bounds in self.state_bounds.items():
            self.state[param] = np.clip(self.state[param], bounds[0], bounds[1])

    def _calculate_reward(self) -> float:
        """Calculate reward based on distance from optimal values and action cost"""
        # Calculate distance-based reward
        total_distance = 0.0

        for param, optimal_value in self.optimal_values.items():
            current_value = self.state[param]

            # Calculate normalized distance from optimal value
            param_range = self.state_bounds[param][1] - self.state_bounds[param][0]
            distance = abs(current_value - optimal_value) / param_range

            total_distance += distance

        # Convert distance to reward: closer to optimal = higher reward
        avg_distance = total_distance / len(self.optimal_values)
        distance_reward = -avg_distance  # Ranges from 0 to -1

        # Calculate action cost penalty
        action_cost = 0.0
        if self.last_actions is not None:
            # L2 norm of actions (encourages minimal intervention)
            action_cost = np.sum(np.square(self.last_actions))  # Sum of squared actions
            action_cost = -action_cost * self.action_cost_weight  # Negative penalty

        # Total reward = distance reward + action cost
        total_reward = distance_reward + action_cost

        return total_reward

    def _is_terminated(self) -> bool:
        """Check if environment should terminate due to extreme conditions"""
        # Terminate if any critical parameter reaches dangerous levels
        critical_conditions = [
            self.state["dissolved_oxygen"] < 1.0,  # Fish kill level
            self.state["ammonia"] > 8.0,  # Toxic level
            self.state["pH"] < 4.5 or self.state["pH"] > 9.5,  # Extreme pH
            self.state["temperature"] > 38
            or self.state["temperature"] < 2,  # Extreme temp
        ]

        return False
        # return any(critical_conditions)

    def _get_normalized_state(self) -> np.ndarray:
        """Get current state normalized to [0, 1] range"""
        normalized = []
        for param, bounds in self.state_bounds.items():
            value = self.state[param]
            normalized_value = (value - bounds[0]) / (bounds[1] - bounds[0])
            normalized.append(normalized_value)
        return np.array(normalized, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state"""
        # Calculate distance from optimal for each parameter
        distances = {}
        for param, optimal_value in self.optimal_values.items():
            distances[param] = abs(self.state[param] - optimal_value)

        # Calculate action cost if actions were taken
        action_cost = 0.0
        if self.last_actions is not None:
            action_cost = np.sum(np.square(self.last_actions)) * self.action_cost_weight

        return {
            "raw_state": self.state.copy(),
            "optimal_values": self.optimal_values.copy(),
            "distances_from_optimal": distances,
            "last_actions": self.last_actions.copy()
            if self.last_actions is not None
            else None,
            "action_cost": action_cost,
            "action_cost_weight": self.action_cost_weight,
            "step": self.current_step,
        }

    def render(self, mode="human"):
        """Render the current state"""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print("Current Water Quality Parameters:")
            for param, value in self.state.items():
                optimal = self.optimal_values[param]
                distance = abs(value - optimal)
                param_name = param.replace("_", " ").title()

                if param == "temperature":
                    print(
                        f"  {param_name}: {value:.2f}°C (optimal: {optimal}°C, distance: {distance:.2f})"
                    )
                elif param == "salinity":
                    print(
                        f"  {param_name}: {value:.2f} ppt (optimal: {optimal} ppt, distance: {distance:.2f})"
                    )
                elif param == "turbidity":
                    print(
                        f"  {param_name}: {value:.2f} NTU (optimal: {optimal} NTU, distance: {distance:.2f})"
                    )
                elif param == "pH":
                    print(
                        f"  {param_name}: {value:.2f} (optimal: {optimal}, distance: {distance:.2f})"
                    )
                elif param == "dissolved_oxygen":
                    print(
                        f"  {param_name}: {value:.2f} mg/L (optimal: {optimal} mg/L, distance: {distance:.2f})"
                    )
                elif param == "ammonia":
                    print(
                        f"  {param_name} (NH3): {value:.2f} mg/L (optimal: {optimal} mg/L, distance: {distance:.2f})"
                    )
            print("-" * 60)


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = WaterEnv()

    # Test the environment
    print("Testing Pond Simulation Environment")
    print("=" * 50)

    # Reset environment
    obs, info = env.reset()
    print("Initial State:")
    env.render()

    # Run a few random actions
    for step in range(5):
        # Random actions between -1 and 1
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)

        print(f"\nAction taken: {actions}")
        print(f"Action magnitude (L2): {np.linalg.norm(actions):.3f}")
        print(f"Reward: {reward:.3f}")
        env.render()

        if terminated or truncated:
            print("Episode ended!")
            break

    print("\nEnvironment test completed!")

    # Print action and observation space info
    print(f"\nAction Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"State bounds: {env.state_bounds}")
    print(f"Action cost weight: {env.action_cost_weight}")

    print("\n" + "=" * 60)
    print("REWARD STRUCTURE EXPLANATION")
    print("=" * 60)
    print("""
Reward = Distance_Reward + Action_Cost

Distance_Reward:
- Ranges from 0 (at optimal values) to -1 (maximum distance)
- Based on normalized distance from optimal values for all parameters

Action_Cost:
- Penalty for taking large actions: -action_cost_weight * sum(action²)
- Encourages minimal intervention when state is near optimal
- Default weight: 0.1 (adjustable in constructor)

Total Reward:
- At optimal state with zero actions: 0 (best possible)
- At optimal state with max actions: 0 - 0.1*6 = -0.6
- At worst state with zero actions: -1
- At worst state with max actions: -1 - 0.6 = -1.6

This encourages the agent to:
1. Get state parameters to optimal values
2. Minimize intervention when already optimal
3. Learn energy-efficient control policies
    """)

    print("\n" + "=" * 60)
    print("TD3 TRAINING EXAMPLE")
    print("=" * 60)
    print("""
# Example TD3 training code (requires stable-baselines3):
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

# Create the environment
env = PondSimulationEnv(max_steps=500)

# Add action noise for exploration (important for TD3)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Initialize TD3 agent
model = TD3(
    "MlpPolicy", 
    env, 
    action_noise=action_noise,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=100000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5
)

# Train the agent
print("Training TD3 agent...")
model.learn(total_timesteps=50000)

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
    """)

    print("\nKey Features for TD3:")
    print("- Continuous action space [-1, 1] for each parameter")
    print("- Smooth reward function based on distance from optimal values")
    print("- Rich state interactions provide complex dynamics to learn")
    print("- Reward ranges from 0 (perfect) to -1 (worst case)")
    print("- Natural stabilization prevents runaway states")
