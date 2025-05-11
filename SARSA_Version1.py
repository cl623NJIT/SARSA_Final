import gymnasium as gym
import numpy as np
import math # Added for epsilon decay

class DefenseEnv(gym.Env):
    """
    Environment for the defender agent.
    """
    # Suggestion 3: Modify h(P) for potentially larger error increase
    def __init__(self, attack_sequence, P_bar=1.0, cost_c=1.0, max_P=10.0, error_increase=1.0):
        super(DefenseEnv, self).__init__()

        # Define action space for defender only: 0 (state 0) or 1 (state 1)
        self.action_space = gym.spaces.Discrete(2)

        # Define observation space: Error covariance (single value)
        # Use max_P for the upper bound
        self.observation_space = gym.spaces.Box(low=0, high=max_P, shape=(1,), dtype=np.float32)

        # Store attack sequence and parameters
        self.attack_sequence = attack_sequence
        self.P_bar = float(P_bar)  # Steady-state error covariance
        self.c = float(cost_c)      # Cost of choosing state 1 (secure channel)
        self.max_P = float(max_P) # Maximum expected P value for state discretization scaling
        self.error_increase = float(error_increase) # How much P increases on successful attack

        self.current_step = 0
        self.max_steps = len(attack_sequence)  # Length of attack sequence

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for newer gym versions
        self.current_step = 0
        self.state = np.array([self.P_bar], dtype=np.float32)  # Initial state
        return self.state, {}

    def h(self, P):
        # Simplified Lyapunov operator - Using the error_increase parameter
        # Ensure P is treated as a scalar float for calculation
        # Clip the result to avoid exceeding max_P excessively in one step
        return np.clip(P + self.error_increase, 0, self.max_P + self.error_increase) # Allow overshoot for index calc

    def step(self, defender_action):
        if self.current_step >= self.max_steps:
             # This should not happen if terminated is handled correctly, but as a safeguard
            # print("Warning: Exceeded max steps!") # Optional warning
            terminated = True
            truncated = False
            # Return state as is, zero reward, flags, empty info
            return self.state, 0.0, terminated, truncated, {}


        attacker_action = self.attack_sequence[self.current_step]
        current_p = float(self.state[0]) # Ensure it's a scalar float

        # Update state based on both actions
        if defender_action == 0 and attacker_action == 1:
            # Packet loss due to successful attack
            next_p = self.h(current_p)
        else:
            # Successful transmission or no attack or defender chose secure channel
            next_p = self.P_bar

        next_state = np.array([next_p], dtype=np.float32)

        # Calculate reward: -trace(P) - c*defender_action
        # Goal is to minimize P (maximize -P) and minimize cost c*action
        reward = -next_p - self.c * defender_action

        # Update step counter
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False # We don't have a separate truncation condition here

        self.state = next_state
        return self.state, reward, terminated, truncated, {}

# --- Helper function for state discretization ---
def discretize_state(state_value, num_bins, max_state_value):
    """Discretizes a continuous state value into a bin index."""
    # Ensure state_value is float
    state_value = float(state_value)
    # Avoid edge case issues with max value
    if state_value >= max_state_value:
        return num_bins - 1
     # Handle potential negative values if environment allows (though unlikely here)
    if state_value < 0:
        return 0
    # Calculate bin index
    bin_index = int((state_value / max_state_value) * num_bins)
     # Clip just in case something unexpected happens
    return max(0, min(bin_index, num_bins - 1))


def sarsa(env, alpha, gamma, initial_epsilon, min_epsilon, epsilon_decay_rate, num_episodes, num_bins):
    """
    Implements the SARSA algorithm with modifications.

    Args:
        env: The Gymnasium environment.
        alpha: Learning rate.
        gamma: Discount factor.
        initial_epsilon: Starting exploration rate.
        min_epsilon: Minimum exploration rate.
        epsilon_decay_rate: Rate at which epsilon decays exponentially.
        num_episodes: Number of training episodes.
        num_bins: Number of bins for state discretization.

    Returns:
        Q: The learned Q-table.
        episode_rewards: List of total rewards for each episode.
        max_p_observed: Maximum P value observed during training.
    """
    # Initialize Q-table: [state_bin, action]
    Q = np.zeros((num_bins, env.action_space.n))
    epsilon = initial_epsilon # Initialize epsilon

    # Track rewards per episode for analysis (optional)
    episode_rewards = []
    # Suggestion 4: Track max P observed
    max_p_observed = 0.0

    for episode in range(num_episodes):
        state, _ = env.reset()
        # Suggestion 4: Update max_p_observed at the start of episode too
        current_p = float(state[0])
        max_p_observed = max(max_p_observed, current_p)
        # Discretize the initial state
        state_idx = discretize_state(current_p, num_bins, env.max_P)

        # Choose initial action (epsilon-greedy)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
             # Handle potential ties in Q-values, especially early on
            best_actions = np.flatnonzero(Q[state_idx] == Q[state_idx].max())
            action = np.random.choice(best_actions)


        terminated = False
        truncated = False
        total_episode_reward = 0

        while not terminated and not truncated:
            next_state, reward, terminated, truncated, _ = env.step(action)
            # Suggestion 4: Update max_p_observed inside the step loop
            next_p_value = float(next_state[0])
            max_p_observed = max(max_p_observed, next_p_value)
            # Discretize the next state
            next_state_idx = discretize_state(next_p_value, num_bins, env.max_P)

            # Choose next action (epsilon-greedy)
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                # Handle potential ties in Q-values
                best_actions = np.flatnonzero(Q[next_state_idx] == Q[next_state_idx].max())
                next_action = np.random.choice(best_actions)

            # Update Q-table using SARSA rule
            Q[state_idx, action] = Q[state_idx, action] + alpha * (
                reward + gamma * Q[next_state_idx, next_action] - Q[state_idx, action]
            )

            # Move to the next state and action
            state_idx = next_state_idx
            action = next_action
            total_episode_reward += reward # Track reward

        # Epsilon Decay (after each episode)
        epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)

        episode_rewards.append(total_episode_reward)
        if (episode + 1) % 1000 == 0: # Print progress less frequently for longer runs
             print(f"Episode {episode + 1}/{num_episodes} finished. Epsilon: {epsilon:.4f}. Max P so far: {max_p_observed:.2f}")


    print(f"\nFinal Epsilon: {epsilon}")
    return Q, episode_rewards, max_p_observed # Return max_p too

# Example usage
if __name__ == "__main__":

    # --- Tunable Parameters ---
    P_BAR = 1.0       # Steady-state error covariance
    COST_C = 0.5      # Cost for secure channel (action 1) (Experiment: 1.0, 0.5, 0.2)
    MAX_P_EXPECTED = 10.0 # Estimate max P for discretization scaling (Increase if max_p_observed exceeds this)
    ERROR_INCREASE = 1.0 # Size of error increase on successful attack (Experiment: 0.5, 1.0, 1.5)

    ALPHA = 0.1                # Learning rate (Experiment: 0.05, 0.1, 0.2)
    GAMMA = 0.9                # Discount factor
    INITIAL_EPSILON = 1.0      # Start with high exploration
    MIN_EPSILON = 0.01         # Minimum exploration rate
    EPSILON_DECAY_RATE = 0.999 # Slower decay for more episodes (adjust: 0.995, 0.999)
    NUM_EPISODES = 20000       # Number of episodes (Experiment: 10000, 20000, 50000)
    NUM_BINS = 50              # Number of state bins (Experiment: 20, 50, 100)

    # Parameters for Attack Sequence Generation
    burst_length = 15  # Number of consecutive attacks (try 10, 15, 20)
    no_attack_length = 5 # Period with no attacks to allow occasional resets
    # --- End Tunable Parameters ---


    # --- Define Attack Sequence ---
    # Strategy: Create bursts of consecutive attacks to force P higher
    pattern = [1] * burst_length + [0] * no_attack_length

    # Make the sequence long enough for the entire training run
    # Estimate steps needed. Max steps per episode is length of attack_sequence in DefenseEnv
    # Let's ensure the sequence is longer than num_episodes * pattern length
    # This heuristic might overestimate, but ensures coverage.
    repeats_needed = (NUM_EPISODES // len(pattern)) + 2 # Add buffer
    attack_sequence = pattern * repeats_needed
    print(f"Generated attack sequence pattern: {burst_length} attacks, {no_attack_length} no-attacks. Total length: {len(attack_sequence)}")
    # --- End Attack Sequence Definition ---


    # Initialize environment
    env = DefenseEnv(
        attack_sequence,
        P_bar=P_BAR,
        cost_c=COST_C,
        max_P=MAX_P_EXPECTED,
        error_increase=ERROR_INCREASE
    )

    # Train the agent
    print("Starting SARSA training...")
    Q_table, rewards, max_p = sarsa(env, ALPHA, GAMMA, INITIAL_EPSILON, MIN_EPSILON, EPSILON_DECAY_RATE, NUM_EPISODES, NUM_BINS)
    print("Training completed!")

    # Print results with better formatting
    print(f"\nHyperparameters:")
    print(f"  P_bar: {P_BAR}, Cost C: {COST_C}, Max P Expected: {MAX_P_EXPECTED}, Error Increase: {ERROR_INCREASE}")
    print(f"  Alpha: {ALPHA}, Gamma: {GAMMA}, Initial Epsilon: {INITIAL_EPSILON}")
    print(f"  Min Epsilon: {MIN_EPSILON}, Decay Rate: {EPSILON_DECAY_RATE}")
    print(f"  Num Episodes: {NUM_EPISODES}, Num Bins: {NUM_BINS}")
    print(f"  Attack Pattern: {burst_length} attacks / {no_attack_length} no-attacks")
    print(f"\nMaximum P value observed during training: {max_p:.4f}")


    # Optional: Print the optimal policy
    optimal_policy = np.argmax(Q_table, axis=1)
    print("\nOptimal policy for each state bin:")
    # Only print policy for a few representative bins if too large
    policy_subset_indices = np.linspace(0, NUM_BINS - 1, min(15, NUM_BINS), dtype=int) # Show more points
    policy_changed = False
    last_action = -1
    for state_bin in policy_subset_indices:
        action = optimal_policy[state_bin]
        action_text = 'Stay in state 0 (Channel 0)' if action == 0 else 'Switch to state 1 (Channel 1)'
        print(f"  State Bin {state_bin:>2}: {action_text}")
        if last_action != -1 and action != last_action:
            policy_changed = True
        last_action = action

    if not policy_changed and len(policy_subset_indices) > 1 :
         print("  (Policy did not change across sampled bins)")


    # You can add plotting here using matplotlib if needed:
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 5))
    # plt.plot(rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward per Episode")
    # plt.title(f"Episode Rewards (Cost C={COST_C}, Error Inc={ERROR_INCREASE})")
    # plt.grid(True)
    # plt.show()