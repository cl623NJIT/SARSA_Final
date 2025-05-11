import gymnasium as gym
import numpy as np
import math

# --- Helper function for state discretization ---
# (Keep the same discretize_state function as before)
def discretize_state(state_value, num_bins, max_state_value):
    """Discretizes a continuous state value into a bin index."""
    state_value = float(state_value)
    if state_value >= max_state_value:
        return num_bins - 1
    if state_value < 0:
        return 0
    bin_index = int((state_value / max_state_value) * num_bins)
    return max(0, min(bin_index, num_bins - 1))

class AttackEnv(gym.Env):
    """
    Environment from Attacker's perspective, now with a DYNAMIC defender.
    Defender Logic:
    - Defends (action 1) on the step immediately following an attack (if possible).
    - Cannot defend for more than 'max_consecutive_defenses' steps in a row.
    - Defaults to not defending (action 0).
    """
    # Removed defense_sequence from init, added defender logic params
    def __init__(self, P_bar=1.0, cost_c_att=1.0, max_P=10.0, error_increase=1.0,
                 max_consecutive_defenses=1, max_episode_steps=100): # Added max steps
        super(AttackEnv, self).__init__()

        self.action_space = gym.spaces.Discrete(2) # 0: no attack, 1: attack
        self.observation_space = gym.spaces.Box(low=0, high=max_P, shape=(1,), dtype=np.float32)

        self.P_bar = float(P_bar)
        self.c = float(cost_c_att) # Attacker's cost of attacking
        self.max_P = float(max_P)
        self.error_increase = float(error_increase)
        self.max_consecutive_defenses = max_consecutive_defenses # Defender constraint
        self._max_episode_steps = max_episode_steps # Max steps per episode

        # Internal state for defender logic
        self.attacker_attacked_last_step = False
        self.consecutive_defenses_count = 0
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.array([self.P_bar], dtype=np.float32)
        # Reset defender state variables
        self.attacker_attacked_last_step = False
        self.consecutive_defenses_count = 0
        return self.state, {}

    def h(self, P):
        # Simplified Lyapunov operator for attack impact
        return np.clip(P + self.error_increase, 0, self.max_P + self.error_increase)

    def step(self, attacker_action):
        """ Simulates one step with dynamic defender logic """
        if self.current_step >= self._max_episode_steps:
             terminated = False # Time limit exceeded = truncated
             truncated = True
             return self.state, 0.0, terminated, truncated, {}

        current_p = float(self.state[0])

        # --- Determine Defender's Action Dynamically ---
        defender_action = 0 # Default: no defense
        if self.attacker_attacked_last_step:
            # Attacker attacked last step, defender *tries* to defend
            if self.consecutive_defenses_count < self.max_consecutive_defenses:
                defender_action = 1 # Defend!
                self.consecutive_defenses_count += 1
            else:
                # Cannot defend due to cost/limit
                defender_action = 0
                self.consecutive_defenses_count = 0 # Reset counter
        else:
            # Attacker did not attack last step, defender doesn't need to react
            defender_action = 0
            self.consecutive_defenses_count = 0 # Reset counter

        # Update for next step's logic: did the attacker attack *this* step?
        self.attacker_attacked_last_step = (attacker_action == 1)
        # --- End Defender Logic ---


        # --- Determine State Transition based on BOTH actions ---
        if attacker_action == 1 and defender_action == 0:
            # Successful attack! Defender was vulnerable.
            next_p = self.h(current_p)
        else:
            # Failed attack (defender defended) or no attack performed
            next_p = self.P_bar

        next_state = np.array([next_p], dtype=np.float32)

        # --- Calculate Reward for Attacker ---
        # Attacker wants to MAXIMIZE this. High P is good, cost 'c' is bad.
        reward = next_p - self.c * attacker_action

        self.current_step += 1
        terminated = False # No explicit termination condition other than time limit
        truncated = self.current_step >= self._max_episode_steps

        self.state = next_state
        return self.state, reward, terminated, truncated, {}

# --- sarsa_attacker function remains the same as the last version ---
# (Includes epsilon decay, returns Q, rewards, max_p)
def sarsa_attacker(env, alpha, gamma, initial_epsilon, min_epsilon,
                   epsilon_decay_rate, num_episodes, num_bins):
    """
    Implements SARSA for the attacker with discretization and epsilon decay.
    (Code is identical to the previous version you have)
    """
    Q = np.zeros((num_bins, env.action_space.n))
    epsilon = initial_epsilon
    episode_rewards = []
    max_p_observed = 0.0

    for episode in range(num_episodes):
        state, _ = env.reset()
        current_p = float(state[0])
        max_p_observed = max(max_p_observed, current_p)
        state_idx = discretize_state(current_p, num_bins, env.max_P)

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            best_actions = np.flatnonzero(Q[state_idx] == Q[state_idx].max())
            action = np.random.choice(best_actions)

        terminated = False
        truncated = False
        total_episode_reward = 0

        # Use truncated flag from the new environment
        while not terminated and not truncated:
            # Environment step now includes dynamic defender logic
            next_state, reward, terminated, truncated, _ = env.step(action)

            next_p_value = float(next_state[0])
            max_p_observed = max(max_p_observed, next_p_value)
            next_state_idx = discretize_state(next_p_value, num_bins, env.max_P)

            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                best_actions = np.flatnonzero(Q[next_state_idx] == Q[next_state_idx].max())
                next_action = np.random.choice(best_actions)

            Q[state_idx, action] = Q[state_idx, action] + alpha * (
                reward + gamma * Q[next_state_idx, next_action] - Q[state_idx, action]
            )

            state_idx = next_state_idx
            action = next_action
            total_episode_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)
        episode_rewards.append(total_episode_reward)

    return Q, episode_rewards, max_p_observed


# --- Example Usage ---
if __name__ == "__main__":
    # Parameters for the Environment and Attacker Training
    P_BAR_ATT = 1.0
    COST_C_ATT = 1.0          # Attacker's cost to attack
    MAX_P_EXPECTED_ATT = 15.0 # Might need higher P if defender logic changes things
    ERROR_INCREASE_ATT = 1.0
    MAX_CONS_DEF = 1          # Defender constraint: Can defend only 1 step after being attacked
    MAX_EP_STEPS = 100        # Max steps per training episode

    ATT_ALPHA = 0.1
    ATT_GAMMA = 0.9
    ATT_INITIAL_EPSILON = 1.0
    ATT_MIN_EPSILON = 0.01
    ATT_EPSILON_DECAY_RATE = 0.999 # Slower decay for potentially more complex learning
    ATT_NUM_EPISODES = 20000      # Increase episodes maybe
    ATT_NUM_BINS = 30             # Bins for attacker state space

    # Initialize the NEW environment with dynamic defender
    env = AttackEnv(
        P_bar=P_BAR_ATT, cost_c_att=COST_C_ATT, max_P=MAX_P_EXPECTED_ATT,
        error_increase=ERROR_INCREASE_ATT,
        max_consecutive_defenses=MAX_CONS_DEF, # Pass defender constraint
        max_episode_steps=MAX_EP_STEPS
    )

    # Train the attacker
    print(f"Starting Attacker Training vs Dynamic Defender (Max Cons. Defenses: {MAX_CONS_DEF})...")
    Q_table, rewards, max_p = sarsa_attacker(
        env, ATT_ALPHA, ATT_GAMMA, ATT_INITIAL_EPSILON, ATT_MIN_EPSILON,
        ATT_EPSILON_DECAY_RATE, ATT_NUM_EPISODES, ATT_NUM_BINS
    )

    print("\nAttacker Training completed!")
    print(f"Number of episodes: {ATT_NUM_EPISODES}, Bins: {ATT_NUM_BINS}")
    print(f"Final Epsilon: {max(ATT_MIN_EPSILON, ATT_INITIAL_EPSILON * (ATT_EPSILON_DECAY_RATE**ATT_NUM_EPISODES)):.4f}")
    print(f"Max P Observed: {max_p:.4f}")

    # --- Analyze results ---
    optimal_policy = np.argmax(Q_table, axis=1)
    print("\nOptimal attack policy for each state bin:")
    policy_subset_indices = np.linspace(0, ATT_NUM_BINS - 1, min(15, ATT_NUM_BINS), dtype=int)
    for state_bin in policy_subset_indices:
        action = optimal_policy[state_bin]
        print(f"  State Bin {state_bin:>2}: {'No attack' if action == 0 else 'Attack'}")

    # Add plotting if desired
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 5))
    # plt.plot(rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward per Episode")
    # plt.title(f"Attacker Rewards vs Dynamic Defender (MaxConsDef={MAX_CONS_DEF})")
    # plt.grid(True)
    # plt.show()