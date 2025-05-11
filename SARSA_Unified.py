import gymnasium as gym
import numpy as np
import math
import logging
import matplotlib.pyplot as plt

# Set up logging to file
logging.basicConfig(filename='training_log.txt', filemode='w', level=logging.INFO, format='%(message)s')

# --- Helper function for state discretization ---
def discretize_state(state_value, num_bins, max_state_value):
    """Discretizes a continuous state value into a bin index."""
    state_value = float(state_value)
    if state_value >= max_state_value:
        return num_bins - 1
    if state_value < 0:
        return 0
    bin_index = int((state_value / max_state_value) * num_bins)
    return max(0, min(bin_index, num_bins - 1))

class CPSEnv(gym.Env):
    def __init__(self, P_bar=0, c_attack=0.25, c_defense=1.0, max_P=30.0, error_increase=1, attack_reward_threshold=0.001):
        super(CPSEnv, self).__init__()
        
        # Define action spaces for both attacker and defender
        self.attacker_action_space = gym.spaces.Discrete(2)  # 0: no attack, 1: attack
        self.defender_action_space = gym.spaces.Discrete(2)  # 0: no defense, 1: defense
        
        # Define observation space: Error covariance (single value)
        self.observation_space = gym.spaces.Box(low=0, high=max_P, shape=(1,), dtype=np.float32)
        
        # System parameters
        self.P_bar = float(P_bar)  # Steady-state error covariance
        self.c_attack = float(c_attack)  # Cost of attacking
        self.c_defense = float(c_defense)  # Cost of defending
        self.max_P = float(max_P)  # Maximum error covariance
        self.error_increase = float(error_increase)  # How much an attack increases error
        self.attack_reward_threshold = attack_reward_threshold
        
        self.current_step = 0
        self.max_steps = 100  # Maximum number of steps per episode
        self.consecutive_attacks = 0
        self.non_attack_steps = 0
        self.reset_attack_reset_threshold = 2  # Number of state 0 actions to reset
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Start with a random value between 0 and, say, max_P/2
        initial_cov = np.random.uniform(0, self.max_P / 2)
        self.state = np.array([initial_cov], dtype=np.float32)
        self.consecutive_attacks = 0
        self.non_attack_steps = 0
        return self.state, {}
    
    def h(self, P):
        # Logarithmic increase with consecutive attacks
        increase = self.error_increase * np.log1p(self.consecutive_attacks)
        return np.clip(P + increase, 0, self.max_P)
    
    def step(self, attacker_action, defender_action):
        # Update state based on both actions
        if attacker_action == 1 and defender_action == 0:
            self.consecutive_attacks += 1
            self.non_attack_steps = 0
            next_state = self.h(self.state)
        else:
            self.non_attack_steps += 1
            if self.non_attack_steps >= self.reset_attack_reset_threshold:
                self.consecutive_attacks = 0
            next_state = np.array([self.P_bar], dtype=np.float32)
            
        # Non-linear cost: quadratic in consecutive attacks
        nonlinear_attack_cost = self.c_attack * (self.consecutive_attacks ** 1.01) if attacker_action == 1 else 0

        # Calculate rewards for both agents
        # Attacker only gets reward if error covariance is above threshold
        if float(next_state[0]) > self.attack_reward_threshold:
            attacker_reward = float(next_state[0]) - nonlinear_attack_cost
        else:
            attacker_reward = -nonlinear_attack_cost  # Only cost, no reward
        
        # Defender wants to minimize error covariance but has defense cost
        defender_reward = -float(next_state[0]) - self.c_defense * defender_action
        
        # Update step counter
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        self.state = next_state
        return self.state, (attacker_reward, defender_reward), terminated, truncated, {}

def train_agents(env, alpha_attacker, alpha_defender, gamma, initial_epsilon, min_epsilon, 
                 epsilon_decay_rate, num_episodes, num_bins, verbose=False):
    # Initialize Q-tables for both agents
    Q_attacker = np.zeros((num_bins, env.attacker_action_space.n))
    Q_defender = np.zeros((num_bins, env.defender_action_space.n))
    
    epsilon = initial_epsilon
    attacker_rewards = []
    defender_rewards = []
    max_p_observed = 0.0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        current_p = float(state[0])
        max_p_observed = max(max_p_observed, current_p)
        state_idx = discretize_state(current_p, num_bins, env.max_P)
        
        # Choose initial actions (epsilon-greedy)
        if np.random.rand() < epsilon:
            attacker_action = env.attacker_action_space.sample()
        else:
            attacker_action = np.argmax(Q_attacker[state_idx])
            
        if np.random.rand() < epsilon:
            defender_action = env.defender_action_space.sample()
        else:
            defender_action = np.argmax(Q_defender[state_idx])
        
        terminated = False
        truncated = False
        total_attacker_reward = 0
        total_defender_reward = 0
        
        if verbose and episode % 1000 == 0:
            logging.info(f"Episode {episode}:")
        while not terminated and not truncated:
            next_state, (attacker_reward, defender_reward), terminated, truncated, _ = env.step(
                attacker_action, defender_action
            )
            
            next_p = float(next_state[0])
            max_p_observed = max(max_p_observed, next_p)
            next_state_idx = discretize_state(next_p, num_bins, env.max_P)
            
            # Choose next actions (epsilon-greedy)
            if np.random.rand() < epsilon:
                next_attacker_action = env.attacker_action_space.sample()
            else:
                next_attacker_action = np.argmax(Q_attacker[next_state_idx])
                
            if np.random.rand() < epsilon:
                next_defender_action = env.defender_action_space.sample()
            else:
                next_defender_action = np.argmax(Q_defender[next_state_idx])
            
            # Update Q-tables
            Q_attacker[state_idx, attacker_action] = Q_attacker[state_idx, attacker_action] + alpha_attacker * (
                attacker_reward + gamma * Q_attacker[next_state_idx, next_attacker_action] - Q_attacker[state_idx, attacker_action]
            )
            
            Q_defender[state_idx, defender_action] = Q_defender[state_idx, defender_action] + alpha_defender * (
                defender_reward + gamma * Q_defender[next_state_idx, next_defender_action] - Q_defender[state_idx, defender_action]
            )
            
            state_idx = next_state_idx
            attacker_action = next_attacker_action
            defender_action = next_defender_action
            
            total_attacker_reward += attacker_reward
            total_defender_reward += defender_reward
            
            if verbose and episode % 1000 == 0:
                logging.info(f"  Step {env.current_step}: State={state_idx}, AttackerAction={attacker_action}, DefenderAction={defender_action}, "
                              f"AttackerReward={attacker_reward:.2f}, DefenderReward={defender_reward:.2f}, "
                              f"ConsecAttacks={env.consecutive_attacks}, NonAttackSteps={env.non_attack_steps}")
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)
        attacker_rewards.append(total_attacker_reward)
        defender_rewards.append(total_defender_reward)
    
    return Q_attacker, Q_defender, attacker_rewards, defender_rewards, max_p_observed

def evaluate_policies(Q_attacker, Q_defender, num_episodes=10, num_bins=30, max_P=10.0, verbose=False):
    env = CPSEnv()
    attacker_wins = 0
    defender_wins = 0
    draws = 0
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        state_idx = discretize_state(float(state[0]), num_bins, max_P)
        total_attacker_reward = 0
        total_defender_reward = 0
        
        terminated = False
        truncated = False
        
        if verbose:
            logging.info(f"Eval Episode {ep}:")
        while not terminated and not truncated:
            # Choose actions based on learned policies
            attacker_action = np.argmax(Q_attacker[state_idx])
            defender_action = np.argmax(Q_defender[state_idx])
            
            next_state, (attacker_reward, defender_reward), terminated, truncated, _ = env.step(
                attacker_action, defender_action
            )
            
            total_attacker_reward += attacker_reward
            total_defender_reward += defender_reward
            state_idx = discretize_state(float(next_state[0]), num_bins, max_P)
            
            if verbose:
                logging.info(f"  Step {env.current_step}: State={state_idx}, AttackerAction={attacker_action}, DefenderAction={defender_action}, "
                              f"AttackerReward={attacker_reward:.2f}, DefenderReward={defender_reward:.2f}, "
                              f"ConsecAttacks={env.consecutive_attacks}, NonAttackSteps={env.non_attack_steps}")
        
        if total_attacker_reward > total_defender_reward:
            attacker_wins += 1
        elif total_defender_reward > total_attacker_reward:
            defender_wins += 1
        else:
            draws += 1
        
        if verbose:
            logging.info(f"  Total Attacker Reward: {total_attacker_reward:.2f}, Total Defender Reward: {total_defender_reward:.2f}")
    
    return attacker_wins, defender_wins, draws

# Example usage
if __name__ == "__main__":
    # Initialize environment and hyperparameters
    P_BAR = -0.5  # Default state rewards defender
    COST_C_ATT = 0.5
    COST_C_DEF = 1.0
    MAX_P = 10.0
    ERROR_INCREASE = 0.5
    ATTACK_REWARD_THRESHOLD = 1.0
    
    env = CPSEnv(
        P_bar=P_BAR,
        c_attack=COST_C_ATT,
        c_defense=COST_C_DEF,
        max_P=MAX_P,
        error_increase=ERROR_INCREASE,
        attack_reward_threshold=ATTACK_REWARD_THRESHOLD
    )
    
    # Training parameters
    ALPHA_ATT = 0.1
    ALPHA_DEF = 0.1
    GAMMA = 0.9
    INITIAL_EPSILON = 1.0
    MIN_EPSILON = 0.01
    EPSILON_DECAY_RATE = 0.9995
    NUM_EPISODES = 1000
    NUM_BINS = 30
    
    print(f"Starting training with parameters:")
    print(f"P_bar: {P_BAR}, Cost_attack: {COST_C_ATT}, Cost_defense: {COST_C_DEF}")
    print(f"Max_P: {MAX_P}, Error_increase: {ERROR_INCREASE}")
    print(f"Learning rates: Attacker={ALPHA_ATT}, Defender={ALPHA_DEF}")
    print(f"Epsilon: {INITIAL_EPSILON}->{MIN_EPSILON} (decay: {EPSILON_DECAY_RATE})")
    print(f"Episodes: {NUM_EPISODES}, State bins: {NUM_BINS}")
    
    # Train both agents
    Q_attacker, Q_defender, attacker_rewards, defender_rewards, max_p = train_agents(
        env, ALPHA_ATT, ALPHA_DEF, GAMMA, INITIAL_EPSILON, MIN_EPSILON,
        EPSILON_DECAY_RATE, NUM_EPISODES, NUM_BINS, verbose=True
    )
    
    # Print results
    print("\nTraining completed!")
    print(f"Number of episodes: {NUM_EPISODES}")
    print(f"Max P observed: {max_p:.4f}")
    print(f"Final epsilon: {max(MIN_EPSILON, INITIAL_EPSILON * (EPSILON_DECAY_RATE**NUM_EPISODES)):.4f}")
    
    # Print optimal policies
    attacker_policy = np.argmax(Q_attacker, axis=1)
    defender_policy = np.argmax(Q_defender, axis=1)
    
    print("\nAttacker's optimal policy (state bin: action):")
    for state, action in enumerate(attacker_policy):
        print(f"{state}: {'No attack' if action == 0 else 'Attack'}")
    
    print("\nDefender's optimal policy (state bin: action):")
    for state, action in enumerate(defender_policy):
        print(f"{state}: {'No defense' if action == 0 else 'Defend'}")
    
    # Evaluate policies
    attacker_wins, defender_wins, draws = evaluate_policies(Q_attacker, Q_defender, num_episodes=100, verbose=True)
    print(f"\nEvaluation results (100 episodes):")
    print(f"Attacker wins: {attacker_wins}")
    print(f"Defender wins: {defender_wins}")
    print(f"Draws: {draws}")
    
    # Optional: Plot rewards over time
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(attacker_rewards, label='Attacker Rewards')
        plt.plot(defender_rewards, label='Defender Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Rewards over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
        # Q-table heatmaps
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(Q_attacker, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Attacker Q-table')
        plt.xlabel('Action')
        plt.ylabel('State Bin')
        plt.subplot(1, 2, 2)
        plt.imshow(Q_defender, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Defender Q-table')
        plt.xlabel('Action')
        plt.ylabel('State Bin')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\nNote: matplotlib not available for plotting rewards or Q-tables") 