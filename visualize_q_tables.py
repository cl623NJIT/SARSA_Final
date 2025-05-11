import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Import updated sarsa_attacker and AttackEnv
from SARSA_DoS import sarsa_attacker, AttackEnv
from SARSA_Version1 import sarsa, DefenseEnv

try:
    from util import plot_return, actions_of_tabular_q
except ImportError:
    print("Warning: util.py not found. Plotting returns and actions might fail.")
    # Define dummy functions if util.py is missing
    def plot_return(return_list, save_plot_name, total_reward):
        print(f"Dummy plot_return called for {save_plot_name}. Avg reward: {np.mean(return_list):.2f}")
        print("Install matplotlib and ensure util.py is present for actual plots.")
    def actions_of_tabular_q(tabular_q):
        print("Dummy actions_of_tabular_q called.")
        if tabular_q is None: return
        n_states, n_actions = tabular_q.shape
        optimal_policy = np.argmax(tabular_q, axis=1)
        policy_subset_indices = np.linspace(0, n_states - 1, min(15, n_states), dtype=int)
        for state_bin in policy_subset_indices:
            action = optimal_policy[state_bin]
            print(f"  State Bin {state_bin:>2}: Action {action}")
        print("Ensure util.py is present to print policy actions properly.")

def plot_q_table(q_table, title, filename):
    """ Plot a heatmap of the Q-table """
    if q_table is None:
        print(f"Skipping plot for {title} as Q-table is None.")
        return
    try:
        plt.figure(figsize=(10, 8))
        n_states, n_actions = q_table.shape
        x_labels = [f'Action {i}' for i in range(n_actions)]
        step = max(1, n_states // 20)
        y_ticks = np.arange(0, n_states, step) + 0.5
        y_labels = [f'StateBin {i}' for i in range(0, n_states, step)]

        sns.heatmap(q_table, annot=False, fmt='.2f', cmap='viridis', linewidths=.5)
        plt.title(title, fontsize=14)
        plt.xlabel('Actions', fontsize=12)
        plt.ylabel('States (Bins)', fontsize=12)
        plt.xticks(np.arange(n_actions) + 0.5, labels=x_labels, rotation=0)
        plt.yticks(y_ticks, labels=y_labels, rotation=0)
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved Q-table plot to {filename}")
        plt.close()
    except Exception as e:
        print(f"Error plotting Q-table '{title}': {e}")

# (Keep imports and other functions like plot_q_table the same)

def main():
    # --- Parameters ---
    P_BAR = 1.0
    MAX_P_EXPECTED = 10.0 # Shared default for max_P scaling
    ERROR_INCREASE = 1.0

    # Defender parameters
    DEF_COST_C = 0.5
    DEF_ALPHA = 0.1; DEF_GAMMA = 0.9
    DEF_INITIAL_EPSILON = 1.0; DEF_MIN_EPSILON = 0.01; DEF_EPSILON_DECAY_RATE = 0.999
    DEF_NUM_EPISODES = 3000; DEF_NUM_BINS = 50
    def_burst_length = 15; def_no_attack_length = 5

    # Attacker parameters
    ATT_COST_C = 1.0
    ATT_ALPHA = 0.1; ATT_GAMMA = 0.9
    ATT_INITIAL_EPSILON = 1.0; ATT_MIN_EPSILON = 0.01; ATT_EPSILON_DECAY_RATE = 0.995
    ATT_NUM_EPISODES = 3000; ATT_NUM_BINS = 20
    ATT_MAX_P_EXPECTED = 10.0 # Can be same or different from defender's view
    # Parameters for the dynamic defender inside AttackEnv
    ATT_MAX_CONS_DEF = 1      # Max consecutive defenses allowed for the dynamic defender
    ATT_MAX_EP_STEPS = 100    # Max steps per episode for attacker training env

    # --- End Parameters ---

    # --- Setup Attack Sequence for Defender Training ---
    def_pattern = [1] * def_burst_length + [0] * def_no_attack_length
    def_repeats = (DEF_NUM_EPISODES // len(def_pattern)) + 2
    attack_sequence_for_def = def_pattern * def_repeats
    print(f"Defender trains against: {def_burst_length} attacks / {def_no_attack_length} no-attacks pattern.")

    # --- Create Environments ---
    try:
        # Defender Environment (still uses attack sequence)
        defense_env = DefenseEnv(
            attack_sequence=attack_sequence_for_def, # Pass sequence here
            P_bar=P_BAR, cost_c=DEF_COST_C,
            max_P=MAX_P_EXPECTED, error_increase=ERROR_INCREASE
        )

        # Attacker Environment (uses dynamic defender logic, NO sequence needed)
        attack_env = AttackEnv(
            # NO defense_sequence here anymore
            P_bar=P_BAR,
            cost_c_att=ATT_COST_C,
            max_P=ATT_MAX_P_EXPECTED, # Use attacker specific max_P
            error_increase=ERROR_INCREASE,
            max_consecutive_defenses=ATT_MAX_CONS_DEF, # Add missing arg
            max_episode_steps=ATT_MAX_EP_STEPS      # Add missing arg
        )
        print(f"Attacker trains against: Dynamic defender (Max Cons. Defenses: {ATT_MAX_CONS_DEF}).")

    except Exception as e:
        print(f"Error initializing environments: {e}")
        return

    # --- Train Defender ---
    # (Defender training code remains the same)
    print("\n--- Training Defender ---")
    defender_q_table, defender_returns, defender_max_p = None, None, None
    try:
        defender_q_table, defender_returns, defender_max_p = sarsa(
            defense_env, DEF_ALPHA, DEF_GAMMA, DEF_INITIAL_EPSILON,
            DEF_MIN_EPSILON, DEF_EPSILON_DECAY_RATE, DEF_NUM_EPISODES, DEF_NUM_BINS
        )
        print(f"Defender training complete. Max P observed: {defender_max_p:.4f}")
    except Exception as e:
        print(f"Error during defender training: {e}")


    # --- Train Attacker ---
    # (Attacker training code remains the same, uses the dynamic attack_env)
    print("\n--- Training Attacker ---")
    attacker_q_table = None
    attacker_returns = None
    attacker_max_p = None
    try:
        attacker_q_table, attacker_returns, attacker_max_p = sarsa_attacker(
            attack_env, ATT_ALPHA, ATT_GAMMA, ATT_INITIAL_EPSILON, ATT_MIN_EPSILON,
            ATT_EPSILON_DECAY_RATE, ATT_NUM_EPISODES, ATT_NUM_BINS
        )
        print(f"Attacker training complete (Bins: {ATT_NUM_BINS}). Max P observed: {attacker_max_p:.4f}")
    except Exception as e:
        print(f"Error during attacker training: {e}")


    # --- Plotting and Policy Printing ---
    # (Remains the same as the previous version)
    print("\n--- Plotting Results ---")
    plot_q_table(defender_q_table, f"Defender Q-table ({DEF_NUM_BINS} bins, {DEF_NUM_EPISODES} eps)", "defender_q_table.png")
    plot_q_table(attacker_q_table, f"Attacker Q-table ({ATT_NUM_BINS} bins, {ATT_NUM_EPISODES} eps)", "attacker_q_table.png")

    print("\n--- Plotting Returns ---")
    if defender_returns:
        avg_defender_reward = np.mean(defender_returns)
        plot_return(defender_returns, "defender_returns", avg_defender_reward)
    else:
        print("Skipping defender returns plot (no data).")

    if attacker_returns:
        avg_attacker_reward = np.mean(attacker_returns)
        plot_return(attacker_returns, "attacker_returns", avg_attacker_reward)
    else:
        print("Skipping attacker returns plot (no data).")

    print("\n--- Optimal Policies ---")
    if defender_q_table is not None:
        print("\nDefender Optimal Policy (Sampled Bins):")
        actions_of_tabular_q(defender_q_table)

    if attacker_q_table is not None:
        print("\nAttacker Optimal Policy (Sampled Bins):")
        actions_of_tabular_q(attacker_q_table)

if __name__ == "__main__":
    main()