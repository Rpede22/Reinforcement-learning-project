import numpy as np
import matplotlib.pyplot as plt
import random
import os

from Algorithm.DDQN import DDQN
from Algorithm.DDQN_MIN import DDQN_MIN
from Algorithm.Q_ENSEMBLE_MIN import Q_ENSEMBLE_MIN
from Algorithm.REDQ import REDQ
from Environment.mountainClimb  import MountainClimb
from Environment.branchingTree import BranchingTree
from Environment.slotMachineChain import SlotMachineChain

def test():
    state_sizes  = [10, 15, 20]
    num_runs     = 5
    num_episodes = 5000
    max_steps    = 5000
    bias_eval_interval = 100
    smoothing_window = 20
    window_size = 100
    
    environment = input(
        "Which environment?\n"
        "Options:\n"
        "  1. MountainClimb\n"
        "  2. BranchingTree\n"
        "  3. SlotMachineChain\n"
        "Enter 1, 2 or 3: "
    )
    agentchoice = input(
        "Which agent?\n"
        "Options:\n"
        "  1. DDQN\n"
        "  2. DDQN_MIN\n"
        "  3. Q_ENSEMBLE_MIN\n"
        "  4. REDQ\n"
        "Enter 1, 2, 3 or 4: "
    )
    if agentchoice in ["3", "4"]:
        ensemble = int(input(
            "How big ensemble size?\n "
            "Min 5\n"
            "Enter: "
            ))
    # Make directory for saving the results
    base_results_dir = "Results"
    env_subdirs = {"1": "MountainClimb", "2": "BranchingTree", "3": "SlotMachineChain"}
    results_dir = os.path.join(base_results_dir, env_subdirs.get(environment, "UnknownEnv"))
    os.makedirs(results_dir, exist_ok=True)

    # Making file names and titles for results
    env_names = {"1": "MountainClimb", "2": "BranchingTree", "3": "SlotMachineChain"}
    agent_names = {"1": "DDQN", "2": "DDQN_MIN", "3": "Q_ENSEMBLE_MIN", "4": "REDQ"}
    env_str = env_names.get(environment, "UnknownEnv")
    agent_str = agent_names.get(agentchoice, "UnknownAgent")

    fig1, ax1 = plt.subplots(figsize=(10, 6))  # for reward
    fig2, ax2 = plt.subplots(figsize=(10, 6))  # for bias

    for length in state_sizes:
        avg_returns = np.zeros(num_episodes)

        if environment == "1":
            env = MountainClimb(
                num_states=length, 
                p_up=0.75, 
                r_camp=0.1, 
                r_goal=1000.0)
        elif environment == "2":
            env = BranchingTree(
                depth=length, 
                mu_leaf=450.0, 
                sigma_leaf=550.0, 
                r_stay=0.01, 
                p_move=0.9)
        elif environment == "3":
            env = SlotMachineChain(
                num_states=length, 
                r_work=0.1, 
                R_goal=1000.0, 
                p_win=0.1, 
                B=200.0, 
                b=10.0)
        else:
            raise ValueError(f"Invalid choice: {environment}")

        if agentchoice == "1":
            agent = DDQN(
                env, 
                alpha=0.9, 
                gamma=0.95, 
                epsilon=1, 
                epsilon_decay=0.999, 
                min_epsilon=0.01)
        elif agentchoice == "2":
            agent = DDQN_MIN(
                env, 
                alpha=0.9, 
                gamma=0.95, 
                epsilon=1, 
                epsilon_decay=0.999, 
                min_epsilon=0.01)
        elif agentchoice == "3":
            agent = Q_ENSEMBLE_MIN(
                env, 
                ensemble_size=ensemble, 
                alpha=0.9, 
                gamma=0.95, 
                epsilon=1, 
                epsilon_decay=0.999, 
                min_epsilon=0.01)
        elif agentchoice == "4":
            agent = REDQ(
                env, 
                ensemble_size=ensemble, 
                alpha=0.9, 
                gamma=0.95, 
                epsilon=1.0, 
                epsilon_decay=0.999, 
                min_epsilon=0.01)
        else:
            raise ValueError(f"Invalid choice: {agentchoice}")
        # More complex environment - another way of interpret bias
        if environment == "2":
            leaf_range = list(env.terminal_states)
            sampled_leaves = random.sample(leaf_range, k=min(100, len(leaf_range)))
            mid_leaf_parents = list(set([(s - 1) // 2 for s in sampled_leaves]))
            sample_size = min(50, len(mid_leaf_parents))
            probe_states = random.sample(mid_leaf_parents, k=sample_size)
        else:
            probe_states = [0, length // 2, length - 1]

        # Loop that finds bias for last x episodes
        overestimation_bias_history = {s: [] for s in probe_states}
        for run in range(num_runs):
            print(f"Run number: {run + 1} of {num_runs}, state length: {length}")
            if agentchoice in ["1", "2"]:
                q1, q2, returns = agent.train(num_episodes=num_episodes, max_steps=max_steps, window_size=window_size)
            else:
                returns = agent.train(num_episodes=num_episodes, max_steps=max_steps, window_size=window_size)

            for ep in range(0, num_episodes, bias_eval_interval):
                for s in overestimation_bias_history.keys():
                    q_values = agent.get_Q_values(s)
                    if np.max(np.abs(q_values)) < 1e-4:
                        continue

                    q_max = np.max(q_values)
                    g_t = agent.evaluate_return(s, max_steps=max_steps)
                    bias = q_max - g_t
                    overestimation_bias_history[s].append(bias)
                if ep % bias_eval_interval == 0:
                    recent_bias_values = [
                        bias_list[-1] for bias_list in overestimation_bias_history.values()
                        if len(bias_list) > 0
                    ]
                    if recent_bias_values:
                        recent_bias = np.mean(recent_bias_values)
                        print(f" [Bias {ep:5d}/{num_episodes}] avg_bias(last {bias_eval_interval} episodes) = {recent_bias:.2f}")

            avg_returns += np.array(returns)

        avg_returns /= num_runs
        rolling_sum_returns = np.convolve(avg_returns, np.ones(window_size), mode='valid')
        ax1.plot(rolling_sum_returns, label=f"Size {length}")

        # Plot average bias across states for this length
        max_len = max(len(b) for b in overestimation_bias_history.values())
        padded_bias_lists = [
            b + [np.nan] * (max_len - len(b))
            for b in overestimation_bias_history.values()
        ]
        bias_len_avg = np.nanmean(padded_bias_lists, axis=0)

        if len(bias_len_avg) >= smoothing_window:
            smoothed_bias = np.convolve(bias_len_avg, np.ones(smoothing_window)/smoothing_window, mode='valid')
        else:
            smoothed_bias = bias_len_avg

        ax2.plot(smoothed_bias, label=f"Size {length}")

    # === Finalize reward plot ===
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(f"Sum of Return {window_size}")
    title_str = f"{agent_str} on {env_str}"
    # Changed depending on which environment is tested
    if environment == "2":
        title_str += f" p_move=0.9"
    if agentchoice in ["3", "4"]:
        title_str += f" Ensemble {ensemble}"
    ax1.set_title(f"Cumulative Reward - {title_str}")
    ax1.legend()
    fig1.savefig(os.path.join(results_dir, f"Returns_{title_str.replace(' ', '_')}.png"))
    plt.close(fig1)

    # === Finalize bias plot ===
    ax2.set_xlabel(f"Episode checkpoint (every {bias_eval_interval} episodes)")
    ax2.set_ylabel("Overestimation Bias")
    ax2.set_title(f"Overestimation Bias - {title_str}")
    ax2.legend()
    fig2.savefig(os.path.join(results_dir, f"Bias_{title_str.replace(' ', '_')}.png"))
    plt.close(fig2)

if __name__ == "__main__":
    test()
