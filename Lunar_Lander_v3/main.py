# ===== Imports & Enums =====
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from enum import Enum

# Action enums
class Act(Enum):
    NO_OP = 0
    LEFT = 1
    MAIN = 2
    RIGHT = 3

# Observation indexes
class Obs(Enum):
    X = 0; Y = 1; VX = 2; VY = 3
    ANGLE = 4; ANGULAR_VELOCITY = 5
    LEFT_LEG_CONTACT = 6; RIGHT_LEG_CONTACT = 7

# ===== Run One Episode =====
def run_episode(agent_function, max_steps=1000, seed=None):
    env = gym.make("LunarLander-v3")
    obs, info = env.reset(seed=seed)

    reward_final = -100  # default crash

    for _ in range(max_steps):
        action = agent_function(obs)
        obs, r, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            left_leg, right_leg = obs[Obs.LEFT_LEG_CONTACT.value], obs[Obs.RIGHT_LEG_CONTACT.value]
            if left_leg == 1.0 and right_leg == 1.0:
                reward_final = 100
            else:
                reward_final = -100
            break

    env.close()
    return reward_final

# ===== Agents =====
def random_agent(obs):
    return random.choice([0, 1, 2, 3])

def simple_reflex_agent(obs):
    return Act.MAIN.value if obs[Obs.VY.value] < -0.3 else Act.NO_OP.value

def better_reflex_agent(obs):
    x, y, vx, vy, angle, ang_vel, left_leg, right_leg = obs
    if left_leg or right_leg:
        return Act.NO_OP.value
    if vy < -0.2:
        return Act.MAIN.value
    if angle > 0.1 or vx > 0.3 or ang_vel > 0.1:
        return Act.LEFT.value
    if angle < -0.1 or vx < -0.3 or ang_vel < -0.1:
        return Act.RIGHT.value
    return Act.NO_OP.value

# ===== Evaluation =====
def run_episodes(agent_function, n=100, seed_base=42):
    return [run_episode(agent_function, seed=seed_base+i) for i in range(n)]

def print_summary(name, rewards):
    rewards = np.array(rewards)
    avg = np.mean(rewards)
    success_count = np.sum(rewards == 100)
    print(f"--- {name} ---")
    print("Rewards per episode:", rewards.tolist())  # tetap tampilkan list
    print(f"Average reward: {avg:.2f}")
    print(f"Success rate: {success_count}/{len(rewards)}")
    print()

def plot_rewards(rewards, title="Rewards per Episode"):
    plt.figure(figsize=(10,4))
    plt.plot(range(1, len(rewards)+1), rewards, marker='o')
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward (-100 crash, +100 safe landing)")
    plt.ylim(-120, 120)
    plt.grid(True)
    plt.show()

def make_summary_table(results_dict):
    rows = []
    for name, rewards in results_dict.items():
        rewards = np.array(rewards)
        rows.append({
            "Agent": name,
            "Average Reward": np.mean(rewards),
            "Success Rate": f"{np.sum(rewards == 100)}/{len(rewards)}"
        })
    return pd.DataFrame(rows)

# ===== Run All Agents =====
N = 100
print("Running Random Agent ...")
rewards_rand = run_episodes(random_agent, n=N, seed_base=1000)

print("Running Simple Reflex Agent ...")
rewards_simple = run_episodes(simple_reflex_agent, n=N, seed_base=2000)

print("Running Better Reflex Agent ...")
rewards_better = run_episodes(better_reflex_agent, n=N, seed_base=3000)

# Summaries per agent
print_summary("Random Agent", rewards_rand)
print_summary("Simple Reflex Agent", rewards_simple)
print_summary("Better Reflex Agent", rewards_better)

# Tabel ringkasan semua agent
df = make_summary_table({
    "Random Agent": rewards_rand,
    "Simple Reflex Agent": rewards_simple,
    "Better Reflex Agent": rewards_better
})
print("=== Comparison Table ===")
print(df.to_string(index=False))

# Plots
plot_rewards(rewards_rand, "Random Agent - Rewards")
plot_rewards(rewards_simple, "Simple Reflex Agent - Rewards")
plot_rewards(rewards_better, "Better Reflex Agent - Rewards")