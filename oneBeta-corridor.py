import gymnasium as gym
import json
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import minigrid
import numpy as np
import sys
import torch
import torch.nn as nn
from collections import defaultdict
from gymnasium import spaces
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
from mpl_toolkits.mplot3d import Axes3D
from torch.distributions import Categorical
from tqdm import tqdm


########################
# Environment Wrappers #
########################

# Env Wapper: Gives the agent [x, y, dir] only.
class AgentStateWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        base_env = env.unwrapped
        self.w = base_env.width
        self.h = base_env.height
        # multi-hot vector of size width + height + 4 (for direction)
        obs_size = self.w + self.h + 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

    def observation(self, obs):
        env = self.env.unwrapped
        x, y = env.agent_pos
        d = env.agent_dir

        one_hot_x = np.zeros(self.w, dtype=np.float32)
        one_hot_x[x] = 1.0

        one_hot_y = np.zeros(self.h, dtype=np.float32)
        one_hot_y[y] = 1.0

        one_hot_d = np.zeros(4, dtype=np.float32)
        one_hot_d[d] = 1.0

        return np.concatenate([one_hot_x, one_hot_y, one_hot_d])


# Env Wrapper: restricts actions to left, right, forward only
class MovementOnlyWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(3)  # 0=left, 1=right, 2=forward

    def action(self, act):
        return act


###############
# Environment #
###############

GOAL_REWARDS = {
    (1, 1): 1.0,  # Zone A — low reward, top-left
    (9, 1): 2.0,  # Zone B — low reward, top-right
    (1, 7): 4.0,  # Zone C — high reward, bottom-left
    (9, 7): 6.0,  # Zone D — highest reward, bottom-right
}

GOAL_POSITIONS = list(GOAL_REWARDS.keys())

# Width / height of the new environment
ENV_W, ENV_H = 11, 9
BOTTLENECK_ROW = 4  # y-coordinate of the internal wall
BOTTLENECK_X = 5  # x-coordinate of the single gap

"""
#  #  #  #  #  #  #  #  #  #  #
#  1                       2  #
#                             #
#                             #
#  #  #  #  #     #  #  #  #  #
#                             #
#                             #
#  4                       6  #
#  #  #  #  #  #  #  #  #  #  #
"""

class ZonedCorridorEnv(MiniGridEnv):
    """
    11x9 grid, split horizontally by a wall at y=4 with a single gap at x=5.
    Four goals, one per quadrant, with graded rewards (1, 2, 4, 6)
    """

    def __init__(self, max_steps=50, **kwargs):
        self.width = ENV_W
        self.height = ENV_H
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            see_through_walls=True,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach the highest-reward goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        # Outer walls
        self.grid.wall_rect(0, 0, width, height)

        # Internal horizontal wall at y=BOTTLENECK_ROW, gap at x=BOTTLENECK_X
        for x in range(1, width - 1):
            if x != BOTTLENECK_X:
                self.put_obj(Wall(), x, BOTTLENECK_ROW)

        # Place goals
        for (gx, gy) in GOAL_POSITIONS:
            self.put_obj(Goal(), gx, gy)

        # Agent starts randomly in either half
        self.place_agent()
        self.agent_dir = 0
        self.mission = "reach the highest-reward goal"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated:
            pos = tuple(self.agent_pos)
            reward = GOAL_REWARDS.get(pos, reward)
        elif not truncated:
            reward = -0.1  # small step penalty
        return obs, reward, terminated, truncated, info


gym.register(
    id="MiniGrid-ZonedCorridor-v0",
    entry_point=ZonedCorridorEnv,
    kwargs={"max_steps": 50},
)


############
# Networks #
############

# Maps an internal representation to distribution over actions
class PolicyNet(nn.Module):
    def __init__(self, rep_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rep_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, state):
        return Categorical(logits=self.net(state))


# Value estimation network
class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


# Time-dependent marginal action model
class MarginalNet(nn.Module):
    def __init__(self, max_steps, act_dim, hidden=64):
        super().__init__()
        self.max_steps = max_steps
        self.net = nn.Sequential(
            nn.Linear(max_steps, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, t: int):
        one_hot = torch.zeros(self.max_steps)
        one_hot[min(t, self.max_steps - 1)] = 1.0
        return Categorical(logits=self.net(one_hot))


#########################
# Grid layout constants #
#########################

W, H = ENV_W, ENV_H
DIR_VECTORS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
DIR_LABELS = {0: "Right", 1: "Down", 2: "Left", 3: "Up"}
ACTION_LABELS = ["Turn L", "Turn R", "Forward"]
ACTION_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]

###########
# Helpers #
###########

def is_free(x, y):
    """Return True for cells where the agent can actually stand."""
    # outer wall ring
    if x <= 0 or x >= W - 1 or y <= 0 or y >= H - 1:
        return False
    if y == BOTTLENECK_ROW and x != BOTTLENECK_X:
        return False
    return True


def make_obs(x, y, d):
    """Build the one-hot observation vector for position (x,y) facing d."""
    oh_x = np.zeros(W, dtype=np.float32)
    oh_x[x] = 1.0
    oh_y = np.zeros(H, dtype=np.float32)
    oh_y[y] = 1.0
    oh_d = np.zeros(4, dtype=np.float32)
    oh_d[d] = 1.0
    return np.concatenate([oh_x, oh_y, oh_d])


@torch.no_grad()
def get_probs(encoder, policy, x, y, d):
    obs = torch.tensor(make_obs(x, y, d), dtype=torch.float32)
    return policy(obs).probs.numpy()


######################
# Numerical Analysis #
######################
def _collect_trajectory_samples(policy, env, num_episodes=20):
    policy.eval()
    all_states, all_actions = [], []

    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                state = torch.tensor(obs, dtype=torch.float32)

                action = policy(state).sample()

                all_states.append(obs.copy())
                all_actions.append(action.item())

                obs, _, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

    return all_states, all_actions


def compute_mutual_information_sa(policy, env, num_episodes=30):
    """Estimate I(S; A) from on-policy trajectories."""
    policy.eval()
    act_dim = env.action_space.n

    all_states, all_actions = _collect_trajectory_samples(policy, env, num_episodes)
    n = len(all_actions)

    # Marginal p(a)
    action_counts = np.bincount(all_actions, minlength=act_dim)
    p_a = action_counts / n
    p_a = np.maximum(p_a, 1e-10)

    # H(A) = -SUM p(a) log p(a)
    h_a = -np.sum(p_a * np.log(p_a))

    cond_entropies = []
    with torch.no_grad():
        for obs in all_states:
            state = torch.tensor(obs, dtype=torch.float32)
            probs = policy(state).probs.numpy()
            h = -np.sum(probs * np.log(probs + 1e-10))
            cond_entropies.append(h)

    h_a_given_s = float(np.mean(cond_entropies))
    i_sa = float(h_a - h_a_given_s)
    return i_sa, h_a_given_s


def compute_policy_entropy(policy, env, num_episodes=30):
    _, h_a_given_s = compute_mutual_information_sa(policy, env, num_episodes)
    return h_a_given_s


def compute_goal_distribution_fractions(goal_hits, num_episodes):
    return {gx: sum(hits) / max(num_episodes, 1) for gx, hits in goal_hits.items()}


def evaluate_policy(policy, env, num_episodes=100):
    """
    Evaluate the trained policy and return statistics.
    """
    policy.eval()

    rewards = []
    steps_list = []
    successes = []
    goal_distribution = defaultdict(int)

    for _ in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        reached_goal = False

        while not done:
            state = torch.tensor(obs, dtype=torch.float32)

            with torch.no_grad():
                action = policy(state).sample()

            obs, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            steps += 1
            done = terminated or truncated

            if terminated:
                reached_goal = True
                pos = tuple(int(v) for v in env.unwrapped.agent_pos)
                if pos in GOAL_REWARDS:
                    key = f"{pos[0]}_{pos[1]}"
                    goal_distribution[key] += 1

        rewards.append(total_reward)
        steps_list.append(steps)
        successes.append(int(reached_goal))

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_steps": np.mean(steps_list),
        "std_steps": np.std(steps_list),
        "success_rate": np.mean(successes),
        "goal_distribution": dict(goal_distribution),
    }


#########################
# Grid layout constants # 
#########################

GOAL_SHADE = {
    (1, 1): "#fadbd8",  # faint red   — R=1
    (9, 1): "#fdebd0",  # faint orange — R=2
    (1, 7): "#d6eaf8",  # faint blue  — R=4
    (9, 7): "#d5f5e3",  # faint green — R=6
}


@torch.no_grad()
def get_probs_direct(policy, x, y, d):
    obs = torch.tensor(make_obs(x, y, d), dtype=torch.float32)
    return policy(obs).probs.numpy()


def draw_grid_panel(ax, policy, direction, act_dim):
    """Draw the corridor map for one facing direction with per-cell action bars."""
    # Background cells
    for y in range(H):
        for x in range(W):
            pos = (x, y)
            if not is_free(x, y):
                color = "#7f8c8d"  # wall
            elif pos in GOAL_REWARDS:
                color = GOAL_SHADE[pos]  # goal cell
            elif y == BOTTLENECK_ROW and x == BOTTLENECK_X:
                color = "#f8f9fa"  # bottleneck gap
            else:
                color = "#ffffff"
            rect = plt.Rectangle(
                (x, H - 1 - y),
                1,
                1,
                facecolor=color,
                edgecolor="#bdc3c7",
                linewidth=0.8,
            )
            ax.add_patch(rect)

    # Action-probability bars 
    bar_h, bar_w, y_off = 0.22, 0.80, 0.39

    for y in range(H):
        for x in range(W):
            if not is_free(x, y):
                continue

            probs = get_probs_direct(policy, x, y, direction)
            by = (H - 1 - y) + y_off
            bx = x + (1 - bar_w) / 2
            cursor = bx

            for p, col in zip(probs, ACTION_COLORS):
                w = bar_w * p
                if w > 0.003:
                    ax.add_patch(
                        plt.Rectangle(
                            (cursor, by), w, bar_h, facecolor=col, edgecolor="none"
                        )
                    )
                cursor += w

            # Label dominant action if confident
            top_idx = int(np.argmax(probs))
            top_p = probs[top_idx]
            if top_p > 0.45:
                ax.text(
                    x + 0.5,
                    (H - 1 - y) + 0.72,
                    f"{ACTION_LABELS[top_idx][0]}{top_p:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.0,
                    color="#2c3e50",
                    fontweight="bold",
                )

    # Goal reward labels
    for (gx, gy), rv in GOAL_REWARDS.items():
        ax.text(
            gx + 0.5,
            (H - 1 - gy) + 0.18,
            f"R={rv:.0f}",
            ha="center",
            va="center",
            fontsize=6.5,
            color="#1a5276",
            fontweight="bold",
        )

    # Bottleneck marker
    ax.text(
        BOTTLENECK_X + 0.5,
        (H - 1 - BOTTLENECK_ROW) + 0.5,
        "⬆⬇",
        ha="center",
        va="center",
        fontsize=7,
        color="#7d6608",
    )

    # Axes formatting
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_title(DIR_LABELS[direction], fontsize=11, fontweight="bold", pad=6)
    ax.set_xticks(range(W + 1))
    ax.set_yticks(range(H + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


# Main training loop with numerical analysis
if __name__ == "__main__":
    BETA_VALUES = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    GAMMA = 0.99
    MAX_STEPS = 30
    all_results = {}
    GOAL_KEYS = [f"{gx}_{gy}" for (gx, gy) in GOAL_POSITIONS]

    for BETA in BETA_VALUES:
        print(f"\n{'='*70}")
        print(f"Training with Beta={BETA}")
        print(f"{'='*70}\n")

        base_env = gym.make("MiniGrid-ZonedCorridor-v0")

        base_env = MovementOnlyWrapper(base_env)
        env = AgentStateWrapper(base_env)

        obs, _ = env.reset()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        # Networks
        policy = PolicyNet(obs_dim, act_dim)
        value = ValueNet(obs_dim)
        marginal_act = MarginalNet(MAX_STEPS, act_dim)

        LR = 3e-4

        pol_optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
        val_optimizer = torch.optim.Adam(value.parameters(), lr=LR)
        mar_act_optimizer = torch.optim.Adam(marginal_act.parameters(), lr=LR)

        NUM_EPISODES = 1000

        total_steps = []
        successes = []
        goal_hits = {k: [] for k in GOAL_KEYS}
        episode_rewards = []
        mi_history = []

        for ep in tqdm(range(NUM_EPISODES), desc=f"Training Beta={BETA}"):
            obs, _ = env.reset()
            state = torch.tensor(obs, dtype=torch.float32)
            done = False
            steps = 0
            t = 0
            reached_goal = False
            hit_goal_key = None
            episode_reward = 0

            while not done:
                act_dist = policy(state)
                action = act_dist.sample()

                # Environment step
                obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                episode_reward += reward

                if terminated:
                    reached_goal = True
                    pos = tuple(int(v) for v in env.unwrapped.agent_pos)
                    hit_goal_key = f"{pos[0]}_{pos[1]}"

                next_state = torch.tensor(obs, dtype=torch.float32)

                # Critic update
                with torch.no_grad():
                    next_val = 0.0 if done else GAMMA * value(next_state)
                    target = reward + next_val
                td_error = target - value(state)

                val_optimizer.zero_grad()
                (0.5 * td_error.pow(2)).backward()
                val_optimizer.step()

                mar_act_dist = marginal_act(t)
                mar_act_nll = -mar_act_dist.log_prob(action)
                mar_act_optimizer.zero_grad()
                mar_act_nll.backward()
                mar_act_optimizer.step()

                log_pi = policy(state).log_prob(action)

                with torch.no_grad():
                    log_q_a = marginal_act(t).log_prob(action)
                    delta = td_error.detach()

                mi_term = log_pi - log_q_a
                actor_loss = -log_pi * delta + BETA * mi_term

                pol_optimizer.zero_grad()
                actor_loss.backward()
                pol_optimizer.step()

                state = next_state
                steps += 1
                t += 1

            total_steps.append(steps)
            successes.append(int(reached_goal))
            episode_rewards.append(episode_reward)

            for k in GOAL_KEYS:
                goal_hits[k].append(1 if hit_goal_key == k else 0)

            if ep % 200 == 0:
                mi_sa, h_a_s = compute_mutual_information_sa(
                    policy, env, num_episodes=10
                )
                mi_history.append((ep, mi_sa, h_a_s))

        print("\nFinal evaluation...")
        eval_stats = evaluate_policy(policy, env, num_episodes=500)

        final_mi_sa, final_h_a_given_s = compute_mutual_information_sa(
            policy, env, num_episodes=500
        )

        goal_fractions = compute_goal_distribution_fractions(goal_hits, NUM_EPISODES)

        last_500_rewards = episode_rewards[-500:]
        last_500_steps = total_steps[-500:]
        results = {
            "beta": BETA,
            "final_mi_sa": final_mi_sa,  # I(S;A) 
            "policy_entropy": final_h_a_given_s,  # H(A|S) 
            "mean_reward": eval_stats["mean_reward"],
            "std_reward": eval_stats["std_reward"],
            "mean_steps": eval_stats["mean_steps"],
            "std_steps": eval_stats["std_steps"],
            "success_rate": eval_stats["success_rate"],
            "goal_distribution": eval_stats["goal_distribution"],
            "goal_fractions": goal_fractions,
            "convergence_reward_mean": float(np.mean(last_500_rewards)),
            "convergence_reward_std": float(np.std(last_500_rewards)),
            "convergence_steps_mean": float(np.mean(last_500_steps)),
            "convergence_steps_std": float(np.std(last_500_steps)),
            "mi_history": mi_history,
            "training_rewards": episode_rewards,
            "training_steps": total_steps,
            "training_successes": successes,
        }

        all_results[f"beta_{BETA}"] = results

        print(f"\n{'-'*70}")
        print(f"Results for Beta={BETA}:")
        print(f"{'-'*70}")
        print(f"I(S;A)  [Beta penalty]:    {final_mi_sa:.4f}")
        print(f"H(A|S)  [policy ent]:   {final_h_a_given_s:.4f}")
        print(f"Goal fractions: { {k: f'{v:.2%}' for k, v in goal_fractions.items()} }")
        print(
            f"Mean Reward:    {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}"
        )
        print(
            f"Mean Steps:     {eval_stats['mean_steps']:.2f} ± {eval_stats['std_steps']:.2f}"
        )
        print(f"Success Rate:   {eval_stats['success_rate']*100:.1f}%")
        print(
            f"Convergence Reward (last 500): {np.mean(last_500_rewards):.2f} ± {np.std(last_500_rewards):.2f}"
        )
        print(f"{'-'*70}\n")

        # Figure 6 – Policy grid visualisation for selected Beta values
        print(f"\nBuilding grid visualisation for Beta={BETA}...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 16))
        fig.suptitle(
            f"S→A Policy  (Beta={BETA}) — action probabilities per cell",
            fontsize=13,
            fontweight="bold",
        )
        for direction, ax in zip(range(4), axes.flat):
            draw_grid_panel(ax, policy, direction, act_dim)  # reuse trained policy

        legend_patches = [
            mpatches.Patch(color=c, label=l)
            for c, l in zip(ACTION_COLORS, ACTION_LABELS)
        ]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=3,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.01),
        )
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        safe_beta = str(BETA).replace(".", "p")
        plt.savefig(
            f"./outputs-onebeta/corridor/policies/onebeta_fig6_policy_grid_b{safe_beta}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved onebeta_fig6_policy_grid_b{safe_beta}.png")
        print(f"{'-'*70}\n")

    # Save JSON
    results_json = {}
    for key, val in all_results.items():
        results_json[key] = {
            "beta": val["beta"],
            "final_mi_sa": float(val["final_mi_sa"]),
            "policy_entropy": float(val["policy_entropy"]),
            "mean_reward": float(val["mean_reward"]),
            "std_reward": float(val["std_reward"]),
            "mean_steps": float(val["mean_steps"]),
            "std_steps": float(val["std_steps"]),
            "success_rate": float(val["success_rate"]),
            "goal_distribution": val["goal_distribution"],
            "goal_fractions": {
                str(k): float(v) for k, v in val["goal_fractions"].items()
            },
            "convergence_reward_mean": float(val["convergence_reward_mean"]),
            "convergence_reward_std": float(val["convergence_reward_std"]),
            "convergence_steps_mean": float(val["convergence_steps_mean"]),
            "convergence_steps_std": float(val["convergence_steps_std"]),
        }

    with open("./outputs-onebeta/corridor/onebeta_numerical_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print("Saved onebeta_numerical_results.json")

    # Plotting
    runs = list(results_json.values())
    betas = np.array([r["beta"] for r in runs])
    rew = np.array([r["mean_reward"] for r in runs])
    rew_std = np.array([r["std_reward"] for r in runs])
    mi_sa = np.array([r["final_mi_sa"] for r in runs])
    pol_ent = np.array([r["policy_entropy"] for r in runs])
    steps = np.array([r["mean_steps"] for r in runs])
    steps_std = np.array([r["std_steps"] for r in runs])
    sr = np.array([r["success_rate"] for r in runs])

    goal_xs = sorted({int(k) for r in runs for k in r["goal_fractions"]})
    goal_frac = {
        gx: np.array([r["goal_fractions"].get(str(gx), 0.0) for r in runs])
        for gx in goal_xs
    }
    goal_colors = {1: "#e74c3c", 3: "#f39c12", 5: "#27ae60"}
    goal_labels = {1: "R=1  (x=1)", 3: "R=2  (x=3)", 5: "R=3  (x=5)"}

    sort_idx = np.argsort(betas)
    bs = betas[sort_idx]
    rews = rew[sort_idx]
    rew_stds = rew_std[sort_idx]
    mi_sas = mi_sa[sort_idx]
    pol_ents = pol_ent[sort_idx]
    stepss = steps[sort_idx]
    steps_stds = steps_std[sort_idx]
    srs = sr[sort_idx]
    gf_s = {gx: goal_frac[gx][sort_idx] for gx in goal_xs}

    LINTHRESH = 0.001
    POINT_COLOR = "#2563eb"

    # Figure 1 – Main 2×2: reward, success, steps, MI vs Beta
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "S→A: Effect of Beta on Agent Performance", fontsize=14, fontweight="bold"
    )

    ax = axes[0, 0]
    ax.errorbar(
        bs,
        rews,
        yerr=rew_stds,
        fmt="-o",
        color=POINT_COLOR,
        capsize=4,
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Beta")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Reward vs Beta")
    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(bs, srs * 100, "-s", color="#16a34a", linewidth=2, markersize=6)
    ax.set_xlabel("Beta")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate vs Beta")
    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.errorbar(
        bs,
        stepss,
        yerr=steps_stds,
        fmt="-^",
        color="#dc2626",
        capsize=4,
        linewidth=2,
        markersize=6,
    )
    ax.set_xlabel("Beta")
    ax.set_ylabel("Mean Steps to Goal")
    ax.set_title("Episode Length vs Beta")
    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(
        bs, mi_sas, "-D", color="#9333ea", linewidth=2, markersize=6, label="I(S;A)"
    )
    ax.fill_between(bs, 0, mi_sas, alpha=0.12, color="#9333ea")
    ax.set_xlabel("Beta")
    ax.set_ylabel("I(S;A)  (nats)")
    ax.set_title("I(S;A) vs Beta ")
    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "./outputs-onebeta/corridor/onebeta_fig1_metrics_vs_beta.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved onebeta_fig1_metrics_vs_beta.png")

    # Figure 2 – I(S;A) and H(A|S) vs Beta  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "S→A: Information-Theoretic Quantities vs Beta", fontsize=13, fontweight="bold"
    )

    ax1.plot(bs, mi_sas, "-o", color="#9333ea", linewidth=2, markersize=6)
    ax1.fill_between(bs, 0, mi_sas, alpha=0.12, color="#9333ea")
    ax1.set_xlabel("Beta")
    ax1.set_ylabel("I(S;A)  (nats)")
    ax1.set_title("I(S;A) vs Beta")
    ax1.set_xscale("symlog", linthresh=LINTHRESH)
    ax1.grid(True, alpha=0.3)

    ax2.plot(bs, pol_ents, "-s", color="#0ea5e9", linewidth=2, markersize=6)
    ax2.fill_between(bs, 0, pol_ents, alpha=0.12, color="#0ea5e9")
    ax2.set_xlabel("Beta")
    ax2.set_ylabel("H(A|S)  (nats)")
    ax2.set_title("Policy Entropy H(A|S) vs Beta\n(0 = deterministic, high = stochastic)")
    ax2.set_xscale("symlog", linthresh=LINTHRESH)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "./outputs-onebeta/corridor/onebeta_fig2_mi_quantities.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved onebeta_fig2_mi_quantities.png")

    # Figure 3 – Goal distribution vs Beta
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "S→A: Goal Visit Rate vs Beta",
        fontsize=13,
        fontweight="bold",
    )

    for gx in goal_xs:
        ax.plot(
            bs,
            gf_s[gx] * 100,
            "-o",
            color=goal_colors.get(gx, "gray"),
            linewidth=2,
            markersize=7,
            label=goal_labels.get(gx, f"x={gx}"),
        )
        ax.fill_between(
            bs, 0, gf_s[gx] * 100, alpha=0.08, color=goal_colors.get(gx, "gray")
        )

    ax.set_xlabel("Beta")
    ax.set_ylabel("Visit Rate (%)")
    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.set_ylim(-3, 103)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "./outputs-onebeta/corridor/onebeta_fig3_goal_distribution.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved onebeta_fig3_goal_distribution.png")

    # Figure 4 – Reward vs steps scatter, colored by Beta
    fig, ax = plt.subplots(figsize=(9, 6))
    norm = plt.Normalize(betas.min(), betas.max())
    sc = ax.scatter(
        steps,
        rew,
        c=betas,
        cmap="coolwarm",
        norm=norm,
        s=80,
        edgecolors="k",
        linewidth=0.5,
        alpha=0.9,
        zorder=3,
    )
    for r in runs:
        ax.annotate(
            f"Beta={r['beta']}",
            (r["mean_steps"], r["mean_reward"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    fig.colorbar(sc, ax=ax, label="Beta")
    ax.set_xlabel("Mean Steps to Goal")
    ax.set_ylabel("Mean Reward")
    ax.set_title("S→A: Reward–Efficiency Trade-off  (colored by Beta)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./outputs-onebeta/corridor/onebeta_fig4_reward_steps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved onebeta_fig4_reward_steps.png")

    # Figure 5 – Reward bar chart with std error bars
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = plt.cm.RdYlGn(rews / (rews.max() + 1e-9))
    ax.bar(range(len(rews)), rews, color=bar_colors, edgecolor="k", linewidth=0.6)
    ax.errorbar(
        range(len(rews)),
        rews,
        yerr=rew_stds,
        fmt="none",
        color="black",
        capsize=5,
        linewidth=1.5,
    )
    ax.set_xticks(range(len(rews)))
    ax.set_xticklabels(
        [f"Beta={r['beta']}" for r in [runs[i] for i in sort_idx]], rotation=30, ha="right"
    )
    ax.set_ylabel("Mean Reward")
    ax.set_title("S→A: Mean Reward ± Std by Beta  (color = reward magnitude)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("./outputs-onebeta/corridor/onebeta_fig5_reward_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved onebeta_fig5_reward_bar.png")
