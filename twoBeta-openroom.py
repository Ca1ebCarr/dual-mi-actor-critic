import gymnasium as gym
import json
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
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

# Maps goal x-coordinate to reward value.
GOAL_REWARDS = {1: 1.0, 3: 2.0, 5: 3.0}
GOAL_POSITIONS = list(GOAL_REWARDS.keys())  # [1, 3, 5]


"""
# # # # # # #
#           #
#           #
#           #
#           #
#           #
# 1 # 2 # 3 #
# # # # # # #
"""
class OpenRoom(MiniGridEnv):
    def __init__(self, max_steps=30, **kwargs):
        self.width = 7
        self.height = 8
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
        return "reach a goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)  
        self.grid.wall_rect(0, 0, width, height) 

        for x in range(1, width - 1): 
            if x not in GOAL_POSITIONS:
                self.put_obj(Wall(), x, 6)

        for gx in GOAL_POSITIONS:
            self.put_obj(Goal(), gx, 6)

        self.place_agent()  # random start space
        self.agent_dir = 0
        self.mission = "reach a goal"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # If the agent just reached a goal, swap in the per-goal reward
        if terminated:
            x, _ = self.agent_pos
            reward = GOAL_REWARDS.get(x, reward)
        elif not truncated:
            reward = -0.1
        return obs, reward, terminated, truncated, info


gym.register(id="MiniGrid-KeyDoor-5x5-v0", entry_point=OpenRoom)


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


# Maps raw state to distribution over latent spaces.
class EncoderNet(nn.Module):
    def __init__(self, obs_dim, rep_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, rep_dim),
        )

    def forward(self, state):
        return Categorical(logits=self.net(state))


# Time-dependent marginal representation model
class MarginalRepNet(nn.Module):
    def __init__(self, max_steps, rep_dim, hidden=64):
        super().__init__()
        self.max_steps = max_steps
        self.net = nn.Sequential(
            nn.Linear(max_steps, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, rep_dim),
        )

    def forward(self, t: int):
        one_hot = torch.zeros(self.max_steps)
        one_hot[min(t, self.max_steps - 1)] = 1.0
        return Categorical(logits=self.net(one_hot))

#########################
# Grid layout constants #
#########################

W, H = 7, 8
GOAL_POS = {1, 3, 5}  # goal x-coords, all at y=6
WALL_ROW6 = {2, 4}  # walls on row 6

# MiniGrid direction indices (dx, dy) for "forward"
# 0=right 1=down 2=left 3=up
DIR_VECTORS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
DIR_LABELS = {0: "Right", 1: "Down", 2: "Left", 3: "Up"}

# MiniGrid action indices
ACTION_LABELS = ["Turn L", "Turn R", "Forward"]
ACTION_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]  # red, blue, green


###########
# Helpers #
###########


def is_free(x, y):
    """Return True for cells where the agent can actually stand."""
    # outer wall ring
    if x <= 0 or x >= W - 1 or y <= 0 or y >= H - 1:
        return False
    # interior walls on row 6
    if y == 6 and x in WALL_ROW6:
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
    """
    Effective action-probability array P(a|s).
    P(a | s) = SUM_x  c_psi(x | s) · pi_theta(a | x)
    """
    obs = torch.tensor(make_obs(x, y, d), dtype=torch.float32)
    enc_dist = encoder(obs)  # Categorical over X
    p_x = enc_dist.probs

    # one-hot inputs for every possible latent code
    rep_dim = p_x.shape[0]
    eye = torch.eye(rep_dim)
    act_probs = policy(eye).probs

    # weighted sum
    marginal_a = (p_x.unsqueeze(1) * act_probs).sum(dim=0)
    return marginal_a.numpy()


######################
# Numerical Analysis #
######################
def _collect_trajectory_samples(encoder, policy, env, num_episodes=20):
    """Roll out full episodes and collect (state, rep, action) tuples."""
    encoder.eval()
    policy.eval()
    all_states, all_reps, all_actions = [], [], []

    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                state = torch.tensor(obs, dtype=torch.float32)
                enc_dist = encoder(state)
                rep = enc_dist.sample()

                rep_onehot = torch.zeros(enc_dist.logits.shape[0])
                rep_onehot[rep] = 1.0
                action = policy(rep_onehot).sample()

                all_states.append(obs.copy())
                all_reps.append(rep.item())
                all_actions.append(action.item())

                obs, _, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

    return all_states, all_reps, all_actions


def compute_mutual_information_xa(encoder, policy, env, num_episodes=30):
    """Estimate I(X; A) from on-policy trajectories."""
    encoder.eval()
    policy.eval()

    _, all_reps, all_actions = _collect_trajectory_samples(
        encoder, policy, env, num_episodes
    )
    n = len(all_actions)
    act_dim = env.action_space.n
    rep_dim = encoder.net[-1].out_features

    # Marginal p(a)
    action_counts = np.bincount(all_actions, minlength=act_dim)
    p_a = action_counts / n
    p_a = np.maximum(p_a, 1e-10)

    # H(A) = -SUM p(a) log p(a)
    h_a = -np.sum(p_a * np.log(p_a))

    # H(A|X) = -SUM_{x,a} p(x,a) log pi(a|x)
    # Precompute pi(a|x) for all x
    eye = torch.eye(rep_dim)
    with torch.no_grad():
        pi_given_x = policy(eye).probs.numpy()  # (rep_dim, act_dim)

    rep_counts = np.bincount(all_reps, minlength=rep_dim)
    p_x = rep_counts / n
    p_x = np.maximum(p_x, 1e-10)

    h_a_given_x = 0.0
    for x_idx in range(rep_dim):
        if p_x[x_idx] < 1e-10:
            continue
        for a_idx in range(act_dim):
            p_xa = p_x[x_idx] * pi_given_x[x_idx, a_idx]
            if p_xa > 1e-12:
                h_a_given_x -= p_xa * np.log(pi_given_x[x_idx, a_idx] + 1e-10)

    return float(h_a - h_a_given_x)


# Helper: collect states with random actions for broad grid coverage
def _collect_states_only(encoder, env, num_episodes=30):
    """Roll out episodes acting greedily on encoder samples, collect states + reps."""
    all_states, all_reps = [], []
    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                state = torch.tensor(obs, dtype=torch.float32)
                enc_dist = encoder(state)
                rep = enc_dist.sample()
                all_states.append(obs.copy())
                all_reps.append(rep.item())
                obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
                done = terminated or truncated
    return all_states, all_reps


def compute_mutual_information_xs(encoder, env, num_episodes=30):
    """Estimate I(X; S) and H(X|S) from trajectories"""
    encoder.eval()
    rep_dim = encoder.net[-1].out_features

    all_states, all_reps = _collect_states_only(encoder, env, num_episodes)
    n = len(all_reps)

    # Marginal p(x)
    rep_counts = np.bincount(all_reps, minlength=rep_dim)
    p_x = rep_counts / n
    p_x = np.maximum(p_x, 1e-10)

    # H(X) = -SUM p(x) log p(x)
    h_x = -np.sum(p_x * np.log(p_x))

    # H(X|S) = E_s[ H(encoder(·|s)) ]
    cond_entropies = []
    with torch.no_grad():
        for obs in all_states:
            state = torch.tensor(obs, dtype=torch.float32)
            probs = encoder(state).probs.numpy()
            h = -np.sum(probs * np.log(probs + 1e-10))
            cond_entropies.append(h)

    h_x_given_s = float(np.mean(cond_entropies))
    i_xs = float(h_x - h_x_given_s)

    return i_xs, h_x_given_s


def compute_effective_latent_codes(encoder, env, num_episodes=30):
    """
    Effective number of latent codes = exp(H(X)), where H(X) is the
    entropy of the marginal latent distribution p(x) = SUM_s p(s) c_psi(x|s).

    Returns value of 1 (collapse) to rep_dim (all codes uniformly)
    """
    encoder.eval()
    rep_dim = encoder.net[-1].out_features
    _, all_reps = _collect_states_only(encoder, env, num_episodes)
    n = len(all_reps)

    rep_counts = np.bincount(all_reps, minlength=rep_dim)
    p_x = rep_counts / n
    p_x = np.maximum(p_x, 1e-10)

    h_x = -np.sum(p_x * np.log(p_x))
    effective_codes = float(np.exp(h_x))
    return effective_codes, int(np.sum(rep_counts > 0))  # eff_N, num_used


def compute_goal_distribution_fractions(goal_hits, num_episodes):
    """
    Convert raw goal hit counts to fractions of total episodes.
    Returns a dict {goal_x: fraction}.
    """
    return {gx: sum(hits) / max(num_episodes, 1) for gx, hits in goal_hits.items()}


def evaluate_policy(encoder, policy, env, num_episodes=100):
    """
    Evaluate the trained policy and return statistics.
    """
    encoder.eval()
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
                enc_dist = encoder(state)
                rep = enc_dist.sample()

                rep_onehot = torch.zeros(enc_dist.logits.shape[0])
                rep_onehot[rep] = 1.0

                act_dist = policy(rep_onehot)
                action = act_dist.sample()

            obs, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            steps += 1
            done = terminated or truncated

            if terminated:
                reached_goal = True
                x_pos = int(env.unwrapped.agent_pos[0])
                if x_pos in GOAL_REWARDS:
                    goal_distribution[x_pos] += 1

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


def plot_state_similarity_matrix_spacial(
    encoder, env, beta_1, beta_2, rep_dim, save_dir="./outputs/openroom/similarity"
):
    """
    Figure 9 - State-space similarity matrix.
    Each cell (s_i, s_j) is coloured by the soft overlap:
        sim(s_i, s_j) = SUM_x  p(x | s_i) · p(x | s_j)

    States are ordered spatially: primary sort = grid position (row-major),
    secondary sort = agent orientation (0-3).
    Walls and invalid states are excluded.
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    encoder.eval()

    width = W 
    height = H 
    n_dirs = 4

    state_obs = []  # observation vectors, one per valid state
    state_labels = []  # (row, col, direction) for axis ticks

    for row in range(height):
        for col in range(width):
            for direction in range(n_dirs):
                if is_free(col, row):

                    one_hot_x = np.zeros(width, dtype=np.float32)
                    one_hot_x[col] = 1.0

                    one_hot_y = np.zeros(height, dtype=np.float32)
                    one_hot_y[row] = 1.0

                    one_hot_d = np.zeros(4, dtype=np.float32)
                    one_hot_d[direction] = 1.0

                    obs_vec = np.concatenate([one_hot_x, one_hot_y, one_hot_d])

                    state_obs.append(obs_vec)
                    state_labels.append((row, col, direction))

    N = len(state_obs)

    # Compute p(x | s) for every valid state
    with torch.no_grad():
        obs_tensor = torch.tensor(
            np.array(state_obs), dtype=torch.float32
        )
        probs = encoder(obs_tensor).probs 

    # Soft overlap matrix  sim[i,j] = SUM_x p(x|s_i)·p(x|s_j)
    sim = (probs @ probs.T).cpu().numpy()

    # Axis labels
    dir_sym = ["→", "↓", "←", "↑"]
    tick_pos = list(range(N))
    tick_labels = [
        f"({r},{c}){dir_sym[d]}" if d == 0 else dir_sym[d]
        for r, c, d in state_labels
    ]

    # Draw the figure
    fig_size = max(8, N * 0.18)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(
        sim,
        cmap="hot",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        interpolation="nearest",
        aspect="auto",
    )

    # Orientation dividers (thin lines every n_dirs states)
    for k in range(0, N, n_dirs):
        ax.axhline(k - 0.5, color="steelblue", linewidth=0.4, alpha=0.5)
        ax.axvline(k - 0.5, color="steelblue", linewidth=0.4, alpha=0.5)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=6)
    ax.set_xlabel("State  sⱼ  (row, col) + orientation block")
    ax.set_ylabel("State  sᵢ  (row, col) + orientation block")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Soft overlap  Σₓ p(x|sᵢ)·p(x|sⱼ)", fontsize=9)
    cbar.ax.axhline(1.0 / rep_dim, color="cyan", linewidth=1.5, linestyle="--")
    cbar.ax.text(2.6, 1.0 / rep_dim, "chance", va="center", fontsize=7, color="cyan")

    ax.set_title(
        f"State-Space Similarity Matrix\n"
        f"BETA_1={beta_1}, BETA_2={beta_2}   |   "
        f"N={N} valid states, {rep_dim} latent codes\n"
        f"Σₓ p(x|sᵢ)·p(x|sⱼ)  —  bright = same latent, dark = different",
        fontsize=10,
        pad=12,
    )

    plt.tight_layout()
    fname = f"{save_dir}/fig9_similarity_b1{beta_1}_b2{beta_2}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")

    encoder.train()
    return sim, state_labels

def plot_state_similarity_matrix_directional(
    encoder, env, beta_1, beta_2, rep_dim, save_dir="./outputs/openroom/similarity"
):
    """
    Figure 10 - State-space similarity matrix.
    Each cell (s_i, s_j) is coloured by the soft overlap:
        sim(s_i, s_j) = SUM_x  p(x | s_i) · p(x | s_j)

    States are ordered spatially: secondary sort = grid position (row-major),
    primary sort = agent orientation (0-3).
    Walls and invalid states are excluded.
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    encoder.eval()

    width = W 
    height = H 
    n_dirs = 4

    state_obs = []  # observation vectors, one per valid state
    state_labels = []  # (row, col, direction) for axis ticks

    for direction in range(n_dirs):
        for row in range(height):
            for col in range(width):
                if is_free(col, row):

                    one_hot_x = np.zeros(width, dtype=np.float32)
                    one_hot_x[col] = 1.0

                    one_hot_y = np.zeros(height, dtype=np.float32)
                    one_hot_y[row] = 1.0

                    one_hot_d = np.zeros(4, dtype=np.float32)
                    one_hot_d[direction] = 1.0

                    obs_vec = np.concatenate([one_hot_x, one_hot_y, one_hot_d])

                    state_obs.append(obs_vec)
                    state_labels.append((row, col, direction))

    N = len(state_obs) 

    # Compute p(x | s) for every valid state
    with torch.no_grad():
        obs_tensor = torch.tensor(
            np.array(state_obs), dtype=torch.float32
        )
        probs = encoder(obs_tensor).probs 

    # Soft overlap matrix  sim[i,j] = SUM_x p(x|s_i)·p(x|s_j)
    sim = (probs @ probs.T).cpu().numpy()

    # Axis labels
    dir_sym = ["→", "↓", "←", "↑"]
    tick_pos = list(range(N))
    tick_labels = [
        f"({r},{c}){dir_sym[d]}"
        for r, c, d in state_labels
    ]

    # Draw the figure
    fig_size = max(8, N * 0.18)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(
        sim,
        cmap="hot",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        interpolation="nearest",
        aspect="auto",
    )

    # Orientation dividers (thin lines every n_dirs states)
    block_size = N // n_dirs
    for k in range(0, N, block_size):
        ax.axhline(k - 0.5, color="steelblue", linewidth=0.4, alpha=0.5)
        ax.axvline(k - 0.5, color="steelblue", linewidth=0.4, alpha=0.5)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=6)
    ax.set_xlabel("State  sⱼ  (row, col) + orientation block")
    ax.set_ylabel("State  sᵢ  (row, col) + orientation block")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Soft overlap  Σₓ p(x|sᵢ)·p(x|sⱼ)", fontsize=9)
    cbar.ax.axhline(1.0 / rep_dim, color="cyan", linewidth=1.5, linestyle="--")
    cbar.ax.text(2.6, 1.0 / rep_dim, "chance", va="center", fontsize=7, color="cyan")

    ax.set_title(
        f"State-Space Similarity Matrix\n"
        f"BETA_1={beta_1}, BETA_2={beta_2}   |   "
        f"N={N} valid states, {rep_dim} latent codes\n"
        f"Σₓ p(x|sᵢ)·p(x|sⱼ)  —  bright = same latent, dark = different",
        fontsize=10,
        pad=12,
    )

    plt.tight_layout()
    fname = f"{save_dir}/fig10_similarity_b1{beta_1}_b2{beta_2}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")

    encoder.train()
    return sim, state_labels


# ---------------------------------------------------------------------------
# Main training loop with numerical analysis
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Hyperparameter grid for beta values
    BETA_VALUES = []
    b = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    for i in range(len(b)):
        for j in range(len(b)):
            BETA_VALUES.append((b[i], b[j]))

    GAMMA = 0.99
    MAX_STEPS = 30  # must match OpenRoom(max_steps=...)
    REP_DIM = 8

    # Store results for each beta configuration
    all_results = {}

    for BETA_1, BETA_2 in BETA_VALUES:
        print(f"\n{'='*70}")
        print(f"Training with BETA_1={BETA_1}, BETA_2={BETA_2}")
        print(f"{'='*70}\n")

        # Environment setup
        base_env = gym.make("MiniGrid-KeyDoor-5x5-v0")
        base_env = MovementOnlyWrapper(base_env)
        env = AgentStateWrapper(base_env)

        # Network dimensions
        obs, _ = env.reset()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        # Networks
        encoder = EncoderNet(obs_dim, REP_DIM)  # c_psi(x | s)
        policy = PolicyNet(REP_DIM, act_dim)  # pi_theta(a | x)
        value = ValueNet(obs_dim)  # V_phi(s)
        marginal_rep = MarginalRepNet(MAX_STEPS, REP_DIM)  # q_xi^t(x)
        marginal_act = MarginalNet(MAX_STEPS, act_dim)  # q_omega^t(a)

        # Optimizers
        LR = 3e-4
        enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
        pol_optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
        val_optimizer = torch.optim.Adam(value.parameters(), lr=LR)
        mar_rep_optimizer = torch.optim.Adam(marginal_rep.parameters(), lr=LR)
        mar_act_optimizer = torch.optim.Adam(marginal_act.parameters(), lr=LR)

        # Training parameters
        GAMMA = 0.99
        NUM_EPISODES = 1000
        window = 50

        # Tracking
        total_steps = []
        successes = []
        goal_hits = {gx: [] for gx in GOAL_REWARDS}
        episode_rewards = []
        mi_history = []  # Track MI over time

        # Training loop
        for ep in tqdm(range(NUM_EPISODES), desc=f"Training BETA_1={BETA_1}, BETA_2={BETA_2}"):
            obs, _ = env.reset()
            state = torch.tensor(obs, dtype=torch.float32)
            done = False
            steps = 0
            t = 0
            reached_goal = False
            hit_goal_x = None
            episode_reward = 0

            while not done:
                # Forward pass
                enc_dist = encoder(state)
                rep = enc_dist.sample()

                rep_onehot = torch.zeros(REP_DIM)
                rep_onehot[rep] = 1.0

                act_dist = policy(rep_onehot)
                action = act_dist.sample()

                # Environment step
                obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                episode_reward += reward

                if terminated:
                    reached_goal = True
                    hit_goal_x = env.unwrapped.agent_pos[0]

                next_state = torch.tensor(obs, dtype=torch.float32)

                # Critic update
                with torch.no_grad():
                    next_val = 0.0 if done else GAMMA * value(next_state)
                    target = reward + next_val
                td_error = target - value(state)

                val_optimizer.zero_grad()
                (0.5 * td_error.pow(2)).backward()
                val_optimizer.step()

                # Marginal-rep update
                mar_rep_dist = marginal_rep(t)
                mar_rep_nll = -mar_rep_dist.log_prob(rep)
                mar_rep_optimizer.zero_grad()
                mar_rep_nll.backward()
                mar_rep_optimizer.step()

                # Marginal-act update
                mar_act_dist = marginal_act(t)
                mar_act_nll = -mar_act_dist.log_prob(action)
                mar_act_optimizer.zero_grad()
                mar_act_nll.backward()
                mar_act_optimizer.step()

                # Encoder + Actor update
                log_c = encoder(state).log_prob(rep)

                rep_oh = torch.zeros(REP_DIM)
                rep_oh[rep] = 1.0
                log_pi = policy(rep_oh).log_prob(action)

                with torch.no_grad():
                    log_q_x = marginal_rep(t).log_prob(rep)
                    log_q_a = marginal_act(t).log_prob(action)
                    delta = td_error.detach()

                # Actor loss
                mi_pi = log_pi - log_q_a
                actor_loss = -log_pi * delta + BETA_2 * mi_pi

                pol_optimizer.zero_grad()
                actor_loss.backward()
                pol_optimizer.step()

                # Encoder loss
                mi_c = log_c - log_q_x
                enc_loss = -log_c * delta + BETA_1 * mi_c

                enc_optimizer.zero_grad()
                enc_loss.backward()
                enc_optimizer.step()

                state = next_state
                steps += 1
                t += 1

            total_steps.append(steps)
            successes.append(int(reached_goal))
            episode_rewards.append(episode_reward)

            for gx in GOAL_REWARDS:
                goal_hits[gx].append(1 if hit_goal_x == gx else 0)

            # Compute MI periodically (both I(X;A) and I(X;S))
            if ep % 200 == 0:
                mi_xa = compute_mutual_information_xa(
                    encoder, policy, env, num_episodes=10
                )
                mi_xs, _ = compute_mutual_information_xs(encoder, env, num_episodes=10)
                eff_n, n_used = compute_effective_latent_codes(
                    encoder, env, num_episodes=10
                )
                mi_history.append((ep, mi_xa, mi_xs, eff_n, n_used))

        # Final evaluation
        print("\nFinal evaluation...")
        eval_stats = evaluate_policy(encoder, policy, env, num_episodes=500)

        # Figure 9: state similarity matrix
        print("Computing state similarity matrix...")
        sim_matrix, sim_labels = plot_state_similarity_matrix_spacial(
            encoder,
            env,
            beta_1=BETA_1,
            beta_2=BETA_2,
            rep_dim=REP_DIM,
            save_dir="./outputs/openroom/similarity/spacial",
        )

        # Figure 10: state similarity matrix 
        print("Computing state similarity matrix...")
        sim_matrix, sim_labels = plot_state_similarity_matrix_directional(
            encoder,
            env,
            beta_1=BETA_1,
            beta_2=BETA_2,
            rep_dim=REP_DIM,
            save_dir="./outputs/openroom/similarity/directional",
        )

        # New comprehensive stats
        final_mi_xa = compute_mutual_information_xa(
            encoder, policy, env, num_episodes=500
        )
        final_mi_xs, final_h_x_given_s = compute_mutual_information_xs(
            encoder, env, num_episodes=500
        )
        final_eff_n, final_n_used = compute_effective_latent_codes(
            encoder, env, num_episodes=500
        )

        # Goal distribution as fractions
        goal_fractions = compute_goal_distribution_fractions(goal_hits, NUM_EPISODES)

        # Calculate convergence metrics
        last_500_rewards = episode_rewards[-500:]
        last_500_steps = total_steps[-500:]

        results = {
            "beta_1": BETA_1,
            "beta_2": BETA_2,
            "final_mi_xa": final_mi_xa,
            "final_mi_xs": final_mi_xs,
            "representation_entropy": final_h_x_given_s,
            "effective_latent_codes": final_eff_n, 
            "latent_codes_used": final_n_used,  
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

        all_results[f"beta_{BETA_1}_{BETA_2}"] = results

        # Print summary
        print(f"\n{'-'*70}")
        print(f"Results for BETA_1={BETA_1}, BETA_2={BETA_2}:")
        print(f"{'-'*70}")
        print(f"I(X;A)  [BETA_2 penalty]:  {final_mi_xa:.4f}")
        print(f"I(X;S)  [BETA_1 penalty]:  {final_mi_xs:.4f}")
        print(f"H(X|S)  [encoder ent]: {final_h_x_given_s:.4f}")
        print(
            f"Effective latent codes: {final_eff_n:.2f} / {REP_DIM}  ({final_n_used} nonzero)"
        )
        print(f"Goal fractions: { {k: f'{v:.2%}' for k, v in goal_fractions.items()} }")
        print(
            f"Mean Reward:    {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}"
        )
        print(
            f"Mean Steps:     {eval_stats['mean_steps']:.2f} ± {eval_stats['std_steps']:.2f}"
        )
        print(f"Success Rate:   {eval_stats['success_rate']*100:.1f}%")
        print(
            f"Convergence Reward (last 500 eps): {np.mean(last_500_rewards):.2f} ± {np.std(last_500_rewards):.2f}"
        )
        print(f"{'-'*70}\n")

    # Save all results to JSON
    results_json = {}
    for key, val in all_results.items():
        results_json[key] = {
            "beta_1": val["beta_1"],
            "beta_2": val["beta_2"],
            "final_mi_xa": float(val["final_mi_xa"]),
            "final_mi_xs": float(val["final_mi_xs"]),
            "representation_entropy": float(val["representation_entropy"]),
            "effective_latent_codes": float(val["effective_latent_codes"]),
            "latent_codes_used": int(val["latent_codes_used"]),
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

    with open("./outputs/openroom/numerical_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nSaved numerical results to numerical_results.json")

    # Extract flat arrays 
    runs = list(results_json.values())
    b1 = np.array([r["beta_1"] for r in runs])
    b2 = np.array([r["beta_2"] for r in runs])
    rew = np.array([r["mean_reward"] for r in runs])
    rew_std = np.array([r["std_reward"] for r in runs])
    mi_xa = np.array([r["final_mi_xa"] for r in runs])
    mi_xs = np.array([r["final_mi_xs"] for r in runs])
    entr = np.array([r["representation_entropy"] for r in runs])
    eff_n = np.array([r["effective_latent_codes"] for r in runs])
    n_used = np.array([r["latent_codes_used"] for r in runs])
    steps = np.array([r["mean_steps"] for r in runs])
    steps_std = np.array([r["std_steps"] for r in runs])
    sr = np.array([r["success_rate"] for r in runs])

    goal_xs = sorted({int(k) for r in runs for k in r["goal_fractions"]})
    goal_frac = {
        gx: np.array([r["goal_fractions"].get(str(gx), 0.0) for r in runs])
        for gx in goal_xs
    }

    # Unique sorted beta values on each axis
    u_b1 = np.array(sorted(set(b1.tolist())))
    u_b2 = np.array(sorted(set(b2.tolist())))

    # Shared helpers 
    goal_colors = {1: "#e74c3c", 3: "#f39c12", 5: "#27ae60"}
    goal_labels = {1: "R=1  (x=1)", 3: "R=5  (x=3)", 5: "R=10 (x=5)"}

    def beta_cmap(n, cmap="plasma"):
        return plt.colormaps[cmap](np.linspace(0.1, 0.85, max(n, 1)))

    b1_colors = beta_cmap(len(u_b1), "plasma")
    b2_colors = beta_cmap(len(u_b2), "viridis")

    def slice_by(fixed_axis, fixed_val, vary_axis, y_arr):
        """
        Return (x_vals, y_vals) for the slice where fixed_axis == fixed_val,
        sorted by vary_axis.
        """
        mask = np.abs(fixed_axis - fixed_val) < 1e-9
        xs = vary_axis[mask]
        ys = y_arr[mask]
        order = np.argsort(xs)
        return xs[order], ys[order]

    def add_slice_lines(
        ax,
        fixed_arr,
        fixed_vals,
        fixed_colors,
        vary_arr,
        y_arr,
        y_err=None,
        marker="o",
        lw=1.8,
        ms=5,
    ):
        """
        Plot one line per unique value of fixed_arr, varying over vary_arr.
        Returns handles for a legend.
        """
        handles = []
        for fv, col in zip(fixed_vals, fixed_colors):
            xs, ys = slice_by(fixed_arr, fv, vary_arr, y_arr)
            if len(xs) == 0:
                continue
            if y_err is not None:
                _, ye = slice_by(fixed_arr, fv, vary_arr, y_err)
                line = ax.errorbar(
                    xs,
                    ys,
                    yerr=ye,
                    fmt=f"-{marker}",
                    color=col,
                    capsize=3,
                    linewidth=lw,
                    markersize=ms,
                    label=f"Beta={fv}",
                )
            else:
                (line,) = ax.plot(
                    xs,
                    ys,
                    f"-{marker}",
                    color=col,
                    linewidth=lw,
                    markersize=ms,
                    label=f"Beta={fv}",
                )
            handles.append(line)
        return handles

    LINTHRESH = 0.001  # symlog linear threshold

    # Figure 1 – Mean Reward: slice by beta2 (left) and slice by beta1 (right)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Mean Reward — separating BETA_1 and BETA_2 effects", fontsize=13, fontweight="bold"
    )

    ax = axes[0]
    handles = add_slice_lines(
        ax, b2, u_b2, b2_colors, b1, rew, y_err=rew_std, marker="o"
    )
    ax.set_xlabel("BETA_1  (encoder compression)")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Fixed BETA_2, varying BETA_1\n(each line = one BETA_2 value)")
    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.legend(
        handles=handles, title="BETA_2", fontsize=7, title_fontsize=8, loc="best", ncol=2
    )
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    handles = add_slice_lines(
        ax, b1, u_b1, b1_colors, b2, rew, y_err=rew_std, marker="s"
    )
    ax.set_xlabel("BETA_2  (policy compression)")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Fixed BETA_1, varying BETA_2\n(each line = one BETA_1 value)")
    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.legend(
        handles=handles, title="BETA_1", fontsize=7, title_fontsize=8, loc="best", ncol=2
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./outputs/openroom/fig1_reward_vs_beta.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig1_reward_vs_beta.png")


    # Figure 2 – 3D scatter: BETA_1 × BETA_2 × Reward 
    fig = plt.figure(figsize=(10, 7))
    ax3d = fig.add_subplot(111, projection="3d")
    sc = ax3d.scatter(
        b1,
        b2,
        rew,
        c=rew,
        cmap="viridis",
        s=60,
        edgecolors="k",
        linewidth=0.4,
        depthshade=True,
    )
    fig.colorbar(sc, ax=ax3d, pad=0.1, shrink=0.6, label="Mean Reward")
    ax3d.set_xlabel("BETA_1", labelpad=8)
    ax3d.set_ylabel("BETA_2", labelpad=8)
    ax3d.set_zlabel("Mean Reward", labelpad=8)
    ax3d.set_xscale("symlog", linthresh=LINTHRESH)
    ax3d.set_yscale("symlog", linthresh=LINTHRESH)
    ax3d.set_title("Mean Reward across full (BETA_1, BETA_2) grid")
    plt.tight_layout()
    plt.savefig("./outputs/openroom/fig2_3d_reward.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig2_3d_reward.png")


    # Figure 3 – Heatmaps: reward, success rate, mean steps on the BETA_1×BETA_2 grid
    def make_grid(z_arr):
        grid = np.full((len(u_b1), len(u_b2)), np.nan)
        for i, r in enumerate(runs):
            ri = np.searchsorted(u_b1, r["beta_1"])
            ci = np.searchsorted(u_b2, r["beta_2"])
            grid[ri, ci] = z_arr[i]
        return grid

    rew_grid = make_grid(rew)
    sr_grid = make_grid(sr * 100)
    steps_grid = make_grid(steps)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("BETA_1 x BETA_2 Grid — heatmaps", fontsize=13, fontweight="bold")

    xticks = range(len(u_b2))
    yticks = range(len(u_b1))
    xlabels = [str(v) for v in u_b2]
    ylabels = [str(v) for v in u_b1]

    def heatmap(ax, grid, title, cmap, fmt=".2f", cbar_label=""):
        im = ax.imshow(
            grid, cmap=cmap, aspect="auto", origin="lower", interpolation="nearest"
        )
        ax.set_xticks(list(xticks))
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(list(yticks))
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_xlabel("BETA_2")
        ax.set_ylabel("BETA_1")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)
        for ri in range(grid.shape[0]):
            for ci in range(grid.shape[1]):
                v = grid[ri, ci]
                if not np.isnan(v):
                    ax.text(
                        ci,
                        ri,
                        f"{v:{fmt}}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if v < np.nanmax(grid) * 0.6 else "black",
                    )

    heatmap(axes[0], rew_grid, "Mean Reward", "YlGn", ".2f", "reward")
    heatmap(axes[1], sr_grid, "Success Rate (%)", "Blues", ".0f", "%")
    heatmap(axes[2], steps_grid, "Mean Steps", "Reds_r", ".1f", "steps")

    plt.tight_layout()
    plt.savefig("./outputs/openroom/fig3_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig3_heatmaps.png")


    # Figure 4 – MI quantities: I(X;S) vs BETA_1 (left) and I(X;A) vs BETA_2 (right)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Mutual Information — separating BETA_1 and BETA_2 effects",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0]
    handles = add_slice_lines(ax, b2, u_b2, b2_colors, b1, mi_xs, marker="s")
    ax.set_xlabel("BETA_1  (encoder compression)")
    ax.set_ylabel("I(X ; S)  (nats)")
    ax.set_title("I(X;S) vs BETA_1\n[BETA_1 is the penalty for this quantity]")
    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.legend(handles=handles, title="BETA_2", fontsize=7, title_fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    handles = add_slice_lines(ax, b1, u_b1, b1_colors, b2, mi_xa, marker="o")
    ax.set_xlabel("BETA_2  (policy compression)")
    ax.set_ylabel("I(X ; A)  (nats)")
    ax.set_title("I(X;A) vs BETA_2\n[BETA_2 is the penalty for this quantity]")
    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.legend(handles=handles, title="BETA_1", fontsize=7, title_fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./outputs/openroom/fig4_mi_quantities.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig4_mi_quantities.png")



    # Figure 5 – Goal distribution: 2×3 grid
    # Rows: fixed BETA_2 varying BETA_1 | fixed BETA_1 varying BETA_2
    # Columns: one per goal (R=1, R=5, R=10)
    fig, axes = plt.subplots(2, len(goal_xs), figsize=(5 * len(goal_xs), 9))
    fig.suptitle(
        "Goal Visit Rate — separating BETA_1 and BETA_2 effects\n"
        "(primary behavioral signature of compression)",
        fontsize=13,
        fontweight="bold",
    )

    for col, gx in enumerate(goal_xs):
        gf = goal_frac[gx] * 100

        # Row 0: fix BETA_2, vary BETA_1
        ax = axes[0, col]
        for fv, col_c in zip(u_b2, b2_colors):
            xs, ys = slice_by(b2, fv, b1, gf)
            if len(xs):
                ax.plot(
                    xs,
                    ys,
                    "-o",
                    color=col_c,
                    linewidth=1.8,
                    markersize=5,
                    label=f"BETA_2={fv}",
                )
        ax.set_xlabel("BETA_1")
        ax.set_ylabel("Visit Rate (%)")
        ax.set_title(f"{goal_labels.get(gx, f'x={gx}')}\nvarying BETA_1")
        ax.set_xscale("symlog", linthresh=LINTHRESH)
        ax.set_ylim(-3, 103)
        ax.grid(True, alpha=0.3)
        if col == len(goal_xs) - 1:
            ax.legend(title="BETA_2", fontsize=6, title_fontsize=7, loc="best")

        # Row 1: fix BETA_1, vary BETA_2
        ax = axes[1, col]
        for fv, col_c in zip(u_b1, b1_colors):
            xs, ys = slice_by(b1, fv, b2, gf)
            if len(xs):
                ax.plot(
                    xs,
                    ys,
                    "-s",
                    color=col_c,
                    linewidth=1.8,
                    markersize=5,
                    label=f"BETA_1={fv}",
                )
        ax.set_xlabel("BETA_2")
        ax.set_ylabel("Visit Rate (%)")
        ax.set_title(f"{goal_labels.get(gx, f'x={gx}')}\nvarying BETA_2")
        ax.set_xscale("symlog", linthresh=LINTHRESH)
        ax.set_ylim(-3, 103)
        ax.grid(True, alpha=0.3)
        if col == len(goal_xs) - 1:
            ax.legend(title="BETA_1", fontsize=6, title_fontsize=7, loc="best")

    plt.tight_layout()
    plt.savefig("./outputs/openroom/fig5_goal_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig5_goal_distribution.png")

    # Figure 6 – Latent compression: exp(H(X)) and H(X|S) vs BETA_1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Encoder Compression vs BETA_1  (each line = fixed BETA_2)",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0]
    for fv, col_c in zip(u_b2, b2_colors):
        xs, ys = slice_by(b2, fv, b1, eff_n)
        _, n_us = slice_by(b2, fv, b1, n_used)
        if len(xs) == 0:
            continue
        ax.plot(
            xs,
            ys,
            "-D",
            color=col_c,
            linewidth=1.8,
            markersize=6,
            label=f"BETA_2={fv} exp(H)",
        )
        ax.plot(xs, n_us, "--^", color=col_c, linewidth=1.0, markersize=4, alpha=0.6)
    ax.axhline(
        REP_DIM, color="gray", linestyle=":", linewidth=1.2, label=f"max={REP_DIM}"
    )
    ax.axhline(4, color="green", linestyle="--", linewidth=1.0, label="zone-optimal=4")
    ax.axhline(1, color="black", linestyle=":", linewidth=1.0, label="collapsed=1")
    ax.set_xlabel("BETA_1")
    ax.set_ylabel("Effective codes  (solid=exp(H), dashed=count)")
    ax.set_title(
        "Effective Latent Codes  exp(H(X))\n"
        "(green dashed = zone-optimal: 4 zones → 4 codes)"
    )
    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    handles = add_slice_lines(ax, b2, u_b2, b2_colors, b1, entr, marker="o")
    ax.set_xlabel("BETA_1")
    ax.set_ylabel("H(X|S)  (nats)")
    ax.set_title("Per-State Encoder Entropy  H(X|S)\n(0=deterministic, high=uncertain)")
    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.legend(handles=handles, title="BETA_2", fontsize=7, title_fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "./outputs/openroom/fig6_latent_compression.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved fig6_latent_compression.png")


    # Figure 7 – Success rate: slice by BETA_2 (left) and slice by BETA_1 (right)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Success Rate — separating BETA_1 and BETA_2 effects", fontsize=13, fontweight="bold"
    )

    ax = axes[0]
    handles = add_slice_lines(ax, b2, u_b2, b2_colors, b1, sr * 100, marker="o")
    ax.set_xlabel("BETA_1")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Fixed BETA_2, varying BETA_1")
    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.set_ylim(-5, 105)
    ax.legend(handles=handles, title="BETA_2", fontsize=7, title_fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    handles = add_slice_lines(ax, b1, u_b1, b1_colors, b2, sr * 100, marker="s")
    ax.set_xlabel("BETA_2")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Fixed BETA_1, varying BETA_2")
    ax.set_xscale("symlog", linthresh=LINTHRESH)
    ax.set_ylim(-5, 105)
    ax.legend(handles=handles, title="BETA_1", fontsize=7, title_fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./outputs/openroom/fig7_success_rate.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig7_success_rate.png")


    # Figure 8 – Reward–Steps scatter, coloured by BETA_1 (left) and BETA_2 (right)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Reward–Efficiency Trade-off", fontsize=13, fontweight="bold")

    for ax, c_arr, c_label in [(axes[0], b1, "BETA_1"), (axes[1], b2, "BETA_2")]:

        norm = mcolors.SymLogNorm(
            linthresh=1e-3,
            linscale=1,
            vmin=min(c_arr),
            vmax=max(c_arr),
        )

        sc = ax.scatter(
            steps,
            rew,
            c=c_arr,
            cmap="coolwarm",
            norm=norm,
            s=60,
            edgecolors="k",
            linewidth=0.4,
            alpha=0.85,
            zorder=3,
        )

        fig.colorbar(sc, ax=ax, label=c_label)
        ax.set_xlabel("Mean Steps to Goal")
        ax.set_ylabel("Mean Reward")
        ax.set_title(f"Coloured by {c_label}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./outputs/openroom/fig8_reward_steps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig8_reward_steps.png")

    print("\nAll figures saved to ./outputs/openroom/")
