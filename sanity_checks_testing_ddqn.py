import os
import torch
import numpy as np
import torch.nn as nn
from mpe2 import simple_spread_v3

# ---- Config ----
model_dir = "./models"
env = simple_spread_v3.parallel_env(render_mode=None, max_cycles=25)
obs, _ = env.reset(seed=123)
agents = env.agents
obs_dim = len(obs[agents[0]])
action_spaces = {agent: env.action_space(agent).n for agent in agents}

# Print landmark sizes
print("Landmark sizes (radii) at environment start:")
for i, landmark in enumerate(env.unwrapped.world.landmarks):
    print(f"  Landmark {i} radius: {landmark.size}")

# Cache agent order and properties
agent_names = env.agents
agent_index_map = {agent: i for i, agent in enumerate(agent_names)}
agents_world = env.unwrapped.world.policy_agents
# Print agent sizes
print("\nAgent sizes:")
for agent, obj in zip(agent_names, agents_world):
    print("  ", agent, "size =", obj.size)


# Q-network (same as training)
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)

# ---- Load Q-networks ----
q_nets = {}
for agent in agents:
    q_net = QNetwork(obs_dim, action_spaces[agent])
    q_net.load_state_dict(torch.load(os.path.join(model_dir, f"{agent}_qnet.pt"), map_location="cpu"))
    q_net.eval()
    q_nets[agent] = q_net

# ---- Helper ----
def get_distances(p, other_ps):
    return [np.linalg.norm(p - op) for op in other_ps]
# Action meaning mapping
ACTION_MEANINGS = {
    0: 'stay',
    1: 'up',
    2: 'down',
    3: 'left',
    4: 'right'
}

# Direction helper
def get_relative_direction(src, dest):
    dx = dest[0] - src[0]
    dy = dest[1] - src[1]
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return 'same'
    if abs(dx) < 1e-3:
        return 'up' if dy > 0 else 'down'
    if abs(dy) < 1e-3:
        return 'right' if dx > 0 else 'left'
    if dx > 0 and dy > 0:
        return 'up-right'
    elif dx > 0 and dy < 0:
        return 'down-right'
    elif dx < 0 and dy > 0:
        return 'up-left'
    else:
        return 'down-left'

# ---- Run Sanity Check ----
print("\nSanity check on trained DDQN Q_i networks for each agent")

# Loop
for t in range(4):  # 4 steps
    print(f"\n--- Step {t} ---")
    actions = {}

    for agent in agents:
        obs_vec = obs[agent]
        pos = obs_vec[0:2]
        vel = obs_vec[2:4]
        landmark_pos = [obs_vec[4 + 2*i:6 + 2*i] for i in range(3)]
        other_agent_pos = [obs[a][0:2] for a in agents if a != agent]

        landmark_dists = get_distances(pos, landmark_pos)
        agent_dists = get_distances(pos, other_agent_pos)

        print(f"{agent} position: {pos}")
        print(f"{agent} velocity: {vel}")
        print(f"{agent} distances from landmarks: {landmark_dists}")
        print(f"{agent} distances from other agents: {agent_dists}")

        # Relative directions
        landmark_dirs = [get_relative_direction(pos, l_pos) for l_pos in landmark_pos]
        agent_dirs = [get_relative_direction(pos, a_pos) for a_pos in other_agent_pos]

        # print(landmark_dists)

        for i, (d, direction) in enumerate(zip(landmark_dists, landmark_dirs)):
            print(f"{agent} -> Landmark {i}: {direction} ({d:.2f})")
        for i, (d, direction) in enumerate(zip(agent_dists, agent_dirs)):
            print(f"{agent} -> Agent {i}: {direction} ({d:.2f})")

        # Q-values
        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_vals = q_nets[agent](obs_tensor).squeeze().numpy()
        best_action = int(np.argmax(q_vals))

        print(f"{agent} Q-values: {q_vals}")
        print(f"{agent} chosen action: {best_action} ({ACTION_MEANINGS[best_action]})")

        actions[agent] = best_action

    # Step env
    next_obs, rewards, terminations, truncations, infos = env.step(actions)
    obs = next_obs

env.close()