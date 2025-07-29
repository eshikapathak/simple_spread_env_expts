import numpy as np
import torch
import matplotlib.pyplot as plt
from mpe2 import simple_spread_v3
import os

# Environment setup
MAX_CYCLES = 25
NUM_TEST_EPISODES = 5

# Load trained Q-networks
model_dir = "./joint_state_models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load environment to get agent info
env = simple_spread_v3.parallel_env(render_mode="human", max_cycles=MAX_CYCLES)
obs, _ = env.reset()
agents = env.agents
obs_dim = len(next(iter(obs.values())))
joint_obs_dim = obs_dim * len(agents)
action_spaces = {agent: env.action_space(agent).n for agent in agents}

# Q-network definition
class QNetwork(torch.nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# Load models
q_nets = {}
for agent in agents:
    q_net = QNetwork(joint_obs_dim, action_spaces[agent]).to(device)
    q_net.load_state_dict(torch.load(os.path.join(model_dir, f"{agent}_qnet.pt"), map_location=device))
    q_net.eval()
    q_nets[agent] = q_net

# Helper
def get_joint_obs(obs_dict, agent_order):
    return np.concatenate([obs_dict[agent] for agent in agent_order])

# Run test episodes
episode_rewards = []
for ep in range(NUM_TEST_EPISODES):
    obs, _ = env.reset()
    total_reward = {agent: 0.0 for agent in agents}

    for t in range(MAX_CYCLES):
        joint_obs = get_joint_obs(obs, agents)
        actions = {}

        with torch.no_grad():
            for agent in agents:
                input_tensor = torch.tensor(joint_obs, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = q_nets[agent](input_tensor).squeeze()
                actions[agent] = int(torch.argmax(q_values).item())

        obs, rewards, terminations, truncations, infos = env.step(actions)

        for agent in agents:
            total_reward[agent] += rewards[agent]

    avg_ep_reward = np.mean(list(total_reward.values()))
    episode_rewards.append(avg_ep_reward)
    print(f"Episode {ep+1}: Avg reward = {avg_ep_reward:.2f}")

# Plot results
plt.figure(figsize=(8, 4))
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Testing Performance of Trained Agents")
plt.grid(True)
plt.show()

env.close()
