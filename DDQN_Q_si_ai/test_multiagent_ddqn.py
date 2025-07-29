import numpy as np
import torch
import torch.nn as nn
from mpe2 import simple_spread_v3
import os
import matplotlib.pyplot as plt

# Evaluation settings
NUM_EVAL_EPISODES = 10
MAX_CYCLES = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Greedy policy
def select_greedy_action(q_net, obs):
    with torch.no_grad():
        q_values = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device))
        return torch.argmax(q_values).item()

# Load environment
env = simple_spread_v3.parallel_env(render_mode="human", max_cycles=MAX_CYCLES)
obs, _ = env.reset()
agents = env.agents
action_spaces = {agent: env.action_space(agent).n for agent in agents}
obs_dim = len(next(iter(obs.values())))

# Load saved Q-networks
q_nets = {}
for agent in agents:
    q_net = QNetwork(obs_dim, action_spaces[agent]).to(device)
    model_path = f"models/{agent}_qnet.pt"
    assert os.path.exists(model_path), f"Missing model for {agent}: {model_path}"
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()
    q_nets[agent] = q_net

# Evaluation
episode_rewards = []

for ep in range(NUM_EVAL_EPISODES):
    obs, _ = env.reset()
    total_reward = {agent: 0.0 for agent in agents}

    for step in range(MAX_CYCLES):
        actions = {}
        for agent in agents:
            actions[agent] = select_greedy_action(q_nets[agent], obs[agent]) #env.action_space(agent).sample() #select_greedy_action(q_nets[agent], obs[agent])
        obs, rewards, terminations, truncations, infos = env.step(actions)
        for agent in agents:
            total_reward[agent] += rewards[agent]

    episode_rewards.append(total_reward)
    print(f"Episode {ep + 1}: ", end="")
    for agent in agents:
        print(f"{agent}: {total_reward[agent]:.2f}", end="  ")
    print()

# Final average per agent
agent_names = list(agents)
avg_rewards = {agent: np.mean([ep[agent] for ep in episode_rewards]) for agent in agents}
print("\nFinal Average Reward Per Agent (over all episodes):")
for agent in agents:
    print(f"{agent}: {avg_rewards[agent]:.2f}")

# Optional: plot
plt.figure(figsize=(6, 4))
plt.bar(agent_names, [avg_rewards[a] for a in agent_names])
plt.ylabel("Avg Reward")
plt.title("Evaluation: Per-Agent Average Reward")
plt.tight_layout()
os.makedirs("logs", exist_ok=True)
plt.savefig("logs/eval_barplot.png")
plt.show()

env.close()
