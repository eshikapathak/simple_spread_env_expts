import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import matplotlib.pyplot as plt
from mpe2 import simple_spread_v3

# Hyperparameters
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 100_000
TARGET_UPDATE_FREQ = 100
EPS_START = 1.0
EPS_END = 0.001
EPS_DECAY = 0.999

NUM_EPISODES = 5000
MAX_CYCLES = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Setup logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
reward_log_path = os.path.join(log_dir, "reward_log.npy")
model_dir = "./models"
os.makedirs(model_dir, exist_ok=True)

# 2. Initialize reward tracking
episode_rewards = []

# Q-network
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


# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32).to(device),
            torch.tensor(a).to(device),
            torch.tensor(r).to(device),
            torch.tensor(np.array(s_), dtype=torch.float32).to(device),
            torch.tensor(d).to(device),
        )

    def __len__(self):
        return len(self.buffer)


def select_action(q_net, obs, epsilon, action_space):
    if random.random() < epsilon:
        return action_space.sample()
    with torch.no_grad():
        q_values = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device))
        return torch.argmax(q_values).item()


# Initialize env and agents
env = simple_spread_v3.parallel_env(render_mode=None, max_cycles=MAX_CYCLES)
obs, _ = env.reset()
agents = env.agents
action_spaces = {agent: env.action_space(agent).n for agent in agents}
# print("obs =", obs)
# print("type(obs) =", type(obs))
obs_dim = len(next(iter(obs.values())))


# Per-agent Q-networks
q_nets = {agent: QNetwork(obs_dim, action_spaces[agent]).to(device) for agent in agents}
target_nets = {agent: QNetwork(obs_dim, action_spaces[agent]).to(device) for agent in agents}
optimizers = {agent: optim.Adam(q_nets[agent].parameters(), lr=LR) for agent in agents}
buffers = {agent: ReplayBuffer(REPLAY_BUFFER_SIZE) for agent in agents}

# Copy weights to target networks
for agent in agents:
    target_nets[agent].load_state_dict(q_nets[agent].state_dict())

epsilon = EPS_START

# Training loop
for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    total_reward = {agent: 0.0 for agent in agents}

    for step in range(MAX_CYCLES):
        actions = {}
        for agent in agents:
            actions[agent] = select_action(q_nets[agent], obs[agent], epsilon, env.action_space(agent))

        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        for agent in agents:
            buffers[agent].push(
                obs[agent], actions[agent], rewards[agent], next_obs[agent], terminations[agent] or truncations[agent]
            )
            total_reward[agent] += rewards[agent]

        obs = next_obs

        for agent in agents:
            if len(buffers[agent]) < BATCH_SIZE:
                continue

            s, a, r, s_, done = buffers[agent].sample(BATCH_SIZE)
            q_vals = q_nets[agent](s).gather(1, a.unsqueeze(1)).squeeze()
            next_actions = torch.argmax(q_nets[agent](s_), dim=1)
            next_q_vals = target_nets[agent](s_).gather(1, next_actions.unsqueeze(1)).squeeze()
            target = r + GAMMA * next_q_vals * (1 - done.float())

            loss = nn.MSELoss()(q_vals, target.detach())
            optimizers[agent].zero_grad()
            loss.backward()
            optimizers[agent].step()

        if step % TARGET_UPDATE_FREQ == 0:
            for agent in agents:
                target_nets[agent].load_state_dict(q_nets[agent].state_dict())

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    avg_reward = np.mean([total_reward[agent] for agent in agents])
    episode_rewards.append(avg_reward)

    if episode % 50 == 0:
        print(f"Episode {episode} | Avg reward: {avg_reward:.2f} | epsilon: {epsilon:.3f}")

# Save model and rewards
np.save(reward_log_path, episode_rewards)
for agent in agents:
    torch.save(q_nets[agent].state_dict(), os.path.join(model_dir, f"{agent}_qnet.pt"))

# Plot learning curve
plt.figure(figsize=(8, 4))
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Multi-Agent DDQN Training")
plt.grid(True)
plt.savefig(os.path.join(log_dir, "learning_curve.png"))
plt.close()

env.close()

