import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import matplotlib.pyplot as plt
from mpe2 import simple_spread_v3  
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Hyperparameters
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 150_000
TARGET_UPDATE_FREQ = 400
EPS_START = 1.0
EPS_END = 0.001
EPS_DECAY = 1
NUM_EPISODES = 10000
MAX_CYCLES = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging & dirs
log_root   = "./joint_state_joint_action_logs"
model_dir  = "./joint_state_joint_action_models"
os.makedirs(log_root,  exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
# build a param string: g{GAMMA}_lr{LR}_bs{BATCH_SIZE}_tu{TARGET_UPDATE_FREQ}_mc{MAX_CYCLES}
param_str = (
    f"g{GAMMA}"
    f"_lr{LR}"
    f"_bs{BATCH_SIZE}"
    f"_tu{TARGET_UPDATE_FREQ}"
    f"_mc{MAX_CYCLES}"
)# add timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# final run ID
run_id   = f"joint_{param_str}_{timestamp}"
log_dir  = os.path.join(log_root, run_id)
os.makedirs(log_dir, exist_ok=True)

# now point SummaryWriter at that specific folder
writer = SummaryWriter(log_dir=log_dir)

reward_log_path = os.path.join(log_dir, "reward_log.npy")

episode_rewards = []

# Helpers
def get_joint_obs(obs_dict, agent_order):
    # returns np array shape [joint_obs_dim]
    return np.concatenate([obs_dict[a] for a in agent_order])

def get_joint_action(actions_dict, agent_order):
    # returns np array shape [joint_action_dim]
    return np.array([actions_dict[a] for a in agent_order], dtype=np.float32)

# Env init
env = simple_spread_v3.parallel_env(render_mode=None, max_cycles=MAX_CYCLES)
obs, _ = env.reset()
agents = env.agents
action_spaces = {ag: env.action_space(ag).n for ag in agents}

# Dimensions
single_obs_dim   = len(next(iter(obs.values())))
joint_obs_dim    = single_obs_dim * len(agents)       
joint_action_dim = len(agents)                         # 3 
input_dim        = joint_obs_dim + joint_action_dim    # 54 + 3 = 57

# Q-network now outputs a single scalar
class QNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # input: [joint_obs_dim + joint_action_dim]
        # output: scalar Q-value
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),  
            nn.ReLU(),
            nn.Linear(512, 256),        
            nn.ReLU(),
            nn.Linear(256, 128),        
            nn.ReLU(),
            nn.Linear(128, 1)          
        )

    def forward(self, x):
        return self.net(x)  # in: [B, input_dim], out: [B, 1]

# Replay Buffer stores joint_obs, joint_action, reward, joint_next_obs, joint_next_action, done
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, ja, r, s_, ja_, done):
        # s:   [joint_obs_dim]
        # ja:  [joint_action_dim]
        # r:   scalar
        # s_:  [joint_obs_dim]
        # ja_: [joint_action_dim]
        # done: bool
        self.buffer.append((s, ja, r, s_, ja_, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, ja, r, s_, ja_, d = zip(*batch)
        return (
            torch.tensor(np.array(s),   dtype=torch.float32).to(device),  # [B, joint_obs_dim]
            torch.tensor(np.array(ja),  dtype=torch.float32).to(device),  # [B, joint_action_dim]
            torch.tensor(r,             dtype=torch.float32).to(device),  # [B]
            torch.tensor(np.array(s_),  dtype=torch.float32).to(device),  # [B, joint_obs_dim]
            torch.tensor(np.array(ja_), dtype=torch.float32).to(device),  # [B, joint_action_dim]
            torch.tensor(d,             dtype=torch.float32).to(device),  # [B]
        )

    def __len__(self):
        return len(self.buffer)

# Epsilon-greedy: for each possible own action, swap into prev_joint_action and eval Q
def select_action(q_net, state, prev_ja, epsilon, action_space, agent_idx):
    # state: [joint_obs_dim], prev_ja: [joint_action_dim]
    if random.random() < epsilon:
        return action_space.sample()
    # greedy: test each candidate a_i
    q_vals = []
    for a_i in range(action_space.n):
        ja_candidate = prev_ja.copy()
        ja_candidate[agent_idx] = a_i
        inp = torch.tensor(np.concatenate([state, ja_candidate]),
                           dtype=torch.float32).unsqueeze(0).to(device)  # [1, input_dim]
        with torch.no_grad():
            q = q_net(inp)  # [1,1]
        q_vals.append(q.item())
    return int(np.argmax(q_vals))

# Instantiate networks, optimizers, buffers
q_nets      = {ag: QNetwork(input_dim).to(device) for ag in agents}
target_nets = {ag: QNetwork(input_dim).to(device) for ag in agents}
optimizers  = {ag: optim.Adam(q_nets[ag].parameters(), lr=LR) for ag in agents}
buffers     = {ag: ReplayBuffer(REPLAY_BUFFER_SIZE) for ag in agents}

# Sync targets
for ag in agents:
    target_nets[ag].load_state_dict(q_nets[ag].state_dict())

epsilon = EPS_START

# Training loop
for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    total_reward = {ag: 0.0 for ag in agents}
    prev_joint_action = np.zeros(joint_action_dim, dtype=np.float32)

    for step in range(MAX_CYCLES):
        joint_obs = get_joint_obs(obs, agents)  # [joint_obs_dim]
        # select actions
        actions = {}
        for i, ag in enumerate(agents):
            actions[ag] = select_action(
                q_nets[ag],
                joint_obs, 
                prev_joint_action, 
                epsilon,
                env.action_space(ag),
                i
            )
        joint_action = get_joint_action(actions, agents)  # [joint_action_dim]

        # step
        next_obs, rewards, terms, truns, infos = env.step(actions)
        joint_next_obs = get_joint_obs(next_obs, agents)   # [joint_obs_dim]
        # compute next_joint_action for buffer
        next_actions = {}
        for i, ag in enumerate(agents):
            next_actions[ag] = select_action(
                q_nets[ag],
                joint_next_obs,
                joint_action,
                epsilon,
                env.action_space(ag),
                i
            )
        joint_next_action = get_joint_action(next_actions, agents)

        # store transitions
        for ag in agents:
            buffers[ag].push(
                joint_obs, joint_action,
                rewards[ag],
                joint_next_obs, joint_next_action,
                terms[ag] or truns[ag]
            )
            total_reward[ag] += rewards[ag]

        obs = next_obs
        prev_joint_action = joint_action

        # learn
        for ag in agents:
            if len(buffers[ag]) < BATCH_SIZE:
                continue

            s, ja, r, s_, ja_, done = buffers[ag].sample(BATCH_SIZE)
            inp      = torch.cat([s,  ja],  dim=1)  # [B, input_dim]
            q_vals   = q_nets[ag](inp).squeeze(1)    # [B]
            next_inp = torch.cat([s_, ja_], dim=1)  # [B, input_dim]
            next_q   = target_nets[ag](next_inp).squeeze(1)  # [B]
            target   = r + GAMMA * next_q * (1 - done)

            loss = nn.MSELoss()(q_vals, target.detach())
            optimizers[ag].zero_grad()
            loss.backward()
            writer.add_scalar(f"{ag}/loss", loss.item(), episode * MAX_CYCLES + step)
            optimizers[ag].step()

        # target sync
        if step % TARGET_UPDATE_FREQ == 0:
            for ag in agents:
                target_nets[ag].load_state_dict(q_nets[ag].state_dict())

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    avg_reward = np.mean(list(total_reward.values()))
    writer.add_scalar("avg_reward", avg_reward, episode)
    episode_rewards.append(avg_reward)
    if episode % 100 == 0:
        print(f"Episode {episode} | Avg reward: {avg_reward:.2f} | eps: {epsilon:.3f}")

writer.close()

# save & plot
np.save(reward_log_path, episode_rewards)
for ag in agents:
    torch.save(q_nets[ag].state_dict(), os.path.join(model_dir, f"{ag}_qnet.pt"))

plt.figure(figsize=(8, 4))
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Multi-Agent Q(s,a) -> scalar")
plt.grid(True)
plt.savefig(os.path.join(log_dir, "learning_curve.png"))
plt.close()

env.close()
