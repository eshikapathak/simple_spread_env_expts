import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from mpe2 import simple_spread_v3
from modules import QNetwork, PotentialNetwork
from utils import ReplayBuffer, get_joint_obs

# --- Hyperparameters ---
GAMMA = 0.99
LR_Q = 1e-4
LR_P = 1e-4
BATCH_SIZE = 128
NUM_EPISODES = 5000
MAX_CYCLES = 50
TARGET_UPDATE_FREQ = 100
TAU = 0.05

log_dir = "./apc_logs"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize env ---
env = simple_spread_v3.parallel_env(render_mode=None, max_cycles=MAX_CYCLES)
obs, _ = env.reset()
agents = env.agents
action_dims = {agent: env.action_space(agent).n for agent in agents}
single_obs_dim = len(next(iter(obs.values())))
joint_obs_dim = single_obs_dim * len(agents)

# Precompute action index slices for splitting joint action
action_sizes = [action_dims[agent] for agent in agents]
action_starts = np.cumsum([0] + action_sizes[:-1])
action_slices = {agent: slice(start, start + action_dims[agent])
                 for agent, start in zip(agents, action_starts)}

# --- Networks ---
# QNetworks take joint state and individual agent action
q_nets = {}
target_q_nets = {}
for agent in agents:
    q_list = []
    target_list = []
    for i in range(2):
        q = QNetwork(joint_obs_dim, action_dims[agent]).to(device)
        target_q = QNetwork(joint_obs_dim, action_dims[agent]).to(device)
        # Load pretrained weights if available
        ckpt_path = os.path.join("trained_q_nets", f"{agent}_q{i}.pth") ## need to give correct path to already trained models
        if os.path.isfile(ckpt_path):
            q.load_state_dict(torch.load(ckpt_path, map_location=device))
            target_q.load_state_dict(q.state_dict())
        q_list.append(q)
        target_list.append(target_q)
    q_nets[agent] = q_list
    target_q_nets[agent] = target_list

potentials = [PotentialNetwork(joint_obs_dim + sum(action_dims.values())).to(device) for _ in range(2)]
target_potentials = [PotentialNetwork(joint_obs_dim + sum(action_dims.values())).to(device) for _ in range(2)]

# Sync remaining targets
for agent in agents:
    for i in range(2):
        target_q_nets[agent][i].load_state_dict(q_nets[agent][i].state_dict())
for i in range(2):
    target_potentials[i].load_state_dict(potentials[i].state_dict())

# --- Optimizers ---
q_opts = {agent: [optim.Adam(q.parameters(), lr=LR_Q) for q in q_nets[agent]] for agent in agents}
p_opts = [optim.Adam(p.parameters(), lr=LR_P) for p in potentials]

# --- Replay Buffer ---
buffer = ReplayBuffer(capacity=100000)

# --- Training Loop ---
episode_rewards = []
for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    total_reward = {agent: 0.0 for agent in agents}

    for step in range(MAX_CYCLES):
        joint_obs = get_joint_obs(obs, agents)
        # Sample random actions
        actions = {agent: env.action_space(agent).sample() for agent in agents} # actions = actor_module.select_actions(joint_obs) -- with policy v itll probably look something liket his
        next_obs, rewards, dones, truncs, infos = env.step(actions)
        joint_next_obs = get_joint_obs(next_obs, agents)
        # Build joint action one-hot for potentials -- should we do one-hot or directly the action number (eg 4)
        joint_action = np.concatenate([np.eye(action_dims[agent])[actions[agent]] for agent in agents])

        buffer.push(joint_obs, joint_action, rewards, joint_next_obs, any(dones.values()) or any(truncs.values()))
        for agent in agents:
            total_reward[agent] += rewards[agent]
        obs = next_obs

        if len(buffer) < BATCH_SIZE:
            continue

        s, a, r_dict, s_, d = buffer.sample(BATCH_SIZE)
        d = d.float()

        # --- Q update per agent ---
        for agent in agents:
            slice_idx = action_slices[agent]
            a_agent = a[:, slice_idx]
            for i in range(2):
                q_pred = q_nets[agent][i](s, a_agent).squeeze()
                with torch.no_grad():
                    next_q = target_q_nets[agent][1 - i](s_, a_agent).squeeze()
                    target = r_dict[agent] + GAMMA * next_q * (1 - d)
                loss_q = nn.MSELoss()(q_pred, target)
                q_opts[agent][i].zero_grad()
                loss_q.backward()
                writer.add_scalar(f"loss/{agent}_q{i}", loss_q.item(), ep * MAX_CYCLES + step)
                q_opts[agent][i].step()

        # --- Potential update ---
        for i in range(2):
            pot_curr = potentials[i](torch.cat([s, a], dim=1)).squeeze()
            pot_next = target_potentials[1 - i](torch.cat([s_, a], dim=1)).squeeze()
            q_term = sum(
                q_nets[agent][i](s, a[:, action_slices[agent]]).squeeze() -
                q_nets[agent][i](s_, a[:, action_slices[agent]]).squeeze()
                for agent in agents)
            pot_loss = nn.MSELoss()(pot_curr - pot_next, q_term)
            p_opts[i].zero_grad()
            pot_loss.backward()
            writer.add_scalar(f"loss/potential{i}", pot_loss.item(), ep * MAX_CYCLES + step)
            p_opts[i].step()

        # --- Soft updates ---
        if step % TARGET_UPDATE_FREQ == 0:
            for agent in agents:
                for i in range(2):
                    for t_param, s_param in zip(target_q_nets[agent][i].parameters(), q_nets[agent][i].parameters()):
                        t_param.data.copy_(TAU * s_param.data + (1 - TAU) * t_param.data)
            for i in range(2):
                for t_param, s_param in zip(target_potentials[i].parameters(), potentials[i].parameters()):
                    t_param.data.copy_(TAU * s_param.data + (1 - TAU) * t_param.data)

    avg_reward = np.mean(list(total_reward.values()))
    writer.add_scalar("avg_reward", avg_reward, ep)
    episode_rewards.append(avg_reward)
    if ep % 100 == 0:
        print(f"Ep {ep}, Avg reward: {avg_reward:.2f}")

writer.close()
env.close()
