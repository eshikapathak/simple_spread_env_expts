# main.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from mpe2 import simple_spread_v3
from datetime import datetime
from utils import ReplayBuffer, get_joint_obs, get_joint_action
from networks import QNetwork, PotentialNetwork

# --- Hyperparameters ---
GAMMA = 0.99
LR_Q = 1e-4
LR_P = 1e-4
BATCH_SIZE = 128
NUM_EPISODES = 5000
MAX_CYCLES = 50
TARGET_UPDATE_FREQ = 100
TAU = 0.05

# Logging & dirs
log_dir = "./apc_logs_jstateaction"
os.makedirs(log_dir, exist_ok=True)
# optional: include hyperparams in run_id
param_str = f"g{GAMMA}_lr{LR_Q}_bs{BATCH_SIZE}_tu{TARGET_UPDATE_FREQ}_mc{MAX_CYCLES}"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_id = f"apc_{param_str}_{timestamp}"
log_dir = os.path.join(log_dir, run_id)
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
joint_action_dim = len(agents)
input_dim = joint_obs_dim + joint_action_dim

# --- Networks ---
q_nets = {}
target_q_nets = {}
for agent in agents:
    q_list, target_list = [], []
    for i in range(2):
        q = QNetwork(input_dim).to(device)         # input: [B, input_dim], output: [B,1]
        target_q = QNetwork(input_dim).to(device)
        ckpt = os.path.join("joint_state_action_models", f"{agent}_q{i}.pth")
        if os.path.isfile(ckpt):
            q.load_state_dict(torch.load(ckpt, map_location=device))
            target_q.load_state_dict(q.state_dict())
            print(f"Loaded {agent}_q{i}")
        q_list.append(q)
        target_list.append(target_q)
    q_nets[agent] = q_list
    target_q_nets[agent] = target_list

potentials = [PotentialNetwork(input_dim).to(device) for _ in range(2)]
target_potentials = [PotentialNetwork(input_dim).to(device) for _ in range(2)]

# Sync targets
for agent in agents:
    for i in range(2):
        target_q_nets[agent][i].load_state_dict(q_nets[agent][i].state_dict())
for i in range(2):
    target_potentials[i].load_state_dict(potentials[i].state_dict())

# Optimizers
q_opts = {agent: [optim.Adam(q.parameters(), lr=LR_Q) for q in q_nets[agent]] for agent in agents}
p_opts = [optim.Adam(p.parameters(), lr=LR_P) for p in potentials]

# Replay Buffer
buffer = ReplayBuffer(capacity=100000)

# Training Loop
for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    total_reward = {agent: 0.0 for agent in agents}
    prev_actions = {agent: 0 for agent in agents}

    for step in range(MAX_CYCLES):
        joint_obs = get_joint_obs(obs, agents)  # [joint_obs_dim]
        actions = {agent: env.action_space(agent).sample() for agent in agents}
        joint_action_idxs = get_joint_action(actions, agents)  # [joint_action_dim]

        next_obs, rewards, dones, truncs, infos = env.step(actions)
        joint_next_obs = get_joint_obs(next_obs, agents)
        next_actions = {agent: env.action_space(agent).sample() for agent in agents}
        joint_next_action_idxs = get_joint_action(next_actions, agents)

        buffer.push(
            joint_obs,
            joint_action_idxs,
            rewards,
            joint_next_obs,
            joint_next_action_idxs,
            any(dones.values()) or any(truncs.values())
        )
        for ag in agents:
            total_reward[ag] += rewards[ag]
        obs = next_obs

        if len(buffer) < BATCH_SIZE:
            continue

        # Sample batch
        s, ja, r_dict, s_, ja_, d = buffer.sample(BATCH_SIZE)
        s, s_, d = s.to(device), s_.to(device), d.to(device)

        # --- Q update ---
        for agent_i, agent in enumerate(agents):
            for i in range(2):
                inp       = torch.cat([s, ja], dim=1)            # [B, input_dim]
                q_pred    = q_nets[agent][i](inp).squeeze(1)     # [B]
                with torch.no_grad():
                    next_inp = torch.cat([s_, ja_], dim=1)        # [B, input_dim]
                    next_q   = target_q_nets[agent][1-i](next_inp).squeeze(1)  # [B]
                    target   = r_dict[agent].to(device) + GAMMA * next_q * (1 - d)
                loss_q = nn.MSELoss()(q_pred, target)
                q_opts[agent][i].zero_grad()
                loss_q.backward()
                q_opts[agent][i].step()
                writer.add_scalar(f"loss/{agent}_q{i}", loss_q.item(), ep*MAX_CYCLES + step)

        # --- Potential update ---
        for i in range(2):
            inp_curr = torch.cat([s, ja], dim=1)            # [B, input_dim]
            inp_next = torch.cat([s_, ja_], dim=1)
            pot_curr = potentials[i](inp_curr).squeeze(1)    # [B]
            pot_next = target_potentials[1-i](inp_next).squeeze(1)

            q_term = torch.zeros_like(pot_curr)
            for j, ag in enumerate(agents):
                q_i      = q_nets[ag][i](inp).squeeze(1)       # [B]
                q_i_next = q_nets[ag][i](inp_next).squeeze(1)  # [B]
                q_term  += (q_i - q_i_next)

            loss_p = nn.MSELoss()(pot_curr - pot_next, q_term)
            p_opts[i].zero_grad()
            loss_p.backward()
            p_opts[i].step()
            writer.add_scalar(f"loss/potential{i}", loss_p.item(), ep*MAX_CYCLES + step)

        # Soft updates
        if step % TARGET_UPDATE_FREQ == 0:
            for ag in agents:
                for i in range(2):
                    for tp, p in zip(target_q_nets[ag][i].parameters(), q_nets[ag][i].parameters()):
                        tp.data.copy_(TAU*p.data + (1-TAU)*tp.data)
            for i in range(2):
                for tp, p in zip(target_potentials[i].parameters(), potentials[i].parameters()):
                    tp.data.copy_(TAU*p.data + (1-TAU)*tp.data)

    avg_reward = np.mean(list(total_reward.values()))
    writer.add_scalar("avg_reward", avg_reward, ep)
    if ep % 100 == 0:
        print(f"Ep {ep}, Avg reward: {avg_reward:.2f}")

writer.close()
env.close()
