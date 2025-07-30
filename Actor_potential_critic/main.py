import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from mpe2 import simple_spread_v3
from networks import QNetwork, PotentialNetwork
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
joint_action_dim = sum(action_dims.values())

# --- Networks ---
q_nets = {}
target_q_nets = {}
for agent in agents:
    q_list, target_list = [], []
    for i in range(2):
        # input: [B, joint_obs_dim] (joint_obs_dim = single_obs_dim * num_agents)
        # output: [B, action_dim_of_this_agent]
        q = QNetwork(joint_obs_dim, action_dims[agent]).to(device) # note: action_dims[agent] = 5
        target_q = QNetwork(joint_obs_dim, action_dims[agent]).to(device)
        ckpt = os.path.join("joint_state_models", f"{agent}_q{i}.pth")
        if os.path.isfile(ckpt):
            q.load_state_dict(torch.load(ckpt, map_location=device))
            target_q.load_state_dict(q.state_dict())
            print("Q loaded")
        q_list.append(q)
        target_list.append(target_q)
    q_nets[agent] = q_list
    target_q_nets[agent] = target_list

# input: concatenation of joint observation and joint one-hot action vector [B, joint_obs_dim + joint_action_dim]
# output: [B, 1]
potentials = [PotentialNetwork(joint_obs_dim + joint_action_dim).to(device) for _ in range(2)] 
target_potentials = [PotentialNetwork(joint_obs_dim + joint_action_dim).to(device) for _ in range(2)]

# Sync targets
for agent in agents:
    for i in range(2):
        target_q_nets[agent][i].load_state_dict(q_nets[agent][i].state_dict()) # redundant, but kept just in case
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
        joint_obs = get_joint_obs(obs, agents) #[54]
        # sample integer actions
        actions = {agent: env.action_space(agent).sample() for agent in agents} # eg [2, 0, 3], not one-hot
        next_obs, rewards, dones, truncs, infos = env.step(actions)
        joint_next_obs = get_joint_obs(next_obs, agents)

        # store integer action indices
        joint_action_idxs = np.array([actions[ag] for ag in agents], dtype=np.int64) # [a0, a1, a2] as ints

        buffer.push(joint_obs, # 54
                    joint_action_idxs, # 3
                    rewards, # dict per agent eg {'agent_0': torch.tensor([...]),  # [B] 'agent_1': torch.tensor([...]), 'agent_2': torch.tensor([...])
                    joint_next_obs, # 54
                    any(dones.values()) or any(truncs.values())) # [B] done flags
        for agent in agents:
            total_reward[agent] += rewards[agent]
        obs = next_obs

        if len(buffer) < BATCH_SIZE:
            continue

        # --- Sample and immediately cast to long on device ---
        s, a_idxs, r_dict, s_, d = buffer.sample(BATCH_SIZE)
        s   = s.to(device) # [B, 54]
        s_  = s_.to(device) # [B, 54]
        d   = d.to(device).float() # [B]
        a_idxs = a_idxs.to(device).long() # [B, 3]

        # --- Q update per agent ---
        ## need to check this parrt, its giving dim issues
        for agent_i, agent in enumerate(agents):
            act_idx = a_idxs[:, agent_i]                # LongTensor [B]
            for i in range(2):
                q_vals = q_nets[agent][i](s)            # [B, 5]
                q_pred = q_vals.gather(1, act_idx.unsqueeze(1)).squeeze(1)  # [B] # qvalue corresponding to the actual action taken

                with torch.no_grad():
                    next_vals = target_q_nets[agent][1-i](s_)  # [B, A]
                    next_q    = next_vals.gather(1, act_idx.unsqueeze(1)).squeeze(1) # [B]
                    target    = r_dict[agent].to(device) + GAMMA * next_q * (1 - d) # [B]

                loss_q = nn.MSELoss()(q_pred, target)
                q_opts[agent][i].zero_grad()
                loss_q.backward()
                q_opts[agent][i].step()
                writer.add_scalar(f"loss/{agent}_q{i}", loss_q.item(), ep*MAX_CYCLES + step)

        # --- Potential update ---
        # rebuild one‐hot on the (now long) indices
        joint_acts_oh = torch.cat([
            nn.functional.one_hot(a_idxs[:, ai], num_classes=action_dims[ag]) #each action becomes [0,0,1,0,0]
            for ai, ag in enumerate(agents)
        ], dim=1).float().to(device)

        for i in range(2):
            inp_curr = torch.cat([s, joint_acts_oh], dim=1) # [B, 54+15 = 69] 
            inp_next = torch.cat([s_, joint_acts_oh], dim=1) 
            pot_curr = potentials[i](inp_curr).squeeze() # [128, 1] becomes [128] after squeeze
            pot_next = target_potentials[1-i](inp_next).squeeze()

            q_term = sum(
                q_nets[ag][i](s).gather(1, a_idxs[:, j].unsqueeze(1)).squeeze(1) ## should we use targets here??
                - q_nets[ag][i](s_).gather(1, a_idxs[:, j].unsqueeze(1)).squeeze(1)
                for j, ag in enumerate(agents)
            )

            pot_loss = nn.MSELoss()(pot_curr - pot_next, q_term)
            p_opts[i].zero_grad()
            pot_loss.backward()
            p_opts[i].step()
            writer.add_scalar(f"loss/potential{i}", pot_loss.item(), ep*MAX_CYCLES + step)

        # --- Soft updates ---
        if step % TARGET_UPDATE_FREQ == 0:
            for agent in agents:
                for i in range(2):
                    for tp, p in zip(target_q_nets[agent][i].parameters(),
                                     q_nets[agent][i].parameters()):
                        tp.data.copy_(TAU*p.data + (1-TAU)*tp.data)
            for i in range(2):
                for tp, p in zip(target_potentials[i].parameters(),
                                 potentials[i].parameters()):
                    tp.data.copy_(TAU*p.data + (1-TAU)*tp.data)

    avg_reward = np.mean(list(total_reward.values()))
    writer.add_scalar("avg_reward", avg_reward, ep)
    episode_rewards.append(avg_reward)
    if ep % 100 == 0:
        print(f"Ep {ep}, Avg reward: {avg_reward:.2f}")

writer.close()
env.close()
