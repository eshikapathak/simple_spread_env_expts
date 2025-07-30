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
BUFFER_SIZE = 100000
MAX_STEPS = 10000      # env steps per iteration
NUM_ITERATIONS = 1000
TARGET_UPDATE_FREQ = 100
TAU = 0.01
BETA = 1.0            # inverse temperature for booltzmann

# Logging setup
log_root = './ppi_logs'
os.makedirs(log_root, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = os.path.join(log_root, timestamp)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Policy improvement: sample joint action via Boltzmann over potential
def sample_joint_action(state, potentials, beta, env, agents, device):
    with torch.no_grad():
        s_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        joint_actions = []
        pot_values = []
        # enumerate all joint actions (cartesian product)
        for a_idxs in np.ndindex(*[env.action_space(ag).n for ag in agents]): # the enumerate is going to make this whole thing very slow but idk how else to do optimally
            ja = np.array(a_idxs, dtype=np.float32)
            inp = torch.cat([s_tensor, torch.tensor(ja, device=device).unsqueeze(0)], dim=1)
            pot_values.append(potentials[0](inp).item())
            joint_actions.append(ja)
        logits = torch.tensor(pot_values, dtype=torch.float32)
        probs = torch.softmax(beta * logits, dim=0).cpu().numpy()
        idx = np.random.choice(len(joint_actions), p=probs)
        choice = joint_actions[idx]
        action = {ag: int(choice[i]) for i, ag in enumerate(agents)}
    return action, choice

# Environment init
env = simple_spread_v3.parallel_env(max_cycles=MAX_STEPS)
obs, _ = env.reset()
agents = env.agents
single_obs_dim = len(obs[agents[0]])
joint_obs_dim = single_obs_dim * len(agents)
joint_action_dim = len(agents)
input_dim = joint_obs_dim + joint_action_dim

# Initialize Q-networks and optimizers
global q_nets, target_q_nets, q_optim
# q_nets = {ag: QNetwork(input_dim).to(device) for ag in agents}
# target_q_nets = {ag: QNetwork(input_dim).to(device) for ag in agents}
# q_optim = {ag: optim.Adam(q_nets[ag].parameters(), lr=LR_Q) for ag in agents}
# for ag in agents:
#     target_q_nets[ag].load_state_dict(q_nets[ag].state_dict())
model_dir = "./joint_state_action_models"
os.makedirs(model_dir, exist_ok=True)
q_nets = {}
target_q_nets = {}
q_optim = {}
for ag in agents:
    # create networks
    q = QNetwork(input_dim).to(device)
    target_q = QNetwork(input_dim).to(device)
    # attempt to load existing model
    ckpt_path = os.path.join(model_dir, f"{ag}_qnet.pt")
    if os.path.isfile(ckpt_path):
        q.load_state_dict(torch.load(ckpt_path, map_location=device))
        target_q.load_state_dict(q.state_dict())
        print(f"Loaded Q-network for {ag} from {ckpt_path}")
    else:
        # initialize target with online weights
        target_q.load_state_dict(q.state_dict())
    # assign
    q_nets[ag] = q
    target_q_nets[ag] = target_q
    q_optim[ag] = optim.Adam(q.parameters(), lr=LR_Q)

# Initialize potentials and optimizers
potentials = [PotentialNetwork(input_dim).to(device) for _ in range(2)]
target_potentials = [PotentialNetwork(input_dim).to(device) for _ in range(2)]
p_optim = [optim.Adam(p.parameters(), lr=LR_P) for p in potentials]
for i in range(2):
    target_potentials[i].load_state_dict(potentials[i].state_dict())

# Replay buffer
buffer = ReplayBuffer(BUFFER_SIZE)

# PPI main loop
for it in range(NUM_ITERATIONS):
    # 1) data collection
    for _ in range(MAX_STEPS):
        state = get_joint_obs(obs, agents)
        action, ja_idx = sample_joint_action(state, potentials, BETA, env, agents, device)
        next_obs, rewards, dones, truncs, _ = env.step(action)
        done = any(dones.values()) or any(truncs.values())
        buffer.push(state, ja_idx, rewards, get_joint_obs(next_obs, agents), done)
        obs = next_obs
        if done:
            obs, _ = env.reset()

    # 2) Q-function update (off-policy TD)
    for _ in range(BATCH_SIZE):
        if len(buffer) < BATCH_SIZE:
            break
        s, ja, r_dict, s_next, done = buffer.sample(BATCH_SIZE)
        s, ja, s_next, done = [t.to(device) for t in (s, ja, s_next, done)]
        for ag in agents:
            # compute target y = r + gamma * Q_target(s', a)
            with torch.no_grad():
                inp_next = torch.cat([s_next, ja], dim=1)
                q_next = target_q_nets[ag](inp_next)
                y = r_dict[ag].to(device) + GAMMA * q_next * (1 - done)
            # prediction
            inp = torch.cat([s, ja], dim=1)
            q_pred = q_nets[ag](inp)
            loss = nn.MSELoss()(q_pred, y)
            q_optim[ag].zero_grad()
            loss.backward()
            q_optim[ag].step()
    # soft update Q-targets
    if it % TARGET_UPDATE_FREQ == 0:
        for ag in agents:
            for tp, p in zip(target_q_nets[ag].parameters(), q_nets[ag].parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

    # 3) Potential function update -- this is what i understand from our discussion today
    for _ in range(BATCH_SIZE):
        if len(buffer) < BATCH_SIZE:
            break
        s, ja, r_dict, s_next, done = buffer.sample(BATCH_SIZE)
        s, ja, s_next = [t.to(device) for t in (s, ja, s_next)]
        for i in range(2):
            # sample an agent and two actions ai, ai_bar 
            idx = random.randrange(len(agents))
            ag = agents[idx]
            ai = ja[:, idx]
            # sample ai_bar != ai
            n_a = env.action_space(ag).n
            ai_bar = torch.randint(0, n_a, ai.shape, device=device)
            mask = (ai_bar == ai)
            while mask.any():
                ai_bar[mask] = torch.randint(0, n_a, (mask.sum().item(),), device=device)
                mask = (ai_bar == ai)
            a_minus = ja.clone()
            # build inputs
            inp_ai = torch.cat([s, ja], dim=1)
            a_minus[:, idx] = ai_bar
            inp_aibar = torch.cat([s, a_minus], dim=1)
            # Q-difference target
            with torch.no_grad():
                y_psi = q_nets[ag](inp_ai) - q_nets[ag](inp_aibar)
            # potential loss
            phi = potentials[i](inp_ai)
            phi_bar = target_potentials[i](inp_aibar)
            loss_p = nn.MSELoss()(phi - phi_bar, y_psi)
            p_optim[i].zero_grad()
            loss_p.backward()
            p_optim[i].step()
    # soft update potentials
    if it % TARGET_UPDATE_FREQ == 0:
        for i in range(2):
            for tp, p in zip(target_potentials[i].parameters(), potentials[i].parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

    # logging
    writer.add_scalar('iteration', it, it)
    if it % 10 == 0:
        print(f'Iteration {it} completed')

writer.close()
env.close()
