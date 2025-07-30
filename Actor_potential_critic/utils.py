# actor_potential_critic/utils.py
import numpy as np
import torch
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, joint_obs, joint_action, reward_dict, joint_next_obs, done):
        self.buffer.append((joint_obs, joint_action, reward_dict, joint_next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r_dicts, s_, d = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(np.array(a), dtype=torch.float32)
        s_ = torch.tensor(np.array(s_), dtype=torch.float32)
        d = torch.tensor(np.array(d), dtype=torch.float32)

        # Convert list of dicts to dict of tensors
        reward_dict = {agent: torch.tensor([r[agent] for r in r_dicts], dtype=torch.float32) for agent in r_dicts[0]}

        return s, a, reward_dict, s_, d

    def __len__(self):
        return len(self.buffer)


def get_joint_obs(obs_dict, agent_order):
    return np.concatenate([obs_dict[agent] for agent in agent_order])
