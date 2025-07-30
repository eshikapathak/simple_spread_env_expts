# utils.py
import numpy as np
import torch
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, joint_obs, joint_action_idxs, reward_dict, joint_next_obs, joint_next_action_idxs, done):
        # joint_obs: [joint_obs_dim]
        # joint_action_idxs: [joint_action_dim] (int indices as floats)
        # reward_dict: dict(agent->float)
        # joint_next_obs: [joint_obs_dim]
        # joint_next_action_idxs: [joint_action_dim]
        # done: bool
        self.buffer.append((joint_obs, joint_action_idxs, reward_dict, joint_next_obs, joint_next_action_idxs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, ja, r_dicts, s_, ja_, d = zip(*batch)

        s      = torch.tensor(np.array(s), dtype=torch.float32)
        ja     = torch.tensor(np.array(ja), dtype=torch.float32)
        s_     = torch.tensor(np.array(s_), dtype=torch.float32)
        ja_    = torch.tensor(np.array(ja_), dtype=torch.float32)
        d      = torch.tensor(np.array(d), dtype=torch.float32)

        # Convert list of dicts to dict of tensors
        reward_dict = {
            agent: torch.tensor([r[agent] for r in r_dicts], dtype=torch.float32)
            for agent in r_dicts[0]
        }

        return s, ja, reward_dict, s_, ja_, d

    def __len__(self):
        return len(self.buffer)


def get_joint_obs(obs_dict, agent_order):
    # returns np array of shape [joint_obs_dim]
    return np.concatenate([obs_dict[agent] for agent in agent_order])


def get_joint_action(actions_dict, agent_order):
    # returns np array of shape [joint_action_dim] with integer action indices
    return np.array([actions_dict[agent] for agent in agent_order], dtype=np.float32)

