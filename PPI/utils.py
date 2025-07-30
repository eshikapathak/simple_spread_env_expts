# utils.py
import numpy as np
import torch
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward_dict, next_state, done):
        # state: np.array [joint_obs_dim]
        # action: np.array [joint_action_dim] integer indices
        # reward_dict: dict(agent->float)
        # next_state: np.array [joint_obs_dim]
        # done: bool
        self.buffer.append((state, action, reward_dict, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards_list, next_states, dones = zip(*batch)

        s = torch.tensor(np.stack(states), dtype=torch.float32)
        a = torch.tensor(np.stack(actions), dtype=torch.float32)
        s_next = torch.tensor(np.stack(next_states), dtype=torch.float32)
        done = torch.tensor(dones, dtype=torch.float32)

        # convert list of dicts to dict of tensors
        reward_dict = {agent: torch.tensor([r[agent] for r in rewards_list], dtype=torch.float32)
                       for agent in rewards_list[0].keys()}

        return s, a, reward_dict, s_next, done

    def __len__(self):
        return len(self.buffer)


def get_joint_obs(obs, agents):
    # returns np array [joint_obs_dim]
    return np.concatenate([obs[ag] for ag in agents])

def get_joint_action(actions, agents):
    # returns np array [joint_action_dim] integer indices
    return np.array([actions[ag] for ag in agents], dtype=np.int64)
