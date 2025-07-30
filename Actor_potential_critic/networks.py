
# networks.py
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # input: joint_obs_dim + joint_action_dim
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
        # x: [B, input_dim]
        # returns [B, 1]
        return self.net(x)

class PotentialNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)  # [B, 1]
