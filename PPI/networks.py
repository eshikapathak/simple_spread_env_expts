# networks.py
import torch.nn as nn

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
        # x: [B, input_dim]
        return self.net(x).squeeze(-1)  # [B] directly

class PotentialNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        # x: [B, input_dim]
        return self.net(x).squeeze(-1)

