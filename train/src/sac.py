from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

from src.networks import SoftModuleBlock, MLP

'''
[
    "left_c",
    "left_t",
    "right_c",
    "right_t",
    "cruise",
    "merge",
    "cut_in",
    "overtake",
    "curve",
]
'''

class DuelingQNet(nn.Module):
    def __init__(self, o_dim, a_dim, z_map_dim=9):
        super().__init__()
        self.softmodule_block_map = SoftModuleBlock(o_dim, z_map_dim, 128, 128, 4, 4, True)
        self.V_layer = nn.Linear(128, 1)
        self.A_layer = MLP(128, a_dim, [128])
        
    def forward(self, o, z_map, collision_prob):
        feature = self.softmodule_block_map(o, z_map, collision_prob)
        value = self.V_layer(feature)
        advantage = self.A_layer(feature)
        advantage -= advantage.mean(-1, keepdim=True)
        return advantage + value
    
class PolicyNet(nn.Module):
    def __init__(self, o_dim, a_dim, z_map_dim=9):
        super().__init__()
        self.softmodule_block_map = SoftModuleBlock(o_dim, z_map_dim, a_dim, 128, 4, 4, True)

    def forward(self, o, z_map, collision_prob):
        logits = self.softmodule_block_map(o, z_map, collision_prob)
        return F.softmax(logits, -1)
    
class DiscreteSAC(nn.Module):
    def __init__(self, o_dim, a_dim, z_map_dim=9, n_critic = 1):
        super(DiscreteSAC, self).__init__()
        self.n_critic = n_critic
        self.a_dim = a_dim

        self.Q_net = nn.ModuleList([DuelingQNet(o_dim, a_dim, z_map_dim) for _ in range(n_critic)])
        self.P_net = PolicyNet(o_dim, a_dim, z_map_dim)
        
        self.temperature = 1.0
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            nn.init.zeros_(module.bias)
            
    def need_grad(self, module: nn.Module, need: bool):
        for p in module.parameters():
            p.requires_grad = need
        
    def forward(self, o, z_map, collision_prob):
        batch_size = o.shape[0]
        Qs = torch.empty((self.n_critic, batch_size, self.a_dim)).to(o.device)
        for i in range(self.n_critic):
            Qs[i, ...] = self.Q_net[i](o, z_map, collision_prob)
        Q = Qs.min(0).values
        P = self.P_net(o, z_map, collision_prob)
        return Q, P
    
    def cal_Q(self, o, z_map, collision_prob):
        batch_size = o.shape[0]
        Qs = torch.empty((self.n_critic, batch_size, self.a_dim)).to(o.device)
        for i in range(self.n_critic):
            Qs[i, ...] = self.Q_net[i](o, z_map, collision_prob)
        return Qs