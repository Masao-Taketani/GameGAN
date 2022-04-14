import torch
from torch import nn


class Generator(nn.Module):

    def __init__(self, z_dim, hidden_dim, use_memory):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.use_memory = use_memory

class DynamicsEngine(nn.Module):

    def __init__(self, z_dim, hidden_dim, state_dim, use_memory, num_action_space, neg_slope, memory_dim=None):
        super(DynamicsEngine, self).__init__()

        a = [nn.Linear(num_action_space, hidden_dim),
             nn.LeakyReLU(neg_slope)]
        z = [nn.Linear(z_dim, hidden_dim),
             nn.LeakyReLU(neg_slope)]
        
        input_dim = hidden_dim * 2
        if use_memory:
            input_dim += memory_dim
        
        

        
    def forward(self, x):
        pass

class Memory(nn.Module):

    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, x):
        pass


class RenderingEngine(nn.Module):

    def __init__(self):
        super(RenderingEngine, self).__init__()
    
    def forward(self, x):
        pass