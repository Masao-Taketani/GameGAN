from turtle import forward
import torch
from torch import nn


class Generator(nn.Module):

    def __init__(self, z_dim, hidden_dim, use_memory):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.use_memory = use_memory


class DynamicsEngine(nn.Module):

    def __init__(self, z_dim, hidden_dim, state_dim, num_a_space, neg_slope, memory_dim=None):
        super(DynamicsEngine, self).__init__()

        
    def forward(self, h, c, s, a, z, m):
        """
        arguments:
            h: h_t-1
            c: c_t-1
            s: s_t
            a: a_t
            z: z_t
            m: m_t-1
        """
        pass


class H(nn.Module):

    def __init__(self, z_dim, hidden_dim, num_a_space, neg_slope, memory_dim=None):
        super(H, self).__init__()
        self.memory_dim = memory_dim

        # get embeddings for a and z
        self.a_emb = [nn.Linear(num_a_space, hidden_dim),
                 nn.LeakyReLU(neg_slope)]
        self.z_emb = [nn.Linear(z_dim, hidden_dim),
                 nn.LeakyReLU(neg_slope)]

        # concat_dim: concat of [a, v(, mem: if memory module is used)]
        concat_dim = hidden_dim * 2
        if not self.memory_dim:
            concat_dim += self.memory_dim

        # two-layered MLP
        self.mlp = nn.Sequential(nn.Linear(concat_dim, concat_dim),
                                 nn.LeakyReLU(neg_slope),
                                 nn.Linear(concat_dim, concat_dim))

        # project h onto the concat_dim
        self.proj_h = nn.Linear(hidden_dim, concat_dim)

    def forward(self, h, c, s, a, z, m):
        if not self.memory_dim:
            concats = [self.a_emb(a), self.z_emb(z), m]
        else:
            concats = [self.a_emb(a), self.z_emb(z)]
        
        

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