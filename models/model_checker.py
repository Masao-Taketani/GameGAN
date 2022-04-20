from generator import *
import torch
from torchsummary import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

z_dim = 32
hidden_dim = 512
# 3 for visdoom
num_a_space = 3
neg_slope = 0.2
# 64x64 for visdoom
img_size = '64x64'
num_inp_channels = 3
memory_dim = 512

def check_dynamics_engine():
    de = DynamicsEngine(z_dim, hidden_dim, num_a_space, neg_slope, img_size, num_inp_channels, memory_dim=memory_dim)
    model = de.to(device)
    h, w = [int(i) for i in img_size.split('x')]
    summary(model, [(hidden_dim,), (hidden_dim,), (num_inp_channels, h, w), (num_a_space,), (z_dim,), (memory_dim,)])


if __name__ == '__main__':
    check_dynamics_engine()