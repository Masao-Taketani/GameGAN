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
# 48x80 for gta
num_inp_channels = 3
memory_dim = 512
batch_size = 2
dataset_name = 'gta'
N = 21
tmp_device = 'cuda'
K = 1
# for vizdoom
#model_arch_dict = {'img_size': (64, 64), 'first_fmap_size': (8, 8), 'in_channels': [512, 256, 128], 'out_channels': [256, 128, 64], 
#         'upsample': [2, 2, 2], 'resolution': [16, 32, 64], 
#         'attention': {16: False, 32: False, 64: True}}
# for gta
model_arch_dict = {'img_size': (48, 80), 'first_fmap_size': (6, 10), 'in_channels': [768, 384, 192, 96, 96], 'out_channels': [384, 192, 96, 96, 96],
         'upsample': [2, 2, 2, 1, 1], 'resolution': [16, 32, 64, 128, 256],
         'attention': {16: False, 32: True, 64: True, 128: False, 256: False}}
use_memory = False


def check_dynamics_engine():
    de = DynamicsEngine(z_dim, hidden_dim, num_a_space, neg_slope, img_size, num_inp_channels, memory_dim=memory_dim)
    model = de.to(device)
    h, w = [int(i) for i in img_size.split('x')]
    summary(model, [(hidden_dim,), (hidden_dim,), (num_inp_channels, h, w), (num_a_space,), (z_dim,), (memory_dim,)])


def check_memory():
    m = Memory(batch_size, dataset_name, hidden_dim, num_a_space, neg_slope, memory_dim, N, device=tmp_device)
    model = m.to(device)
    summary(model, [(hidden_dim,), (hidden_dim,), (num_a_space,), (N ** 2,), (N ** 2, memory_dim)])


def check_rendering_engine():
    re = RenderingEngine(batch_size, hidden_dim, K, model_arch_dict, use_memory)
    model = re.to(device)
    summary(model, [(hidden_dim,)])


if __name__ == '__main__':
    check_dynamics_engine()
    check_memory()
    check_rendering_engine()