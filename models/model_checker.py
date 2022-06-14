from generator import *
from discriminator import *
import torch
from torchsummary import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

z_dim = 32
hidden_dim = 512
# num_a_space=3 is used for both vizdoom and gta
num_a_space = 3
neg_slope = 0.2
num_inp_channels = 3
batch_size = 2
N = 21
tmp_device = 'cuda'
K = 1

## with memory
#memory_dim = 512
## without memory
memory_dim = None
use_memory = False

## for vizdoom
#img_size = '64x64'
#dataset_name = 'vizdoom'
# for generator
#gen_model_arch_dict = {'img_size': (64, 64), 'first_fmap_size': (8, 8), 'in_channels': [512, 256, 128], 
#                       'out_channels': [256, 128, 64], 
#                       'upsample': [2, 2, 2], 'resolution': [16, 32, 64], 
#                       'attention': {16: False, 32: False, 64: True}}
# for discriminator
#disc_model_arch_dict = {'in_channels': [3, 16, 32, 64, 128], 
#                        'out_channels': [16, 32, 64, 128, 256], 
#                        'downsample': [True, True, True, True, False], 
#                        'resolution': [32, 16, 8, 4, 4], 
#                        'attention': {4: False, 8: False, 16: False, 32: False, 64: True}}

## for gta
img_size = '48x80'
dataset_name = 'gta'
# for generator
gen_model_arch_dict = {'img_size': (48, 80), 'first_fmap_size': (6, 10), 'in_channels': [768, 384, 192, 96, 96], 
                       'out_channels': [384, 192, 96, 96, 96],
                       'upsample': [2, 2, 2, 1, 1], 'resolution': [16, 32, 64, 128, 256],
                       'attention': {16: False, 32: True, 64: True, 128: False, 256: False}}
# for discriminator
disc_model_arch_dict = {'in_channels':   [3, 16, 32, 64, 64, 64, 128, 128], 
                        'out_channels': [16, 32, 64, 64, 64, 128, 128, 256], 
                        'downsample': [True, True, False, False, True, True, False, False], 
                        'resolution': [64, 32, 16, 8, 4, 4, 4, 4], 
                        'attention': {4: False, 8: False, 16: False, 32: True, 64: True, 128: False}}
action_space = 3
temporal_window = 32
# check disc input size for imgs, actions, num_warmup_frames, real_frames, pred_neg_act



def check_dynamics_engine():
    de = DynamicsEngine(z_dim, hidden_dim, num_a_space, neg_slope, img_size, num_inp_channels, memory_dim=memory_dim)
    model = de.to(device)
    h, w = [int(i) for i in img_size.split('x')]
    if memory_dim is not None:
        # with memory
        summary(model, [(hidden_dim,), (hidden_dim,), (num_inp_channels, h, w), (num_a_space,), (z_dim,), (memory_dim,)])
    else:
        # without memory
        summary(model, [(hidden_dim,), (hidden_dim,), (num_inp_channels, h, w), (num_a_space,), (z_dim,)])


def check_memory():
    m = Memory(batch_size, dataset_name, hidden_dim, num_a_space, neg_slope, 512, N, device=tmp_device)
    model = m.to(device)
    summary(model, [(hidden_dim,), (hidden_dim,), (num_a_space,), (N ** 2,), (N ** 2, 512)])


def check_rendering_engine():
    re = RenderingEngine(batch_size, hidden_dim, K, gen_model_arch_dict, use_memory)
    model = re.to(device)
    summary(model, [(hidden_dim,)])


def check_discriminator():
    disc = Discriminator(batch_size, disc_model_arch_dict, action_space, img_size, hidden_dim, neg_slope, 
                         temporal_window)
    model = disc.to(device)
    summary(model, [(hidden_dim,)])


if __name__ == '__main__':
    check_dynamics_engine()
    check_memory()
    check_rendering_engine()