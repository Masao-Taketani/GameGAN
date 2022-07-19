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
img_size = (48, 80)
# check disc input size for imgs, actions, num_warmup_frames, real_frames, pred_neg_act
#images: torch.Size([31, 3, 48, 80])
#actions: 32
#actions[0]: torch.Size([1, 3])
#states: 32
#states[0]: torch.Size([1, 3, 48, 80])
#warm_up: 16
#neg_actions: 32
#neg_actions[0]: torch.Size([1, 3])



# generator tests
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


# discriminator tests
def check_single_disc():
    singl_disc = SingleImageDiscriminator(disc_model_arch_dict)
    model = singl_disc.to(device)
    summary(model, [(3, 48, 80)])


def check_act_cond_disc():
    act_cond_disc = ActionConditionedDiscriminator(action_space, img_size, hidden_dim, neg_slope, z_dim, True)
    model = act_cond_disc.to(device)
    summary(model, [(3,), (256, 3, 5), (256, 3, 5), (3,)])


def check_tempo_disc():
    tempo_disc = TemporalDiscriminator(batch_size, img_size, temporal_window, neg_slope, debug=True)
    model = tempo_disc.to(device)
    summary(model, [(256, 3, 5), (16,)])


if __name__ == '__main__':
    # generator tests
    #print('generator tests')
    #check_dynamics_engine()
    #check_memory()
    #check_rendering_engine()

    # discriminator tests
    print('discriminator tests')
    check_single_disc()
    check_act_cond_disc()
    check_tempo_disc()