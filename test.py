import torch

from options.test_options import TestOptions
from models.model_modules import get_gen_model_arch_dict, get_disc_model_arch_dict
import utils


def run_simulator(test_opts):
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0') if use_gpu else torch.device('cpu')

    # To avoid GPU RAM surge when loading a model checkpoint.
    # Reference: https://pytorch.org/docs/stable/generated/torch.load.html
    model_path = torch.load(test_opts.model_path, map_location='cpu')
    train_opts = model_path['opts']

    if train_opts.dataset_name == 'gta':
        # 0: left, 1: no action, 2: right
        resized_img_size = (320, 192)
        left = [1, 0, 0]
        stay = [0, 1, 0]
        right = [0, 0, 1]
    else:
        raise NotImplementedError()
    
    num_components = 2 if train_opts.memory_dim is not None else 1
    if train_opts.dataset_name == 'gta':
        num_action_spaces = 3
    
    gen_model_arch_dict = get_gen_model_arch_dict(train_opts.dataset_name, num_components)
    disc_model_arch_dict = get_disc_model_arch_dict(train_opts.dataset_name)
    img_size = [int(size) for size in train_opts.img_size.split('x')]

    gen, _ = utils.create_models(train_opts, use_gpu, num_action_spaces, img_size, 
                                 gen_model_arch_dict, device, disc_model_arch_dict)


if __name__ == '__main__':
    opts = TestOptions().parse()
    run_simulator(opts)