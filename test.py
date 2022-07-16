import torch
import keyboard
import cv2
import numpy as np

from options.test_options import TestOptions
from models.model_modules import get_gen_model_arch_dict, get_disc_model_arch_dict
import utils


# Initial image is always given for inference
WARMUP_STEPS = 1


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

    # Load an initial image
    img = utils.BGR2RGB(test_opts.init_img_path)
    img = utils.make_channels_first_and_normalize_img(img)

    imgs = [torch.tensor([img], dtype=torch.float32).cuda()]
    acts = [torch.tensor(stay, dtype=torch.float32).cuda()]

    utils.set_grads(gen, False)
    gen.eval()

    gen_img = None

    # simulator loop
    while True:
        act_label = ''
        if keyboard.is_pressed('e'):
            exit()
        elif gen_img is None:
            # Run warmup to get initial values
            # warmup is set to 0, so initial image is going to be used as input
            gen_img, warmup_h_c, M, alpha_prev, m_vec_prev, out_imgs, zs, alphas, fine_mask_list, map_list, \
               unmasked_base_imgs, alpha_losses = gen.run_warmup_phase(imgs, acts, WARMUP_STEPS)
            h, c = warmup_h_c

            # TODO: need to implement the rest

        
        elif keyboard.is_pressed('a'):
            act = torch.tensor([left], dtype=torch.float32).cuda()
            hidden_action = -1
        elif keyboard.is_pressed('d'):
            act = torch.tensor([right], dtype=torch.float32).cuda()
            hidden_action = 1
        else:
            act = torch.tensor([stay], dtype=torch.float32).cuda()
            hidden_action = 0

        x_next, h_next, c_next, z, M, alpha, m_vec, fine_masks, maps, base_imgs, alpha_loss \
                                 = gen.proceed_step(gen_img, h, c, act, M, alpha_prev, m_vec_prev)

    

if __name__ == '__main__':
    opts = TestOptions().parse()
    run_simulator(opts)