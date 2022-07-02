import torch
from torch.autograd import Variable
from torch import distributions
from torch import autograd
import numpy as np

from models.generator import Generator
from models.discriminator import Discriminator


def make_channels_first_and_normalize_img(img):
    # convert img from channels last to channels first and normalize it from -1.0 to 1.0
    img = (np.transpose(img, axes=(2, 0, 1)) / 255.0).astype('float32')
    return (img - 0.5) / 0.5


def make_channels_last_and_denormalize_img(img):
    # convert img from channels first to channels last and denormalize it
    img = (np.transpose(img, axes=(1, 2, 0)))
    return (((img + 1.0) / 2.0) * 255.0).to(torch.int64)


def make_label_idx_to_onehot(label_idx, num_action_space):
    return np.eye(num_action_space)[label_idx].astype('float32')


def to_gpu(data):
    if isinstance(data, list):
        return [elem.cuda() for elem in data]
    else:
        return data.cuda()


def to_variable(data, use_gpu):
    if use_gpu:
        if isinstance(data, list):
            return [Variable(elem).cuda() for elem in data]
        else:
            return Variable(data).cuda()
    else:
        if isinstance(data, list):
            return [Variable(elem) for elem in data]
        else:
            return Variable(data)


def get_random_noise_dist(z_dim, dist_type='gaussian'):
    # True dist for gaussian comes from standard normal dist
    if dist_type == 'gaussian':
        mu = torch.zeros(z_dim)
        std = torch.ones(z_dim)
        random_noise_dist = distributions.Normal(mu, std)
    elif dist_type == 'uniform':
        lower_limit = torch.zeros(z_dim)
        upper_limit = torch.ones(z_dim)
        random_noise_dist = distributions.Uniform(lower_limit, upper_limit)
    else:
        raise NotImplementedError

    return random_noise_dist


def set_grads(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def save_grad(name, grads):
    def hook(grad):
        grads[name] = grad
    return hook


def compute_grad2(d_out, x_in, allow_unused=False, batch_size=None, use_gpu=True, ns=1):
    # Reference:
    # https://github.com/LMescheder/GAN_stability/blob/master/gan_training/train.py
    if d_out is None:
        return to_variable(torch.FloatTensor([0]), use_gpu)
    if batch_size is None:
        batch_size = x_in.size(0)

    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True,
        allow_unused=allow_unused
    )[0]

    grad_dout2 = grad_dout.pow(2)
    reg = grad_dout2.view(batch_size, -1).sum(1) * (ns * 1.0 / 6)
    return reg


def create_models(opts, use_gpu, num_action_spaces, img_size, gen_model_arch_dict, device,\
                  disc_model_arch_dict):
    gen = Generator(opts.batch_size, opts.z_dim, opts.hidden_dim, use_gpu, num_action_spaces, 
                    opts.neg_slope, img_size, opts.num_inp_channels, gen_model_arch_dict,
                    opts.dataset_name, 21, device, opts.memory_dim)

    disc = Discriminator(opts.batch_size, disc_model_arch_dict, num_action_spaces, img_size, 
                         opts.hidden_dim, opts.neg_slope, opts.temporal_window)

    return gen, disc


def get_optim(net, lr, include=None, exclude=None, model_name=''):
    if type(net) is list:
        params = net
    else:
        params = net.parameters()
        if exclude is not None:
            params = []
            for name, W in net.named_parameters():
                if exclude in name:
                    print(model_name + ', Exclude: ' + name)
                else:
                    params.append(W)
                    print(model_name + ', Include: ' + name)
        if include is not None:
            params = []
            for name, W in net.named_parameters():
                if include in name:
                    params.append(W)
                    print(model_name + ', Include: ' + name)

    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.0, 0.9))

    return optimizer