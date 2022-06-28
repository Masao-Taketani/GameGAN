import torch
from torch.autograd import Variable
from torch import distributions
import numpy as np


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