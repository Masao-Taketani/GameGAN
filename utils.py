import torch
from torch.autograd import Variable
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


def to_variable(args, use_gpu):
    if use_gpu:
        if len(args) > 1:
            return [Variable(elem).cuda() for elem in args]
        else:
            return Variable(args[0]).cuda()
    else:
        if len(args) > 1:
            return [Variable(elem) for elem in args]
        else:
            return Variable(args[0])