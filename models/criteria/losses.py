import torch


def generator_hinge_loss(disc_out):
    return -torch.mean(disc_out)