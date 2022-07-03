import torch
import torch.nn.functional as F


def generator_hinge_loss(disc_out):
    # This loss can be negative
    return -torch.mean(disc_out)


def discriminator_hinge_loss(logits, is_real):
    # This loss is always greater than or equal to 0
    if is_real:
        # As for real data, logits are cripped when they are larger than 1
        return torch.mean(F.relu(1. - logits))
    else:
        # As for fake data, logits are cripped when they are smaller than -1
        return torch.mean(F.relu(1. + logits))