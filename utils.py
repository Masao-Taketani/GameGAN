import numpy as np


def make_channels_first_and_normalize_img(img):
    # convert img from channels last to channels first and normalize it from -1.0 to 1.0
    img = (np.transpose(img, axes=(2, 0, 1)) / 255.0).astype('float32')
    return (img - 0.5) / 0.5


def make_label_idx_to_onehot(label_idx, num_action_space):
    return np.eye(num_action_space)[label_idx].astype('float32')