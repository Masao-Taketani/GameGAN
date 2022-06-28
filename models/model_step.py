import torch

import utils


def run_generator_step(gen, disc, x_real, a, warmup_steps, epoch, to_train):
    utils.set_grads(gen, True)
    utils.set_grads(disc, True)
    
    # select train or eval mode for batch norm, dropout, etc.
    if to_train:
        gen.train()
        disc.train()
    else:
        gen.eval()
        disc.eval()

        gen_out_dic = gen(x_real, a, warmup_steps, epoch)
        # shape of input: [(total steps - 1) * bs, 3, h, w]
        # shape of a[:-1]: [(bs, action_space) * (total steps - 1)]
        disc_out_dic = disc(torch.cat(gen_out_dic['out_imgs'], dim=0), a[:-1], warmup_steps, x_real)