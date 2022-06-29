import torch
import torch.nn.functional as F

import utils
from criteria import losses


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

    loss_dict = {}

    gen_out = gen(x_real, a, warmup_steps, epoch)
    # shape of input: [(total steps - 1) * bs, 3, h, w]
    # shape of a[:-1]: [(bs, action_space) * (total steps - 1)]
    disc_out = disc(torch.cat(gen_out['out_imgs'], dim=0), a[:-1], warmup_steps, x_real)

    # Single image discriminator loss
    gen_single_img_loss = losses.generator_hinge_loss(disc_out['full_frame_preds'])
    loss_dict['gen_single_img_loss'] = gen_single_img_loss

    # Action-conditioned discriminator loss
    gen_act_cond_loss = losses.generator_hinge_loss(disc_out['act_preds'])
    loss_dict['gen_act_cond_loss'] = gen_act_cond_loss

    # Temporal discriminator
    # get hierarchical temporal discriminator logits
    gen_avg_tempo_loss = 0
    hier_levels = len(disc_out['tempo_preds'])
    for i in range(hier_levels):
        tmp_tempo_loss = losses.generator_hinge_loss(disc_out['tempo_preds'][i])
        loss_dict[f'gen_tempo_loss{i}'] = tmp_tempo_loss
        gen_total_tempo_loss += tmp_tempo_loss
    gen_avg_tempo_loss = gen_avg_tempo_loss / hier_levels

    # Action loss
    a_real = torch.cat(a[:len(gen_out['out_imgs'])], dim=0)
    _, act_idxes = torch.max(a_real, 1)
    # As for F.cross_entropy, preds should be logits and targets can be indexes
    act_loss = F.cross_entropy(disc_out['act_recon'], act_idxes)
    loss_dict['act_loss'] = act_loss

    # Info loss
    z_real = torch.cat(gen_out['zs'], dim=0)
    info_loss = F.mse_loss(disc_out['z_recon'], z_real)
    loss_dict['info_loss'] = info_loss

    # Image reconstruction loss


    # feature reconstruction loss