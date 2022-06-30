import torch
import torch.nn.functional as F

import utils
from criteria import losses


def run_generator_step(gen, disc, x_real, a, warmup_steps, epoch, to_train, use_memory=False):
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
    disc_fake_out = disc(torch.cat(gen_out['out_imgs'], dim=0), a[:-1], warmup_steps, x_real)

    # Single image discriminator loss
    gen_single_img_loss = losses.generator_hinge_loss(disc_fake_out['full_frame_preds'])
    loss_dict['gen_single_img_loss'] = gen_single_img_loss

    # Action-conditioned discriminator loss
    gen_act_cond_loss = losses.generator_hinge_loss(disc_fake_out['act_preds'])
    loss_dict['gen_act_cond_loss'] = gen_act_cond_loss

    # Temporal discriminator
    # get hierarchical temporal discriminator logits
    gen_avg_tempo_loss = 0
    hier_levels = len(disc_fake_out['tempo_preds'])
    for i in range(hier_levels):
        tmp_tempo_loss = losses.generator_hinge_loss(disc_fake_out['tempo_preds'][i])
        loss_dict[f'gen_tempo_loss{i}'] = tmp_tempo_loss
        gen_avg_tempo_loss += tmp_tempo_loss
    gen_avg_tempo_loss = gen_avg_tempo_loss / hier_levels

    # Action loss
    a_real = torch.cat(a[:len(gen_out['out_imgs'])], dim=0)
    _, act_idxes = torch.max(a_real, 1)
    # As for F.cross_entropy, preds should be logits and targets can be indexes
    act_loss = F.cross_entropy(disc_fake_out['act_recon'], act_idxes)
    loss_dict['act_loss'] = act_loss

    # Info loss
    z_real = torch.cat(gen_out['zs'], dim=0)
    info_loss = F.mse_loss(disc_fake_out['z_recon'], z_real)
    loss_dict['info_loss'] = info_loss

    # Image reconstruction loss
    # total time steps for x and x_hat: total_steps - 1
    x = torch.cat(x_real[1:len(gen_out['out_imgs']) + 1], dim=0)
    x_hat = torch.cat(gen_out['out_imgs'], dim=0)
    recon_loss = F.mse_loss(x_hat, x)
    loss_dict['recon_loss'] = recon_loss
    
    # Feature reconstruction loss
    disc_real_out = disc(x, a[:-1], warmup_steps, x_real)
    # feat is not included for the generator loss
    feat = disc_real_out['pred_frame_fmaps'].detach()
    feat_hat = disc_fake_out['pred_frame_fmaps']
    # L1 loss is used in the original code.
    feat_loss = F.l1_loss(feat, feat_hat)
    loss_dict['feat_loss'] = feat_loss

    # Cycle loss
    # Include the loss when the memory module and the specialized rendering engine are used.
    # The loss is not used to update the rendering engine. Instead, it is used to update the
    # memory and dynamics engine modules for better temporal consistency.
    if use_memory:
        reverse_gen_imgs = torch.cat(gen_out['reverse_imgs'][::-1], dim=0)
        num_reverse = len(gen_out['reverse_imgs'])
        reverse_reals = [comp[0] for comp in gen_out['unmasked_base_imgs']]
        gen_out['reverse_real'] = reverse_reals[:num_reverse]
        reverse_real = torch.cat(gen_out['reverse_real'], dim=0)
        cycle_loss = F.mse_loss(reverse_gen_imgs, reverse_real)
        loss_dict['cycle_loss'] = cycle_loss

        # Memory regularization
        loss_dict['memory_reg']