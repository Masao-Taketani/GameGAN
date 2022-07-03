import torch
import torch.nn.functional as F

import utils
from criteria import losses


def run_generator_step(gen, disc, gen_tempo_optim, gen_graphic_optim, disc_optim, x_real, a,\
                       warmup_steps, epoch, to_train, opts):
    # make all parameters of the genenerator and the discriminator learnable
    utils.set_grads(gen, True)
    utils.set_grads(disc, True)
    
    # select train or eval mode for batch norm, dropout, etc.
    if to_train:
        gen.train()
        disc.train()
    else:
        gen.eval()
        disc.eval()

    gen_tempo_optim.zero_grad()
    gen_graphic_optim.zero_grad()
    disc_optim.zero_grad()

    total_loss = 0
    loss_dict = {}

    gen_out = gen(x_real, a, warmup_steps)
    # shape of input: [(total steps - 1) * bs, 3, h, w]
    #gen_imgs = torch.cat(gen_out['out_imgs'], dim=0)
    # shape of a[:-1]: [(bs, action_space) * (total steps - 1)]
    disc_fake_out = disc(torch.cat(gen_out['out_imgs'], dim=0), a[:-1], warmup_steps, x_real)

    # Single image discriminator loss
    gen_single_img_loss = losses.generator_hinge_loss(disc_fake_out['full_frame_preds'])
    loss_dict['gen_single_img_loss'] = gen_single_img_loss
    total_loss += gen_single_img_loss

    # Action-conditioned discriminator loss
    gen_act_cond_loss = losses.generator_hinge_loss(disc_fake_out['act_preds'])
    loss_dict['gen_act_cond_loss'] = gen_act_cond_loss
    total_loss += gen_act_cond_loss

    # Temporal discriminator
    # get hierarchical temporal discriminator logits
    gen_avg_tempo_loss = 0
    hier_levels = len(disc_fake_out['tempo_preds'])
    for i in range(hier_levels):
        tmp_tempo_loss = losses.generator_hinge_loss(disc_fake_out['tempo_preds'][i])
        loss_dict[f'gen_tempo_loss{i}'] = tmp_tempo_loss
        gen_avg_tempo_loss += tmp_tempo_loss
    gen_avg_tempo_loss = gen_avg_tempo_loss / hier_levels
    total_loss += gen_avg_tempo_loss

    # Action loss
    # Here a[:-1] is used for action labels since image differences between t+1 and t are 
    # based on actions time t = 0, ..., total_steps - 2
    a_real = torch.cat(a[:-1], dim=0)
    _, act_idxes = torch.max(a_real, 1)
    # As for F.cross_entropy, preds should be logits and targets can be indexes
    act_loss = F.cross_entropy(disc_fake_out['act_recon'], act_idxes)
    loss_dict['act_loss'] = act_loss
    total_loss += act_loss

    # Info loss
    z_real = torch.cat(gen_out['zs'], dim=0)
    info_loss = F.mse_loss(disc_fake_out['z_recon'], z_real)
    loss_dict['info_loss'] = info_loss
    total_loss += opts.lambda_I * info_loss

    # Image reconstruction loss
    # total time steps for x and x_hat: total_steps - 1
    x = torch.cat(x_real[1:len(gen_out['out_imgs']) + 1], dim=0)
    x_hat = torch.cat(gen_out['out_imgs'], dim=0)
    recon_loss = F.mse_loss(x_hat, x)
    loss_dict['recon_loss'] = recon_loss
    total_loss += opts.lambda_r * recon_loss
    
    # Feature reconstruction loss
    disc_real_out = disc(x, a[:-1], warmup_steps, x_real)
    # feat is not included for the generator loss
    feat = disc_real_out['pred_frame_fmaps'].detach()
    feat_hat = disc_fake_out['pred_frame_fmaps']
    # L1 loss is used in the original code.
    feat_loss = F.l1_loss(feat, feat_hat)
    loss_dict['feat_loss'] = feat_loss
    total_loss += opts.lambda_f * feat_loss

    # Cycle loss
    # Include the loss when the memory module and the specialized rendering engine are used.
    # The loss is not used to update the rendering engine. Instead, it is used to update the
    # memory and dynamics engine modules for better temporal consistency.
    if opts.memory_dim:
        reverse_gen_imgs = torch.cat(gen_out['reverse_imgs'][::-1], dim=0)
        num_reverse = len(gen_out['reverse_imgs'])
        reverse_reals = [comp[0] for comp in gen_out['unmasked_base_imgs']]
        gen_out['reverse_real'] = reverse_reals[:num_reverse]
        reverse_real = torch.cat(gen_out['reverse_real'], dim=0)
        cycle_loss = F.mse_loss(reverse_gen_imgs, reverse_real)
        loss_dict['cycle_loss'] = cycle_loss

        # Memory regularization
        avg_alpha_loss = gen_out['avg_alpha_loss']
        loss_dict['memory_reg'] = avg_alpha_loss
        total_loss += opts.lambda_mem_reg * avg_alpha_loss
        total_loss.backward(retain_graph=True)


    # If it is to train, update the parameters
    if to_train:
        grads = {}
        x_hat.register_hook(utils.save_grad('gen_adv_input', grads))

        if opts.memory_dim:
            # Caluculate the grads to only update tempo module
            (total_loss + opts.lambda_c * cycle_loss).backward(retain_graph=True)
            tempo_grads = []
            for param in gen_tempo_optim.param_groups[0]['params']:
                tempo_grads.append(param.grad.clone())

            gen_tempo_optim.zero_grad()
            gen_graphic_optim.zero_grad()

            total_loss.backward()

            # Replace tempo grads with the calculated grads
            for param, tempo_grad in zip(gen_tempo_optim.param_groups[0]['params'], tempo_grads):
                param.grad.detach()
                del param.grad
                param.grad = tempo_grad

            torch.cuda.empty_cache()
        else:
            total_loss.backward()

        gen_tempo_optim.step()
        gen_graphic_optim.step()
    
    return loss_dict, total_loss, gen_out, grads


def run_discriminator_step(gen, disc, gen_tempo_optim, gen_graphic_optim, disc_optim, x_real, a,\
                           neg_a, warmup_steps, opts, gen_out):
    # make all parameters of the discriminator learnable and of the generator unlearnable
    utils.set_grads(disc, True)
    utils.set_grads(gen, False)

    gen.train()
    disc.train()

    gen_tempo_optim.zero_grad()
    gen_graphic_optim.zero_grad()
    disc_optim.zero_grad()

    total_loss = 0
    loss_dict = {}

    # make all input variables learnable
    x_real = [tmp.requires_grad_() for tmp in x_real]
    a = [tmp.requires_grad_() for tmp in a]
    neg_a = [tmp.requires_grad_() for tmp in neg_a]


    ### [1] Calculate loss for real data
    real_inputs = torch.cat(x_real[1:], dim=0).requires_grad_()
    disc_real_out = disc(real_inputs, a[:-1], warmup_steps, x_real, neg_a)

    # Single image discriminator loss
    disc_real_single_img_loss = losses.discriminator_hinge_loss(disc_real_out['full_frame_preds'], True)
    loss_dict['disc_real_single_img_loss'] = disc_real_single_img_loss
    total_loss += disc_real_single_img_loss

    # Action-conditioned discriminator loss (including negative actions)
    disc_act_cond_loss = losses.discriminator_hinge_loss(disc_real_out['act_preds'], True)
    loss_dict['disc_act_cond_loss'] = disc_act_cond_loss
    total_loss += disc_act_cond_loss
    disc_neg_act_cond_loss = losses.discriminator_hinge_loss(disc_real_out['neg_act_preds'], False)
    loss_dict['disc_neg_act_cond_loss'] = disc_neg_act_cond_loss
    total_loss += disc_neg_act_cond_loss

    # Temporal discriminator
    # get hierarchical temporal discriminator logits
    disc_avg_tempo_loss = 0
    hier_levels = len(disc_real_out['tempo_preds'])
    for i in range(hier_levels):
        tmp_tempo_loss = losses.discriminator_hinge_loss(disc_real_out['tempo_preds'][i], True)
        loss_dict[f'disc_real_tempo_loss{i}'] = tmp_tempo_loss
        disc_avg_tempo_loss += tmp_tempo_loss
    disc_avg_tempo_loss = disc_avg_tempo_loss / hier_levels
    total_loss += disc_avg_tempo_loss

    # Action loss
    # Here a[:-1] is used for action labels since image differences between t+1 and t are 
    # based on actions time t = 0, ..., total_steps - 2
    a_real = torch.cat(a[:-1], dim=0)
    _, act_idxes = torch.max(a_real, 1)
    # As for F.cross_entropy, preds should be logits and targets can be indexes
    act_loss = F.cross_entropy(disc_real_out['act_recon'], act_idxes)
    loss_dict['act_loss'] = act_loss
    total_loss += act_loss

    # gradient penalty for real data
    reg = 0
    reg += 0.33 * utils.compute_grad2(disc_real_out['act_preds'], real_inputs, ns=opts.total_steps).mean()
    reg += 0.33 * utils.compute_grad2(disc_real_out['full_frame_preds'], real_inputs, ns=opts.total_steps).mean()
    reg += 0.33 * utils.compute_grad2(disc_real_out['act_recon'], real_inputs, ns=opts.total_steps).mean()
    reg_temporal = 0
    for i in range(hier_levels):
        tmp_loss = utils.compute_grad2(disc_real_out['tempo_preds'][i], real_inputs, ns=opts.total_steps).mean()
        reg_temporal += tmp_loss
    reg_temporal = reg_temporal / hier_levels
    loss_dict['disc_real_tempo_gp'] = reg_temporal
    loss_dict['disc_real_gp'] = reg
    loss += reg + opts.gamma * reg_temporal


    ### [2] Calculate loss for fake data
    # since we are not training the generator this time, the input tensor needs to be detached.
    disc_fake_out = disc(torch.cat(gen_out['out_imgs'], dim=0).detach(), a[:-1], warmup_steps, x_real)

    # Single image discriminator loss
    disc_fake_single_img_loss = losses.discriminator_hinge_loss(disc_fake_out['full_frame_preds'], False)
    loss_dict['disc_fake_single_img_loss'] = disc_fake_single_img_loss
    total_loss += disc_fake_single_img_loss

    # Action-conditioned discriminator loss
    disc_fake_act_cond_loss = losses.discriminator_hinge_loss(disc_fake_out['act_preds'], False)
    loss_dict['disc_fake_act_cond_loss'] = disc_fake_act_cond_loss
    total_loss += disc_fake_act_cond_loss

    # Temporal discriminator
    # get hierarchical temporal discriminator logits
    disc_fake_avg_tempo_loss = 0
    for i in range(hier_levels):
        tmp_tempo_loss = losses.discriminator_hinge_loss(disc_fake_out['tempo_preds'][i], False)
        loss_dict[f'disc_fake_tempo_loss{i}'] = tmp_tempo_loss
        disc_fake_avg_tempo_loss += tmp_tempo_loss
    disc_fake_avg_tempo_loss = disc_fake_avg_tempo_loss / hier_levels
    total_loss += disc_fake_avg_tempo_loss

    # Info loss
    z_real = torch.cat(gen_out['zs'], dim=0)
    disc_fake_info_loss = F.mse_loss(disc_fake_out['z_recon'], z_real)
    loss_dict['disc_fake_info_loss'] = disc_fake_info_loss
    total_loss += disc_fake_info_loss

    # Update the discriminator
    total_loss.backward()
    disc_optim.step()
    utils.set_grads(disc, False)

    return loss_dict