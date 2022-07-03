import torch
from torch.autograd import Variable
from torch import distributions
from torch import autograd
import numpy as np
import math

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


def add_histogram(ml, writer, step):
    for key, model in ml.items():
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            if value.grad is None:
                print('@@@@@@@@@@@@@' + key + '/' + tag + ' has no grad.')
            else:
               writer.add_histogram('grad/'+key+'/'+tag, value.grad, step)


def save_model(fname, epoch, netG, netD, opts):
    outdict = {'epoch': epoch, 'netG': netG.state_dict(), 'netD': netD.state_dict(), 'opts': opts}
    torch.save(outdict, fname)


def save_optim(fname, epoch, optG_temporal, optG_graphic, optD):
    outdict = {'epoch': epoch, 'optG_temporal': optG_temporal.state_dict(), 
               'optG_graphic': optG_graphic.state_dict(), 'optD': optD.state_dict()}
    torch.save(outdict, fname)


def draw_output(gout, states, warm_up, opts, vutils, vis_num_row, normalize, logger, it, num_vis, tag='images'):
    img_size = opts.img_size
    _, _, h, w = states[0].size()

    if warm_up > 0:
        warm_up_states = torch.cat(states[:warm_up], dim=1)
        warm_up_states = warm_up_states[0:num_vis].view(warm_up * num_vis, opts.num_inp_channels, h, w)
        if opts.penultimate_tanh:
            warm_up_states = rescale(warm_up_states)
        warm_up_states = torch.clamp(warm_up_states, 0, 1.0)
        x = vutils.make_grid(
            warm_up_states, nrow=(warm_up) // vis_num_row,
            normalize=normalize, scale_each=normalize
        )
        logger.add_image(tag + '_output/WARMUPImage', x, it)

    states_ = torch.cat(states[warm_up:], dim=1)
    states_ = states_[0:num_vis].view((opts.num_steps - warm_up) * num_vis, opts.num_inp_channels, h, w)
    if opts.penultimate_tanh:
        states_ = rescale(states_)
    states_ = torch.clamp(states_, 0, 1.0)
    x = vutils.make_grid(
        states_, nrow=(opts.num_steps - warm_up) // vis_num_row,
        normalize=normalize, scale_each=normalize
    )
    logger.add_image(tag + '_output/GTImage', x, it)


    x_gen = gout['out_imgs']
    x_gen = torch.cat(x_gen, dim=1)
    x_gen = x_gen[0:num_vis].view(len(gout['out_imgs']) * num_vis, opts.num_inp_channels, h, w)
    if opts.penultimate_tanh:
        x_gen = rescale(x_gen)
    x_gen = torch.clamp(x_gen, 0, 1.0)
    x = vutils.make_grid(
        x_gen, nrow=len(gout['out_imgs']) // vis_num_row,
        normalize=normalize, scale_each=normalize
    )
    logger.add_image(tag + '_output/GenImage', x, it)



    mem_h = int(math.sqrt(opts.memory_h))
    mem_w = opts.memory_h // mem_h

    if 'reverse_out_imgs' in gout and len(gout['reverse_out_imgs']) > 0:

        x_rev = torch.cat(gout['reverse_real'], dim=1)
        x_rev = x_rev[0:num_vis].view(len(gout['reverse_real']) * num_vis, opts.num_inp_channels, h, w)
        # x_rev = torch.clamp(x_rev, 0, 1.0)
        if opts.penultimate_tanh:
            x_rev = rescale(x_rev)
        x = vutils.make_grid(
            x_rev, nrow=len(gout['reverse_real']) // vis_num_row,
            normalize=normalize, scale_each=normalize
        )
        logger.add_image(tag + '_rev_output/RevInputImage', x, it)

        x_rev = torch.cat(gout['reverse_out_imgs'], dim=1)
        x_rev = x_rev[0:num_vis].view(len(gout['reverse_out_imgs']) * num_vis, opts.num_inp_channels, h, w)
        # x_rev = torch.clamp(x_rev, 0, 1.0)
        if opts.penultimate_tanh:
            x_rev = rescale(x_rev)
        x = vutils.make_grid(
            x_rev, nrow=len(gout['reverse_out_imgs']) // vis_num_row,
            normalize=normalize, scale_each=normalize
        )
        logger.add_image(tag + '_rev_output/RevOutputImage', x, it)

        if opts.do_memory:
            rev_alpha = torch.clamp(torch.cat(gout['reverse_alphas'], dim=1), 0, 1.0)
            rev_alpha = rev_alpha[0:num_vis].view(len(gout['reverse_alphas']) * num_vis, 1, mem_w, mem_h)
            x = vutils.make_grid(
                rev_alpha, nrow=len(gout['reverse_alphas']) // vis_num_row, normalize=False, scale_each=False
            )
            logger.add_image(tag + '_rev_memory/reverse_alphas', x, it)
            if 'sec_reverse_alphas' in gout and len(gout['sec_reverse_alphas']) > 0:
                rev_alpha = torch.clamp(torch.cat(gout['sec_reverse_alphas'], dim=1), 0, 1.0)
                rev_alpha = rev_alpha[0:num_vis].view(len(gout['sec_reverse_alphas']) * num_vis, 1, mem_w,
                                                      mem_h)
                x = vutils.make_grid(
                    rev_alpha, nrow=len(gout['sec_reverse_alphas']), normalize=False, scale_each=False
                )
                logger.add_image(tag + '_rev_memory/sec_reverse_alphas', x, it)

    if opts.do_memory:
        alpha = torch.clamp(torch.cat(gout['alphas'], dim=1), 0, 1.0)
        alpha = alpha[0:num_vis].view(len(gout['alphas']) * num_vis, 1, mem_w, mem_h)
        x = vutils.make_grid(
            alpha, nrow=len(gout['alphas']) // vis_num_row, normalize=False, scale_each=False
        )
        logger.add_image(tag + '_memory/alphas', x, it)

        # import pdb; pdb.set_trace();
        if 'kernels' in gout:
            kernels = torch.clamp(torch.cat(gout['kernels'], dim=1), 0, 1.0)
            kernels = kernels[0:num_vis].view(len(gout['kernels']) * num_vis, 1, mem_w, mem_h)
            x = vutils.make_grid(
                kernels, nrow=len(gout['kernels']) // vis_num_row, normalize=False, scale_each=False
            )
            logger.add_image(tag + '_memory/kernels', x, it)

    maps = gout['fine_masks']

    if len(maps) > 0:
        for cur_component in range(len(gout['unmasked_base_imgs'][0])):
            gather_recon_maps = []
            len_episode = len(gout['unmasked_base_imgs'])
            for cur_step in range(len_episode):
                gather_recon_maps.append(
                    F.interpolate(gout['unmasked_base_imgs'][cur_step][cur_component], size=img_size,
                                  mode='bilinear'))

            gather_recon_maps = torch.cat(gather_recon_maps, dim=1)

            gather_recon_maps = gather_recon_maps[0:num_vis].view(len_episode * num_vis, opts.num_inp_channels,
                                                                  img_size[0], img_size[1])
            x = vutils.make_grid(
                gather_recon_maps, nrow=len_episode // vis_num_row, normalize=normalize,
                scale_each=normalize
            )
            logger.add_image(tag + '_graphics/recon_x_map' + str(cur_component), x, it)
            if len(gout['reverse_out_imgs']) > 0:
                gather_recon_maps = []
                len_episode = len(gout['rev_unmasked_base_imgs'])
                for cur_step in range(len_episode):
                    gather_recon_maps.append(
                        F.interpolate(gout['rev_unmasked_base_imgs'][cur_step][cur_component], size=img_size,
                                      mode='bilinear'))

                gather_recon_maps = torch.cat(gather_recon_maps, dim=1)
                gather_recon_maps = gather_recon_maps[0:num_vis].view(len_episode * num_vis,
                                                                      opts.num_inp_channels,
                                                                      img_size[0], img_size[1])
                x = vutils.make_grid(
                    gather_recon_maps, nrow=len_episode // vis_num_row, normalize=normalize,
                    scale_each=normalize
                )
                logger.add_image(tag + '_rev_graphics/recon_x_map' + str(cur_component), x, it)

        for cur_component in range(len(maps[0])):
            if len(maps[0]) == 0:
                break
            gather_maps = []
            for cur_step in range(len(maps)):
                gather_maps.append(maps[cur_step][cur_component])

            gather_maps = torch.cat(gather_maps, dim=1)
            gather_maps = gather_maps[0:num_vis].view(len(maps) * num_vis, 1, gather_maps.size(2),
                                                      gather_maps.size(3))
            x = vutils.make_grid(
                gather_maps, nrow=len(maps) // vis_num_row, normalize=False, scale_each=False
            )
            logger.add_image(tag + '_graphics/Map' + str(cur_component), x, it)

            if 'init_maps' in gout:
                gather_maps = []
                init_maps = gout['init_maps']
                if len(init_maps)> 0 and len(init_maps[0]) > 0 and len(init_maps[0][0]) > 0:
                    for cur_step in range(len(init_maps)):

                        gather_maps.append(init_maps[cur_step][cur_component])

                    gather_maps = torch.cat(gather_maps, dim=1)
                    gather_maps = gather_maps[0:num_vis].view(len(init_maps) * num_vis, 1, gather_maps.size(2),
                                                              gather_maps.size(3))
                    x = vutils.make_grid(
                        gather_maps, nrow=len(init_maps) // vis_num_row, normalize=False, scale_each=False
                    )
                    logger.add_image(tag + '_graphics/init_Map' + str(cur_component), x, it)

            if len(gout['reverse_out_imgs']) > 0:
                gather_maps = []
                if len(gout['reverse_fine_masks']) > 0 and len(gout['reverse_fine_masks'][0]) > 0 and len(gout['reverse_fine_masks'][0][0]) > 0:
                    for cur_step in range(len(gout['reverse_fine_masks'])):
                        gather_maps.append(gout['reverse_fine_masks'][cur_step][cur_component])

                    gather_maps = torch.cat(gather_maps, dim=1)
                    gather_maps = gather_maps[0:num_vis].view(len(gout['reverse_fine_masks']) * num_vis, 1,
                                                              gather_maps.size(2),
                                                              gather_maps.size(3))
                    x = vutils.make_grid(
                        gather_maps, nrow=len(gout['reverse_fine_masks']) // vis_num_row, normalize=False,
                        scale_each=False
                    )
                    logger.add_image(tag + '_rev_graphics/Map' + str(cur_component), x, it)


def rescale(x):
    return (x + 1) * 0.5