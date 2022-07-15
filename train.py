import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import os

import utils
from data import create_custom_dataloader
from options.train_options import TrainOptions
from models.model_step import run_generator_step, run_discriminator_step
from models.model_modules import get_gen_model_arch_dict, get_disc_model_arch_dict


def train(opts):
    """if benchmark is True, no reproducibility, but higher performance.
    else deterministically select an algorithm, possibly at the cost of 
    reduced performance.
    """
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda:0') if use_gpu else torch.device('cpu')
    torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(opts.log_dir)

    num_components = 2 if opts.memory_dim is not None else 1
    if opts.dataset_name == 'gta':
        num_action_spaces = 3
    
    gen_model_arch_dict = get_gen_model_arch_dict(opts.dataset_name, num_components)
    disc_model_arch_dict = get_disc_model_arch_dict(opts.dataset_name)
    img_size = [int(size) for size in opts.img_size.split('x')]

    gen, disc = utils.create_models(opts, use_gpu, num_action_spaces, img_size, 
                                    gen_model_arch_dict, device, disc_model_arch_dict)
    key = 're'
    # Generator optimization for dynamics engine
    gen_tempo_optim = utils.get_optim(gen, opts.lr, exclude=key, model_name='gen_tempo_optim')
    # Generator optimization for graphics engine
    gen_graphic_optim = utils.get_optim(gen, opts.lr, include=key, model_name='gen_graphic_optim')
    disc_optim = utils.get_optim(disc, opts.lr)

    # train_loader shuffles dataset
    train_loader = create_custom_dataloader(opts.dataset_name, True, opts.batch_size, 
                                                opts.num_workers, opts.pin_memory, 
                                                num_action_spaces, 
                                                opts.split_ratio, opts.datapath)
    # val_loader does not shuffle dataset
    val_loader = create_custom_dataloader(opts.dataset_name, False, opts.batch_size, 
                                          opts.num_workers, opts.pin_memory, 
                                          num_action_spaces, 
                                          opts.split_ratio, opts.datapath)
    
    # set initial warm-up steps
    num_train_steps = 0
    num_val_steps = 0
    vis_num_row = 1
    if opts.total_steps > 29:
        vis_num_row = 3
    num_vis = 1
    normalize = True

    data_iters = []
    data_iters.append(iter(train_loader))
    train_len = len(data_iters[0])
    log_iter = max(1,int(train_len // 10))
    del data_iters

    start_epoch = 0
    if opts.resume_epoch is not None:
        start_epoch = opts.resume_epoch

    for epoch in range(start_epoch, opts.num_epochs):
        print(f'[epoch {epoch}]')
        updated_warmup_steps = utils.update_warmup_steps(opts, epoch)
        # clear gpu memory cache
        torch.cuda.empty_cache()

        for step, (imgs, acts, neg_acts) in enumerate(train_loader):
            num_train_steps += 1
            if use_gpu:
                imgs = utils.to_gpu(imgs)
                acts = utils.to_gpu(acts)
                neg_acts = utils.to_gpu(neg_acts)
            #print(len(imgs), len(acts), len(neg_acts))
            #print(imgs[0].shape, acts[0].shape, neg_acts[0].shape)

            gen_loss_dict, gen_total_loss, gen_out, grads = run_generator_step(gen, disc, gen_tempo_optim,\
                                                                               gen_graphic_optim, disc_optim,
                                                                               imgs, acts, updated_warmup_steps, epoch,\
                                                                               True, opts)
            
            disc_loss_dict = run_discriminator_step(gen, disc, gen_tempo_optim, gen_graphic_optim, disc_optim,\
                                                   imgs, acts, neg_acts, updated_warmup_steps, opts, gen_out)
            # (tmp) for debugging
            #break
            
            # For logging
            with torch.no_grad():
                if step == 0:
                    utils.add_histogram({'gen': gen, 'disc': disc}, writer, num_train_steps)

                loss_str = 'Generator [epoch %d, step %d / %d] ' % (epoch, step, train_len)
                for k, v in gen_loss_dict.items():
                    if not (type(v) is float):
                        if (step % log_iter) == 0:
                            writer.add_scalar('losses/' + k, v.data.item(), num_train_steps)
                        loss_str += k + ': ' + str(v.data.item()) + ', '
                print(loss_str)

                if (step % log_iter) == 0:
                    # logging visualization
                    utils.draw_output(gen_out, imgs, updated_warmup_steps, opts, vutils, vis_num_row, normalize, writer,
                                        num_train_steps,
                                        num_vis, tag='trn_images')
            

            loss_str = 'Discriminator [epoch %d, step %d / %d] ' % (epoch, step, train_len)
            for k, v in disc_loss_dict.items():
                if not type(v) is float:
                    if (step % log_iter) == 0:
                        writer.add_scalar('losses/' + k, v.data.item(), num_train_steps)
                    loss_str += k + ': ' + str(v.data.item()) + ', '
            print(loss_str)
            del gen_loss_dict, gen_total_loss, gen_out, grads, imgs, acts, neg_acts, disc_loss_dict

        print('Validation epoch %d...' % epoch)
        torch.cuda.empty_cache()

        max_vis = 10
        for step, (imgs, acts, neg_acts) in enumerate(val_loader):
            num_val_steps += 1
            if use_gpu:
                imgs = utils.to_gpu(imgs)
                acts = utils.to_gpu(acts)
                neg_acts = utils.to_gpu(neg_acts)

            gen.eval()
            if step < max_vis:
                with torch.no_grad():
                    gen_loss_dict, gen_total_loss, gen_out, _ = run_generator_step(gen, disc, gen_tempo_optim,\
                                                                               gen_graphic_optim, disc_optim,
                                                                               imgs, acts, updated_warmup_steps, epoch,\
                                                                               False, opts)

                    writer.add_scalar('val_losses/recon_loss', gen_loss_dict['recon_loss'], num_val_steps)
                    utils.draw_output(gen_out, imgs, updated_warmup_steps, opts, vutils, vis_num_row, normalize, writer, num_val_steps,
                                        num_vis, tag='val_images')
                del gen_loss_dict, gen_total_loss, gen_out
            else:
                break

        print('Saving checkpoint')
        utils.save_model(os.path.join(opts.log_dir, 'model' + str(epoch) + '.pt'), epoch, gen, disc, opts)
        utils.save_optim(os.path.join(opts.log_dir, 'optim' + str(epoch) + '.pt'), epoch, gen_tempo_optim,
                            gen_graphic_optim, disc_optim)



if __name__ == '__main__':
    opts = TrainOptions().parse()
    train(opts)