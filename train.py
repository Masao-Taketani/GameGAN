import torch
from data import create_custom_dataloader

import utils
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

    num_components = 2 if opts.memory_dim is not None else 1
    if opts.dataset_name == 'gta':
        num_action_spaces = 3
    
    gen_model_arch_dict = get_gen_model_arch_dict(opts.dataset_name, num_components)
    disc_model_arch_dict = get_disc_model_arch_dict(opts.dataset_name)
    img_size = [int(size) for size in opts.img_size.split('x')]
    print('img_size', img_size)

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
                                                opts.split_ratio, opts.dirpath)
    # val_loader does not shuffle dataset
    val_loader = create_custom_dataloader(opts.dataset_name, False, opts.batch_size, 
                                          opts.num_workers, opts.pin_memory, 
                                          num_action_spaces, 
                                          opts.split_ratio, opts.dirpath)
    
    # set initial warm-up steps
    warmup_steps = opts.warmup_steps

    for epoch in range(opts.num_epochs):
        print(f'[epoch {epoch}]')
        data_iters = []
        data_iters.append(iter(train_loader))
        train_len = len(data_iters[0])
        # clear gpu memory cache
        torch.cuda.empty_cache()

        for imgs, acts, neg_acts in train_loader:
            if use_gpu:
                imgs = utils.to_gpu(imgs)
                acts = utils.to_gpu(acts)
                neg_acts = utils.to_gpu(neg_acts)
            #print(len(imgs), len(acts), len(neg_acts))
            #print(imgs[0].shape, acts[0].shape, neg_acts[0].shape)

            loss_dict, total_loss, gen_out, grads = run_generator_step(gen, disc, gen_tempo_optim,\
                                                                       gen_graphic_optim, disc_optim,
                                                                       imgs, acts, warmup_steps, epoch,\
                                                                       True, opts)



if __name__ == '__main__':
    opts = TrainOptions().parse()
    train(opts)