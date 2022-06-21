import torch
from data import create_custom_dataloader

import utils
from options.train_options import TrainOptions


def train(opts):
    """if benchmark is True, no reproducibility, but higher performance.
    else deterministically select an algorithm, possibly at the cost of 
    reduced performance.
    """
    use_gpu = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    if opts.data_name == 'gta':
        num_action_spaces = 3

    # train_loader shuffles dataset
    train_loader = create_custom_dataloader(opts.data_name, True, opts.batch_size, 
                                                opts.num_workers, opts.pin_memory, 
                                                num_action_spaces, 
                                                opts.split_ratio, opts.dirpath)
    # val_loader does not shuffle dataset
    val_loader = create_custom_dataloader(opts.data_name, False, opts.batch_size, 
                                          opts.num_workers, opts.pin_memory, 
                                          num_action_spaces, 
                                          opts.split_ratio, opts.dirpath)

    for epoch in range(opts.num_epochs):
        print(f'epoch {epoch}')
        data_iters = []
        data_iters.append(iter(train_loader))
        train_len = len(data_iters[0])
        # clear gpu memory cache
        torch.cuda.empty_cache()

        for imgs, acts, neg_acts in train_loader:
            imgs = utils.to_variable(imgs, use_gpu)
            acts = utils.to_variable(acts, use_gpu)
            neg_acts = utils.to_variable(neg_acts, use_gpu)
            print(len(imgs), len(acts), len(neg_acts))
            print(imgs[0].shape, acts[0].shape, neg_acts[0].shape)


if __name__ == '__main__':
    opts = TrainOptions().parse()
    train(opts)