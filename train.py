import torch
from data import create_custom_dataloader

import utils


def train(opts):
    """if benchmark is True, no reproducibility, but higher performance.
    else deterministically select an algorithm, possibly at the cost of 
    reduced performance.
    """
    torch.backends.cudnn.benchmark = True

    # train_loader shuffles dataset
    train_loader = create_custom_dataloader(opts.data_name, True, opts.batch_size, 
                                                opts.num_workers, opts.pin_memory, 
                                                opts.num_action_spaces, 
                                                opts.split_ratio, opts.dirpath)
    # val_loader does not shuffle dataset
    val_loader = create_custom_dataloader(opts.data_name, False, opts.batch_size, 
                                          opts.num_workers, opts.pin_memory, 
                                          opts.num_action_spaces, 
                                          opts.split_ratio, opts.dirpath)

    for epoch in range(opts.num_epochs):
        print(f'epoch {epoch}')
        data_iters = []
        data_iters.append(iter(train_loader))
        train_len = len(data_iters[0])
        # clear gpu memory cache
        torch.cuda.empty_cache()

        for imgs, acts, neg_acts in train_loader:
            imgs = utils.to_variable(imgs, opts.use_gpu)
            acts = utils.to_variable(acts, opts.use_gpu)
            neg_acts = utils.to_variable(neg_acts, opts.use_gpu)



if __name__ == '__main__':
    train()