from ctypes import util
from torch.utils import data
from pathlib import Path
import gzip
import pickle
import random
import numpy as np
import utils


def create_custom_dataloader(data_name, to_train, batch_size, num_workers, pin_memory, shuffle, 
                             num_action_spaces, split_ratio=0.9):
    if data_name == 'gta':
        dataset = GTADataset(to_train, split_ratio, num_action_spaces=num_action_spaces)
    else:
        raise Exception('spacify one of the supported dataset name: [gta]')
    
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                                  pin_memory=pin_memory, shuffle=shuffle, drop_last=True)



class GTADataset(data.dataset):

    def __init__(self, to_train, split_ratio, dirpath='./', ext='.pickle.gz', num_steps=32, 
                 num_action_spaces=3):
        self.num_steps = num_steps
        self.num_action_spaces = num_action_spaces
        self.samples = []
        dpath_plib = Path(dirpath)
        dflies = dpath_plib.glob("*" + ext)
        num_files = len(dflies)
        dflist = list(dflies)[:int(num_files * split_ratio)] if to_train else \
                 list(dflies)[int(num_files * split_ratio):]
        
        for df in dflist:
            dfpath = dpath_plib / df
            self.samples.append(dfpath)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with gzip.open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)

        imgs, actions, neg_actions = [], [], []
        ep_len = len(data['observations']) - self.num_steps

        # randomly pick a startpoint of sequential steps
        start_step = random.randint(0, ep_len - self.num_steps)

        for i in range(self.num_steps):
            x_t = data['observations'][start_step + i]
            a_idx = data['actions'][start_step + i]

            x_t = utils.make_channels_first_and_normalize_img(x_t)
            a_t = utils.make_label_idx_to_onehot(a_idx, self.num_action_spaces)

            neg_a_idx = random.randint(0, self.num_action_spaces - 1)
            while neg_a_idx == a_idx:
                neg_a_idx = random.randint(0, self.num_action_spaces - 1)
            neg_a_t = utils.make_label_idx_to_onehot(neg_a_idx, self.num_action_spaces)

            imgs.append(x_t)
            actions.append(a_t)
            neg_actions.append(neg_a_t)

        return imgs, actions, neg_actions