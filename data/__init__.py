from sklearn.utils import shuffle
from torch.utils import data
from pathlib import Path
import gzip
import pickle
import random
import utils


def create_custom_dataloader(data_name, to_train, batch_size, num_workers, pin_memory, num_action_spaces, 
                             split_ratio=0.9, dirpath=None):
    
    shuffle = True if to_train else False

    if data_name == 'gta':
        if not dirpath:
            dataset = GTADataset(to_train, split_ratio, num_action_spaces=num_action_spaces)
        else:
            dataset = GTADataset(to_train, split_ratio, num_action_spaces=num_action_spaces, dirpath=dirpath)
    else:
        raise Exception('spacify one of the supported dataset name: [gta]')
    
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                                  pin_memory=pin_memory, shuffle=shuffle, drop_last=True)
    
    return data_loader


class GTADataset(data.Dataset):

    def __init__(self, to_train, split_ratio=0.9, ext='.pickle.gz', num_steps=32, 
                 num_action_spaces=3, dirpath='./../'):
        self.num_steps = num_steps
        self.num_action_spaces = num_action_spaces
        self.samples = []
        dpath_plib = Path(dirpath)
        dflist = list(dpath_plib.glob("*" + ext))
        num_files = len(list(dflist))
        dflist = dflist[:int(num_files * split_ratio)] if to_train else \
                 dflist[int(num_files * split_ratio):]
                 
        for df in dflist:
            dfpath = str(df)
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
            #print(x_t)
            #print(a_t)
            #print(neg_a_t)
            imgs.append(x_t)
            actions.append(a_t)
            neg_actions.append(neg_a_t)

        return imgs, actions, neg_actions