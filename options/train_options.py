from argparse import ArgumentParser


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.set_opts()

    def set_opts(self):
        self.parser.add_argument('--data_name', default='gta', type=str, help='select dataset among gta, vizdoom')
        self.parser.add_argument('--batch_size', default=1, type=int)
        self.parser.add_argument('--num_workers', default=1, type=int, help='used for dataloader')
        self.parser.add_argument('--pin_memory', default=False, action="store_true", help='used for dataloader')
        self.parser.add_argument('--split_ratio', default=1, type=float, help='ratio for train and valid')
        self.parser.add_argument('--dirpath', default='datasets/gta/', type=str, help='default datapath')
        self.parser.add_argument('--num_epochs', default=20, type=int, help='total training epochs')
        self.parser.add_argument('--z_dim', default=32, type=int, help='random noise dimension for GAN')
    
    def parse(self):
        opts = self.parser.parse_args()
        return opts