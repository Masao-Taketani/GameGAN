from argparse import ArgumentParser


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.set_opts()

    def set_opts(self):
        self.parser.add_argument('--data_name', default='gta', type=str, help='select dataset among gta, vizdoom')
        self.parser.add_argument('--total_steps', default=32, type=int, help='total steps used for each sequence')
        self.parser.add_argument('--batch_size', default=1, type=int)
        self.parser.add_argument('--num_workers', default=1, type=int, help='used for dataloader')
        self.parser.add_argument('--pin_memory', default=False, action="store_true", help='used for dataloader')
        self.parser.add_argument('--split_ratio', default=1, type=float, help='ratio for train and valid')
        self.parser.add_argument('--dirpath', default='datasets/gta/', type=str, help='default datapath')
        self.parser.add_argument('--num_epochs', default=20, type=int, help='total training epochs')
        self.parser.add_argument('--z_dim', default=32, type=int, help='random noise dimension for GAN')
        self.parser.add_argument('--neg_slope', default=0.2, type=float, help='slope for negative values of leaky ReLU')
        self.parser.add_argument('--no_gpu', action='store_false', help='specify when gpu is not used')
        # For loss calculation
        self.parser.add_argument('--lambda_I', default=30.0, type=float, help='Info loss multiplier')
        self.parser.add_argument('--lambda_r', default=0.05, type=float, help='Image reconstruction loss multiplier')
        self.parser.add_argument('--lambda_f', default=10.0, type=float, help='Feature reconstruction loss multiplier')
        self.parser.add_argument('--lambda_mem_reg', default=0.000075, type=float, help='Memory regularization multiplier')
        self.parser.add_argument('--lambda_c', default=0.05, type=float, help='Cycle loss multiplier')
        self.parser.add_argument('--gamma', default=10.0, type=float, help='gradient penalty multiplier for real data')
    
    def parse(self):
        opts = self.parser.parse_args()
        return opts