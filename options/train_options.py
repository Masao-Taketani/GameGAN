from argparse import ArgumentParser


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.set_opts()

    def set_opts(self):
        self.parser.add_argument('--dataset_name', default='gta', type=str, help='select dataset among gta, vizdoom')
        self.parser.add_argument('--log_dir', default='logs', type=str, help='specify a log directory')
        self.parser.add_argument('--num_workers', default=1, type=int, help='used for dataloader')
        self.parser.add_argument('--pin_memory', default=False, action="store_true", help='used for dataloader')
        self.parser.add_argument('--split_ratio', default=0.9, type=float, help='ratio for train and valid')
        self.parser.add_argument('--datapath', default='datasets/gta/', type=str, help='default datapath')
        self.parser.add_argument('--num_epochs', default=60, type=int, help='total training epochs')
        self.parser.add_argument('--penultimate_tanh', action='store_true', default=True)
        # model hyper-parameters
        self.parser.add_argument('--img_size', default='48x80', type=str, help='gta: 48x80, vizdoom: 64x64')
        self.parser.add_argument('--total_steps', default=32, type=int, help='total steps used for each sequence')
        self.parser.add_argument('--warmup_steps', default=16, type=int, help='number of initial warm-up steps')
        self.parser.add_argument('--batch_size', default=1, type=int)
        self.parser.add_argument('--z_dim', default=32, type=int, help='random noise dimension for GAN')
        self.parser.add_argument('--neg_slope', default=0.2, type=float, help='slope for negative values of leaky ReLU')
        self.parser.add_argument('--no_gpu', action='store_false', help='specify when gpu is not used')
        self.parser.add_argument('--memory_dim', default=None, type=int, help='number of dimension for the memory module.\
                                                                               use None not to use it')
        self.parser.add_argument('--hidden_dim', default=512, type=int, help='number of dimension for hidden state')
        self.parser.add_argument('--num_inp_channels', default=3, type=int, help='number of image input channels')
        self.parser.add_argument('--temporal_window', default=32, type=int, help='used to decide hierarchical temporal levels')
        self.parser.add_argument('--lr', default=1e-4, type=float, help='learning rate used for both the generator and discriminator')
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