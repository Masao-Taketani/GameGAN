from tokenize import Single
import torch
from torch import nn

from models.model_modules import DResBlock, SA, SN


class SingleImageDiscriminator(nn.Module):

    def __init__(self, model_arch_dict, activation=nn.ReLU(inplace=False), out_dim=1):
        super(SingleImageDiscriminator, self).__init__()
        """
        [vizdoom]
        {'in_channels': [3, 16, 32, 64, 128], 
         'out_channels': [16, 32, 64, 128, 256], 
         'downsample': [True, True, True, True, False], 
         'resolution': [32, 16, 8, 4, 4], 
         'attention': {4: False, 8: False, 16: False, 32: False, 64: True}}
        [gta]
        {'in_channels':   [3, 16, 32, 64, 64, 64, 128, 128], 
         'out_channels': [16, 32, 64, 64, 64, 128, 128, 256], 
         'downsample': [True, True, False, False, True, True, False, False], 
         'resolution': [64, 32, 16, 8, 4, 4, 4, 4], 
         'attention': {4: False, 8: False, 16: False, 32: True, 64: True, 128: False}}
        """

        self.model_arch_dict = model_arch_dict
        self.activation = activation

        self.blocks = []
        for i in range(len(self.model_arch_dict['out_channels'])):
            self.blocks.append(DResBlock(model_arch_dict['in_channels'][i],
                                         model_arch_dict['out_channels'][i],
                                         activation,
                                         model_arch_dict['downsample'][i],
                                         use_preactivation=(i > 0)))
        
            if model_arch_dict['attention'][model_arch_dict['resolution'][i]]:
                self.blocks.append(SA(model_arch_dict['out_channels'][i]))

        self.module_list = nn.ModuleList(self.blocks)
        self.last_linear = SN(nn.Linear(model_arch_dict['out_channels'][-1], 
                                        out_dim, 
                                        bias=True))

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):
                # orthogonal initalization is used in the original code
                nn.init.orthogonal_(module.weight)

    def forward(self, x):
        h = x
        for f in self.module_list:
            h = f(h)
        h = self.activation(h)
        # Apply global sum pooling as in SN-GAN. It returns (bs, out_channels)
        out = torch.sum(h, [2, 3])
        out = self.linear(out)

        return out, h