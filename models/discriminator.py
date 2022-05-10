from tokenize import Single
import torch
from torch import nn


class SingleImageDiscriminator(nn.Module):

    def __init__(self, model_arch_dict,):
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
        for i in range(len(self.model_arch_dict['out_channels'])):
            