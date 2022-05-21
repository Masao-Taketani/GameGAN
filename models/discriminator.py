from tokenize import Single
import torch
from torch import nn

from models.model_modules import DResBlock, SA, SN, Reshape


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
        # shape of h: (bs, out_channels, h, w)
        h = self.activation(h)
        # Apply global sum pooling as in SN-GAN. It returns (bs, out_channels)
        out = torch.sum(h, [2, 3])
        out = self.last_linear(out)

        return out, h


class ActionConditionedDiscriminator(nn.Module):

    def __init__(self, batch_size, action_space, img_size, hidden_dim, neg_slope, model_arch_dict, neg_action=None):
        super(ActionConditionedDiscriminator, self).__init__()
        # In the original code, 256 is always used for the dim
        self.batch_size = batch_size
        dim = 256
        kernel_size = (3, 5) if img_size[0] == 48 and img_size[1] == 80 else 4

        self.action_emb = nn.Linear(action_space, dim)
        # In the original code, BatchNorm is not used for block1 and block2
        self.conv_for_x_t0_t1 = nn.Sequential(SN(nn.Conv2d(hidden_dim, dim,
                                                 kernel_size=kernel_size,
                                                 padding=0)),
                                                 nn.LeakyReLU(neg_slope),
                                                 Reshape((-1, dim)))
        self.linear_given_act = nn.Sequential(SN(nn.Linear(hidden_dim, hidden_dim)),
                                                 nn.LeakyReLU(neg_slope),
                                                 SN(nn.Linear(hidden_dim, 1)))
        
        self.single_disc = SingleImageDiscriminator(model_arch_dict)
        
    def forward(self, imgs, actions, num_warmup_frames, real_frames, ):
        # shape of imgs: (total steps - 1) * bs, c, h, w)
        # shape of actions: [(bs, action_space) * num_steps]
        # shape of real_frames: [(bs, 3, h, w) * num_steps]

        # change of shape after concat: [(bs, 3, h, w) * warmup_steps] -> (bs * warmup_steps, 3, h, w)
        warmup_real_frames = torch.cat(real_frames[:num_warmup_frames], dim=0)
        # as for the input of the discriminator, it combines warmup_real_frames, which consist of warmup steps of
        # dataset frames, and imgs, which consist of total steps of generated or dataset frames.
        full_frame_pred, bottom_fmaps = self.single_disc(torch.cat([warmup_real_frames, imgs], dim=0))
        # only use the results correspoing to the imgs
        full_frame_pred = full_frame_pred[num_warmup_frames*self.batch_size:]
        x_t1_fmaps = bottom_fmaps[num_warmup_frames*self.batch_size:]
        # taking fmaps of warmup steps and the rest of generated of real frames except the last step
        x_t0_fmaps = torch.cat([bottom_fmaps[:num_warmup_frames*self.batch_size],
                                                 bottom_fmaps[(num_warmup_frames*2-1)*self.batch_size:-self.batch_size]],
                                                 dim=0)
        act_emb = self.action_emb(torch.cat(actions, dim=0))
        merged_fmaps = self.conv_for_x_t0_t1(torch.cat([x_t0_fmaps, x_t1_fmaps], dim=1))
        consistent_pred_given_act = self.linear_given_act(torch.cat([act_emb, merged_fmaps], dim=1))


class TemporalDiscriminator():