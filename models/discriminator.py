from tokenize import Single
import torch
from torch import neg, neg_, nn
from torch.nn.utils import spectral_norm as SN
from models.model_modules import DResBlock, SA, Reshape


class Discriminator(nn.Module):
    """
    This class consists of a single image discriminator, action-conditioned discriminator, 
    and temporal discriminator
    """

    def __init__(self, batch_size, model_arch_dict, action_space, img_size, hidden_dim, neg_slope, 
                 temporal_window):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.neg_slope = neg_slope
        self.temporal_window = temporal_window

        self.single_disc = SingleImageDiscriminator(model_arch_dict)
        self.act_cond_disc = ActionConditionedDiscriminator(action_space, img_size, hidden_dim, neg_slope)
        self.tempo_disc = TemporalDiscriminator(self.batch_size, self.img_size, self.temporal_window, self.neg_slope)

    def forward(self, imgs, actions, num_warmup_frames, real_frames, neg_actions=None):
        # shape of imgs: (total steps - 1) * bs, c, h, w)
        # shape of actions: [(bs, action_space) * num_steps]
        # shape of real_frames: [(bs, 3, h, w) * num_steps]

        # change of shape after concat: [(bs, 3, h, w) * warmup_steps] -> (bs * warmup_steps, 3, h, w)
        warmup_real_frames = torch.cat(real_frames[:num_warmup_frames], dim=0)

        # as for the input of the discriminator, it combines warmup_real_frames, which consist of warmup steps of
        # dataset frames, and imgs, which consist of total steps of generated or dataset frames.
        full_frame_preds, bottom_fmaps = self.single_disc(torch.cat([warmup_real_frames, imgs], dim=0))
        
        # only use the results correspoing to the imgs
        full_frame_preds = full_frame_preds[num_warmup_frames*self.batch_size:]
        x_t1_fmaps = bottom_fmaps[num_warmup_frames*self.batch_size:]
        
        # take fmaps of warmup steps and the rest of generated or real frames except the last step
        x_t0_fmaps = torch.cat([bottom_fmaps[:num_warmup_frames*self.batch_size],
                                bottom_fmaps[(num_warmup_frames*2-1)*self.batch_size:-self.batch_size]],
                                dim=0)

        act_preds, neg_act_preds, act_recon, z_recon = self.act_cond_disc(actions, x_t0_fmaps, x_t1_fmaps, neg_actions)

        tempo_preds = self.tempo_disc(bottom_fmaps, num_warmup_frames)

        out = {}
        out['pred_frame_fmaps'] = x_t1_fmaps[:(len(real_frames)-1)*self.batch_size]
        out['act_preds'] = act_preds
        out['full_frame_preds'] = full_frame_preds
        out['tempo_preds'] = tempo_preds
        out['neg_act_preds'] = neg_act_preds
        out['act_recon'] = act_recon
        out['z_recon'] = z_recon
        return out


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

    def __init__(self, action_space, img_size, hidden_dim, neg_slope, debug=False):
        super(ActionConditionedDiscriminator, self).__init__()
        # In the original code, 256 is always used for the dim
        self.action_space = action_space
        self.debug = debug

        dim = 256
        kernel_size = (3, 5) if img_size[0] == 48 and img_size[1] == 80 else 4
        act_z = 32

        self.action_emb = nn.Linear(action_space, dim)
        # In the original code, BatchNorm is not used for block1 and block2
        # in the original paper, conv_for_x_t0_t1 acts as φ and ψ
        self.conv_for_x_t0_t1 = nn.Sequential(SN(nn.Conv2d(hidden_dim, dim,
                                                           kernel_size=kernel_size,
                                                           padding=0)),
                                              nn.LeakyReLU(neg_slope),
                                              Reshape((-1, dim)))
        self.last_linear_given_act = nn.Sequential(SN(nn.Linear(hidden_dim, hidden_dim)),
                                                   nn.LeakyReLU(neg_slope),
                                                   SN(nn.Linear(hidden_dim, 1)))

        self.reconstruct_action_z = nn.Linear(dim, action_space + act_z)
        
    def forward(self, actions, x_t0_fmaps, x_t1_fmaps, neg_actions):
        neg_act_preds = None
        
        # predict for potive samples
        if not self.debug:
            act_emb = self.action_emb(torch.cat(actions, dim=0))
        else:
            act_emb = self.action_emb(actions)
            print('act_emb:', act_emb.shape)
            print('x_t0_fmaps:', x_t0_fmaps.shape)
            print('x_t1_fmaps:', x_t1_fmaps.shape)
        transit_vecs = self.conv_for_x_t0_t1(torch.cat([x_t0_fmaps, x_t1_fmaps], dim=1))
        act_preds = self.last_linear_given_act(torch.cat([act_emb, transit_vecs], dim=1))
        
        # predict for negative samples
        if neg_actions is not None:
            if not self.debug:
                neg_act_emb = self.action_emb(torch.cat(neg_actions, dim=0))
            else:
                neg_act_emb = self.action_emb(neg_actions)
                print('neg_act_emb:', neg_act_emb.shape)
            neg_act_preds = self.last_linear_given_act(torch.cat([neg_act_emb, transit_vecs], dim=1))

        # calculate reconstruction of actions and zs for an info loss and action loss 
        act_z_recon = self.reconstruct_action_z(transit_vecs)
        act_recon = act_z_recon[:, :self.action_space]
        z_recon = act_z_recon[:, self.action_space:]

        if self.debug:
            print('act_preds:', act_preds.shape)
            print('neg_act_preds:', neg_act_preds.shape)
            print('act_recon:', act_recon.shape)
            print('z_recon:', z_recon.shape)
        return act_preds, neg_act_preds, act_recon, z_recon


class TemporalDiscriminator(nn.Module):

    def __init__(self, batch_size, img_size, temporal_window, neg_slope, num_filters=16, debug=False):
        super(TemporalDiscriminator, self).__init__()
        self.batch_size = batch_size
        self.debug = debug

        # arch hyper-params
        in_channels = num_filters * 16
        base_channels = 64
        if img_size[0] == 48 and img_size[1] == 80:
            kernel_size1 = (2, 2, 4)
            kernel_size2 = (3, 2, 2)
        else:
            kernel_size1 = (2, 2, 2)
            kernel_size2 = (3, 3, 3)
        stride1 = (1, 1, 1)
        stride2 = (2, 1, 1)
        first_logit_kernel = (2, 1, 1)
        first_logit_stride = (1, 1, 1)
        kernel_size3 = (3, 1, 1)
        stride3 = (1, 1, 1)
        second_logit_kernel = (3, 1, 1)
        second_logit_stride = (1, 1, 1)
        kernel_size4 = (3, 1, 1)
        stride4 = (2, 1, 1)
        third_logit_kernel = (4, 1, 1)
        third_logit_stride = (2, 1, 1)

        layers, logits = [], []

        # BatchNorms are not used in the original code
        first_layers = nn.Sequential(SN(nn.Conv3d(in_channels, base_channels, 
                                                  kernel_size1, stride1)),
                                     nn.LeakyReLU(neg_slope),
                                     SN(nn.Conv3d(base_channels, base_channels * 2, 
                                                  kernel_size2, stride2)),
                                     nn.LeakyReLU(neg_slope))
        
        first_logit = SN(nn.Conv3d(base_channels * 2, 1, first_logit_kernel, 
                                   first_logit_stride))
        layers.append(first_layers)
        logits.append(first_logit)

        if temporal_window >= 12:
            second_layers = nn.Sequential(SN(nn.Conv3d(base_channels * 2, base_channels * 4, 
                                                       kernel_size3, stride3)),
                                          nn.LeakyReLU(neg_slope))
            
            second_logit = SN(nn.Conv3d(base_channels * 4, 1, second_logit_kernel, 
                                        second_logit_stride))
            layers.append(second_layers)
            logits.append(second_logit)

        if temporal_window >= 18:
            third_layers = nn.Sequential(SN(nn.Conv3d(base_channels * 4, base_channels * 8, 
                                                       kernel_size4, stride4)),
                                          nn.LeakyReLU(neg_slope))
            
            third_logit = SN(nn.Conv3d(base_channels * 8, 1, third_logit_kernel, 
                                       third_logit_stride))
            layers.append(third_layers)
            logits.append(third_logit)

        self.conv3d_layers = nn.ModuleList(layers)
        self.conv3d_logits = nn.ModuleList(logits)

    def forward(self, bottom_fmaps, num_warmup_frames):
        """nn.Conv3d
        input: (bs, c_in, d_in, h_in, w_in) output: (bs, c_out, d_out, h_out, w_out)
        """
        if self.debug:
            bottom_fmaps = torch.randn(47, 256, 3, 5).cuda()
            num_warmup_frames = 16
            self.batch_size = 1

        # slice fmaps of warmup steps
        sliced_fmaps = []
        for fm in bottom_fmaps[:num_warmup_frames * self.batch_size].split(self.batch_size, dim=0):
            sliced_fmaps.append(fm)
        # slice fmaps of corresponding generated or real frames
        for fm in bottom_fmaps[(2 * num_warmup_frames - 1) * self.batch_size:].split(self.batch_size, dim=0):
            sliced_fmaps.append(fm)

        # create an input for conv3d. shape: (bs, c, d, h, w) (d is for temporal)
        inp = torch.stack(sliced_fmaps, dim=2)

        tempo_preds = []
        # use an input tensor shaped as [1, 256, 32, 3, 5] for debugging
        for layers, logit in zip(self.conv3d_layers, self.conv3d_logits):
            inp = layers(inp)
            pred = logit(inp)
            tempo_preds.append(pred.view(self.batch_size, -1))

        return tempo_preds        