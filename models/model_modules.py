import torch
from torch import nn
from torch.nn.utils import spectral_norm as SN
from torch.nn import functional as F
import math


class H(nn.Module):
    """process action, stochastic variable(, retrieved memory vector: if memory is used)"""
    def __init__(self, z_dim, hidden_dim, num_a_space, neg_slope, memory_dim=None):
        super(H, self).__init__()
        self.memory_dim = memory_dim

        # get embeddings for a and z
        self.embed_a = nn.Sequential(nn.Linear(num_a_space, hidden_dim),
                                     nn.LeakyReLU(neg_slope))
        self.embed_z = nn.Sequential(nn.Linear(z_dim, hidden_dim),
                                     nn.LeakyReLU(neg_slope))

        # concat_dim: concat of [a, v(, mem: if memory module is used)]
        self.concat_dim = hidden_dim * 2
        if self.memory_dim:
            self.concat_dim += self.memory_dim

        # two-layered MLP
        self.mlp = nn.Sequential(nn.Linear(self.concat_dim, self.concat_dim),
                                 nn.LeakyReLU(neg_slope),
                                 nn.Linear(self.concat_dim, self.concat_dim))


    def forward(self, a, z, m):
        a_emb = self.embed_a(a)
        z_emb = self.embed_z(z)

        if self.memory_dim:
            concats = torch.cat([a_emb, z_emb, m], dim=1)
        else:
            concats = torch.cat([a_emb, z_emb], dim=1)

        return self.mlp(concats)


class Reshape(nn.Module):
    """Reshape layer"""
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, inp):
        return inp.view(self.shape)


class C(nn.Module):
    """image encoder"""
    def __init__(self, hidden_dim, neg_slope, img_size, num_inp_channels):
        """
        img_size:
            64x64: 5 layered convs(for VizDoom)
            84x84: 6 layered convs(for Pacman)
        """
        super(C, self).__init__()
        model_type = self.check_conv_type(img_size)
        if model_type == 'vizdoom':
            convs = [nn.Conv2d(num_inp_channels, hidden_dim // 8, 4, 1, 1),
                     nn.LeakyReLU(neg_slope),
                     nn.Conv2d(hidden_dim // 8, hidden_dim // 8, 3, 2, 0),
                     nn.LeakyReLU(neg_slope),
                     nn.Conv2d(hidden_dim // 8, hidden_dim // 8, 3, 2, 0),
                     nn.LeakyReLU(neg_slope),
                     nn.Conv2d(hidden_dim // 8, hidden_dim // 8, 3, 2, 0),
                     nn.LeakyReLU(neg_slope),
                     Reshape((-1, 7 * 7 * (hidden_dim // 8))),
                     nn.Linear(7 * 7 * (hidden_dim // 8), hidden_dim),
                     nn.LeakyReLU(neg_slope)]
        elif model_type == 'pacman':
            convs = [nn.Conv2d(num_inp_channels, hidden_dim // 8, 3, 2, 0),
                     nn.LeakyReLU(neg_slope),
                     nn.Conv2d(hidden_dim // 8, hidden_dim // 8, 3, 1, 0),
                     nn.LeakyReLU(neg_slope),
                     nn.Conv2d(hidden_dim // 8, hidden_dim // 8, 3, 2, 0),
                     nn.LeakyReLU(neg_slope),
                     nn.Conv2d(hidden_dim // 8, hidden_dim // 8, 3, 1, 0),
                     nn.LeakyReLU(neg_slope),
                     nn.Conv2d(hidden_dim // 8, hidden_dim // 8, 3, 2, 0),
                     nn.LeakyReLU(neg_slope),
                     Reshape((-1, 8 * 8 * (hidden_dim // 8))),
                     nn.Linear( 8 * 8 * (hidden_dim // 8), hidden_dim),
                     nn.LeakyReLU(neg_slope)]
        
        self.img_encoder = nn.Sequential(*convs)

    def check_conv_type(self, img_size):
        img_shape = [int(size) for size in img_size.split('x')]
        if img_shape[0] == img_shape[1]:
            if img_shape[0] == 64:
                model_type = 'vizdoom'
            elif img_shape[0] == 84:
                model_type = 'pacman'
            else:
                raise ValueError(f"img_size {img_size} cannot be accepted")
        else:
            # TODO: implement the case when img_size is not square
            raise ValueError(f"img_size {img_size} cannot be accepted")
        return model_type

    def forward(self, inp):
        return self.img_encoder(inp)


class ActionLSTM(nn.Module):

    def __init__(self, hidden_dim, neg_slope, concat_dim):
        super(ActionLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        # 4 * hidden_dim is used for i_t, f_t, o_t, c_t
        # self.W_v = nn.Sequential([nn.Linear(concat_dim, 4 * hidden_dim)])
        # this is used in the original repo
        self.W_v = nn.Sequential(nn.Linear(concat_dim, concat_dim),
                                 nn.LeakyReLU(neg_slope),
                                 nn.Linear(concat_dim, 4 * hidden_dim))
        self.W_s = nn.Sequential(nn.Linear(hidden_dim, 4 * hidden_dim))

        self.init_Ws()

    def init_Ws(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, c_prev, v, s):
        weighted_v_s = self.W_v(v) + self.W_s(s)
        i_t = weighted_v_s[:, :self.hidden_dim].sigmoid()
        f_t = weighted_v_s[:, self.hidden_dim: 2* self.hidden_dim].sigmoid()
        o_t = weighted_v_s[:, 2 * self.hidden_dim: 3 * self.hidden_dim].sigmoid()
        candidate_state = weighted_v_s[:, 3 * self.hidden_dim: 4 * self.hidden_dim].tanh()
        c_t = f_t * c_prev + i_t * candidate_state
        h_t = o_t * c_t.tanh()
        return h_t, c_t


class REResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsample_sacle_factor=2):
        super(REResBlock, self).__init__()
        self.upsample_scale_factor = upsample_sacle_factor
        self.instance_norm_1 = nn.InstanceNorm2d(in_channels)
        self.instance_norm_2 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.sn_conv2d_1 = SN(nn.Conv2d(in_channels, out_channels, kernel_size))
        self.sn_conv2d_2 = SN(nn.Conv2d(out_channels, out_channels, kernel_size))
        self.use_1x1conv = True if in_channels != out_channels else False
        if self.use_1x1conv:
            self.conv2d1x1 = SN(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))

    def forward(self, inp):
        x = self.instance_norm_1(inp)
        x = self.relu(x)
        # upscale each height and width for each feature map
        x = F.interpolate(x, scale_factor=self.upsample_scale_factor)
        x = self.sn_conv2d_1(x)
        x = self.instance_norm_2(x)
        x = self.relu(x)
        x = self.sn_conv2d_2(x)

        # for a skip connection
        inp = F.interpolate(inp, scale_factor=self.upsample_scale_factor)
        if self.use_1x1conv:
            inp = self.conv2d1x1(inp)
        return x + inp