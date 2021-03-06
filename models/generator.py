import sys
import os
from unittest.mock import NonCallableMagicMock
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as SN
import numpy as np

from models.model_modules import H, C, ActionLSTM, REResBlock, SA
import utils


class Generator(nn.Module):

    def __init__(self, batch_size, z_dim, hidden_dim, use_gpu, num_a_space, 
                 neg_slope, img_size, num_inp_channels, model_arch_dict,
                 dataset_name='gta', N=21, device='cuda', memory_dim=None):
                 # when the memory module is not used, set None to memory_dim
        super(Generator, self).__init__()
        self.batch_size = batch_size
        # z_dist is used to sample z, so that the info loss can be calculated
        self.z_dist = utils.get_random_noise_dist(z_dim)
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.memory_dim = memory_dim
        # K: number of components
        K = 1

        self.de = DynamicsEngine(z_dim, hidden_dim, num_a_space, neg_slope, 
                                 img_size, num_inp_channels, memory_dim)
        
        if self.memory_dim is not None:
            self.memory = Memory(batch_size, dataset_name, hidden_dim, 
                                 num_a_space, neg_slope, memory_dim,
                                 N, devide=device)
            K = 2
            self.cycle_start_epoch = 0

        self.re = RenderingEngine(batch_size, hidden_dim, K, model_arch_dict, 
                                  memory_dim, nn.ReLU(inplace=False))        

    def proceed_step(self, x, h, c, a, M=None, alpha_prev=None, m_vec_prev=None):
        # x: 1 step image
        x = x.detach()
        z = self.z_dist.sample((self.batch_size,))
        if self.use_gpu:
            z = utils.to_gpu(z)
        h_next, c_next = self.de(h, c, x, a, z, m_vec_prev)

        if self.memory_dim is not None:
            components, M, alpha = self.memory(h_next, h, a, alpha_prev, M)
            m_vec = components[0]
        else:
            components = h
            alpha = None
            m_vec = None

        x_next, fine_masks, maps, base_imgs = self.re(components)

        alpha_loss = 0
        if self.memory_dim is not None:
            for i in range(1, len(fine_masks)):
                alpha_loss += (fine_masks[i].abs().sum() / self.batch_size)

        # everything to the right of c_next belongs to Memory module
        return x_next, h_next, c_next, z, M, alpha, m_vec, fine_masks, maps, base_imgs, alpha_loss

    def run_warmup_phase(self, x_real, a, warmup_steps, M=None, alpha_prev=None, m_vec_prev=None):
        h, c = self.de.init_hidden_state_and_cell(self.batch_size)
        h = utils.to_variable(h, self.use_gpu)
        c = utils.to_variable(c, self.use_gpu)

        if self.memory_dim is not None:
            # When the memory is used and M, alpha_prev, and m_vec_prev are None,
            # initialize them
            if M is None and alpha_prev is None and m_vec_prev is None:
                M = self.memory().init_M()
                alpha_prev = torch.zeros(self.batch_size, self.memory.N ** 2)
                # put 1.0 in the center of alpha_prev
                alpha_prev[:, (self.memory.N // 2) * self.memory.N + self.memory.N // 2] = 1.0
                m_vec_prev = torch.zeros(self.batch_size, self.memory_dim)
                M = utils.to_variable(M, self.use_gpu)
                alpha_prev = utils.to_variable(alpha_prev, self.use_gpu)
                m_vec_prev = utils.to_variable(m_vec_prev, self.use_gpu)
        
        x_next = None
        out_imgs = []
        zs = []
        alphas = []
        fine_mask_list = []
        unmasked_base_imgs = []
        map_list = []
        alpha_losses = 0
        for i in range(warmup_steps):
            # To conduct warmups, always use real images for a specified warm-up step(s)
            x_i = x_real[i]
            a_i = a[i]
            x_next, h, c, z, M, alpha_prev, m_vec_prev, fine_masks, maps, base_imgs, alpha_loss = self.proceed_step(x_i, 
                                                                                                            h, c, a_i, M, 
                                                                                                            alpha_prev, 
                                                                                                            m_vec_prev)
            out_imgs.append(x_next)
            zs.append(z)
            alphas.append(alpha_prev)
            fine_mask_list.append(fine_masks)
            map_list.append(maps)
            unmasked_base_imgs.append(base_imgs)
            alpha_losses += alpha_loss

        warmup_h_c = [h, c]
        # when warm-up is not conducted at all, set x0 as the initial state
        if x_next is None: x_next = x_real[0]

        return x_next, warmup_h_c, M, alpha_prev, m_vec_prev, out_imgs, zs, alphas, fine_mask_list, map_list, \
               unmasked_base_imgs, alpha_losses

    def forward(self, x_real, a, warmup_steps):
        # run warm-up phase
        x, warmup_h_c, M, alpha_prev, m_vec_prev, out_imgs, zs, alphas, fine_mask_list, map_list, \
               unmasked_base_imgs, alpha_losses = self.run_warmup_phase(x_real, a, warmup_steps)

        h, c = warmup_h_c
        end_steps = len(a) - 1
        # run post warm-up phase
        for i in range(warmup_steps, end_steps):
            x, h, c, z, M, alpha_prev, m_vec_prev, fine_masks, maps, base_imgs, alpha_loss \
                                    = self.proceed_step(x, h, c, a[i], M, alpha_prev, m_vec_prev)

            out_imgs.append(x)
            zs.append(z)
            alphas.append(alpha_prev)
            fine_mask_list.append(fine_masks)
            map_list.append(maps)
            unmasked_base_imgs.append(base_imgs)
            alpha_losses += alpha_loss
        avg_alpha_loss = alpha_losses / end_steps

        reverse_alphas, reverse_out_imgs, reverse_fine_masks, reverse_base_imgs = [], [], [], []
        if self.memory_dim is not None:
            # i: len(alphas) - 1, ..., 2, 1, 0
            for i in range(len(alphas) - 1, -1, -1):
                # shape of M: (bs, N ** 2, memory_dim)
                # shape of alpha: (bs, N ** 2)
                read_vec = self.memory.read(alphas[i], M)
                _, reverse_fine_mask, _, reverse_unmasked_base_img = \
                                                self.re([read_vec, torch.zeros_like(read_vec)])
                reverse_alphas.append(alphas[i])
                reverse_out_imgs.append(reverse_unmasked_base_img[2] * fine_mask_list[i][0])
                reverse_fine_masks.append(reverse_fine_mask)
                reverse_base_imgs.append(reverse_unmasked_base_img)

        out = {}
        out['out_imgs'] = out_imgs
        out['zs'] = zs
        out['alphas'] = alphas
        out['fine_masks'] = fine_mask_list
        out['maps'] = map_list
        out['unmasked_base_imgs'] = unmasked_base_imgs
        out['avg_alpha_loss'] = avg_alpha_loss
        out['reverse_alphas'] = reverse_alphas
        out['reverse_out_imgs'] = reverse_out_imgs
        out['reverse_fine_masks'] = reverse_fine_masks
        out['reverse_base_imgs'] = reverse_base_imgs
        return out


class DynamicsEngine(nn.Module):

    def __init__(self, z_dim, hidden_dim, num_a_space, neg_slope, img_size, 
                 num_inp_channels, memory_dim=None):
        super(DynamicsEngine, self).__init__()
        self.H = H(z_dim, hidden_dim, num_a_space, neg_slope, memory_dim)
        self.C = C(hidden_dim, neg_slope, img_size, num_inp_channels)
        self.W_s = nn.Sequential(nn.Linear(hidden_dim, 4 * hidden_dim))
        self.action_lstm = ActionLSTM(hidden_dim, neg_slope, self.H.concat_dim)

    def init_hidden_state_and_cell(self, batch_size):
        # initialize the hidden state and the cell state to be 0s
        return torch.zeros(batch_size, self.action_lstm.hidden_dim), torch.zeros(batch_size, self.action_lstm.hidden_dim)
        
    def forward(self, h, c, x, a, z, m_vec_prev=None):
        """
        arguments:
            h: h_t-1
            c: c_t-1
            x: x_t
            a: a_t
            z: z_t
            m_vec_prev: retrieved memory vector in the previous step
        """
        H = self.H(a, z, m_vec_prev)
        weighted_s = self.W_s(self.C(x))
        h_t, c_t = self.action_lstm(h, c, H, weighted_s)
        return h_t, c_t


class Memory(nn.Module):

    def __init__(self, batch_size, dataset_name, hidden_dim, num_a_space, neg_slope, memory_dim,
                 N, device, use_h_for_gate=False, devide_scale_for_softmax=0.1):
        super(Memory, self).__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.memory_dim = memory_dim
        self.N = N
        self.device = device
        self.use_h_for_gate = use_h_for_gate
        self.devide_scale_for_softmax = devide_scale_for_softmax

        self.K = nn.Sequential(nn.Linear(num_a_space, memory_dim),
                               nn.LeakyReLU(neg_slope),
                               nn.Linear(memory_dim, 9))
        self.G = nn.Sequential(nn.Linear(hidden_dim, memory_dim),
                               nn.LeakyReLU(neg_slope),
                               nn.Linear(memory_dim, 1),
                               nn.Sigmoid())
        self.E = nn.Sequential(nn.Linear(hidden_dim, 3 * memory_dim))
        # register M state as constant
        self.register_buffer('M_bias', torch.Tensor(self.N ** 2, self.memory_dim))

    def init_M(self):
        std = 1 / (np.sqrt(self.N * self.N + self.memory_dim))
        # nn.init.uniform_ updates self.M_bias tensor
        nn.init.uniform_(self.M_bias, -std, std)
        return self.M_bias.clone().repeat(self.batch_size, 1, 1)

    def conv2d_for_alpha_and_w(self, alpha_prev, w):
        """
        inputs:
            shape of alpha_prev: [batch_size, 1, N, N]
            shape of w: [batch_size, 1, 3, 3]
        outputs:
            shape of alpha_t: [batch_size, 1, N, N]
        """
        inputs = alpha_prev.view(1, self.batch_size, self.N, self.N)
        return F.conv2d(inputs, w, padding=1, groups=self.batch_size)

    def write(self, erase_vec, add_vec, alpha_t, M):
        # (bs, N**2 , 1) * (bs, 1, memory_dim) -> (bs, N**2, memory_dim)
        alpha_erase = torch.bmm(alpha_t.unsqueeze(-1), erase_vec.unsqueeze(1))
        # (bs, N**2 , 1) * (bs, 1, memory_dim) -> (bs, N**2, memory_dim)
        alpha_add = torch.bmm(alpha_t.unsqueeze(-1), add_vec.unsqueeze(1))
        return M * (1 - alpha_erase) + alpha_add

    def read(self, alpha_t, M):
        # (bs, 1, N**2) * (bs, N**2, memory_dim) -> (bs, 1, memory_dim)
        m_t = torch.bmm(alpha_t.unsqueeze(1), M)
        # shape of m_t: (bs, memory_dim)
        return m_t.squeeze(1)

    def forward(self, h, h_prev, a, alpha_prev, M):
        if self.use_h_for_gate:
            g_input = h
        else:
            # I don't know why, but this one is used as default
            h_norm = F.normalize(h, dim=1)
            h_prev_norm = F.normalize(h_prev, dim=1)
            g_input = h_norm - h_prev_norm
        
        w = self.K(a).view(-1, 1, 3, 3)

        # for the following flipping thing, I don't why this is done. Nothing mentioned
        # in the paper, but the original code does it
        a_flipped = a.cpu().numpy()
        _, a_idxes = torch.max(a, 1)
        a_idxes = a_idxes.long().cpu().numpy()
        mask = np.zeros((self.batch_size, 1))
        for i in range(self.batch_size):
            if 'gta' in self.dataset_name:
                # 0: left, 1: no action, 2: right
                if a_idxes[i] == 0:
                    a_flipped[i][2] = 1.0
                    a_flipped[i][0] = 0.0
                    mask[i][0] = 1.0
        mask = torch.tensor(mask, device=self.device).float().view(-1, 1, 1, 1)
        a_flipped = torch.tensor(a_flipped, device=self.device).float()

        w_flipped = torch.flip(self.K(a_flipped).view(-1, 1, 3, 3), [2, 3])
        w = (1 - mask) * w + mask * w_flipped
        w = F.softmax(w.view(self.batch_size, -1) / self.devide_scale_for_softmax, dim=1)
        w = w.view(self.batch_size, 1, 3, 3)

        g = self.G(g_input)
        alpha_t = self.conv2d_for_alpha_and_w(alpha_prev.view(self.batch_size, 
                                                            1, 
                                                            self.N, 
                                                            self.N), 
                                            w)
        # (batch_size, 1, N, N) -> (batch_size, N ** 2)
        alpha_t = alpha_t.view(self.batch_size, -1)
        alpha_t = alpha_t * g + alpha_prev * (1 - g)
        erase_add_vecs = self.E(h)
        erase_vec = erase_add_vecs[:, :self.memory_dim]
        erase_vec = erase_vec.sigmoid()
        add_vec = erase_add_vecs[:, self.memory_dim:2 * self.memory_dim]
        # comp_vec is used for the rendering engine
        comp_vec = erase_add_vecs[:, 2 * self.memory_dim:]

        M = self.write(erase_vec, add_vec, alpha_t, M)
        m_t = self.read(alpha_t, M)
        # the first list is used for the rendering engine
        return [m_t, comp_vec], M, alpha_t


class RenderingEngine(nn.Module):

    def __init__(self, batch_size, hidden_dim, K, model_arch_dict, use_memory, activation=nn.ReLU(inplace=False)):
        super(RenderingEngine, self).__init__()
        self.batch_size = batch_size
        self.K = K
        self.model_arch_dict = model_arch_dict

        """
        [vizdoom]
        when K=1
        {'img_size': (64, 64), 'first_fmap_size': (8, 8), 'in_channels': [512, 256, 128], 'out_channels': [256, 128, 64], 
         'upsample': [2, 2, 2], 'resolution': [16, 32, 64], 
         'attention': {16: False, 32: False, 64: True}}
        when K=2
        {'img_size': (64, 64), 'first_fmap_size': (8, 8), 'in_channels': [256, 128, 64], 'out_channels': [128, 64, 32], 
        'upsample': [2, 2, 2], 'resolution': [16, 32, 64], 
        'attention': {16: False, 32: False, 64: True}}
        [gta]
        k = 1
        {'img_size': (48, 80), 'first_fmap_size': (6, 10), 'in_channels': [768, 384, 192, 96, 96], 'out_channels': [384, 192, 96, 96, 96],
         'upsample': [2, 2, 2, 1, 1], 'resolution': [16, 32, 64, 128, 256],
         'attention': {16: False, 32: True, 64: True, 128: False, 256: False}}
        K = 2
        {'img_size': (48, 80), 'first_fmap_size': (6, 10), 'in_channels': [256, 128, 64, 32, 32], 'out_channels': [128, 64, 32, 32, 32], 
         'upsample': [2, 2, 2, 1, 1], 'resolution': [16, 32, 64, 128, 256], 
         'attention': {16: False, 32: True, 64: True, 128: False, 256: False}}
        """
        self.blocks = []
        for i in range(self.K):
            if self.K == 1:
                self.sn_linear = SN(nn.Linear(hidden_dim, 
                    model_arch_dict['in_channels'][0] * model_arch_dict['first_fmap_size'][0] * model_arch_dict['first_fmap_size'][1],
                    bias=True))

            for j in range(len(self.model_arch_dict['out_channels'])):
                upsample_scale_factor = self.model_arch_dict['upsample'][j]
                
                if j == 0:
                    in_channels = self.model_arch_dict['in_channels'][0] if self.K == 1 else 512
                else:
                    in_channels = self.model_arch_dict['in_channels'][j]
                
                self.blocks.append(REResBlock(in_channels, self.model_arch_dict['out_channels'][j], 
                                              upsample_sacle_factor=upsample_scale_factor, activation=activation))

                if self.model_arch_dict['attention'][self.model_arch_dict['resolution'][j]] and self.K == 1:
                    self.blocks.append(SA(self.model_arch_dict['out_channels'][j]))
            
            self.blocks.extend([activation, SN(nn.Conv2d(self.model_arch_dict['out_channels'][-1], 3, 3, padding=1))])

        self.module_list = nn.ModuleList(self.blocks)
        #if K == 1:
        #    if use_memory:

        self.init_weights()
    
    def init_weights(self):
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):
                # orthogonal initalization is used in the original code
                nn.init.orthogonal_(module.weight)

    def forward(self, components):
        if self.K == 1:
            # simple rendering engine
            # components: h_t
            h = self.sn_linear(components)
            h = h.view(h.shape[0], -1, self.model_arch_dict['first_fmap_size'][0], self.model_arch_dict['first_fmap_size'][1])
            for f in self.module_list:
                h = f(h)
            rendering_img = torch.tanh(h)
            return rendering_img, [], [], []
        else:
            # rendering engine for disentangling static and dynamic components
            # components: [m_t, other_vec]
            rendering_img = 0
            fine_masks = None
            maps = NonCallableMagicMock
            unmasked_base_imgs = []
            return rendering_img, fine_masks, maps, unmasked_base_imgs