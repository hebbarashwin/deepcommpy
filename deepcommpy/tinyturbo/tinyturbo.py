import torch
import torch.nn as nn
import os 

class Turbo_subnet(nn.Module):
    def __init__(self, block_len, init_type = 'ones', one_weight = False):
        super(Turbo_subnet, self).__init__()

        assert init_type in ['ones', 'random', 'gaussian'], "Invalid init type"
        if init_type == 'ones':
            self.w1 = nn.parameter.Parameter(torch.ones((1, block_len)))
            self.w2 = nn.parameter.Parameter(torch.ones((1, block_len)))
            self.w3 = nn.parameter.Parameter(torch.ones((1, block_len)))
        elif init_type == 'random':
            self.w1 = nn.parameter.Parameter(torch.rand((1, block_len)))
            self.w2 = nn.parameter.Parameter(torch.rand((1, block_len)))
            self.w3 = nn.parameter.Parameter(torch.rand((1, block_len)))
        elif init_type == 'gaussian':
            self.w1 = nn.parameter.Parameter(0.001* torch.randn((1, block_len)))
            self.w2 = nn.parameter.Parameter(0.001*torch.randn((1, block_len)))
            self.w3 = nn.parameter.Parameter(0.001*torch.randn((1, block_len)))

        if one_weight:
            self.w3 = self.w1
            self.w2 = self.w1

    def forward(self, L_ext, L_sys, L_int):

        x = self.w1 * L_ext - self.w2 * L_sys - self.w3 * L_int

        return x

class TinyTurbo(nn.Module):
    def __init__(self, block_len, num_iter, init_type = 'ones', type = 'scale'):
        super(TinyTurbo, self).__init__()

        """
        Initialize weights for TinyTurbo
        Weight entanglement described in paper: 'scale'

        Other settings are ablation studies.
        """
        self.normal = nn.ModuleList()
        self.interleaved = nn.ModuleList()

        assert type in ['normal', 'normal_common', 'same_all', 'same_iteration', 'scale', 'scale_common', 'same_scale_iteration', 'same_scale', 'one_weight', 'default']

        if init_type == 'default' or 'type' == 'default':
            # Load pretrained TinyTurbo weights.
            for ii in range(num_iter):
                self.normal.append(Turbo_subnet(1, 'ones'))
                self.interleaved.append(Turbo_subnet(1, 'ones'))
            script_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint = torch.load(os.path.join(script_dir, "Results", "tinyturbo", "models", "weights.pt"))
            weights = checkpoint['weights']
            self.load_state_dict(weights)
        else:
            if type == 'normal':
                for ii in range(num_iter):
                    self.normal.append(Turbo_subnet(block_len, init_type))
                    self.interleaved.append(Turbo_subnet(block_len, init_type))

            if type == 'normal_common':
                for ii in range(num_iter):
                    net = Turbo_subnet(block_len, init_type)
                    self.normal.append(net)
                    self.interleaved.append(net)

            elif type == 'same_all':
                net = Turbo_subnet(block_len, init_type)
                for ii in range(num_iter):
                    self.normal.append(net)
                    self.interleaved.append(net)

            elif type == 'same_iteration':
                normal_net = Turbo_subnet(block_len, init_type)
                interleaved_net = Turbo_subnet(block_len, init_type)

                for ii in range(num_iter):
                    self.normal.append(normal_net)
                    self.interleaved.append(interleaved_net)

            elif type == 'scale':
                for ii in range(num_iter):
                    self.normal.append(Turbo_subnet(1, init_type))
                    self.interleaved.append(Turbo_subnet(1, init_type))

            elif type == 'scale_common':
                for ii in range(num_iter):
                    net = Turbo_subnet(1, init_type)
                    self.normal.append(net)
                    self.interleaved.append(net)

            elif type == 'same_scale':
                net = Turbo_subnet(1, init_type)
                for ii in range(num_iter):
                    self.normal.append(net)
                    self.interleaved.append(net)

            elif type == 'same_scale_iteration':
                net_normal = Turbo_subnet(1, init_type)
                net_interleaved = Turbo_subnet(1, init_type)
                for ii in range(num_iter):
                    self.normal.append(net_normal)
                    self.interleaved.append(net_interleaved)

            elif type == 'one_weight':
                net = Turbo_subnet(1, init_type, one_weight = True)
                for ii in range(num_iter):
                    self.normal.append(net)
                    self.interleaved.append(net)