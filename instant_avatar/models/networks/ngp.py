import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import tinycudann as tcnn
import numpy as np

EPS = 1e-3

class TruncExp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x.clamp(max=15, min=-15))

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(max=15, min=-15))


class NeRFNGPNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # self.encoder =  tcnn.NetworkWithInputEncoding(
        #     n_input_dims=3,
        #     n_output_dims=16,
        #     encoding_config={
        #         "otype": "HashGrid",
        #         "n_levels": 16,
        #         "n_features_per_level": 2,
        #         "log2_hashmap_size": 19,
        #         "base_resolution": 16,
        #         "per_level_scale": 1.5,
        #     },
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 1,
        #     }
        # )

        # self.color_net = tcnn.Network(
        #     n_input_dims=15,
        #     n_output_dims=3,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "Sigmoid",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 2,
        #     },
        # )
        # self.sigma_activ = lambda x: torch.relu(x).float()
        # self.sigma_activ = TruncExp.apply
        self.register_buffer("center", torch.FloatTensor(opt.center))
        self.register_buffer("scale", torch.FloatTensor(opt.scale))
        self.opt = opt

        # constants
        L = 16; F = 2; log2_T = 19; N_min = 16; 
        b = np.exp(np.log(2048/N_min)/(L-1))
        
        # canonical ----------------------------------------------
        # canonical positional encoding
        self.encoder = tcnn.NetworkWithInputEncoding(
            n_input_dims=3, n_output_dims=32,
            encoding_config={
                "otype": "Grid",
                "type": "Hash",
                "n_levels": L,
                "n_features_per_level": F,
                "log2_hashmap_size": log2_T,
                "base_resolution": N_min,
                "per_level_scale": b,
                "interpolation": "Linear"
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": 64,
                "n_hidden_layers": 1
            }
        )
        
        # self.cnl_dir_encoder = tcnn.Encoding(
        #     n_input_dims=3,
        #     encoding_config={
        #         "otype": "SphericalHarmonics",
        #         "degree": 4
        #     }
        # )
        
        self.color_net = tcnn.Network(
            n_input_dims=32, n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2
            }
        )
        
        self.sigma_net = tcnn.Network(
            n_input_dims=32, n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2
            }
        )

    def initialize(self, bbox):
        if hasattr(self, "bbox"):
            return
        c = (bbox[0] + bbox[1]) / 2
        s = (bbox[1] - bbox[0])
        self.center = c
        self.scale = s
        self.bbox = bbox

    def forward(self, x, d, cond=None):
        # normalize pts and view_dir to [0, 1]
        x = (x - self.center) / self.scale + 0.5
        # assert x.min() >= -EPS and x.max() < 1 + EPS
        x = x.clamp(min=0, max=1)
        x = self.encoder(x)
        # x = self.density_net(x)
        # sigma = x[..., 0]
        # color = self.color_net(x[..., 1:])
        sigma = self.sigma_net(x)[:, 0]
        color = self.color_net(x)
        # sigma = self.sigma_activ(sigma)
        return color.float(), sigma.float()
