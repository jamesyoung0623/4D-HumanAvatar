import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.network_util import ConvDecoder3D

class MotionWeightVolumeDecoder(nn.Module):
    def __init__(self, cfg):
        super(MotionWeightVolumeDecoder, self).__init__()
        self.embedding_size = 256
        self.volume_size = 32
        self.const_embedding = nn.Parameter(torch.randn(self.embedding_size), requires_grad=True)
        self.decoder = ConvDecoder3D(embedding_size=self.embedding_size)

        #self.n_components = 1
        #self.xys = [nn.Parameter(torch.randn((25, self.volume_size, self.volume_size, 1)), requires_grad=True) for _ in range(self.n_components)]
        #self.zs = [nn.Parameter(torch.randn((25, 1, 1, self.volume_size)), requires_grad=True) for _ in range(self.n_components)]

        #self.yzs = [nn.Parameter(torch.randn((25, 1, self.volume_size, self.volume_size)), requires_grad=True) for _ in range(self.n_components)]
        #self.xs = [nn.Parameter(torch.randn((25, self.volume_size, 1, 1)), requires_grad=True) for _ in range(self.n_components)]

        #self.xzs = [nn.Parameter(torch.randn((25, self.volume_size, 1, self.volume_size)), requires_grad=True) for _ in range(self.n_components)]
        #self.ys = [nn.Parameter(torch.randn((25, 1, self.volume_size, 1)), requires_grad=True) for _ in range(self.n_components)]

    def forward(self, motion_weights_priors, **_):
        #self.xyz = torch.zeros((25, self.volume_size, self.volume_size, self.volume_size))
        #for i in range(self.n_components):
            #self.xyz += self.zs[i]*self.xys[i]
            #self.xyz += self.xs[i]*self.yzs[i]
            #self.xyz += self.ys[i]*self.xzs[i]
            #self.xyz += torch.einsum('bp,bqr->bpqr', self.zs[i], self.xys[i])
            #self.xyz += torch.einsum('bp,bqr->bpqr', self.xs[i], self.yzs[i])
            #self.xyz += torch.einsum('bp,bqr->bpqr', self.ys[i], self.xzs[i])

        #decoded_weights = F.softmax(self.xyz.view(1, 25, self.volume_size, self.volume_size, self.volume_size).cuda() + torch.log(motion_weights_priors), dim=1)
        #decoded_weights = F.softmax(self.decoder(self.xyz) + torch.log(motion_weights_priors), dim=1)
        decoded_weights = F.softmax(self.decoder(self.const_embedding) + torch.log(motion_weights_priors), dim=1)
        return decoded_weights