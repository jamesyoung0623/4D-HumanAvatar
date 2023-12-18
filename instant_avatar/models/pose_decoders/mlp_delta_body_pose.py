import torch
import torch.nn as nn
import tinycudann as tcnn
from instant_avatar.utils.network_util import initseq, RodriguesModule

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


class BodyPoseRefiner(nn.Module):
    def __init__(self):
        super(BodyPoseRefiner, self).__init__()
        self.rvec_net = tcnn.Network(
            n_input_dims=72, n_output_dims=72,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": 128,
                "n_hidden_layers": 4
            }
        )

        self.transl_net = tcnn.Network(
            n_input_dims=3, n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": 64,
                "n_hidden_layers": 2
            }
        )

        self.rodriguez = RodriguesModule()
    
        
    def forward(self, body_params):
        global_orient = body_params["global_orient"] 
        body_pose = body_params["body_pose"] 
        transl = body_params["transl"]

        """
        rvec = torch.cat((global_orient, body_pose), dim=1)
        delta_rvec = self.rvec_net(rvec)

        rmat = axis_angle_to_matrix(rvec.view(-1, 3)).view(-1, 24, 3, 3)
        delta_rmat = axis_angle_to_matrix(delta_rvec.view(-1, 3)).view(-1, 24, 3, 3)
        
        rmat = torch.matmul(rmat.type(torch.HalfTensor).cuda().reshape(-1, 3, 3), delta_rmat.type(torch.HalfTensor).cuda().reshape(-1, 3, 3))
         
        for i in range(rmat.shape[0]):
            rvec = matrix_to_axis_angle(rmat[i])
            if i == 0:
                body_params['global_orient'][0] = rvec
            else:
                body_params['body_pose'][0][3*(i-1):3*i] = rvec
        """
        transl += self.transl_net(transl)
        body_params['transl'] = transl

        return body_params