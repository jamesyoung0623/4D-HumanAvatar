import torch
import torch.nn as nn
import tinycudann as tcnn
from instant_avatar.utils.network_util import initseq, RodriguesModule

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


class BodyPoseRefiner(nn.Module):
    def __init__(self):
        super(BodyPoseRefiner, self).__init__()
        self.global_orient_coef = nn.Parameter(torch.zeros((3)), requires_grad=True)

        self.body_pose_coef = nn.Parameter(torch.zeros((3)), requires_grad=True)

        self.transl_coef = nn.Parameter(torch.zeros((3)), requires_grad=True)

        # init_val = 1e-5

        # self.global_orient_net[0].weight.data.uniform_(-init_val, init_val)
        # self.global_orient_net[0].bias.data.zero_()
        
        # self.body_pose_net[0].weight.data.uniform_(-init_val, init_val)
        # self.body_pose_net[0].bias.data.zero_()

        # self.transl_net[0].weight.data.uniform_(-init_val, init_val)
        # self.transl_net[0].bias.data.zero_()

        self.rodriguez = RodriguesModule()
    
        
    def forward(self, global_orient, body_pose, transl):
        # print(global_orient)
        global_orient = global_orient[:, 1] + torch.matmul(self.global_orient_coef, global_orient)
        # print(global_orient)
        # print(body_pose)
        body_pose = body_pose[:, 1] + torch.matmul(self.body_pose_coef, body_pose)
        # print(body_pose)
        # print(transl)
        transl = transl[:, 1] + torch.matmul(self.transl_coef, transl)
        # print(transl)
        # breakpoint()
   
        # rvec = torch.cat((global_orient, body_pose), dim=1)
        # delta_rvec = self.rvec_net(rvec)

        # rmat = axis_angle_to_matrix(rvec.view(-1, 3)).view(-1, 24, 3, 3)
        # delta_rmat = axis_angle_to_matrix(delta_rvec.view(-1, 3)).view(-1, 24, 3, 3)
        
        # rmat = torch.matmul(rmat.type(torch.HalfTensor).cuda().reshape(-1, 3, 3), delta_rmat.type(torch.HalfTensor).cuda().reshape(-1, 3, 3))
         
        # for i in range(rmat.shape[0]):
        #     rvec = matrix_to_axis_angle(rmat[i])
        #     if i == 0:
        #         body_params['global_orient'][0] = rvec
        #     else:
        #         body_params['body_pose'][0][3*(i-1):3*i] = rvec

        return global_orient, body_pose, transl