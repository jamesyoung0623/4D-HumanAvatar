import torch
import torch.nn as nn

class SMPLParamEmbedding(nn.Module):
    '''optimize SMPL params on the fly'''
    def __init__(self, **kwargs) -> None:
        super().__init__()
        # fill in init value
        for k, v in kwargs.items():
            setattr(self, k, nn.Embedding.from_pretrained(v, freeze=False))
        self.keys = ['betas', 'global_orient', 'body_pose', 'transl']

    def forward(self, idx):
        # if idx == 0:
        #     global_orient = torch.cat((self.global_orient(idx), self.global_orient(idx), self.global_orient(idx+1)), dim=0)
        #     body_pose = torch.cat((self.body_pose(idx), self.body_pose(idx), self.body_pose(idx+1)), dim=0)
        #     transl = torch.cat((self.transl(idx), self.transl(idx), self.transl(idx+1)), dim=0)
        # elif idx == self.global_orient.num_embeddings - 1:
        #     global_orient = torch.cat((self.global_orient(idx-1), self.global_orient(idx), self.global_orient(idx)), dim=0)
        #     body_pose = torch.cat((self.body_pose(idx-1), self.body_pose(idx), self.body_pose(idx)), dim=0)
        #     transl = torch.cat((self.transl(idx-1), self.transl(idx), self.transl(idx)), dim=0)
        # else:
        #     global_orient = torch.cat((self.global_orient(idx-1), self.global_orient(idx), self.global_orient(idx+1)), dim=0)
        #     body_pose = torch.cat((self.body_pose(idx-1), self.body_pose(idx), self.body_pose(idx+1)), dim=0)
        #     transl = torch.cat((self.transl(idx-1), self.transl(idx), self.transl(idx+1)), dim=0)

        global_orient = self.global_orient(idx)
        body_pose = self.body_pose(idx)
        transl = self.transl(idx)
        
        return {
            'betas': self.betas(torch.zeros_like(idx)),
            'global_orient': global_orient,
            'body_pose': body_pose,
            'transl': transl
        }

    def tv_loss(self):
        breakpoint()
        global_orient_tv_loss = torch.zeros(()).cuda()
        body_pose_tv_loss = torch.zeros(()).cuda()
        transl_tv_loss = torch.zeros(()).cuda()

        for i in range(len(self.global_orient)-1):
            global_orient_tv_loss += (self.global_orient(i) - self.global_orient(i+1)).square().mean()
            body_pose_tv_loss += 10*(self.body_pose(i) - self.body_pose(i+1)).square().mean()
            transl_tv_loss += (self.transl(i) - self.transl(i+1)).square().mean()

        return global_orient_tv_loss.type(torch.float32), body_pose_tv_loss.type(torch.float32), transl_tv_loss.type(torch.float32)
    