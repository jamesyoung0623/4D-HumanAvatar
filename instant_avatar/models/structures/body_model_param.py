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
        return {
            'betas': self.betas(torch.zeros_like(idx)),
            'global_orient': self.global_orient(idx),
            'body_pose': self.body_pose(idx),
            'transl': self.transl(idx)
        }

    def tv_loss(self, idx):
        #global_orient_loss = 0
        body_pose_tv_loss = torch.zeros(()).cuda()
        transl_tv_loss = torch.zeros(()).cuda()

        N = len(self.global_orient.weight)
        window_size = 5

        idx_list = torch.linspace(0, N-1, steps=N)
        idx_list = idx_list[idx:idx+window_size].clip(max=N - 1).type(torch.LongTensor).cuda()

        for i in range(len(idx_list)-1):
            #global_orient_loss += (self.global_orient(idx_list[i]) - self.global_orient(idx_list[i+1])).square().mean()
            #body_pose_tv_loss += 10*(self.body_pose(idx_list[i]) - self.body_pose(idx_list[i+1])).square().mean()
            transl_tv_loss += (self.transl(idx_list[i]) - self.transl(idx_list[i+1])).square().mean()

        return body_pose_tv_loss.type(torch.float32), transl_tv_loss.type(torch.float32)
    