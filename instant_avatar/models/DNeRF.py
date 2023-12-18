from .structures.body_model_param import SMPLParamEmbedding
from ..deformers.smpl_deformer import SMPLDeformer
from .structures.utils import Rays
from .networks.ngp import NeRFNGPNet
import torch
import numpy as np
import pytorch_lightning as pl
import hydra
import cv2
import os
import glob
import torch.nn.functional as F

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full


from pytorch3d import transforms

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

import logging
logger = logging.getLogger('instant-avatar.DNeRF')
logger.addHandler(logging.FileHandler('DNeRF.log'))

class DNeRFModel(pl.LightningModule):
    def __init__(self, opt, datamodule) -> None:
        super().__init__()
        self.automatic_optimization = False

        self.net_coarse = hydra.utils.instantiate(opt.network)

        if opt.optimize_SMPL.enable:
            self.SMPL_param = SMPLParamEmbedding(**datamodule.trainset.get_SMPL_params())
            # Download and load checkpoints
            download_models(CACHE_DIR_4DHUMANS)
            self.hmr2_model, self.hmr2_model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

            del self.hmr2_model.discriminator
            del self.hmr2_model.keypoint_3d_loss
            del self.hmr2_model.keypoint_2d_loss
            del self.hmr2_model.smpl_parameter_loss

            # Setup HMR2.0 model
            self.hmr2_model = self.hmr2_model.cuda()
            self.hmr2_model.eval()   

            # self.hmr2_model.train()   
            for name, param in self.hmr2_model.named_parameters():
                if name.startswith('backbone'):
                    param.requires_grad = False
                elif 'transformer' in name:
                    param.requires_grad = False
                elif 'deccam' in name:
                    param.requires_grad = False

        self.deformer = hydra.utils.instantiate(opt.deformer)
        self.loss_fn = hydra.utils.instantiate(opt.loss)

        self.renderer = hydra.utils.instantiate(opt.renderer, smpl_init=opt.get('smpl_init', False))
        self.renderer.initialize(len(datamodule.trainset))

        self.datamodule = datamodule
        self.opt = opt
        self.step = 0

        self.PSNR = PeakSignalNoiseRatio().cuda()
        self.SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
        self.LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()

        self.psnr_list = []
        self.ssim_list = []
        self.lpips_list = []


    def configure_optimizers(self):
        # use one optimizer with different learning rate for params
        params, body_model_params, hmr2_params, encoding_params = [], [], [], []
        for (name, param) in self.named_parameters():
            if name.startswith('loss_fn'):
                continue

            if name.startswith('SMPL_param'):
                body_model_params.append(param)
            elif name.startswith('hmr2_model'):
                hmr2_params.append(param)
            elif 'encoder' in name:
                encoding_params.append(param)
            else:
                params.append(param)
        
        optimizer = torch.optim.Adam([
            {'params': encoding_params},
            {'params': params},
            {'params': body_model_params, 'lr': self.opt.optimize_SMPL.get('lr', 5e-4)},
            {'params': hmr2_params, 'lr': self.opt.optimize_HMR2.get('lr', 5e-4)},
        ], **self.opt.optimizer)


        max_epochs = self.opt.scheduler.max_epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - epoch / max_epochs) ** 1.5)

        # additional configure for gradscaler
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1024.0)
        return [optimizer], [scheduler]

    def forward(self, batch, eval_mode=False):
        eval_mode = not self.training
        rays = Rays(o=batch['rays_o'], d=batch['rays_d'], near=batch['near'], far=batch['far'])
        self.deformer.transform_rays_w2s(rays)
        use_noise = self.step < 1000 and not self.opt.optimize_SMPL.get('is_refine', False) and not eval_mode
        return self.renderer(rays, lambda x, _: self.deformer(x, self.net_coarse, eval_mode), eval_mode=eval_mode, noise=1 if use_noise else 0, bg_color=batch.get('bg_color', None))

    @torch.no_grad()
    def render_image_fast(self, batch, img_size):
        if hasattr(self, 'SMPL_param') and self.opt.optimize_SMPL.get('is_refine', False):
            # batch['img_cv2'] = batch['img_cv2'].cpu().numpy().squeeze()
            # batch['bbox'] = batch['bbox'].cpu().numpy()

            # dataset = ViTDetDataset(self.hmr2_model_cfg, batch['img_cv2'], batch['bbox'])
            # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
                
            # hmr2_batch = next(iter(dataloader))
                    
            # hmr2_batch = recursive_to(hmr2_batch, torch.device('cuda'))

            # out = self.hmr2_model(hmr2_batch)

            # pred_cam = out['pred_cam']
            # box_center = hmr2_batch["box_center"].float()
            # box_size = hmr2_batch["box_size"].float()
            # img_size = hmr2_batch["img_size"].float()

            # scaled_focal_length = 2664
            # batch['transl'] = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)

            # hmr2_body_params = out['pred_smpl_params']

            # batch['global_orient'] = transforms.matrix_to_axis_angle(hmr2_body_params['global_orient']).reshape(1, -1)
            # batch['body_pose'] = transforms.matrix_to_axis_angle(hmr2_body_params['body_pose']).reshape(1, -1)

            # update betas if use SMPLDeformer
            if isinstance(self.deformer, SMPLDeformer):
                # batch['betas'] = hmr2_body_params['betas']
                batch['betas'] = self.SMPL_param(batch['idx'])['betas']
            
            batch['global_orient'] = self.SMPL_param(batch['idx'])['global_orient']
            batch['body_pose'] = self.SMPL_param(batch['idx'])['body_pose']
            batch['transl'] = self.SMPL_param(batch['idx'])['transl']

            # update near & far with refined SMPL
            dist = torch.norm(batch['transl'], dim=-1, keepdim=True).detach()
            batch['near'][:] = dist - 1
            batch['far'][:] = dist + 1

        self.deformer.prepare_deformer(batch)
        if hasattr(self.renderer, 'density_grid_test'):
            self.renderer.density_grid_test.initialize(self.deformer, self.net_coarse)

        d = self.forward(batch, eval_mode=True)
        rgb = d['rgb_coarse'].reshape(-1, *img_size, 3)
        depth = d['depth_coarse'].reshape(-1, *img_size)
        alpha = d['alpha_coarse'].reshape(-1, *img_size)
        counter = d['counter_coarse'].reshape(-1, *img_size)
        return rgb, depth, alpha, counter

    def update_density_grid(self):
        N = 1 if self.opt.get('smpl_init', False) else 20
        if self.step % N == 0 and hasattr(self.renderer, 'density_grid_train'):
            density, valid = self.renderer.density_grid_train.update(self.deformer, self.net_coarse, self.step)
            reg = N * density[~valid].mean()
            if self.step < 500:
                reg += 0.5 * density.mean()
            return reg
        else:
            return None

    def training_step(self, batch, *args, **kwargs):
        if hasattr(self, 'SMPL_param'):
            # batch['img_cv2'] = batch['img_cv2'].cpu().numpy().squeeze()
            # batch['bbox'] = batch['bbox'].cpu().numpy()

            # dataset = ViTDetDataset(self.hmr2_model_cfg, batch['img_cv2'], batch['bbox'])
            # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
                
            # hmr2_batch = next(iter(dataloader))
                    
            # hmr2_batch = recursive_to(hmr2_batch, torch.device('cuda'))

            # out = self.hmr2_model(hmr2_batch)

            # pred_cam = out['pred_cam']
            # box_center = hmr2_batch["box_center"].float()
            # box_size = hmr2_batch["box_size"].float()
            # img_size = hmr2_batch["img_size"].float()

            # scaled_focal_length = 2664
            # batch['transl'] = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)

            # hmr2_body_params = out['pred_smpl_params']

            # batch['global_orient'] = transforms.matrix_to_axis_angle(hmr2_body_params['global_orient']).reshape(1, -1)
            # batch['body_pose'] = transforms.matrix_to_axis_angle(hmr2_body_params['body_pose']).reshape(1, -1)

            # update betas if use SMPLDeformer
            if isinstance(self.deformer, SMPLDeformer):
                # batch['betas'] = hmr2_body_params['betas']
                batch['betas'] = self.SMPL_param(batch['idx'])['betas']
            
            batch['global_orient'] = self.SMPL_param(batch['idx'])['global_orient']
            batch['body_pose'] = self.SMPL_param(batch['idx'])['body_pose']
            batch['transl'] = self.SMPL_param(batch['idx'])['transl']

            # update near & far with refined SMPL
            dist = torch.norm(batch['transl'], dim=-1, keepdim=True).detach()
            batch['near'][:] = dist - 1
            batch['far'][:] = dist + 1

        self.renderer.idx = int(batch['idx'][0])
        self.deformer.prepare_deformer(batch)
        reg = self.update_density_grid()
        if isinstance(self.net_coarse, NeRFNGPNet):
            self.net_coarse.initialize(self.deformer.bbox)
        
        predicts = self.forward(batch, eval_mode=False)
        losses = self.loss_fn(predicts, batch)

        # body_pose_loss, transl_loss = self.SMPL_param.tv_loss(batch['idx'])

        # losses['body_pose_loss'] = body_pose_loss
        # losses['transl_loss'] = transl_loss
        # losses['loss'] += body_pose_loss
        # losses['loss'] += transl_loss

        if not (reg is None or self.opt.optimize_SMPL.get('is_refine', False)):
            losses['reg'] = reg
            losses['loss'] += reg

        for k, v in losses.items():
            self.log(f'train/{k}', v)

        self.step += 1

        if self.automatic_optimization:
            return losses['loss']
        else:
            loss = losses['loss']
            optimizer = self.optimizers(False)

            try:
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            except Exception as e:
                logger.warning(e)

    def on_validation_epoch_end(self, *args, **kwargs):
        # self.deformer.initialized = False
        scheduler = self.lr_schedulers()
        scheduler.step()

        psnr = sum(self.psnr_list)/len(self.psnr_list)
        ssim = sum(self.ssim_list)/len(self.ssim_list)
        lpips = sum(self.lpips_list)/len(self.lpips_list)
        
        print('PSNR: {:.2f}, SSIM: {:.4f}, LPIPS: {:.2f}'.format(psnr, ssim, lpips))

        self.log(f"val/PSNR", psnr, on_epoch=True)
        self.log(f"val/SSIM", ssim, on_epoch=True)
        self.log(f"val/LPIPS", lpips, on_epoch=True)
        
        self.psnr_list = []
        self.ssim_list = []
        self.lpips_list = []

    # def on_validation_epoch_start(self, *args, **kwargs):
    #     self.deformer.initialized = False

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        img_size = self.datamodule.valset.image_shape
        rgb, depth, alpha, counter = self.render_image_fast(batch, img_size)

        rgb_gt = batch['rgb'].reshape(-1, *img_size, 3)
        alpha_gt = batch['alpha'].reshape(-1, *img_size)

        losses = {
            # add regular losses
            'rgb_loss': (rgb - rgb_gt).square().mean(),
            'counter_avg': counter.mean(),
            'counter_max': counter.max(),
        }
        for k, v in losses.items():
            self.log(f'val/{k}', v, on_epoch=True)
        
        self.psnr_list.append(self.PSNR(rgb.permute(0, 3, 1, 2), rgb_gt.permute(0, 3, 1, 2)).item())
        self.ssim_list.append(self.SSIM(rgb.permute(0, 3, 1, 2), rgb_gt.permute(0, 3, 1, 2)).item())
        self.lpips_list.append(1000*self.LPIPS(rgb.permute(0, 3, 1, 2), rgb_gt.permute(0, 3, 1, 2)).item())

        # extra visualization for debugging
        os.makedirs('animation/progression/', exist_ok=True)
        # visualize heatmap (blue ~ 0, red ~ 1)
        errmap = (rgb - rgb_gt).square().sum(-1).sqrt().cpu().numpy()[0] / np.sqrt(3)
        errmap = cv2.applyColorMap((errmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        errmap_rgb = torch.from_numpy(errmap).to(rgb.device)[None] / 255

        errmap = (alpha - alpha_gt).abs().cpu().numpy()[0]
        errmap = cv2.applyColorMap((errmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        errmap_alpha = torch.from_numpy(errmap).to(rgb.device)[None] / 255

        img = torch.cat([rgb_gt.squeeze(), rgb.squeeze(), errmap_rgb.squeeze(), errmap_alpha.squeeze()], dim=1)

        # visualize novel pose
        # batch['global_orient'][:] = 0
        batch['body_pose'][:] = 0
        batch['body_pose'][:, 2] = 0.5
        batch['body_pose'][:, 5] = -0.5

        dist = torch.sqrt(torch.square(batch['transl']).sum(-1))
        batch['near'] = torch.ones_like(batch['rays_d'][..., 0]) * (dist - 1)
        batch['far'] = torch.ones_like(batch['rays_d'][..., 0]) * (dist + 1)

        rgb_cano, *_ = self.render_image_fast(batch, img_size)
        rgb_cano = rgb_cano.squeeze()

        for angle in [0.5*np.pi, np.pi, 1.5*np.pi]:
            R = cv2.Rodrigues(np.array([0, angle, 0]))[0]
            R_new = cv2.Rodrigues(batch['global_orient'][:].cpu().numpy())[0]
            R_new = R @ R_new
            R_new = cv2.Rodrigues(R_new)[0].astype(np.float32)
            batch['global_orient'][:] = torch.FloatTensor(R_new).reshape(1, 3).cuda()
            rgb_new, *_ = self.render_image_fast(batch, img_size)
            rgb_cano = torch.cat([rgb_cano, rgb_new.squeeze()], dim=1)

        img = torch.cat([img, rgb_cano], dim=0)
        cv2.imwrite(f'animation/progression/Step_{self.step:06d}_{batch_idx:03d}.png', img.cpu().numpy() * 255)

        return losses

    @torch.no_grad()
    def test_step(self, batch, batch_idx, *args, **kwargs):
        img_size = self.datamodule.testset.image_shape
        rgb, *_ = self.render_image_fast(batch, img_size)
        rgb_gt = batch['rgb'].reshape(-1, *img_size, 3)
        errmap = (rgb - rgb_gt).square().sum(-1).sqrt().cpu().numpy()[0] / np.sqrt(3)
        errmap = cv2.applyColorMap((errmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        errmap = torch.from_numpy(errmap).to(rgb.device)[None] / 255

        if batch_idx == 0:
            os.makedirs('test/', exist_ok=True)

        # save for later evaluation
        img = torch.cat([rgb_gt, rgb, errmap], dim=2)
        cv2.imwrite(f'test/{batch_idx}.png', img.cpu().numpy()[0] * 255)


    ######################
    # DATA RELATED HOOKS #
    ######################
    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
