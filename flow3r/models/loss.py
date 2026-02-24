import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
import math

from ..utils.geometry import homogenize_points, se3_inverse, depth_edge
from ..utils.alignment import align_points_scale
from ..utils.flow_utils import batched_pi3_motion_flow, visualize_flow, calculate_flow_metrics
from datasets import __HIGH_QUALITY_DATASETS__, __MIDDLE_QUALITY_DATASETS__

import os
import trimesh
from trimesh.viewer.notebook import scene_to_html
import numpy as np
# ---------------------------------------------------------------------------
# Some functions from MoGe
# ---------------------------------------------------------------------------

def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)

def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))

# ---------------------------------------------------------------------------
# PointLoss: Scale-invariant Local Pointmap
# ---------------------------------------------------------------------------

class PointLoss(nn.Module):
    def __init__(self, local_align_res=4096, train_conf=False, expected_dist_thresh=0.02):
        super().__init__()
        self.local_align_res = local_align_res
        self.criteria_local = nn.L1Loss(reduction='none')

        self.train_conf = train_conf
        if self.train_conf:
            self.prepare_segformer()
            self.conf_loss_fn = torch.nn.BCEWithLogitsLoss()
            self.expected_dist_thresh = expected_dist_thresh

    def prepare_segformer(self):
        from pi3.models.segformer.model import EncoderDecoder
        self.segformer = EncoderDecoder()
        self.segformer.load_state_dict(torch.load('ckpts/segformer.b0.512x512.ade.160k.pth', map_location=torch.device('cpu'), weights_only=False)['state_dict'])
        self.segformer = self.segformer.cuda()

    def predict_sky_mask(self, imgs):
        with torch.no_grad():
            output = self.segformer.inference_(imgs)
            output = output == 2
        return output

    def prepare_ROE(self, pts, mask, target_size=4096):
        B, N, H, W, C = pts.shape
        output = []
        
        for i in range(B):
            valid_pts = pts[i][mask[i]]

            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
                # NOTE: Is is important to use nearest interpolate. Linear interpolate will lead to unstable result!
                valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
            else:
                valid_pts = torch.ones((target_size, C), device=valid_pts.device)

            output.append(valid_pts)

        return torch.stack(output, dim=0)
    
    def noraml_loss(self, points, gt_points, mask):
        not_edge = ~depth_edge(gt_points[..., 2], rtol=0.03)
        mask = torch.logical_and(mask, not_edge)

        leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
        upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
        leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
        downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
        rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

        gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
        gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
        gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
        gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
        gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

        mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

        loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

        loss = loss.mean() / (4 * max(points.shape[-3:-1]))

        return loss

    def forward(self, pred, gt):
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        details = dict()
        final_loss = 0.0

        B, N, H, W, _ = pred_local_pts.shape

        weights_ = gt_local_pts[..., 2]
        weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, valid_masks, dim=(-2, -1), keepdim=True))
        weights_ = 1 / (weights_ + 1e-6)

        # alignment
        with torch.no_grad():
            xyz_pred_local = self.prepare_ROE(pred_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_gt_local = self.prepare_ROE(gt_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_weights_local = self.prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()[:, :, 0]

            S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            S_opt_local[S_opt_local <= 0] *= -1

        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts

        # local point loss
        local_pts_loss = self.criteria_local(aligned_local_pts[valid_masks].float(), gt_local_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

        # conf loss
        if self.train_conf:
            pred_conf = pred['conf']

            # probability loss
            valid = local_pts_loss.detach().mean(-1, keepdims=True) < self.expected_dist_thresh
            local_conf_loss = self.conf_loss_fn(pred_conf[valid_masks], valid.float())

            sky_mask = self.predict_sky_mask(gt['imgs'].reshape(B*N, 3, H, W)).reshape(B, N, H, W)
            sky_mask[valid_masks] = False
            if sky_mask.sum() == 0:
                sky_mask_loss = 0.0 * aligned_local_pts.mean()
            else:
                sky_mask_loss = self.conf_loss_fn(pred_conf[sky_mask], torch.zeros_like(pred_conf[sky_mask]))
            
            final_loss += 0.05 * (local_conf_loss + sky_mask_loss)
            details['local_conf_loss'] = (local_conf_loss + sky_mask_loss)

        final_loss += local_pts_loss.mean()
        details['local_pts_loss'] = local_pts_loss.mean()

        # normal loss
        normal_batch_id = [i for i in range(len(gt['dataset_names'])) if gt['dataset_names'][i] in __HIGH_QUALITY_DATASETS__ + __MIDDLE_QUALITY_DATASETS__]
        if len(normal_batch_id) == 0:
            normal_loss =  0.0 * aligned_local_pts.mean()
        else:
            normal_loss = self.noraml_loss(aligned_local_pts[normal_batch_id], gt_local_pts[normal_batch_id], valid_masks[normal_batch_id])
            final_loss += normal_loss.mean()
        details['normal_loss'] = normal_loss.mean()

        # [Optional] Global Point Loss
        if 'global_points' in pred and pred['global_points'] is not None:
            gt_pts = gt['global_points']

            pred_global_pts = pred['global_points'] * S_opt_local.view(B, 1, 1, 1, 1)
            global_pts_loss = self.criteria_local(pred_global_pts[valid_masks].float(), gt_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

            final_loss += global_pts_loss.mean()
            details['global_pts_loss'] = global_pts_loss.mean()

        return final_loss, details, S_opt_local

class PointLosswMask(nn.Module):
    def __init__(self, local_align_res=4096, train_conf=False, expected_dist_thresh=0.02):
        super().__init__()
        self.local_align_res = local_align_res
        self.criteria_local = nn.L1Loss(reduction='none')
        self.flow_only_datasets = ["spatialvid", "epickitchens"] 

        self.train_conf = train_conf
        if self.train_conf:
            self.prepare_segformer()
            self.conf_loss_fn = torch.nn.BCEWithLogitsLoss()
            self.expected_dist_thresh = expected_dist_thresh

    def prepare_segformer(self):
        from pi3.models.segformer.model import EncoderDecoder
        self.segformer = EncoderDecoder()
        self.segformer.load_state_dict(torch.load('ckpts/segformer.b0.512x512.ade.160k.pth', map_location=torch.device('cpu'), weights_only=False)['state_dict'])
        self.segformer = self.segformer.cuda()

    def predict_sky_mask(self, imgs):
        with torch.no_grad():
            output = self.segformer.inference_(imgs)
            output = output == 2
        return output

    def prepare_ROE(self, pts, mask, target_size=4096):
        B, N, H, W, C = pts.shape
        output = []
        
        for i in range(B):
            valid_pts = pts[i][mask[i]]

            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
                # NOTE: Is is important to use nearest interpolate. Linear interpolate will lead to unstable result!
                valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
            else:
                valid_pts = torch.ones((target_size, C), device=valid_pts.device)

            output.append(valid_pts)

        return torch.stack(output, dim=0)
    
    def noraml_loss(self, points, gt_points, mask):
        not_edge = ~depth_edge(gt_points[..., 2], rtol=0.03)
        mask = torch.logical_and(mask, not_edge)

        leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
        upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
        leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
        downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
        rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

        gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
        gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
        gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
        gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
        gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

        mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

        loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

        loss = loss.mean() / (4 * max(points.shape[-3:-1]))

        return loss

    def forward(self, pred, gt, dataset_names):
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks'].clone()
        
        for i, name in enumerate(dataset_names):
            if name in self.flow_only_datasets:
                valid_masks[i] = False

        details = dict()
        final_loss = 0.0

        B, N, H, W, _ = pred_local_pts.shape

        weights_ = gt_local_pts[..., 2]
        weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, valid_masks, dim=(-2, -1), keepdim=True))
        weights_ = 1 / (weights_ + 1e-6)

        # alignment
        with torch.no_grad():
            xyz_pred_local = self.prepare_ROE(pred_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_gt_local = self.prepare_ROE(gt_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_weights_local = self.prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()[:, :, 0]

            S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            S_opt_local[S_opt_local <= 0] *= -1

        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts

        # local point loss
        local_pts_loss = self.criteria_local(aligned_local_pts[valid_masks].float(), gt_local_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

        # conf loss
        if self.train_conf:
            pred_conf = pred['conf']

            # probability loss
            valid = local_pts_loss.detach().mean(-1, keepdims=True) < self.expected_dist_thresh
            local_conf_loss = self.conf_loss_fn(pred_conf[valid_masks], valid.float())

            sky_mask = self.predict_sky_mask(gt['imgs'].reshape(B*N, 3, H, W)).reshape(B, N, H, W)
            sky_mask[valid_masks] = False
            
            # also mask sky for flow_only datasets
            for i, name in enumerate(dataset_names):
                if name in self.flow_only_datasets:
                    sky_mask[i] = False

            if sky_mask.sum() == 0:
                sky_mask_loss = 0.0 * aligned_local_pts.mean()
            else:
                sky_mask_loss = self.conf_loss_fn(pred_conf[sky_mask], torch.zeros_like(pred_conf[sky_mask]))
            
            final_loss += 0.05 * (local_conf_loss + sky_mask_loss)
            details['local_conf_loss'] = (local_conf_loss + sky_mask_loss)

        final_loss += local_pts_loss.mean()
        details['local_pts_loss'] = local_pts_loss.mean()

        # normal loss
        normal_batch_id = [i for i in range(len(dataset_names)) 
                           if dataset_names[i] in __HIGH_QUALITY_DATASETS__ + __MIDDLE_QUALITY_DATASETS__ 
                           and dataset_names[i] not in self.flow_only_datasets]
        if len(normal_batch_id) == 0:
            normal_loss =  0.0 * aligned_local_pts.mean()
        else:
            normal_loss = self.noraml_loss(aligned_local_pts[normal_batch_id], gt_local_pts[normal_batch_id], valid_masks[normal_batch_id])
            final_loss += normal_loss.mean()
        details['normal_loss'] = normal_loss.mean()

        # [Optional] Global Point Loss
        if 'global_points' in pred and pred['global_points'] is not None:
            gt_pts = gt['global_points']

            pred_global_pts = pred['global_points'] * S_opt_local.view(B, 1, 1, 1, 1)
            global_pts_loss = self.criteria_local(pred_global_pts[valid_masks].float(), gt_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

            final_loss += global_pts_loss.mean()
            details['global_pts_loss'] = global_pts_loss.mean()

        return final_loss, details, S_opt_local

# ---------------------------------------------------------------------------
# CameraLoss: Affine-invariant Camera Pose
# ---------------------------------------------------------------------------

class CameraLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:  
            R_err: rotation angular error 
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err.mean()         # [0, 3.14]
    
    def forward(self, pred, gt, scale):
        pred_pose = pred['camera_poses']
        gt_pose = gt['camera_poses']

        B, N, _, _ = pred_pose.shape

        pred_pose_align = pred_pose.clone()
        pred_pose_align[..., :3, 3] *=  scale.view(B, 1, 1)
        
        pred_w2c = se3_inverse(pred_pose_align)
        gt_w2c = se3_inverse(gt_pose)
        
        pred_w2c_exp = pred_w2c.unsqueeze(2)
        pred_pose_exp = pred_pose_align.unsqueeze(1)
        
        gt_w2c_exp = gt_w2c.unsqueeze(2)
        gt_pose_exp = gt_pose.unsqueeze(1)
        
        pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)
        gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)

        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)

        t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
        R_pred = pred_rel_all[..., :3, :3][:, mask, ...]
        
        t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
        R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

        trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)
        
        rot_loss = self.rot_ang_loss(
            R_pred.reshape(-1, 3, 3), 
            R_gt.reshape(-1, 3, 3)
        )
        
        total_loss = self.alpha * trans_loss + rot_loss

        return total_loss, dict(trans_loss=trans_loss, rot_loss=rot_loss)

class CameraLosswMask(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha
        self.flow_only_datasets = ["spatialvid", "epickitchens"] 

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:  
            R_err: rotation angular error 
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err.mean()         # [0, 3.14]
    
    def forward(self, pred, gt, scale, dataset_names):
        pred_pose = pred['camera_poses']
        gt_pose = gt['camera_poses']

        B, N, _, _ = pred_pose.shape

        valid_batch_indices = [i for i, name in enumerate(dataset_names) if name not in self.flow_only_datasets]
        
        # Filter valid batches
        pred_pose = pred_pose[valid_batch_indices]
        gt_pose = gt_pose[valid_batch_indices]
        scale = scale[valid_batch_indices]
        
        # Recalculate B for the valid batch size
        B = len(valid_batch_indices)

        pred_pose_align = pred_pose.clone()
        pred_pose_align[..., :3, 3] *=  scale.view(B, 1, 1)
        
        pred_w2c = se3_inverse(pred_pose_align)
        gt_w2c = se3_inverse(gt_pose)
        
        pred_w2c_exp = pred_w2c.unsqueeze(2)
        pred_pose_exp = pred_pose_align.unsqueeze(1)
        
        gt_w2c_exp = gt_w2c.unsqueeze(2)
        gt_pose_exp = gt_pose.unsqueeze(1)
        
        pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)
        gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)

        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)

        t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
        R_pred = pred_rel_all[..., :3, :3][:, mask, ...]
        
        t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
        R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

        trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)
        
        rot_loss = self.rot_ang_loss(
            R_pred.reshape(-1, 3, 3), 
            R_gt.reshape(-1, 3, 3)
        )
        
        total_loss = self.alpha * trans_loss + rot_loss

        return total_loss, dict(trans_loss=trans_loss, rot_loss=rot_loss)

# ---------------------------------------------------------------------------
# FlowLoss: Flow Loss
# ---------------------------------------------------------------------------

class FlowLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def generalized_charbonnier_loss(self,x, alpha=0.5, c=0.24):
        """ Generalized Charbonnier Loss from UFM paper.
        
        Args:
            x: Tensor of residuals, e.g. (pred - gt)
            alpha: shape parameter (default: 0.5)
            c: scale parameter (default: 0.24)
        
        Returns:
            Tensor of loss values (same shape as x)
        """
        abs_diff = x / c
        alpha_minus_2 = alpha - 2.0
        denom = abs(alpha_minus_2)

        loss = (abs_diff ** 2 / denom + 1.0) ** (alpha / 2.0) - 1.0
        loss = (abs(alpha_minus_2) / alpha) * loss
        return loss

    def ufm_flow_loss(self, flow_pred, flow_gt, covis_mask):
        """
        Args:
            flow_pred: [B*S, 2, H, W]
            flow_gt:   [B*S, 2, H, W]
            covis_mask: [B*S, 1, H, W]
        Returns:
            Scalar loss value
        """
        # Flatten everything
        mask = covis_mask.reshape(-1)    # [N]
        flow_pred = flow_pred.permute(0, 2, 3, 1).reshape(-1, 2)  # [N, 2]
        flow_pred = flow_pred[mask > 0.5] 
        flow_gt = flow_gt.permute(0, 2, 3, 1).reshape(-1, 2)    # [N, 2]
        flow_gt = flow_gt[mask > 0.5]

        # Compute L2 per pixel
        diff = flow_pred - flow_gt  # [N, 2]
        l2 = torch.norm(diff, dim=1)  # [N]

        # Charbonnier loss
        # robust_loss = torch.sqrt(l2 ** 2 + epsilon ** 2)  # [N]
        robust_loss = self.generalized_charbonnier_loss(l2, alpha=0.5, c=0.24)

        return robust_loss.mean()
   
    def flow_loss(self, pred_motion_flow, gt_motion_flow, covis_masks, sampled_pairs):
        """
        Compute the motion loss.
        
        Parameters:
        pred_motion_flow (torch.Tensor): Predicted motion flows, shape (B, S, H, W, 2)
        pred_motion_depth (torch.Tensor): Predicted motion depths, shape (B, S, H, W, 1)
        gt_motion_flow (torch.Tensor): Ground truth motion flows, shape (B, S, H, W, 2)
        covis_masks (torch.Tensor): Covisibility masks for motion flows, shape (B, S, H, W)
        depths (torch.Tensor): Depth maps for the images, shape (B, N, H, W)
        
        Returns:
        torch.Tensor: Computed loss
        """
        # Calculate the difference between predicted and ground truth motion flows
        B, S, H, W, _ = pred_motion_flow.shape
        pred_motion_flow = pred_motion_flow.permute(0,1,4,2,3).reshape(B * S, -1, H, W)  # (B*S, 2, H, W)
        gt_motion_flow = gt_motion_flow.permute(0,1,4,2,3).reshape(B * S, -1, H, W)  # (B*S, 2, H, W)
        covis_masks = covis_masks.unsqueeze(2).reshape(B * S, -1, H, W)  # (B*S, 1, H, W)
        y_indices = sampled_pairs[:, :, 1]  # (B, S)
        batch_idx = torch.arange(B).unsqueeze(1).expand(B, S).to(y_indices.device)  # (B, S)
        # compute flow loss
        flow_loss = self.ufm_flow_loss(pred_motion_flow, gt_motion_flow, covis_masks)
        return flow_loss

    def forward(self, pred, motion_coords, covis_masks, all_sampled_pairs):
        pred_motion_flow = pred['flow']
        flow_loss = self.flow_loss(pred_motion_flow, motion_coords, covis_masks, all_sampled_pairs)
        return flow_loss, dict(flow_loss=flow_loss)

# ---------------------------------------------------------------------------
# Final Loss
# ---------------------------------------------------------------------------

class Pi3Loss(nn.Module):
    def __init__(
        self,
        train_conf=False,
    ):
        super().__init__()
        self.point_loss = PointLoss(train_conf=train_conf)
        self.camera_loss = CameraLoss()
        self.flow_loss = FlowLoss()

    def prepare_gt(self, gt):
        gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1)
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1)

        B, N, H, W, _ = gt_pts.shape

        # transform to first frame camera coordinate
        w2c_target = se3_inverse(poses[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        # normalize points
        valid_batch = masks.sum([-1, -2, -3]) > 0
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_pts[valid_batch].clone()
            all_pts[~masks[valid_batch]] = 0
            all_pts = all_pts.reshape(B_, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]
        
        dataset_names = gt[0]['dataset']

        return dict(
            imgs = torch.stack([view['img'] for view in gt], dim=1),
            global_points=gt_pts,
            local_points=gt_local_pts,
            valid_masks=masks,
            camera_poses=poses,
            dataset_names=dataset_names
        )
    
    def normalize_pred(self, pred, gt):
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['valid_masks']

        # normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        local_points  = local_points / norm_factor[..., None, None, None, None]

        if 'global_points' in pred and pred['global_points'] is not None:
            pred['global_points'] /= norm_factor[..., None, None, None, None]

        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized

        return pred

    def forward(self, pred, gt_raw, iters=None, output_dir=None, motion_coords=None, covis_masks=None, all_sampled_pairs=None, flow_only=False, accelerator=None, mode='train'):
        final_loss = 0.0
        details = dict()
        if not flow_only:
            gt = self.prepare_gt(gt_raw)
            pred = self.normalize_pred(pred, gt)

            # Local Point Loss
            point_loss, point_loss_details, scale = self.point_loss(pred, gt)
            final_loss += point_loss
            details.update(point_loss_details)

            # Camera Loss
            camera_loss, camera_loss_details = self.camera_loss(pred, gt, scale)
            final_loss += camera_loss * 0.1
            details.update(camera_loss_details)

        # Flow Loss
        flow_loss, flow_loss_details = self.flow_loss(pred, motion_coords, covis_masks, all_sampled_pairs)
        final_loss += flow_loss * 0.1
        details.update(flow_loss_details)

        if iters is not None:
            if iters % 1000 == 0 and mode == 'train' and accelerator.is_main_process:
                with torch.no_grad():
                    # visualize_points(pred, gt, iters, output_dir)
                    print("Visualize the flow of gt and pred...")
                    intrinsics = torch.stack([view['camera_intrinsics'] for view in gt_raw], dim=1)
                    true_shape = gt_raw[0]['true_shape']
                    images = torch.stack([view['img'] for view in gt_raw], dim=1)
                    pred_pi3_flow = batched_pi3_motion_flow(pred["points"], pred["camera_poses"], intrinsics, all_sampled_pairs, true_shape) # (B, S, H, W, 2)
                    visualize_flow(pred["flow"], motion_coords, covis_masks, all_sampled_pairs, images, pred_pi3_flow, iters, accelerator)
        if mode == 'test':
            # calculate the metrics
            with torch.no_grad():
                intrinsics = torch.stack([view['camera_intrinsics'] for view in gt_raw], dim=1)
                true_shape = gt_raw[0]['true_shape']
                pred_pi3_flow = batched_pi3_motion_flow(pred["points"], pred["camera_poses"], intrinsics, all_sampled_pairs, true_shape) # (B, S, H, W, 2)
                aepe, aepe_5px, aepe_pi3, aepe_5px_pi3 = calculate_flow_metrics(pred["flow"], motion_coords, covis_masks, all_sampled_pairs, pred_pi3_flow)
                details['aepe'] = torch.tensor(aepe)
                details['aepe_5px'] = torch.tensor(aepe_5px)
                details['aepe_pi3'] = torch.tensor(aepe_pi3)
                details['aepe_5px_pi3'] = torch.tensor(aepe_5px_pi3)
        
        return final_loss, details

class Pi3FlowLoss(nn.Module):
    def __init__(
        self,
        train_conf=False,
    ):
        super().__init__()
        self.point_loss = PointLosswMask(train_conf=train_conf)
        self.camera_loss = CameraLosswMask()
        self.flow_loss = FlowLoss()
        # all datasets have flow-supervision (with omniworld mask=0), unsupervised datasets only have flow supervision
        self.flow_only_datasets = ["spatialvid", "epickitchens"] 

    def prepare_gt(self, gt):
        gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1)
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1)

        B, N, H, W, _ = gt_pts.shape

        # transform to first frame camera coordinate
        w2c_target = se3_inverse(poses[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        # normalize points
        valid_batch = masks.sum([-1, -2, -3]) > 0
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_pts[valid_batch].clone()
            all_pts[~masks[valid_batch]] = 0
            all_pts = all_pts.reshape(B_, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]
        
        dataset_names = [gt[i]['dataset'] for i in range(len(gt))]

        return dict(
            imgs = torch.stack([view['img'] for view in gt], dim=1),
            global_points=gt_pts,
            local_points=gt_local_pts,
            valid_masks=masks,
            camera_poses=poses,
            dataset_names=dataset_names
        )
    
    def normalize_pred(self, pred, gt):
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['valid_masks']

        # normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        local_points  = local_points / norm_factor[..., None, None, None, None]

        if 'global_points' in pred and pred['global_points'] is not None:
            pred['global_points'] /= norm_factor[..., None, None, None, None]
        
        if 'points' in pred and pred['points'] is not None:
            pred['points'] /= norm_factor[..., None, None, None, None]

        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized

        return pred

    def forward(self, pred, gt_raw, iters=None, output_dir=None, motion_coords=None, covis_masks=None, all_sampled_pairs=None, flow_only=False, accelerator=None, mode='train'):
        final_loss = 0.0
        details = dict()
        # determine if we skip the point and camera loss
        skip_point_and_camera_loss = True
        dataset_names = gt_raw[0]['dataset']
        for dataset_name in dataset_names:
            if dataset_name not in self.flow_only_datasets:
                skip_point_and_camera_loss = False
                break
        
        gt = self.prepare_gt(gt_raw)
        pred = self.normalize_pred(pred, gt)
        if not flow_only and not skip_point_and_camera_loss:
        # if not flow_only:
            # Local Point Loss
            point_loss, point_loss_details, scale = self.point_loss(pred, gt, dataset_names)
            
            # CHECK POINT LOSS
            if not torch.isfinite(point_loss) or point_loss_details['local_pts_loss'] < 0.0:
                print(f"[Warning] Point Loss is {point_loss.item()}, zeroing it.", flush=True)
                # Use nan_to_num to ensure we get a clean 0.0 even if input has NaNs
                point_loss = torch.nan_to_num(pred['local_points']).sum() * 0.0
                # clean up nan details
                for k, v in point_loss_details.items():
                    if isinstance(v, torch.Tensor) and not torch.isfinite(v):
                        point_loss_details[k] = torch.tensor(0.0, device=v.device, dtype=v.dtype)
            

            final_loss += point_loss
            details.update(point_loss_details)

            # Camera Loss
            camera_loss, camera_loss_details = self.camera_loss(pred, gt, scale, dataset_names)
            
            # CHECK CAMERA LOSS
            if not torch.isfinite(camera_loss):
                print(f"[Warning] Camera Loss is {camera_loss.item()}, zeroing it.", flush=True)
                camera_loss = torch.nan_to_num(pred['camera_poses']).sum() * 0.0
                for k, v in camera_loss_details.items():
                    if isinstance(v, torch.Tensor) and not torch.isfinite(v):
                        camera_loss_details[k] = torch.tensor(0.0, device=v.device, dtype=v.dtype)

            final_loss += camera_loss * 0.1
            details.update(camera_loss_details)

        # Flow Loss
        flow_loss, flow_loss_details = self.flow_loss(pred, motion_coords, covis_masks, all_sampled_pairs)
        
        # CHECK FLOW LOSS
        if not torch.isfinite(flow_loss):
            print(f"[Warning] Flow Loss is {flow_loss.item()}, zeroing it.", flush=True)
            flow_loss = torch.nan_to_num(pred['flow']).sum() * 0.0
            for k, v in flow_loss_details.items():
                if isinstance(v, torch.Tensor) and not torch.isfinite(v):
                    flow_loss_details[k] = torch.tensor(0.0, device=v.device, dtype=v.dtype)

        final_loss += flow_loss * 0.01
        details.update(flow_loss_details)

        # Handle nan/inf loss
        if not torch.isfinite(final_loss):
            print(f"[Error] Final Loss is {final_loss.item()} after summation, skipping step.", flush=True)
            # Dummy loss using multiple outputs
            final_loss = torch.nan_to_num(final_loss) * 0.0
            for k, v in details.items():
                if isinstance(v, torch.Tensor):
                    details[k] = torch.tensor(0.0, device=v.device, dtype=v.dtype)
                else:
                    details[k] = 0.0

        if skip_point_and_camera_loss:
            # print("Skip the point and camera loss for the flow-only datasets.")
            point_loss_details = dict(normal_loss=torch.tensor(0.0, device=final_loss.device,dtype=final_loss.dtype), local_pts_loss=torch.tensor(0.0, device=final_loss.device,dtype=final_loss.dtype))
            camera_loss_details = dict(trans_loss=torch.tensor(0.0, device=final_loss.device,dtype=final_loss.dtype), rot_loss=torch.tensor(0.0, device=final_loss.device,dtype=final_loss.dtype))
            details.update(point_loss_details)
            details.update(camera_loss_details)

        if iters is not None:
            if iters % 1000 == 0 and mode == 'train' and accelerator.is_main_process:
                with torch.no_grad():
                    print("Visualize the points of gt and pred...")
                    visualize_points(pred, gt, iters, output_dir)
                    print("Visualize the flow of gt and pred...")
                    intrinsics = torch.stack([view['camera_intrinsics'] for view in gt_raw], dim=1)
                    true_shape = gt_raw[0]['true_shape']
                    images = torch.stack([view['img'] for view in gt_raw], dim=1)
                    pred_pi3_flow = batched_pi3_motion_flow(pred["points"], pred["camera_poses"], intrinsics, all_sampled_pairs, true_shape) # (B, S, H, W, 2)
                    visualize_flow(pred["flow"], motion_coords, covis_masks, all_sampled_pairs, images, pred_pi3_flow, iters, accelerator, dataset_names)
        if mode == 'test':
            # calculate the metrics
            with torch.no_grad():
                intrinsics = torch.stack([view['camera_intrinsics'] for view in gt_raw], dim=1)
                true_shape = gt_raw[0]['true_shape']
                pred_pi3_flow = batched_pi3_motion_flow(pred["points"], pred["camera_poses"], intrinsics, all_sampled_pairs, true_shape) # (B, S, H, W, 2)
                aepe, aepe_5px, aepe_pi3, aepe_5px_pi3 = calculate_flow_metrics(pred["flow"], motion_coords, covis_masks, all_sampled_pairs, pred_pi3_flow)
                details['aepe'] = torch.tensor(aepe)
                details['aepe_5px'] = torch.tensor(aepe_5px)
                details['aepe_pi3'] = torch.tensor(aepe_pi3)
                details['aepe_5px_pi3'] = torch.tensor(aepe_5px_pi3)
        
        return final_loss, details


def visualize_points(pred, gt, iteration, output_dir):
    # get points and masks
    global_points_gt = gt['global_points']
    global_points_pred = pred['points']
    local_points_gt = gt['local_points']
    local_points_pred = pred['local_points']
    # print("local_points_pred.shape: ", local_points_pred.shape)
    # camera_poses_pred = pred['camera_poses']
    # print("camera_poses_pred.shape: ", camera_poses_pred.shape)
    valid_masks = gt['valid_masks']
    images = gt['imgs']
    dataset_name = gt['dataset_names']

    # get corresponding points and masks
    valid_mask = torch.argwhere(valid_masks[0].reshape(-1) > 0.5)[:, 0]
    global_points_gt = global_points_gt[0].reshape(-1, 3)[valid_mask]
    global_points_pred = global_points_pred[0].reshape(-1, 3)[valid_mask]
    local_points_gt = local_points_gt[0].reshape(-1, 3)[valid_mask]
    local_points_pred = local_points_pred[0].reshape(-1, 3)[valid_mask]

    # visualize
    out_dir_global = os.path.join(output_dir, "eval_global_points", f"{iteration}_{dataset_name[0]}")
    # out_dir_local = os.path.join(output_dir, "eval_local_points", f"{iteration}_{dataset_name[0]}")
    os.makedirs(out_dir_global, exist_ok=True)
    # os.makedirs(out_dir_local, exist_ok=True)
    with torch.no_grad():
        # visualize GT
        gt_global = visualize_trimesh_html(images[0], global_points_gt.cpu().numpy(), valid_mask.cpu())
        with open(os.path.join(out_dir_global,f"global_points_gt.html"), 'w') as html_file:
            html_file.write(gt_global)
        # glbscene2.export(file_obj=os.path.join(out_dir, f"train_gt_{self.iteration:08d}_{seq_name}.glb"))
        # visualize pred
        pred_global = visualize_trimesh_html(images[0], global_points_pred.cpu().numpy(), valid_mask.cpu())
        with open(os.path.join(out_dir_global,f"global_points_pred.html"), 'w') as html_file:
            html_file.write(pred_global)
        # glbscene3.export(file_obj=os.path.join(out_dir, f"train_pred_{self.iteration:08d}_{seq_name}.glb"))
    
        # print("Visualize the local points of gt and pred...")
        # gt_local = visualize_trimesh_html(images[0], local_points_gt.cpu().numpy(), valid_mask.cpu())
        # with open(os.path.join(out_dir_local,f"local_points_gt.html"), 'w') as html_file:
        #     html_file.write(gt_local)
        # pred_local = visualize_trimesh_html(images[0], local_points_pred.cpu().numpy(), valid_mask.cpu())
        # with open(os.path.join(out_dir_local,f"local_points_pred.html"), 'w') as html_file:
        #     html_file.write(pred_local)
    
def visualize_trimesh_html(
    images,
    vertices_3d,
    masks,
) -> trimesh.Scene:

    print("Building pointmap")
    # denormalize images
    # images_denorm = denormalize(images)
    images_denorm = images.cpu().numpy()
    # Handle different image formats - check if images need transposing
    if images_denorm.ndim == 4 and images_denorm.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images_denorm, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images_denorm
    colors_rgb = (colors_rgb.reshape(-1, 3)[masks] * 255).astype(np.uint8)
    # vertices_3d = vertices_3d[masks]


    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)

    scene_3d.add_geometry(point_cloud_data)
    html_str = scene_to_html(scene_3d)
    # html = scene_3d.show(viewer='notebook')
    return html_str
    # print("Pointmap built")
    # return scene_3d

