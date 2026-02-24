import torch
import numpy as np
import os
from PIL import Image
import flow_vis 
from .geometry import se3_inverse, homogenize_points
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb

def warp_image_with_flow(source_image, source_mask, target_image, flow) -> np.ndarray:
    """
    Warp the target to source image using the given flow vectors.
    Flow vectors indicate the displacement from source to target.

    Args:
    source_image: np.ndarray of shape (H, W, 3), normalized to [0, 1]
    target_image: np.ndarray of shape (H, W, 3), normalized to [0, 1]
    flow: np.ndarray of shape (H, W, 2)
    source_mask: non_occluded mask represented in source image.

    Returns:
    warped_image: target_image warped according to flow into frame of source image
    np.ndarray of shape (H, W, 3), normalized to [0, 1]

    """
    # assert source_image.shape[-1] == 3
    # assert target_image.shape[-1] == 3

    assert flow.shape[-1] == 2

    # Get the shape of the source image
    height, width = source_image.shape[:2]
    target_height, target_width = target_image.shape[:2]

    # Create mesh grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Apply flow displacements
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    x_new = np.clip(x + flow_x, 0, target_width - 1) + 0.5
    y_new = np.clip(y + flow_y, 0, target_height - 1) + 0.5

    x_new = (x_new / target_image.shape[1]) * 2 - 1
    y_new = (y_new / target_image.shape[0]) * 2 - 1

    warped_image = F.grid_sample(
        torch.from_numpy(target_image).permute(2, 0, 1)[None, ...].float(),
        torch.from_numpy(np.stack([x_new, y_new], axis=-1)).float()[None, ...],
        mode="bilinear",
        align_corners=False,
    )

    warped_image = warped_image[0].permute(1, 2, 0).numpy()

    if source_mask is not None:
        warped_image = warped_image * (source_mask > 0.5)[..., None]

    return warped_image

def ndc_to_pixel_coords(coords_ndc: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Convert coordinates from NDC space back to pixel space.
    
    Args:
        coords_ndc: [..., H, W, 2], coordinates in NDC space (x_ndc, y_ndc)
        H, W: image dimensions
        
    Returns:
        coords_px: [..., H, W, 2], coordinates in pixel space (x_pix, y_pix)
    """
    coords_px = coords_ndc.clone()
    
    # Convert x: NDC [1, -1] -> pixel [0, W-1]
    coords_px[..., 0] = (1.0 - coords_ndc[..., 0]) * max(W - 1, 1) / 2.0
    
    # Convert y: NDC [1, -1] -> pixel [0, H-1]
    coords_px[..., 1] = (1.0 - coords_ndc[..., 1]) * max(H - 1, 1) / 2.0
    
    return coords_px

def coords_to_flow(coords: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Convert coordinates to flow by subtracting source pixel coordinates.
    
    Args:
        coords: [..., H, W, 2], target coordinates (where pixels from source appear)
        H, W: image dimensions
        
    Returns:
        flow: [..., H, W, 2], optical flow (displacement vectors)
    """
    device = coords.device
    
    # Create source coordinate grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    source_coords = torch.stack([grid_x, grid_y], dim=-1).float()  # (H, W, 2)
    
    # Compute flow as target - source
    flow = coords - source_coords
    
    return flow

def flow_to_coords(flow: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Convert optical flow to absolute target coordinates.
    
    Args:
        flow: [..., H, W, 2], optical flow (displacement vectors)
        H, W: image dimensions
        
    Returns:
        coords: [..., H, W, 2], absolute target coordinates (pixel positions in target image)
    """
    device = flow.device

    # Create source coordinate grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    source_coords = torch.stack([grid_x, grid_y], dim=-1).float()  # (H, W, 2)

    # Compute absolute target coordinates
    coords = flow + source_coords

    return coords

def ndc_pixels_to_flow(flow_ndc: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Convert optical flow from NDC space back to pixel space.

    Args:
        flow_ndc: [..., H, W, 2], optical flow in NDC (dx_ndc, dy_ndc),
                PyTorch3D NDC convention: +x left, +y up, origin at image center.
        H, W: image height and width.

    Returns:
        flow_px:  [..., H, W, 2], optical flow in pixel space (dx_pix, dy_pix),
                screen convention: +x right, +y down, origin at top-left.
    """
    # Inverse of: dx_ndc = -2/(W-1)*dx_pix, dy_ndc = -2/(H-1)*dy_pix
    sx = 2.0 / max(W - 1, 1)
    sy = 2.0 / max(H - 1, 1)

    flow_px = flow_ndc.clone()
    flow_px[..., 0] = - flow_ndc[..., 0] / sx   # dx_pix
    flow_px[..., 1] = - flow_ndc[..., 1] / sy   # dy_pix
    return flow_px

def coords_pixels_to_ndc(coords_px: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    PyTorch3D convention:
    - NDC space: x ∈ [1, -1] (+x left), y ∈ [-1, 1] (+y up), origin at center
    - Pixel space: x ∈ [0, W-1] (+x right), y ∈ [0, H-1] (+y down), origin at top-left
    """
    coords_ndc = coords_px.clone()
    
    # Convert x: pixel [0, W-1] (left→right) -> NDC [1, -1] (left→right in NDC means 1→-1)
    coords_ndc[..., 0] = 1.0 - (coords_px[..., 0] / max(W - 1, 1)) * 2.0
    
    # Convert y: pixel [0, H-1] (top→bottom) -> NDC [1, -1] (top→bottom in NDC means 1→-1)
    coords_ndc[..., 1] = 1.0 - (coords_px[..., 1] / max(H - 1, 1)) * 2.0
    
    return coords_ndc


def batched_pi3_motion_flow(world_points, camera_poses, camera_intrinsics, sampled_pairs, image_size):
    """
    Compute batched motion flow from img1 to img2 using world points and camera pose encodings.
    
    Args:
        world_points: (B, N, H, W, 3) predicted world points per image.
        camera_poses: (B, N, 4, 4) extrinsics for each frame, camera-to-world.
        camera_intrinsics: (B, N, 3, 3) camera intrinsics for each frame.
        sampled_pairs: (B, P, 2) image pairs to compute flow between.
        image_size: int, image height/width.
        
    Returns:
        flow: (B, P, H, W, 2) motion flows, (x, y) in pixel coordinates
    """
    B, N, H, W, _ = world_points.shape
    P = sampled_pairs.shape[1]
    device = world_points.device

    # Gather source points
    # (B, P)
    src_idx = sampled_pairs[..., 0]
    # (B, P, 1, 1, 1) -> (B, P, H, W, 3)
    # Expand indices to gather along N dimension
    src_idx_exp = src_idx.view(B, P, 1, 1, 1).expand(B, P, H, W, 3)
    src_points = torch.gather(world_points, 1, src_idx_exp)

    # Gather target poses and intrinsics
    # (B, P)
    tgt_idx = sampled_pairs[..., 1]
    
    tgt_poses = torch.gather(camera_poses, 1, tgt_idx.view(B, P, 1, 1).expand(B, P, 4, 4))
    tgt_intrinsics = torch.gather(camera_intrinsics, 1, tgt_idx.view(B, P, 1, 1).expand(B, P, 3, 3))

    # Transform points to target camera frame
    w2c_tgt = se3_inverse(tgt_poses)
    src_points_homo = homogenize_points(src_points)
    
    # P_cam = T_w2c @ P_world
    # (B, P, 4, 4) @ (B, P, H, W, 4) -> (B, P, H, W, 4)
    pts_cam = torch.einsum('bpij,bphwj->bphwi', w2c_tgt, src_points_homo)[..., :3]

    # Project to image plane
    # P_img = K @ P_cam
    # (B, P, 3, 3) @ (B, P, H, W, 3) -> (B, P, H, W, 3)
    pts_img = torch.einsum('bpij,bphwj->bphwi', tgt_intrinsics, pts_cam)
    
    # Normalize to pixels
    uv_tgt = pts_img[..., :2] / (pts_img[..., 2:3] + 1e-6)

    # Generate source pixel coordinates
    # print("image_size is: ", image_size)
    H_img, W_img = image_size[0]
        
    scale_h = H_img / H
    scale_w = W_img / W
    
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Map grid to image coordinates (assuming center of pixels/patches)
    uv_src = torch.stack([
        (x + 0.5) * scale_w - 0.5,
        (y + 0.5) * scale_h - 0.5
    ], dim=-1) # (H, W, 2)
    
    uv_src = uv_src.view(1, 1, H, W, 2).expand(B, P, -1, -1, -1)

    return uv_tgt - uv_src
    

def visualize_flow(pred_motion_coords, motion_coords, covis_masks, sampled_pairs, images, pred_pi3_flow, iteration, accelerator, dataset_names):
    # visualize gt images, gt flow, pred flow, flow computed from predicted cameras and points
    path = f"/ocean/projects/cis250013p/zcong/pi3/outputs/flow_vis/{iteration}"
    if not os.path.exists(path):
        os.makedirs(path)
    
    with torch.no_grad():
        # Get dimensions
        B, num_pairs = sampled_pairs.shape[0], sampled_pairs.shape[1]
        H, W = motion_coords[0, 0].shape[0], motion_coords[0, 0].shape[1]
        
        # Process all pairs for all batches
        for batch_idx in range(B):
            dataset_name = dataset_names[batch_idx]
            for pair_idx in range(num_pairs):
                if pair_idx > 1: break
                # Get pair indices
                pairs = sampled_pairs[batch_idx, pair_idx].cpu().numpy()  # (2,)
                img1 = images[batch_idx, pairs[0]].cpu().numpy()
                img2 = images[batch_idx, pairs[1]].cpu().numpy()
                
                # Convert ground truth coordinates to flow
                gt_coords_ndc = motion_coords[batch_idx, pair_idx]  # NDC coordinates
                gt_coords_pixel = ndc_to_pixel_coords(gt_coords_ndc, H, W)  # Convert to pixel coordinates
                flow_tensor = coords_to_flow(gt_coords_pixel, H, W).float().cpu()  # (H, W, 2)
                flow = flow_tensor.numpy()  # (H, W, 2)
                
                covis_mask = covis_masks[batch_idx, pair_idx].float().cpu().numpy()  # (H, W)
                masked_flow = flow * covis_mask[..., None]
                
                # Convert predicted coordinates to flow
                pred_coords_ndc = pred_motion_coords[batch_idx, pair_idx]  # NDC coordinates
                pred_coords_pixel = ndc_to_pixel_coords(pred_coords_ndc, H, W)  # Convert to pixel coordinates
                pred_flow = coords_to_flow(pred_coords_pixel, H, W).float().cpu().numpy()  # (H, W, 2)
                masked_pred_flow = pred_flow * covis_mask[..., None]

                pi3_flow = pred_pi3_flow[batch_idx, pair_idx].float().cpu().numpy()  # (H, W, 2)
                masked_pi3_flow = pi3_flow * covis_mask[..., None]

                # warp img1 to img2
                # first compute gt warpping
                img1_np = np.transpose(img1, (1, 2, 0))  # [H, W, 3]
                img2_np = np.transpose(img2, (1, 2, 0))  # [H, W, 3]
                warped_img_gt = warp_image_with_flow(img1_np, covis_mask, img2_np, flow)
                warped_img_gt = warped_img_gt.clip(0, 1)
                warped_img_gt = Image.fromarray((warped_img_gt * 255).astype(np.uint8))
                # compute prediction warping
                warped_img_pred = warp_image_with_flow(img1_np, covis_mask, img2_np, pred_flow)
                warped_img_pred = warped_img_pred.clip(0, 1)
                warped_img_pred = Image.fromarray((warped_img_pred * 255).astype(np.uint8))
                # compute pi3 warping
                warped_img_pi3 = warp_image_with_flow(img1_np, covis_mask, img2_np, pi3_flow)
                warped_img_pi3 = warped_img_pi3.clip(0, 1)
                warped_img_pi3 = Image.fromarray((warped_img_pi3 * 255).astype(np.uint8))
                
                # visualize images
                img_array1 = np.transpose(img1, (1, 2, 0))
                img1_pil = Image.fromarray((img_array1 * 255).astype(np.uint8))
                img_array2 = np.transpose(img2, (1, 2, 0))
                img2_pil = Image.fromarray((img_array2 * 255).astype(np.uint8))

                # Calculate AEPE metrics
                # Only calculate on valid covisible pixels
                valid_mask = covis_mask > 0
                if np.sum(valid_mask) > 0:
                    # AEPE for predicted flow vs GT flow
                    flow_diff_pred = np.sqrt(np.sum((masked_pred_flow - masked_flow) ** 2, axis=-1))
                    aepe_pred = np.mean(flow_diff_pred[valid_mask])
                    aepe_5px_pred = np.mean(flow_diff_pred[valid_mask] < 5.0) * 100  # percentage
                    
                    # AEPE for pi3 flow vs GT flow
                    flow_diff_pi3 = np.sqrt(np.sum((masked_pi3_flow - masked_flow) ** 2, axis=-1))
                    aepe_pi3 = np.mean(flow_diff_pi3[valid_mask])
                    aepe_5px_pi3 = np.mean(flow_diff_pi3[valid_mask] < 5.0) * 100  # percentage
                else:
                    aepe_pred = float('inf')
                    aepe_5px_pred = 0.0
                    aepe_pi3 = float('inf')
                    aepe_5px_pi3 = 0.0

                # visualize flow
                flow_vis_image_gt = flow_vis.flow_to_color(masked_flow)
                flow_pil = Image.fromarray(flow_vis_image_gt.astype(np.uint8))
                flow_vis_image_pred = flow_vis.flow_to_color(masked_pred_flow)
                flow_pred_pil = Image.fromarray(flow_vis_image_pred.astype(np.uint8))
                flow_vis_image_pi3 = flow_vis.flow_to_color(masked_pi3_flow)
                flow_pi3_pil = Image.fromarray(flow_vis_image_pi3.astype(np.uint8))

                # Create metrics text
                metrics_text = {
                    'pred_aepe': aepe_pred,
                    'pred_5px_pct': aepe_5px_pred,
                    'pi3_aepe': aepe_pi3,
                    'pi3_5px_pct': aepe_5px_pi3,
                    'covis_ratio': float(np.mean(covis_mask)) * 100,
                    'pairs': pairs,
                    'dataset': dataset_name,
                }

                # Save individual visualization and log to wandb
                save_path = os.path.join(path, f"motion_flow_grid_batch_{batch_idx}_pair_{pair_idx}_imgs_{pairs[0]}_{pairs[1]}_iter_{iteration:08d}.png")
                visualize_motion_grid_nodepth_with_metrics(
                    img1_pil, img2_pil, flow_pil, flow_pred_pil, flow_pi3_pil, 
                    warped_img_gt, warped_img_pred, warped_img_pi3,
                    metrics_text,
                    save_path=save_path,
                    pair_idx = pair_idx,
                    step=iteration,
                    log_to_wandb=True,  # We'll handle wandb logging separately
                    accelerator=accelerator,
                    dataset_name=dataset_name
                )

def visualize_motion_grid_nodepth_with_metrics(img1, img2, flow_pil, flow_pred_pil, flow_pi3_pil, warped_img_gt, warped_img_pred, warped_img_pi3, metrics_text, pair_idx, save_path="motion_flow_grid.png", step=None, log_to_wandb=True, accelerator=None, dataset_name=None):
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    # images
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title(f"Image {metrics_text['pairs'][0]}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img2)
    axes[0, 1].set_title(f"Image {metrics_text['pairs'][1]}")
    axes[0, 1].axis("off")
    
    # Add overall metrics in the third subplot
    axes[0, 2].text(0.1, 0.9, f"{metrics_text['dataset']} Pair: {metrics_text['pairs'][0]} → {metrics_text['pairs'][1]}", 
                    fontsize=14, fontweight='bold', transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.8, f"Covis Ratio: {metrics_text['covis_ratio']:.1f}%", 
                    fontsize=12, transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.7, "Pred Flow Metrics:", 
                    fontsize=12, fontweight='bold', color='blue', transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.6, f"AEPE: {metrics_text['pred_aepe']:.3f}", 
                    fontsize=11, color='blue', transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.5, f"<5px: {metrics_text['pred_5px_pct']:.1f}%", 
                    fontsize=11, color='blue', transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.4, "Pi3 Flow Metrics:", 
                    fontsize=12, fontweight='bold', color='red', transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.3, f"AEPE: {metrics_text['pi3_aepe']:.3f}", 
                    fontsize=11, color='red', transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.2, f"<5px: {metrics_text['pi3_5px_pct']:.1f}%", 
                    fontsize=11, color='red', transform=axes[0, 2].transAxes)
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].axis("off")

    # GT flow and Pred flow
    axes[1, 0].imshow(flow_pil)
    axes[1, 0].set_title("GT Motion Flow")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(flow_pred_pil)
    axes[1, 1].set_title(f"Predicted Flow\nAEPE: {metrics_text['pred_aepe']:.3f}, <5px: {metrics_text['pred_5px_pct']:.1f}%")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(flow_pi3_pil)
    axes[1, 2].set_title(f"Pi3 Flow\nAEPE: {metrics_text['pi3_aepe']:.3f}, <5px: {metrics_text['pi3_5px_pct']:.1f}%")
    axes[1, 2].axis("off")

    # GT warp and Pred warp
    axes[2, 0].imshow(warped_img_gt)
    axes[2, 0].set_title("GT Warped Image")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(warped_img_pred)
    axes[2, 1].set_title("Pred Warped Image")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(warped_img_pi3)
    axes[2, 2].set_title("PI3 Warped Image")
    axes[2, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if log_to_wandb:
        accelerator.log({f"Visualization_{pair_idx}": wandb.Image(save_path)}, step=step)
    plt.close()

def calculate_flow_metrics(pred_motion_coords, motion_coords, covis_masks, sampled_pairs, pred_pi3_flow):
    with torch.no_grad():
        # Get dimensions
        B, num_pairs = sampled_pairs.shape[0], sampled_pairs.shape[1]
        H, W = motion_coords[0, 0].shape[0], motion_coords[0, 0].shape[1]
        aepe_pred, aepe_5px_pred, aepe_pi3, aepe_5px_pi3 = [], [], [], []
        
        # Process all pairs for all batches
        for batch_idx in range(B):
            for pair_idx in range(num_pairs):
                # Convert ground truth coordinates to flow
                gt_coords_ndc = motion_coords[batch_idx, pair_idx]  # NDC coordinates
                gt_coords_pixel = ndc_to_pixel_coords(gt_coords_ndc, H, W)  # Convert to pixel coordinates
                flow_tensor = coords_to_flow(gt_coords_pixel, H, W).float().cpu()  # (H, W, 2)
                flow = flow_tensor.numpy()  # (H, W, 2)
                
                covis_mask = covis_masks[batch_idx, pair_idx].float().cpu().numpy()  # (H, W)
                masked_flow = flow * covis_mask[..., None]
                
                # Convert predicted coordinates to flow
                pred_coords_ndc = pred_motion_coords[batch_idx, pair_idx]  # NDC coordinates
                pred_coords_pixel = ndc_to_pixel_coords(pred_coords_ndc, H, W)  # Convert to pixel coordinates
                pred_flow = coords_to_flow(pred_coords_pixel, H, W).float().cpu().numpy()  # (H, W, 2)
                masked_pred_flow = pred_flow * covis_mask[..., None]

                pi3_flow = pred_pi3_flow[batch_idx, pair_idx].float().cpu().numpy()  # (H, W, 2)
                masked_pi3_flow = pi3_flow * covis_mask[..., None]
                
                # Calculate AEPE metrics
                # Only calculate on valid covisible pixels
                valid_mask = covis_mask > 0
                if np.sum(valid_mask) > 0:
                    # AEPE for predicted flow vs GT flow
                    flow_diff_pred = np.sqrt(np.sum((masked_pred_flow - masked_flow) ** 2, axis=-1))
                    aepe_pred.append(np.mean(flow_diff_pred[valid_mask]))
                    aepe_5px_pred.append(np.mean(flow_diff_pred[valid_mask] < 5.0) * 100)  # percentage
                    
                    # AEPE for pi3 flow vs GT flow
                    flow_diff_pi3 = np.sqrt(np.sum((masked_pi3_flow - masked_flow) ** 2, axis=-1))
                    aepe_pi3.append(np.mean(flow_diff_pi3[valid_mask]))
                    aepe_5px_pi3.append(np.mean(flow_diff_pi3[valid_mask] < 5.0) * 100)  # percentage
                else:
                    aepe_pred.append(float('inf'))
                    aepe_5px_pred.append(0.0)
                    aepe_pi3.append(float('inf'))
                    aepe_5px_pi3.append(0.0)

        # print("aepe 5px pi3 is",aepe_5px_pi3)
        return np.mean(aepe_pred), np.mean(aepe_5px_pred), np.mean(aepe_pi3), np.mean(aepe_5px_pi3)