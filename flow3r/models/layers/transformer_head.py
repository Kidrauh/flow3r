from .attention import FlashAttentionRope, FlashCrossAttentionRope
from .block import BlockRope, CrossBlockRope
from ..dinov2.layers import Mlp
import torch
import torch.nn as nn
from functools import partial
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from flow3r.models.flow_head.utils import create_uv_grid, position_grid_to_embed
   
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        need_project=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                # attn_class=MemEffAttentionRope,
                attn_class=FlashAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, xpos=None, return_intermediate=False):
        hidden = self.projects(hidden)
        intermediate = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=xpos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=xpos)
            
            if return_intermediate:
                intermediate.append(hidden)

        out = self.linear_out(hidden)
        
        if return_intermediate:
            return out, intermediate[-4:]
            
        return out

class LinearPts3d (nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=3,):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(dec_embed_dim, (output_dim)*self.patch_size**2)

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape
        # print("--------------------------------")
        # print("pointhead")
        # print("H, W is", H, W)
        # print("hw is", S)
        # print("patch_h is", H//self.patch_size)
        # print("patch_w is", W//self.patch_size)
        # print("--------------------------------")

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # permute + norm depth
        return feat.permute(0, 2, 3, 1)

class LinearFlow2d (nn.Module):
    """ 
    Linear head for flow 2D with MLP fusion of camera and patch features
    Each token outputs: - 16x16 2D flow
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=2, camera_dim=512, num_heads=8, rope=None):
        super().__init__()
        self.patch_size = patch_size
        self.dec_embed_dim = dec_embed_dim
        self.camera_dim = camera_dim

        # Position embedding for camera features (to distinguish first and second camera)
        self.camera_pos_embed = nn.Parameter(torch.randn(2, 1, camera_dim))
        nn.init.normal_(self.camera_pos_embed, std=0.02)

        # Projection to match camera feature dimension to patch feature dimension
        # self.camera_proj = nn.Linear(camera_dim, dec_embed_dim)

        # MLP to fuse camera features and patch features
        self.mlp = nn.Sequential(
            nn.Linear(2*camera_dim + dec_embed_dim, 2*dec_embed_dim),
            nn.ReLU(),
            nn.Linear(2*dec_embed_dim, 2*dec_embed_dim),
            nn.ReLU(),
            nn.Linear(2*dec_embed_dim, dec_embed_dim),
        )

        # Final projection to output dimension
        self.proj = nn.Linear(dec_embed_dim, (output_dim)*self.patch_size**2)

    def forward(self, patch_hidden, camera_hidden, pair_indices, img_shape, B, N):
        """
        Args:
            patch_hidden: (B*N, hw, dec_embed_dim) - motion decoder output
            camera_hidden: (B*N, hw, camera_dim) - camera decoder output
            pair_indices: Tensor of shape (B, S, 2) or list of tuples
                         If Tensor (B, S, 2): indices are (i, j) relative to each batch
                         If list: [(b1, i1, j1), ...] or [(i1, j1), ...]
            img_shape: (H, W)
            B: batch size
            N: sequence length (number of images)
        Returns:
            flow: (total_pairs, H, W, 2)
        """
        H, W = img_shape
        hw = patch_hidden.shape[1]
        
        # Reshape from (B*N, hw, dim) to (B, N, hw, dim)、
        # print("!!!!!now inside the LinearFlow2d forward function")
        patch_hidden = patch_hidden.reshape(B, N, hw, self.dec_embed_dim)
        camera_hidden = camera_hidden.reshape(B, N, hw, self.camera_dim)
        # print("the shape of patch_hidden is", patch_hidden.shape)
        # print("the shape of camera_hidden is", camera_hidden.shape)
        
        # Handle Tensor input (B, S, 2)
        if isinstance(pair_indices, torch.Tensor) and pair_indices.dim() == 3:
            # pair_indices shape: (B, S, 2)
            # We can use advanced indexing for efficiency
            
            # Create batch indices: (B, S)
            S = pair_indices.shape[1]
            batch_idx = torch.arange(B, device=pair_indices.device).unsqueeze(1).expand(B, S)
            
            # Extract indices for i and j images: (B, S)
            idx_i = pair_indices[:, :, 0]
            idx_j = pair_indices[:, :, 1]
            
            # Extract patch features: (B, S, hw, dim)
            patch_feat = patch_hidden[batch_idx, idx_i]
            # print("the shape of patch_feat is", patch_feat.shape)
            
            # Extract camera features: (B, S, hw, dim)
            camera_i = camera_hidden[batch_idx, idx_i]
            camera_j = camera_hidden[batch_idx, idx_j]
            # print("the shape of camera_i is", camera_i.shape)
            # print("the shape of camera_j is", camera_j.shape)
            # Add position encoding
            camera_i = camera_i + self.camera_pos_embed[0]
            camera_j = camera_j + self.camera_pos_embed[1]
            # print("the shape of camera_i after position encoding is", camera_i.shape)
            # print("the shape of camera_j after position encoding is", camera_j.shape)
            # Project camera features
            # camera_i = self.camera_proj(camera_i)
            # camera_j = self.camera_proj(camera_j)
            # print("the shape of camera_i after projection is", camera_i.shape)
            # print("the shape of camera_j after projection is", camera_j.shape)
            # Concatenate camera features and patch features: (B, S, hw, 3*dim)
            concat_features = torch.cat([camera_i, camera_j, patch_feat], dim=-1)

            # Flatten B and S dimensions
            total_pairs = B * S
            input_features = concat_features.reshape(total_pairs, hw, 2*self.camera_dim + self.dec_embed_dim)

        else:
            raise ValueError("Invalid pair_indices type")
            
        # Apply MLP
        fused_features = self.mlp(input_features)
        # print("the shape of fused_features after reshape is", fused_features.shape)
        # Project to output dimension
        feat = self.proj(fused_features)  # (total_pairs, patch_hw, output_dim * patch_size^2)
        # print("the shape of feat is", feat.shape)
        # Reshape and apply pixel shuffle
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        # print("--------------------------------")
        # print("H, W is", H, W)
        # print("hw is", hw)
        # print("patch_h is", patch_h)
        # print("patch_w is", patch_w)
        # print("--------------------------------")
        feat = feat.transpose(-1, -2).reshape(total_pairs, -1, patch_h, patch_w)
        feat = F.pixel_shuffle(feat, self.patch_size)  # (total_pairs, output_dim, H, W)
        # print("the shape of feat after pixel shuffle is", feat.shape)
        
        # Permute to (total_pairs, H, W, output_dim)
        return feat.permute(0, 2, 3, 1).reshape(B, S, H, W, -1)
    
class DPTFlow2d (nn.Module):
    """ 
    Simplified DPT head for flow 2D with only one layer input
    Each token outputs: - 16x16 2D flow
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=2, camera_dim=512, rope=None, features=256):
        super().__init__()
        self.patch_size = patch_size
        self.dec_embed_dim = dec_embed_dim
        self.camera_dim = camera_dim

        # Projection to match camera feature dimension to patch feature dimension
        # self.camera_proj = nn.Linear(camera_dim, dec_embed_dim)

        # MLP to fuse camera features and patch features
        self.mlp = nn.Sequential(
            nn.Linear(2*camera_dim + dec_embed_dim, 2*dec_embed_dim),
            nn.ReLU(),
            nn.Linear(2*dec_embed_dim, 2*dec_embed_dim),
            nn.ReLU(),
            nn.Linear(2*dec_embed_dim, dec_embed_dim),
        )

        self.norm = nn.LayerNorm(dec_embed_dim)

        self.project = nn.Conv2d(dec_embed_dim, features, kernel_size=1, stride=1, padding=0)
        self.refine_low = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(features, features, 3, padding=1),
            nn.GELU(),
        )
        self.refine_high = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(features, features, 3, padding=1),
            nn.GELU(),
        )
        self.out_head = nn.Sequential(
            nn.Conv2d(features, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, output_dim, 1),
        )

        # Final projection to output dimension
        # self.proj = nn.Linear(dec_embed_dim, (output_dim)*self.patch_size**2)

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed


    def forward(self, patch_hidden, camera_hidden, pair_indices, img_shape, B, N):
        """
        Args:
            patch_hidden: (B*N, hw, dec_embed_dim) - motion decoder output
            camera_hidden: (B*N, hw, camera_dim) - camera decoder output
            pair_indices: Tensor of shape (B, S, 2) or list of tuples
                         If Tensor (B, S, 2): indices are (i, j) relative to each batch
                         If list: [(b1, i1, j1), ...] or [(i1, j1), ...]
            img_shape: (H, W)
            B: batch size
            N: sequence length (number of images)
        Returns:
            flow: (total_pairs, H, W, 2)
        """
        H, W = img_shape
        hw = patch_hidden.shape[1]
        
        # Reshape from (B*N, hw, dim) to (B, N, hw, dim)、
        # print("!!!!!now inside the LinearFlow2d forward function")
        patch_hidden = patch_hidden.reshape(B, N, hw, self.dec_embed_dim)
        camera_hidden = camera_hidden.reshape(B, N, hw, self.camera_dim)
        
        # Handle Tensor input (B, S, 2)
        S = pair_indices.shape[1]
        batch_idx = torch.arange(B, device=pair_indices.device).unsqueeze(1).expand(B, S)
        
        # Extract indices for i and j images: (B, S)
        idx_i = pair_indices[:, :, 0]
        idx_j = pair_indices[:, :, 1]
        
        # Extract patch features: (B, S, hw, dim)
        patch_feat = patch_hidden[batch_idx, idx_i]
        # print("the shape of patch_feat is", patch_feat.shape)
        
        # Extract camera features: (B, S, hw, dim)
        camera_i = camera_hidden[batch_idx, idx_i]
        camera_j = camera_hidden[batch_idx, idx_j]
        # Concatenate camera features and patch features: (B, S, hw, 3*dim)
        concat_features = torch.cat([camera_i, camera_j, patch_feat], dim=-1)

        # Flatten B and S dimensions
        total_pairs = B * S
        input_features = concat_features.reshape(total_pairs, hw, 2*self.camera_dim + self.dec_embed_dim)

        # Apply MLP
        fused = self.mlp(input_features)  # (T, hw, dec_embed_dim)

        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        assert hw == patch_h * patch_w, (hw, patch_h, patch_w)
        fused = self.norm(fused)
        feat = fused.transpose(1, 2).reshape(total_pairs, self.dec_embed_dim, patch_h, patch_w)  # (T,D,h,w)

        feat = self.project(feat)              # (T,features,h,w)
        feat = self._apply_pos_embed(feat, W, H)
        feat = self.refine_low(feat)

        feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=True)
        feat = self._apply_pos_embed(feat, W, H)
        feat = self.refine_high(feat)

        flow = self.out_head(feat)  # (T,2,H,W)
        return flow.permute(0, 2, 3, 1).reshape(B, S, H, W, -1)
     
class ContextTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
    ):
        super().__init__()

        self.projects_x = nn.Linear(in_dim, dec_embed_dim)
        self.projects_y = nn.Linear(in_dim, dec_embed_dim)

        self.blocks = nn.ModuleList([
            CrossBlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                # attn_class=MemEffAttentionRope, 
                # cross_attn_class=MemEffCrossAttentionRope,
                attn_class=FlashAttentionRope, 
                cross_attn_class=FlashCrossAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, context, xpos=None, ypos=None):
        hidden = self.projects_x(hidden)
        context = self.projects_y(context)

        for i, blk in enumerate(self.blocks):
            hidden = blk(hidden, context, xpos=xpos, ypos=ypos)

        out = self.linear_out(hidden)

