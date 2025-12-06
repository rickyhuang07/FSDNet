"""
FSDNet: Fusion of RPSP path and spacial path for binary classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchvision.models import resnet50, ResNet50_Weights
logger = logging.getLogger(__name__)

class RadialPowerSpectrumPooling(nn.Module):
    """
    RadialPowerSpectrumPooling (RPSP) for extracting global frequency signatures.
    """

    def __init__(self, radii_count: int = 128, max_radius: float = 100.0):
        super().__init__()
        self.radii_count = radii_count
        self.max_radius = max_radius

    @staticmethod
    def _fftshift2d(x: torch.Tensor) -> torch.Tensor:
        """Shift zero-frequency component to the center of the spectrum."""
        b, h, w = x.shape
        x = x.reshape(b, h, w)
        return torch.fft.fftshift(x, dim=(-2, -1))

    def forward(self, power_spectrum: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RPSP.

        Args:
            power_spectrum: 2D power spectrum tensor of shape (B, H, W)

        Returns:
            1D feature vector of shape (B, radii_count)
        """
        device = power_spectrum.device
        batch_size, height, width = power_spectrum.shape

        # Center DC to middle
        centered = self._fftshift2d(power_spectrum)

        # Radius map
        y = torch.arange(height, device=device) - (height // 2)
        x = torch.arange(width, device=device) - (width // 2)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        radius = torch.sqrt(xx.float() ** 2 + yy.float() ** 2)

        # Determine max usable radius
        max_r = min(self.max_radius, float(min(height, width) // 2))

        # Bin edges and indices
        bin_edges = torch.linspace(0.0, max_r, steps=self.radii_count + 1, device=device)
        eps = 1e-6
        clamped_radius = torch.clamp(radius, min=bin_edges[0] + eps, max=bin_edges[-1] - eps)
        bin_indices = torch.bucketize(clamped_radius.reshape(-1), bin_edges, right=False) - 1

        # Accumulate sums per bin
        flat_power = centered.reshape(batch_size, -1)
        features = torch.zeros(batch_size, self.radii_count, device=device)
        features = features.scatter_add(1, bin_indices.unsqueeze(0).expand(batch_size, -1), flat_power)

        # Normalize by number of pixels per bin
        ones = torch.ones_like(clamped_radius).reshape(-1)
        counts = torch.zeros(self.radii_count, device=device)
        counts = counts.scatter_add(0, bin_indices, ones)
        counts = torch.clamp(counts, min=1.0)
        features = features / counts.unsqueeze(0)
        return features

class TwoLayerProj(nn.Module):
    """Two-layer projection net: Linear -> ReLU -> Linear (+ optional LayerNorm)."""
    def __init__(self, in_dim: int, mid_dim: int, out_dim: int, use_ln: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, out_dim),
        )
        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(out_dim)
        else:
            self.ln = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.ln(x)

class FeatureFuseMLP(nn.Module):
    """Two-layer fusion MLP: proj_out -> hidden -> fuse_out (+ optional LayerNorm)."""
    def __init__(self, in_dim: int, mid_dim: int, out_dim: int, use_ln: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, out_dim),
        )
        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(out_dim)
        else:
            self.ln = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.ln(x)

class FSDNet(nn.Module):
    """
    Fuses RPSP features with Spatial features before classification.
    Now projects both paths through 2-layer MLPs, normalizes them, applies
    a learned softmax gate, then fuses with a 2-layer MLP.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- RPSP Path ---
        self.rpsp = RadialPowerSpectrumPooling(
            radii_count=config.model.rpsp_radii_count,
            max_radius=config.model.rpsp_max_radius
        )

        # --- Spatial Path (backbone) ---
        pretrained = getattr(config.model, "pretrained", True)

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.spatial = resnet50(weights=weights)
        self._spatial_feature_dim = 2048
        self.spatial.fc = nn.Identity()


        # --- Fusion / projection hyperparams (config fallbacks provided) ---
        proj_mid = getattr(config.model, "proj_mid_dim", 256)
        proj_out = getattr(config.model, "proj_out_dim", 512)
        fuse_mid = getattr(config.model, "fuse_mid_dim", 1024)
        fuse_out = getattr(config.model, "fuse_out_dim", 512)
        use_ln = getattr(config.model, "use_layernorm_in_proj", True)

        # Two-layer projections
        self.rpsp_proj = TwoLayerProj(in_dim=config.model.rpsp_radii_count,
                                      mid_dim=proj_mid,
                                      out_dim=proj_out,
                                      use_ln=use_ln)
        self.backbone_proj = TwoLayerProj(in_dim=self._spatial_feature_dim,
                                          mid_dim=proj_mid,
                                          out_dim=proj_out,
                                          use_ln=use_ln)

        # Fusion MLP
        self.fuse_mlp = FeatureFuseMLP(in_dim=proj_out*2, mid_dim=fuse_mid, out_dim=fuse_out, use_ln=use_ln)

        # --- Classifier now takes fused features ---
        self.classifier = nn.Sequential(
            nn.Dropout(config.model.dropout_rate),
            nn.Linear(fuse_out, 512),
            nn.ReLU(),
            nn.Dropout(config.model.dropout_rate),
            nn.Linear(512, config.model.num_classes)
        )

    def _rgb_to_ycbcr(self, rgb: torch.Tensor) -> torch.Tensor:
        r, g, b = rgb[:,0:1], rgb[:,1:2], rgb[:,2:3]
        y  = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128.0/255.0 - 0.168736*r - 0.331264*g + 0.5*b
        cr = 128.0/255.0 + 0.5*r - 0.418688*g - 0.081312*b
        return torch.cat([y, cb, cr], dim=1)

    def _apply_fft(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft2(x, dim=(-2,-1))

    def _compute_power_spectrum(self, fft_output: torch.Tensor) -> torch.Tensor:
        return fft_output.real ** 2 + fft_output.imag ** 2

    def forward(self, rgb_image: torch.Tensor) -> torch.Tensor:
        # Denormalize input from ImageNet stats back to [0,1]
        mean = rgb_image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = rgb_image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        rgb_01 = torch.clamp(rgb_image * std + mean, 0.0, 1.0)

        # --- RPSP Path ---
        ycbcr = self._rgb_to_ycbcr(rgb_01)
        fft_ycbcr = self._apply_fft(ycbcr)
        power_spectrum = torch.log1p(self._compute_power_spectrum(fft_ycbcr))
        merged_spectrum = power_spectrum.mean(dim=1)
        rpsp_features = self.rpsp(merged_spectrum)
        # Standardize RPSP (per-sample)
        rpsp_features = (rpsp_features - rpsp_features.mean(dim=1, keepdim=True)) / \
                        (rpsp_features.std(dim=1, keepdim=True) + 1e-6)

        # --- Spatial Path ---
        spatial_features = self.spatial(rgb_image)  # (B, spatial_dim)

        # --- Project both through 2-layer MLPs ---
        p_r = self.rpsp_proj(rpsp_features)          # (B, proj_out)
        p_b = self.backbone_proj(spatial_features)   # (B, proj_out)

        # L2-normalize projections to encourage comparable magnitudes
        eps = 1e-6
        p_r = p_r / (p_r.norm(dim=-1, keepdim=True) + eps)
        p_b = p_b / (p_b.norm(dim=-1, keepdim=True) + eps)

        # Concatenate projections
        fused = torch.cat([p_r, p_b], dim=-1)  # (B, proj_out * 2)

        # Final fuse MLP
        fused = self.fuse_mlp(fused)  # (B, fuse_out)

        # Classifier
        logits = self.classifier(fused)

        return logits

    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  

        
