"""
Official-style SOTA model adapters for JRTIP training.

These classes replace the previous lightweight proxies while using public model
implementations where available.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]


class TorchvisionSwin3DClassifier(nn.Module):
    """Video Swin Transformer 3D from torchvision."""

    def __init__(self, pretrained: bool = False, freeze_backbone: bool = False) -> None:
        super().__init__()
        from torchvision.models.video import Swin3D_T_Weights, swin3d_t

        weights = Swin3D_T_Weights.DEFAULT if pretrained else None
        self.backbone = swin3d_t(weights=weights)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, 2)
        self.implementation = "torchvision_swin3d_t"
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("head."):
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class OfficialJOSENetClassifier(nn.Module):
    """
    JOSENet architecture from external/JOSENet.

    The public JOSENet classifier returns one binary logit and is trained with
    BCEWithLogitsLoss in the official primary task.
    """

    def __init__(self, freeze_backbone: bool = False) -> None:
        super().__init__()
        josenet_root = REPO_ROOT / "external" / "JOSENet"
        if not (josenet_root / "architectures.py").exists():
            raise RuntimeError(
                "Official JOSENet source is missing. Expected external/JOSENet/architectures.py."
            )
        if str(josenet_root) not in sys.path:
            sys.path.insert(0, str(josenet_root))
        import architectures  # type: ignore

        args = SimpleNamespace(dropout=0.2, dropout3d=0.2)
        self.model = architectures.FGN(
            architectures.FGN_RGB(args),
            architectures.FGN_FLOW(args),
            architectures.FGN_MERGE_CLASSIFY(args),
        )
        self.implementation = "official_josenet_fgn_binary_scratch"
        self.binary_logits = True
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "fc3" not in name:
                    param.requires_grad = False

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        rgb, flow = inputs
        return self.model(rgb, flow).flatten()


class OfficialVideoMambaClassifier(nn.Module):
    """
    OpenGVLab VideoMamba adapter.

    This requires the official CUDA extensions mamba_ssm and causal_conv1d.
    If those extensions are not installed, construction fails explicitly.
    """

    def __init__(
        self,
        num_frames: int = 16,
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if pretrained:
            raise RuntimeError(
                "Pretrained VideoMamba checkpoint loading is not configured. "
                "Run without --pretrained or provide an official checkpoint integration first."
            )
        repo = REPO_ROOT / "external" / "OpenGVLab_VideoMamba"
        mamba_root = repo / "mamba"
        video_sm = repo / "videomamba" / "video_sm"
        if not (video_sm / "models" / "videomamba.py").exists():
            raise RuntimeError(
                "Official VideoMamba source is missing. Expected external/OpenGVLab_VideoMamba."
            )
        for path in (mamba_root, video_sm, video_sm / "models"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        try:
            from models.videomamba import VisionMamba  # type: ignore
        except ModuleNotFoundError as exc:
            if exc.name in {"mamba_ssm", "causal_conv1d"}:
                raise RuntimeError(
                    "Official VideoMamba needs CUDA extension dependencies that are not installed: "
                    f"{exc.name}. Install/build mamba_ssm and causal_conv1d for this conda env, "
                    "then rerun the same command."
                ) from exc
            raise

        # The public Windows environment has no supported Triton wheel, so use
        # the official VideoMamba implementation with PyTorch LayerNorm instead
        # of Triton fused add+norm. The Mamba CUDA selective-scan extension is
        # still required and remains the core VideoMamba operator.
        self.model = VisionMamba(
            patch_size=16,
            embed_dim=192,
            depth=24,
            rms_norm=False,
            residual_in_fp32=True,
            fused_add_norm=False,
            num_classes=2,
            img_size=224,
            num_frames=int(num_frames),
            kernel_size=1,
            ssm_cfg={"use_fast_path": False},
            drop_path_rate=0.1,
            fc_drop_rate=0.0,
            use_checkpoint=False,
            checkpoint_num=0,
        )
        self.implementation = "opengvlab_videomamba_visionmamba_scratch"
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("head."):
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
