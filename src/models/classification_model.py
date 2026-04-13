"""
Frozen-LLM time-series classification model.

Architecture: Input projection -> Frozen backbone (with NumLoRA/LoRA) -> Classification head.
"""

import torch
import torch.nn as nn


class LLMClassificationModel(nn.Module):
    """Frozen-LLM time-series classification model.

    Args:
        backbone: HuggingFace causal LM (frozen, with PEFT applied).
        patch_dim: Dimension of each input patch (patch_size * n_features).
        hidden_dim: LLM hidden dimension.
        n_classes: Number of output classes.
    """

    def __init__(
        self,
        backbone: nn.Module,
        patch_dim: int,
        hidden_dim: int,
        n_classes: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.n_classes = n_classes

        self.input_proj = nn.Linear(patch_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (batch, n_patches, patch_dim).

        Returns:
            logits: (batch, n_classes).
        """
        embeds = self.input_proj(patches)

        backbone_out = self.backbone(
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
        )

        # Pool: mean of last hidden state across patches
        hidden = backbone_out.hidden_states[-1]  # (B, n_patches, hidden_dim)
        pooled = hidden.mean(dim=1)  # (B, hidden_dim)

        logits = self.classifier(pooled)  # (B, n_classes)
        return logits
