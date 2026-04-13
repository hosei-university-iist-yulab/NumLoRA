"""
Frozen-LLM imputation model.

Architecture: Input projection -> Frozen LLM backbone (with NumLoRA/LoRA) -> Output head.
The backbone processes numerical patches as if they were token embeddings.
"""

import torch
import torch.nn as nn


class LLMImputationModel(nn.Module):
    """Frozen-LLM time-series imputation model.

    Patches of shape (batch, n_patches, patch_dim) are projected into the
    LLM's hidden dimension, processed by the frozen backbone with PEFT
    adaptation, and decoded back to the original feature space.

    Args:
        backbone: A HuggingFace causal LM (already frozen, with NumLoRA/LoRA applied).
        patch_dim: Dimension of each input patch (patch_size * n_features).
        hidden_dim: LLM hidden dimension.
        window_size: Full window length.
        n_features: Number of time-series features.
        patch_size: Size of each temporal patch.
    """

    def __init__(
        self,
        backbone: nn.Module,
        patch_dim: int,
        hidden_dim: int,
        window_size: int = 96,
        n_features: int = 7,
        patch_size: int = 16,
    ):
        super().__init__()
        self.backbone = backbone
        self.window_size = window_size
        self.n_features = n_features
        self.patch_size = patch_size
        self.n_patches = window_size // patch_size

        # Input projection: patch_dim -> hidden_dim
        self.input_proj = nn.Linear(patch_dim, hidden_dim)

        # Output head: hidden_dim -> patch_dim (reconstruct patches)
        self.output_head = nn.Linear(hidden_dim, patch_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (batch, n_patches, patch_dim) — masked input patches.

        Returns:
            reconstructed: (batch, window_size, n_features) — imputed output.
        """
        batch_size = patches.shape[0]

        # Project patches into LLM hidden space
        embeds = self.input_proj(patches)  # (B, n_patches, hidden_dim)

        # Pass through frozen backbone (bypass the embedding layer)
        # Use inputs_embeds instead of input_ids to feed numerical embeddings
        backbone_out = self.backbone(
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
        )

        # Use last hidden state (not logits) for reconstruction
        hidden = backbone_out.hidden_states[-1]  # (B, n_patches, hidden_dim)

        # Decode back to patch space
        decoded = self.output_head(hidden)  # (B, n_patches, patch_dim)

        # Reshape patches back to (B, window_size, n_features)
        reconstructed = decoded.reshape(
            batch_size, self.n_patches, self.patch_size, self.n_features
        )
        reconstructed = reconstructed.reshape(batch_size, self.window_size, self.n_features)

        return reconstructed
