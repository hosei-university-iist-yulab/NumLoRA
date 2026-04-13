"""
Frozen-LLM forecasting model.

Architecture: Input projection -> Frozen backbone (with NumLoRA/LoRA) -> Forecast head.
Lookback patches are projected into the LLM hidden space, processed by the backbone,
and decoded to predict the future horizon.
"""

import torch
import torch.nn as nn


class LLMForecastingModel(nn.Module):
    """Frozen-LLM time-series forecasting model.

    Args:
        backbone: HuggingFace causal LM (frozen, with PEFT applied).
        patch_dim: Dimension of each input patch (patch_size * n_features).
        hidden_dim: LLM hidden dimension.
        horizon: Forecast horizon length.
        n_features: Number of time-series features.
    """

    def __init__(
        self,
        backbone: nn.Module,
        patch_dim: int,
        hidden_dim: int,
        horizon: int = 96,
        n_features: int = 7,
    ):
        super().__init__()
        self.backbone = backbone
        self.horizon = horizon
        self.n_features = n_features

        self.input_proj = nn.Linear(patch_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, horizon * n_features)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (batch, n_patches, patch_dim) — lookback patches.

        Returns:
            forecast: (batch, horizon, n_features) — predicted future values.
        """
        batch_size = patches.shape[0]

        embeds = self.input_proj(patches)

        backbone_out = self.backbone(
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
        )

        # Use last token's hidden state for forecasting (autoregressive style)
        last_hidden = backbone_out.hidden_states[-1][:, -1, :]  # (B, hidden_dim)

        forecast_flat = self.output_head(last_hidden)  # (B, horizon * n_features)
        forecast = forecast_flat.reshape(batch_size, self.horizon, self.n_features)

        return forecast
