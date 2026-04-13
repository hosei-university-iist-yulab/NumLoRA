"""
NumLoRA: Prior-Free Parameter-Efficient Fine-Tuning for Numerical Data.

Core module implementing NumLoRALinear — a drop-in replacement for LoRA
that adds Scale-Shift Renormalisation (SSR), Magnitude-Aware Initialisation
(MAI), and Continuous-Token Gradient Scaling (CTGS).

Backbone-agnostic: works with any HuggingFace transformer (GPT-2, Llama,
Phi, Qwen, SmolLM, Mistral, etc.).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NumLoRALinear(nn.Module):
    """Drop-in replacement for a frozen nn.Linear with NumLoRA adaptation.

    Forward: h = W_0 x + diag(alpha) B A diag(beta) x + gamma

    At init (alpha=1, beta=1, gamma=0, B=0): reduces exactly to W_0 x.
    At inference: merges into W_merged, b_merged for zero overhead.

    Args:
        in_features: Input dimension of the original linear layer.
        out_features: Output dimension of the original linear layer.
        rank: LoRA rank r.
        bias: Whether the original layer has a bias.
        enable_ssr: Enable Scale-Shift Renormalisation.
        enable_ctgs: Enable Continuous-Token Gradient Scaling.
        ctgs_eps: Epsilon for CTGS denominator stability.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        bias: bool = True,
        enable_ssr: bool = True,
        enable_ctgs: bool = True,
        ctgs_eps: float = 1e-8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.enable_ssr = enable_ssr
        self.enable_ctgs = enable_ctgs
        self.ctgs_eps = ctgs_eps

        # Frozen base weight (set externally via from_linear)
        self.weight = None  # will be set by from_linear()
        self.base_bias = None

        # LoRA factors
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # SSR parameters
        if enable_ssr:
            self.alpha = nn.Parameter(torch.ones(out_features))
            self.beta = nn.Parameter(torch.ones(in_features))
            self.gamma = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_buffer("alpha", torch.ones(out_features))
            self.register_buffer("beta", torch.ones(in_features))
            self.register_buffer("gamma", torch.zeros(out_features))

        # CTGS learnable scale (one per layer instance)
        if enable_ctgs:
            self.ctgs_c = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("ctgs_c", torch.ones(1))

        # Default init (overridden by MAI if calibrated)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_A.data /= math.sqrt(rank)

        # State flags
        self._merged = False
        self._ctgs_hook_handle = None
        self._ctgs_x_norm = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 8,
                    enable_ssr: bool = True, enable_ctgs: bool = True):
        """Create NumLoRALinear from an existing nn.Linear, freezing its weight."""
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            bias=linear.bias is not None,
            enable_ssr=enable_ssr,
            enable_ctgs=enable_ctgs,
        )
        # Freeze original weight
        layer.weight = linear.weight
        layer.weight.requires_grad_(False)
        if linear.bias is not None:
            layer.base_bias = linear.bias
            layer.base_bias.requires_grad_(False)
        # Move trainable params to same device as frozen weight
        device = linear.weight.device
        layer.to(device)
        return layer

    def _ctgs_hook_fn(self, grad: torch.Tensor) -> torch.Tensor:
        """Scale gradient of lora_A by c / (||x|| + eps). Uses cached x_norm."""
        if self._ctgs_x_norm is None:
            return grad
        scale = self.ctgs_c / (self._ctgs_x_norm + self.ctgs_eps)
        return grad * scale

    def _register_ctgs_hook(self):
        """Register CTGS hook once (not per forward call)."""
        if self._ctgs_hook_handle is None and self.enable_ctgs:
            self._ctgs_hook_handle = self.lora_A.register_hook(self._ctgs_hook_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear
        h = F.linear(x, self.weight, self.base_bias)

        if self._merged:
            return h

        # LoRA path: diag(alpha) @ B @ A @ diag(beta) @ x + gamma
        x_scaled = x * self.beta  # [..., in_features]
        lora_out = F.linear(x_scaled, self.lora_A)  # [..., rank]
        lora_out = F.linear(lora_out, self.lora_B)   # [..., out_features]
        lora_out = lora_out * self.alpha              # [..., out_features]

        # Cache x_norm for CTGS backward hook (updated each forward, hook registered once)
        if self.enable_ctgs and self.training:
            self._ctgs_x_norm = x.detach().norm(dim=-1, keepdim=True).mean()
            self._register_ctgs_hook()

        h = h + lora_out + self.gamma
        return h

    def calibrate_mai(self, activation_var: float):
        """Magnitude-Aware Initialisation: set lora_A variance from activation stats.

        Args:
            activation_var: Variance of activations at this layer, measured
                from a single forward pass on a calibration batch.
        """
        std = 1.0 / math.sqrt(self.rank * max(activation_var, 1e-8))
        nn.init.normal_(self.lora_A, mean=0.0, std=std)
        # Reset B to zero (standard LoRA convention)
        nn.init.zeros_(self.lora_B)

    def merge_weights(self):
        """Merge SSR + LoRA into base weight for zero-overhead inference."""
        if self._merged:
            return
        with torch.no_grad():
            # delta_W = diag(alpha) @ B @ A @ diag(beta)
            delta_W = self.lora_B @ self.lora_A          # [out, in]
            delta_W = self.alpha.unsqueeze(1) * delta_W   # scale rows
            delta_W = delta_W * self.beta.unsqueeze(0)    # scale cols
            self.weight.data += delta_W

            # Merge gamma into bias
            if self.base_bias is not None:
                self.base_bias.data += self.gamma.data
            else:
                # Create a new bias parameter from gamma
                self.base_bias = nn.Parameter(self.gamma.data.clone(),
                                              requires_grad=False)
        self._merged = True

    def unmerge_weights(self):
        """Reverse merge (for switching back to training)."""
        if not self._merged:
            return
        with torch.no_grad():
            delta_W = self.lora_B @ self.lora_A
            delta_W = self.alpha.unsqueeze(1) * delta_W
            delta_W = delta_W * self.beta.unsqueeze(0)
            self.weight.data -= delta_W

            if self.base_bias is not None:
                self.base_bias.data -= self.gamma.data
        self._merged = False

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"ssr={self.enable_ssr}, ctgs={self.enable_ctgs}, "
            f"trainable={self.num_trainable_params:,}"
        )
