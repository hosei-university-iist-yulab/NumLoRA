"""
NumLoRA Integration with HuggingFace PEFT
==========================================

Draft implementation for native PEFT integration. Once merged into the
peft library, users can do:

    from peft import get_peft_model
    from numlora.integrations.peft_numlora import NumLoRAConfig

    config = NumLoRAConfig(r=8, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(base_model, config)

This module defines NumLoRAConfig and NumLoRAModel following the peft
library's conventions (PeftConfig, PeftModel subclasses).

Status: Draft — not yet submitted as a PR to huggingface/peft.
Will be submitted after paper acceptance.

Author: Mesabo (https://github.com/mesabo)
Repository: https://github.com/hosei-university-iist-yulab/NumLoRA
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set

import torch
import torch.nn as nn

try:
    from peft import PeftConfig, PeftModel, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


@dataclass
class NumLoRAConfig:
    """Configuration for NumLoRA (CTGS-enhanced LoRA).

    Extends standard LoRA with Continuous-Token Gradient Scaling.
    Compatible with any HuggingFace model supported by peft.

    Args:
        r: LoRA rank (same as standard LoRA).
        target_modules: List of module names to adapt (e.g., ["q_proj", "v_proj"]).
        lora_alpha: LoRA scaling factor (default: same as r).
        lora_dropout: Dropout probability for LoRA layers.
        enable_ctgs: Enable Continuous-Token Gradient Scaling (the core NumLoRA fix).
        ctgs_lr: Learning rate for CTGS scalars (typically 3x the LoRA lr).
        enable_ssr: Enable Scale-Shift Renormalisation (optional, not recommended).
        enable_mai: Enable Magnitude-Aware Initialisation (optional, minor benefit).
        bias: Bias handling ("none", "all", "lora_only").

    Example:
        config = NumLoRAConfig(r=8, target_modules=["q_proj", "v_proj"])
        model = apply_numlora_peft(base_model, config)
    """

    r: int = 8
    target_modules: Optional[List[str]] = None
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.0
    enable_ctgs: bool = True
    ctgs_lr: float = 3e-3
    enable_ssr: bool = False
    enable_mai: bool = False
    bias: str = "none"

    def __post_init__(self):
        if self.lora_alpha is None:
            self.lora_alpha = self.r

    def to_lora_config(self):
        """Convert to a standard peft LoraConfig (for the base LoRA part)."""
        if not PEFT_AVAILABLE:
            raise ImportError("peft library required: pip install peft")
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
        )


def apply_numlora_peft(model, config: NumLoRAConfig):
    """Apply NumLoRA to a HuggingFace model using peft + CTGS hooks.

    This is a bridge between the peft library and NumLoRA's CTGS hook.
    It applies standard LoRA via peft, then adds CTGS backward hooks
    to the LoRA-A matrices.

    Args:
        model: A HuggingFace pretrained model.
        config: NumLoRAConfig instance.

    Returns:
        model: The adapted model with LoRA + CTGS hooks.
        ctgs_params: List of CTGS scalar parameters (for separate optimizer group).
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft library required: pip install peft")

    from peft import get_peft_model

    # Step 1: Apply standard LoRA via peft
    lora_config = config.to_lora_config()
    model = get_peft_model(model, lora_config)

    # Step 2: Add CTGS hooks to LoRA-A matrices
    ctgs_params = []
    ctgs_hooks = {}

    if config.enable_ctgs:
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and isinstance(module.lora_A, nn.ModuleDict):
                for adapter_name, lora_a_linear in module.lora_A.items():
                    # Create learnable CTGS scalar for this layer
                    ctgs_c = nn.Parameter(torch.ones(1, device=lora_a_linear.weight.device))
                    ctgs_params.append(ctgs_c)

                    # Store x_norm cache on the module
                    module._ctgs_x_norm = None
                    module._ctgs_c = ctgs_c

                    # Forward hook to cache input norm
                    def _fwd_hook(mod, inp, out):
                        x = inp[0] if isinstance(inp, tuple) else inp
                        mod._ctgs_x_norm = x.detach().norm(dim=-1, keepdim=True).mean()

                    module.register_forward_hook(_fwd_hook)

                    # Backward hook on lora_A weight to scale gradient
                    def _make_bwd_hook(mod):
                        def _bwd_hook(grad):
                            if mod._ctgs_x_norm is not None:
                                scale = mod._ctgs_c / (mod._ctgs_x_norm + 1e-8)
                                return grad * scale
                            return grad
                        return _bwd_hook

                    lora_a_linear.weight.register_hook(_make_bwd_hook(module))

    return model, ctgs_params


def get_optimizer_groups(model, ctgs_params, lr=1e-3, ctgs_lr=3e-3):
    """Create optimizer parameter groups with separate LR for CTGS scalars.

    Args:
        model: The peft-adapted model.
        ctgs_params: CTGS scalar parameters from apply_numlora_peft.
        lr: Learning rate for LoRA parameters.
        ctgs_lr: Learning rate for CTGS scalars.

    Returns:
        List of parameter group dicts for torch.optim.AdamW.

    Example:
        optimizer = torch.optim.AdamW(
            get_optimizer_groups(model, ctgs_params, lr=1e-3, ctgs_lr=3e-3)
        )
    """
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora" in n]
    return [
        {"params": lora_params, "lr": lr},
        {"params": ctgs_params, "lr": ctgs_lr},
    ]
