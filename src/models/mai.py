"""
Magnitude-Aware Initialisation (MAI) for NumLoRA.

Runs a single forward pass on a calibration batch through the frozen backbone,
records per-layer activation variance, and calls calibrate_mai() on each
NumLoRALinear layer. This replaces LoRA's fixed A ~ N(0, 1/r) with a
per-layer calibrated variance A ~ N(0, 1/(r * sigma^2_ell)).

Cost: one forward pass (~2 seconds). Zero extra parameters.
"""

from collections.abc import Mapping
from typing import Dict, List

import torch
import torch.nn as nn

from src.models.numlora import NumLoRALinear


def collect_activation_variances(
    model: nn.Module,
    calibration_input: dict,
    device: torch.device,
) -> Dict[str, float]:
    """Run one forward pass and record activation variance at each NumLoRALinear.

    Args:
        model: The model with NumLoRALinear layers already inserted.
        calibration_input: A single batch dict compatible with the model's
            forward() (e.g., {"input_ids": ..., "attention_mask": ...} for
            HuggingFace models, or raw tensor for custom wrappers).
        device: Device to run calibration on.

    Returns:
        Dict mapping layer name -> activation variance (float).
    """
    variances: Dict[str, float] = {}
    hooks: List[torch.utils.hooks.RemovableHook] = []

    def make_hook(name: str):
        def hook_fn(module, input, output):
            # input is a tuple; take the first tensor
            x = input[0] if isinstance(input, tuple) else input
            variances[name] = x.detach().float().var().item()
        return hook_fn

    # Register hooks on all NumLoRALinear layers
    for name, module in model.named_modules():
        if isinstance(module, NumLoRALinear):
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    # Forward pass (no grad needed)
    model.eval()
    with torch.no_grad():
        if isinstance(calibration_input, Mapping):
            moved = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in calibration_input.items()
            }
            model(**moved)
        else:
            model(calibration_input.to(device))

    # Remove hooks
    for h in hooks:
        h.remove()

    return variances


def calibrate_numlora(
    model: nn.Module,
    calibration_input: dict,
    device: torch.device,
) -> Dict[str, float]:
    """Full MAI calibration: measure variances and reinitialise lora_A per layer.

    Args:
        model: Model with NumLoRALinear layers.
        calibration_input: One batch for the forward pass.
        device: CUDA device.

    Returns:
        Dict of layer name -> variance (for logging/debugging).
    """
    variances = collect_activation_variances(model, calibration_input, device)

    for name, module in model.named_modules():
        if isinstance(module, NumLoRALinear) and name in variances:
            module.calibrate_mai(variances[name])

    return variances
