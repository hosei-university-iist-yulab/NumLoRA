"""
apply_numlora() — Generic wrapper to inject NumLoRA into any HuggingFace model.

Replaces selected nn.Linear layers with NumLoRALinear, keeping the rest frozen.
Backbone-agnostic: auto-detects target modules from known architecture families
(GPT-2, Llama, Phi, Qwen, SmolLM, Mistral) or accepts explicit target names.

Usage:
    from transformers import AutoModelForCausalLM
    from src.models.apply import apply_numlora

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    numlora_layers = apply_numlora(model, rank=8)
"""

from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

from src.models.numlora import NumLoRALinear

# GPT-2 uses transformers.pytorch_utils.Conv1D instead of nn.Linear
try:
    from transformers.pytorch_utils import Conv1D as HFConv1D
except ImportError:
    HFConv1D = None


def _conv1d_to_linear(conv1d) -> nn.Linear:
    """Convert HuggingFace Conv1D to nn.Linear (they differ only in weight layout).

    Conv1D stores weight as (in_features, out_features).
    nn.Linear stores weight as (out_features, in_features).
    """
    in_features, out_features = conv1d.weight.shape
    linear = nn.Linear(in_features, out_features, bias=conv1d.bias is not None)
    linear.weight = nn.Parameter(conv1d.weight.T.contiguous())
    if conv1d.bias is not None:
        linear.bias = conv1d.bias
    return linear

# Known target module patterns per architecture family
TARGET_MODULES = {
    "gpt2": {"c_attn", "c_proj", "c_fc"},
    "gpt_neox": {"query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"},
    "llama": {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
    "mistral": {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
    "phi": {"q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"},
    "phi3": {"qkv_proj", "o_proj", "gate_up_proj", "down_proj"},
    "qwen2": {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
    "smollm": {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
}


def _detect_arch(model: nn.Module) -> Optional[str]:
    """Detect the architecture family from model config or class name."""
    config = getattr(model, "config", None)
    if config is not None:
        model_type = getattr(config, "model_type", "").lower()
        if model_type in TARGET_MODULES:
            return model_type
        # Fuzzy matching for variants
        for arch in TARGET_MODULES:
            if arch in model_type:
                return arch

    # Fallback: check class name
    cls_name = type(model).__name__.lower()
    for arch in TARGET_MODULES:
        if arch in cls_name:
            return arch

    return None


def _get_parent_and_attr(model: nn.Module, name: str):
    """Get parent module and attribute name for a dotted module path."""
    parts = name.rsplit(".", 1)
    if len(parts) == 1:
        return model, parts[0]
    parent_name, attr = parts
    parent = dict(model.named_modules())[parent_name]
    return parent, attr


def apply_numlora(
    model: nn.Module,
    rank: int = 8,
    target_modules: Optional[Set[str]] = None,
    enable_ssr: bool = True,
    enable_ctgs: bool = True,
    freeze_backbone: bool = True,
) -> Dict[str, NumLoRALinear]:
    """Replace target nn.Linear layers with NumLoRALinear in-place.

    Args:
        model: Any nn.Module (typically a HuggingFace pretrained model).
        rank: LoRA rank.
        target_modules: Set of layer name suffixes to replace (e.g., {"q_proj", "v_proj"}).
            If None, auto-detects from architecture family.
        enable_ssr: Enable Scale-Shift Renormalisation.
        enable_ctgs: Enable Continuous-Token Gradient Scaling.
        freeze_backbone: Freeze all parameters before injecting NumLoRA.

    Returns:
        Dict mapping full layer name -> NumLoRALinear instance.
    """
    # Auto-detect target modules if not specified
    if target_modules is None:
        arch = _detect_arch(model)
        if arch is None:
            raise ValueError(
                "Could not auto-detect model architecture. "
                "Pass target_modules explicitly, e.g., "
                "target_modules={'q_proj', 'k_proj', 'v_proj', 'o_proj'}"
            )
        target_modules = TARGET_MODULES[arch]

    # Freeze backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad_(False)

    # Find and replace target layers
    replaced: Dict[str, NumLoRALinear] = {}
    # Collect names first to avoid mutating dict during iteration
    targets: List[str] = []
    linear_types = (nn.Linear,)
    if HFConv1D is not None:
        linear_types = (nn.Linear, HFConv1D)

    for name, module in model.named_modules():
        if isinstance(module, linear_types):
            # Check if any target suffix matches the layer name
            layer_name = name.split(".")[-1]
            if layer_name in target_modules:
                targets.append(name)

    for name in targets:
        module = dict(model.named_modules())[name]
        # Convert Conv1D to Linear if needed (GPT-2 uses Conv1D)
        if HFConv1D is not None and isinstance(module, HFConv1D):
            module = _conv1d_to_linear(module)
        numlora_layer = NumLoRALinear.from_linear(
            module, rank=rank, enable_ssr=enable_ssr, enable_ctgs=enable_ctgs
        )
        parent, attr = _get_parent_and_attr(model, name)
        setattr(parent, attr, numlora_layer)
        replaced[name] = numlora_layer

    return replaced


def get_numlora_params(model: nn.Module) -> Dict[str, list]:
    """Collect NumLoRA parameters grouped by learning rate.

    Returns dict with keys:
        "lora": [lora_A, lora_B] — use base lr (e.g., 1e-3)
        "ssr": [alpha, beta, gamma] — use higher lr (e.g., 1e-2)
        "ctgs": [ctgs_c] — use higher lr (e.g., 1e-2)
    """
    groups = {"lora": [], "ssr": [], "ctgs": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name or "lora_B" in name:
            groups["lora"].append(param)
        elif any(k in name for k in ("alpha", "beta", "gamma")):
            groups["ssr"].append(param)
        elif "ctgs_c" in name:
            groups["ctgs"].append(param)
    return groups


def count_params(model: nn.Module) -> Dict[str, int]:
    """Count total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": 100 * trainable / max(total, 1),
    }
