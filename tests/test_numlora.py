"""Unit tests for NumLoRA core module."""

import math

import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.numlora import NumLoRALinear
from src.models.apply import apply_numlora, get_numlora_params, count_params
from src.models.mai import collect_activation_variances, calibrate_numlora


# ---------------------------------------------------------------------------
# NumLoRALinear basic tests
# ---------------------------------------------------------------------------

class TestNumLoRALinear:
    """Test the core NumLoRALinear layer."""

    def test_init_equals_identity(self):
        """At init (B=0, alpha=1, beta=1, gamma=0), NumLoRA = frozen linear."""
        linear = nn.Linear(64, 32)
        numlora = NumLoRALinear.from_linear(linear, rank=4)

        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            y_base = linear(x)
            y_numlora = numlora(x)

        torch.testing.assert_close(y_numlora, y_base, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        """Output shape matches input batch x seq x out_features."""
        linear = nn.Linear(128, 64)
        numlora = NumLoRALinear.from_linear(linear, rank=8)
        x = torch.randn(4, 20, 128)
        y = numlora(x)
        assert y.shape == (4, 20, 64)

    def test_trainable_params(self):
        """Only LoRA + SSR + CTGS params are trainable."""
        linear = nn.Linear(768, 768)
        numlora = NumLoRALinear.from_linear(linear, rank=8)

        # LoRA: A (8*768) + B (768*8) = 12288
        # SSR: alpha (768) + beta (768) + gamma (768) = 2304
        # CTGS: c (1)
        expected = 8 * 768 + 768 * 8 + 768 + 768 + 768 + 1
        assert numlora.num_trainable_params == expected

    def test_frozen_weight(self):
        """Base weight is frozen (no grad)."""
        linear = nn.Linear(64, 32)
        numlora = NumLoRALinear.from_linear(linear, rank=4)
        assert not numlora.weight.requires_grad
        if numlora.base_bias is not None:
            assert not numlora.base_bias.requires_grad

    def test_disable_ssr(self):
        """With SSR disabled, alpha/beta/gamma are buffers not parameters."""
        linear = nn.Linear(64, 32)
        numlora = NumLoRALinear.from_linear(linear, rank=4, enable_ssr=False)
        param_names = {n for n, _ in numlora.named_parameters()}
        assert "alpha" not in param_names
        assert "beta" not in param_names
        assert "gamma" not in param_names

    def test_disable_ctgs(self):
        """With CTGS disabled, ctgs_c is a buffer not a parameter."""
        linear = nn.Linear(64, 32)
        numlora = NumLoRALinear.from_linear(linear, rank=4, enable_ctgs=False)
        param_names = {n for n, _ in numlora.named_parameters()}
        assert "ctgs_c" not in param_names


# ---------------------------------------------------------------------------
# Weight merging tests
# ---------------------------------------------------------------------------

class TestMerge:
    """Test weight merging for zero-overhead inference."""

    def test_merge_produces_same_output(self):
        """Merged forward should match unmerged forward."""
        linear = nn.Linear(64, 32, bias=True)
        numlora = NumLoRALinear.from_linear(linear, rank=4)

        # Simulate some training (randomise LoRA params)
        with torch.no_grad():
            numlora.lora_A.normal_(0, 0.1)
            numlora.lora_B.normal_(0, 0.1)
            numlora.alpha.fill_(1.2)
            numlora.beta.fill_(0.9)
            numlora.gamma.fill_(0.05)

        x = torch.randn(2, 10, 64)
        numlora.eval()

        with torch.no_grad():
            y_unmerged = numlora(x)

        numlora.merge_weights()
        with torch.no_grad():
            y_merged = numlora(x)

        torch.testing.assert_close(y_merged, y_unmerged, atol=1e-4, rtol=1e-4)

    def test_unmerge_restores(self):
        """Unmerge should restore the original base weight."""
        linear = nn.Linear(64, 32, bias=True)
        original_weight = linear.weight.data.clone()
        original_bias = linear.bias.data.clone()

        numlora = NumLoRALinear.from_linear(linear, rank=4)
        with torch.no_grad():
            numlora.lora_A.normal_(0, 0.1)
            numlora.lora_B.normal_(0, 0.1)

        numlora.merge_weights()
        numlora.unmerge_weights()

        torch.testing.assert_close(numlora.weight.data, original_weight, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(numlora.base_bias.data, original_bias, atol=1e-5, rtol=1e-5)

    def test_no_bias_merge(self):
        """Merge works even when original linear has no bias."""
        linear = nn.Linear(64, 32, bias=False)
        numlora = NumLoRALinear.from_linear(linear, rank=4)
        with torch.no_grad():
            numlora.lora_A.normal_(0, 0.1)
            numlora.lora_B.normal_(0, 0.1)
            numlora.gamma.fill_(0.01)

        x = torch.randn(2, 5, 64)
        numlora.eval()
        with torch.no_grad():
            y_unmerged = numlora(x)
        numlora.merge_weights()
        with torch.no_grad():
            y_merged = numlora(x)

        torch.testing.assert_close(y_merged, y_unmerged, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# MAI tests
# ---------------------------------------------------------------------------

class TestMAI:
    """Test Magnitude-Aware Initialisation."""

    def test_calibrate_changes_lora_a(self):
        """calibrate_mai should change lora_A values."""
        linear = nn.Linear(64, 32)
        numlora = NumLoRALinear.from_linear(linear, rank=4)
        old_A = numlora.lora_A.data.clone()

        numlora.calibrate_mai(activation_var=10.0)

        assert not torch.equal(numlora.lora_A.data, old_A)

    def test_calibrate_scales_inversely(self):
        """Higher activation variance -> smaller lora_A std."""
        linear = nn.Linear(64, 32)

        nl_low = NumLoRALinear.from_linear(linear, rank=4)
        nl_low.calibrate_mai(activation_var=1.0)

        nl_high = NumLoRALinear.from_linear(nn.Linear(64, 32), rank=4)
        nl_high.calibrate_mai(activation_var=100.0)

        std_low = nl_low.lora_A.data.std().item()
        std_high = nl_high.lora_A.data.std().item()
        assert std_low > std_high, "Higher variance should produce smaller init std"


# ---------------------------------------------------------------------------
# apply_numlora tests (backbone-agnostic)
# ---------------------------------------------------------------------------

class TestApplyNumLoRA:
    """Test apply_numlora on simple models and HuggingFace models."""

    def test_simple_model(self):
        """Works on a plain nn.Module with named Linear layers."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
                self.mlp = nn.Linear(64, 32)

            def forward(self, x):
                return self.mlp(self.q_proj(x) + self.v_proj(x))

        model = SimpleModel()
        replaced = apply_numlora(
            model, rank=4,
            target_modules={"q_proj", "v_proj"}
        )

        assert len(replaced) == 2
        assert isinstance(model.q_proj, NumLoRALinear)
        assert isinstance(model.v_proj, NumLoRALinear)
        assert isinstance(model.mlp, nn.Linear)  # not replaced

    def test_param_groups(self):
        """get_numlora_params returns correct grouping."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)

            def forward(self, x):
                return self.q_proj(x)

        model = SimpleModel()
        apply_numlora(model, rank=4, target_modules={"q_proj"})
        groups = get_numlora_params(model)

        assert len(groups["lora"]) == 2   # lora_A, lora_B
        assert len(groups["ssr"]) == 3    # alpha, beta, gamma
        assert len(groups["ctgs"]) == 1   # ctgs_c

    def test_count_params(self):
        """count_params reports trainable vs frozen correctly."""
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 64)

            def forward(self, x):
                return self.fc(x)

        model = SmallModel()
        apply_numlora(model, rank=4, target_modules={"fc"})
        counts = count_params(model)

        assert counts["trainable"] > 0
        assert counts["frozen"] > 0
        assert counts["total"] == counts["trainable"] + counts["frozen"]


# ---------------------------------------------------------------------------
# Gradient flow test
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Verify gradients flow through LoRA path but not through frozen weights."""

    def test_lora_grads_flow(self):
        """lora_A and lora_B receive gradients during backward."""
        linear = nn.Linear(64, 32)
        numlora = NumLoRALinear.from_linear(linear, rank=4)
        numlora.train()

        x = torch.randn(2, 5, 64, requires_grad=True)
        y = numlora(x)
        loss = y.sum()
        loss.backward()

        assert numlora.lora_A.grad is not None
        assert numlora.lora_B.grad is not None
        assert numlora.alpha.grad is not None
        assert numlora.gamma.grad is not None

    def test_frozen_no_grad(self):
        """Frozen base weight should not accumulate gradients."""
        linear = nn.Linear(64, 32)
        numlora = NumLoRALinear.from_linear(linear, rank=4)
        numlora.train()

        x = torch.randn(2, 5, 64, requires_grad=True)
        y = numlora(x)
        loss = y.sum()
        loss.backward()

        assert numlora.weight.grad is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
