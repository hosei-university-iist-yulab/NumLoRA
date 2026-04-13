"""
NumLoRA Test Suite
==================

Comprehensive tests for all NumLoRA components. Each test class maps to a
specific source module and validates its public interface without modifying
any production code.

Module mapping:
    TestNumLoRALinear     -> src/models/numlora.py (NumLoRALinear layer)
    TestMerge             -> src/models/numlora.py (merge/unmerge weights)
    TestMAI               -> src/models/mai.py (Magnitude-Aware Initialisation)
    TestApplyNumLoRA      -> src/models/apply.py (apply_numlora wrapper)
    TestGradientFlow      -> src/models/numlora.py (CTGS backward hook)
    TestImputationModel   -> src/models/imputation_model.py (frozen-LLM imputation)
    TestForecastingModel  -> src/models/forecasting_model.py (frozen-LLM forecasting)
    TestClassificationModel -> src/models/classification_model.py (frozen-LLM classification)
    TestDataset           -> src/data/dataset.py (imputation dataset + masking)
    TestForecastingData   -> src/data/forecasting.py (forecasting dataset)
    TestClassificationData -> src/data/classification.py (classification dataset)
    TestMetrics           -> scripts/experiments/train.py (MAE, MSE, MRE metrics)

Run all:  pytest tests/test_numlora.py -v
Run one:  pytest tests/test_numlora.py::TestNumLoRALinear -v
"""

import math
import os
import sys
import tempfile

import pytest
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.numlora import NumLoRALinear
from src.models.apply import apply_numlora, get_numlora_params, count_params
from src.models.mai import collect_activation_variances, calibrate_numlora
from src.models.imputation_model import LLMImputationModel
from src.models.forecasting_model import LLMForecastingModel
from src.models.classification_model import LLMClassificationModel
from src.data.dataset import TimeSeriesImputationDataset, load_ett, create_datasets
from src.data.forecasting import ForecastingDataset, load_ett_forecasting, create_forecasting_datasets
from src.data.classification import TSClassificationDataset


# ═══════════════════════════════════════════════════════════════════════
# NumLoRALinear (src/models/numlora.py)
# ═══════════════════════════════════════════════════════════════════════

class TestNumLoRALinear:
    """Tests for the core NumLoRALinear layer.

    Validates: construction from nn.Linear, init-identity property,
    output shape, parameter counts, and SSR/CTGS toggle behaviour.
    """

    def test_init_equals_identity(self):
        """At init (B=0, alpha=1, beta=1, gamma=0), output matches frozen linear."""
        linear = nn.Linear(64, 32)
        numlora = NumLoRALinear.from_linear(linear, rank=4)
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            torch.testing.assert_close(numlora(x), linear(x), atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        """Output shape: (batch, seq, out_features)."""
        numlora = NumLoRALinear.from_linear(nn.Linear(128, 64), rank=8)
        assert numlora(torch.randn(4, 20, 128)).shape == (4, 20, 64)

    def test_trainable_params(self):
        """Trainable = LoRA(A,B) + SSR(alpha,beta,gamma) + CTGS(c)."""
        numlora = NumLoRALinear.from_linear(nn.Linear(768, 768), rank=8)
        expected = 8 * 768 + 768 * 8 + 768 + 768 + 768 + 1
        assert numlora.num_trainable_params == expected

    def test_frozen_weight(self):
        """Base weight and bias are frozen (requires_grad=False)."""
        numlora = NumLoRALinear.from_linear(nn.Linear(64, 32), rank=4)
        assert not numlora.weight.requires_grad
        if numlora.base_bias is not None:
            assert not numlora.base_bias.requires_grad

    def test_disable_ssr(self):
        """With SSR disabled, alpha/beta/gamma are buffers, not parameters."""
        numlora = NumLoRALinear.from_linear(nn.Linear(64, 32), rank=4, enable_ssr=False)
        param_names = {n for n, _ in numlora.named_parameters()}
        assert "alpha" not in param_names
        assert "beta" not in param_names
        assert "gamma" not in param_names

    def test_disable_ctgs(self):
        """With CTGS disabled, ctgs_c is a buffer, not a parameter."""
        numlora = NumLoRALinear.from_linear(nn.Linear(64, 32), rank=4, enable_ctgs=False)
        param_names = {n for n, _ in numlora.named_parameters()}
        assert "ctgs_c" not in param_names

    def test_no_bias_linear(self):
        """Works correctly when original linear has no bias."""
        numlora = NumLoRALinear.from_linear(nn.Linear(64, 32, bias=False), rank=4)
        y = numlora(torch.randn(2, 5, 64))
        assert y.shape == (2, 5, 32)

    def test_different_ranks(self):
        """Supports various rank values."""
        for rank in [1, 2, 4, 8, 16, 32]:
            numlora = NumLoRALinear.from_linear(nn.Linear(64, 64), rank=rank)
            assert numlora.lora_A.shape == (rank, 64)
            assert numlora.lora_B.shape == (64, rank)


# ═══════════════════════════════════════════════════════════════════════
# Weight Merging (src/models/numlora.py)
# ═══════════════════════════════════════════════════════════════════════

class TestMerge:
    """Tests for weight merging (zero-overhead inference).

    Validates: merged output matches unmerged, unmerge restores original,
    and merge works with and without bias.
    """

    def _make_trained_numlora(self, in_f=64, out_f=32, rank=4, bias=True):
        """Helper: create a NumLoRALinear with non-trivial trained weights."""
        linear = nn.Linear(in_f, out_f, bias=bias)
        numlora = NumLoRALinear.from_linear(linear, rank=rank)
        with torch.no_grad():
            numlora.lora_A.normal_(0, 0.1)
            numlora.lora_B.normal_(0, 0.1)
            numlora.alpha.fill_(1.2)
            numlora.beta.fill_(0.9)
            numlora.gamma.fill_(0.05)
        return numlora

    def test_merge_produces_same_output(self):
        """Merged forward matches unmerged forward."""
        numlora = self._make_trained_numlora()
        x = torch.randn(2, 10, 64)
        numlora.eval()
        with torch.no_grad():
            y_before = numlora(x)
        numlora.merge_weights()
        with torch.no_grad():
            y_after = numlora(x)
        torch.testing.assert_close(y_after, y_before, atol=1e-4, rtol=1e-4)

    def test_unmerge_restores(self):
        """Unmerge restores the original base weight."""
        linear = nn.Linear(64, 32, bias=True)
        w_orig = linear.weight.data.clone()
        b_orig = linear.bias.data.clone()
        numlora = NumLoRALinear.from_linear(linear, rank=4)
        with torch.no_grad():
            numlora.lora_A.normal_(0, 0.1)
            numlora.lora_B.normal_(0, 0.1)
        numlora.merge_weights()
        numlora.unmerge_weights()
        torch.testing.assert_close(numlora.weight.data, w_orig, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(numlora.base_bias.data, b_orig, atol=1e-5, rtol=1e-5)

    def test_no_bias_merge(self):
        """Merge works when original linear has no bias (gamma creates one)."""
        numlora = self._make_trained_numlora(bias=False)
        x = torch.randn(2, 5, 64)
        numlora.eval()
        with torch.no_grad():
            y_before = numlora(x)
        numlora.merge_weights()
        with torch.no_grad():
            y_after = numlora(x)
        torch.testing.assert_close(y_after, y_before, atol=1e-4, rtol=1e-4)

    def test_double_merge_idempotent(self):
        """Calling merge twice does not corrupt weights."""
        numlora = self._make_trained_numlora()
        numlora.merge_weights()
        w_after_first = numlora.weight.data.clone()
        numlora.merge_weights()
        torch.testing.assert_close(numlora.weight.data, w_after_first)


# ═══════════════════════════════════════════════════════════════════════
# MAI (src/models/mai.py)
# ═══════════════════════════════════════════════════════════════════════

class TestMAI:
    """Tests for Magnitude-Aware Initialisation.

    Validates: calibrate_mai changes lora_A, higher variance produces
    smaller init std, and B is reset to zero after calibration.
    """

    def test_calibrate_changes_lora_a(self):
        """calibrate_mai modifies lora_A values."""
        numlora = NumLoRALinear.from_linear(nn.Linear(64, 32), rank=4)
        old_A = numlora.lora_A.data.clone()
        numlora.calibrate_mai(activation_var=10.0)
        assert not torch.equal(numlora.lora_A.data, old_A)

    def test_calibrate_scales_inversely(self):
        """Higher activation variance produces smaller lora_A std."""
        nl_low = NumLoRALinear.from_linear(nn.Linear(64, 32), rank=4)
        nl_low.calibrate_mai(activation_var=1.0)
        nl_high = NumLoRALinear.from_linear(nn.Linear(64, 32), rank=4)
        nl_high.calibrate_mai(activation_var=100.0)
        assert nl_low.lora_A.data.std().item() > nl_high.lora_A.data.std().item()

    def test_calibrate_resets_b(self):
        """After calibration, lora_B is zero (standard LoRA convention)."""
        numlora = NumLoRALinear.from_linear(nn.Linear(64, 32), rank=4)
        with torch.no_grad():
            numlora.lora_B.fill_(1.0)
        numlora.calibrate_mai(activation_var=5.0)
        assert torch.all(numlora.lora_B == 0)


# ═══════════════════════════════════════════════════════════════════════
# apply_numlora (src/models/apply.py)
# ═══════════════════════════════════════════════════════════════════════

class TestApplyNumLoRA:
    """Tests for the generic apply_numlora wrapper.

    Validates: correct layer replacement, parameter grouping, counting,
    and that non-target layers are left untouched.
    """

    def _make_simple_model(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
                self.mlp = nn.Linear(64, 32)
            def forward(self, x):
                return self.mlp(self.q_proj(x) + self.v_proj(x))
        return SimpleModel()

    def test_replaces_target_layers(self):
        """Only target modules are replaced with NumLoRALinear."""
        model = self._make_simple_model()
        replaced = apply_numlora(model, rank=4, target_modules={"q_proj", "v_proj"})
        assert len(replaced) == 2
        assert isinstance(model.q_proj, NumLoRALinear)
        assert isinstance(model.v_proj, NumLoRALinear)
        assert isinstance(model.mlp, nn.Linear)

    def test_param_groups(self):
        """get_numlora_params returns lora, ssr, and ctgs groups."""
        model = self._make_simple_model()
        apply_numlora(model, rank=4, target_modules={"q_proj"})
        groups = get_numlora_params(model)
        assert len(groups["lora"]) == 2
        assert len(groups["ssr"]) == 3
        assert len(groups["ctgs"]) == 1

    def test_count_params(self):
        """count_params reports trainable vs frozen correctly."""
        model = self._make_simple_model()
        apply_numlora(model, rank=4, target_modules={"q_proj"})
        counts = count_params(model)
        assert counts["trainable"] > 0
        assert counts["frozen"] > 0
        assert counts["total"] == counts["trainable"] + counts["frozen"]

    def test_freeze_backbone(self):
        """With freeze_backbone=True, all original params are frozen."""
        model = self._make_simple_model()
        apply_numlora(model, rank=4, target_modules={"q_proj"}, freeze_backbone=True)
        for name, param in model.named_parameters():
            if "numlora" not in name and "lora" not in name and "alpha" not in name \
               and "beta" not in name and "gamma" not in name and "ctgs" not in name:
                assert not param.requires_grad, f"{name} should be frozen"


# ═══════════════════════════════════════════════════════════════════════
# Gradient Flow (src/models/numlora.py — CTGS hook)
# ═══════════════════════════════════════════════════════════════════════

class TestGradientFlow:
    """Tests for CTGS backward hook and gradient flow.

    Validates: LoRA params receive gradients, frozen weights do not,
    and CTGS hook does not accumulate across forward passes.
    """

    def test_lora_grads_flow(self):
        """lora_A, lora_B, alpha, gamma receive gradients during backward."""
        numlora = NumLoRALinear.from_linear(nn.Linear(64, 32), rank=4)
        numlora.train()
        x = torch.randn(2, 5, 64, requires_grad=True)
        loss = numlora(x).sum()
        loss.backward()
        assert numlora.lora_A.grad is not None
        assert numlora.lora_B.grad is not None
        assert numlora.alpha.grad is not None
        assert numlora.gamma.grad is not None

    def test_frozen_no_grad(self):
        """Frozen base weight does not accumulate gradients."""
        numlora = NumLoRALinear.from_linear(nn.Linear(64, 32), rank=4)
        numlora.train()
        x = torch.randn(2, 5, 64, requires_grad=True)
        numlora(x).sum().backward()
        assert numlora.weight.grad is None

    def test_ctgs_hook_no_accumulation(self):
        """CTGS hook runs once per backward, not once per forward call."""
        numlora = NumLoRALinear.from_linear(nn.Linear(64, 32), rank=4)
        numlora.train()
        x = torch.randn(2, 5, 64, requires_grad=True)
        # Multiple forward passes
        for _ in range(5):
            numlora(x)
        loss = numlora(x).sum()
        loss.backward()
        # If hooks accumulated, gradient would be scaled 6 times -> explode
        assert numlora.lora_A.grad.abs().max().item() < 1e6


# ═══════════════════════════════════════════════════════════════════════
# Imputation Model (src/models/imputation_model.py)
# ═══════════════════════════════════════════════════════════════════════

class TestImputationModel:
    """Tests for LLMImputationModel.

    Uses a tiny mock backbone (no real LLM) to validate shapes and
    forward pass without GPU dependency.
    """

    def _make_mock_backbone(self, hidden_dim=32):
        """Create a minimal mock that mimics HF output structure."""
        class MockBackbone(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.proj = nn.Linear(d, d)
            def forward(self, inputs_embeds=None, **kwargs):
                h = self.proj(inputs_embeds)
                class Out:
                    hidden_states = [h]
                return Out()
        return MockBackbone(hidden_dim)

    def test_output_shape(self):
        """Output shape matches (batch, window_size, n_features)."""
        backbone = self._make_mock_backbone(32)
        # patch_dim = patch_size * n_features = 8 * 2 = 16, window = 24, n_patches = 3
        model = LLMImputationModel(backbone, patch_dim=16, hidden_dim=32,
                                    window_size=24, n_features=2, patch_size=8)
        patches = torch.randn(2, 3, 16)  # 3 patches of dim 16
        out = model(patches)
        assert out.shape == (2, 24, 2)

    def test_backward_works(self):
        """Loss backpropagates through the model."""
        backbone = self._make_mock_backbone(32)
        model = LLMImputationModel(backbone, patch_dim=16, hidden_dim=32,
                                    window_size=24, n_features=2, patch_size=8)
        patches = torch.randn(2, 3, 16)
        loss = model(patches).sum()
        loss.backward()
        assert model.input_proj.weight.grad is not None


# ═══════════════════════════════════════════════════════════════════════
# Forecasting Model (src/models/forecasting_model.py)
# ═══════════════════════════════════════════════════════════════════════

class TestForecastingModel:
    """Tests for LLMForecastingModel.

    Validates output shape (batch, horizon, n_features) and backward pass
    using a mock backbone.
    """

    def _make_mock_backbone(self, hidden_dim=32):
        class MockBackbone(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.proj = nn.Linear(d, d)
            def forward(self, inputs_embeds=None, **kwargs):
                h = self.proj(inputs_embeds)
                class Out:
                    hidden_states = [h]
                return Out()
        return MockBackbone(hidden_dim)

    def test_output_shape(self):
        """Output: (batch, horizon, n_features)."""
        backbone = self._make_mock_backbone(32)
        model = LLMForecastingModel(backbone, patch_dim=16, hidden_dim=32,
                                     horizon=96, n_features=7)
        patches = torch.randn(2, 6, 16)
        assert model(patches).shape == (2, 96, 7)

    def test_different_horizons(self):
        """Works for short (96), medium (192), long (336) horizons."""
        backbone = self._make_mock_backbone(32)
        for horizon in [96, 192, 336]:
            model = LLMForecastingModel(backbone, patch_dim=16, hidden_dim=32,
                                         horizon=horizon, n_features=7)
            out = model(torch.randn(1, 6, 16))
            assert out.shape == (1, horizon, 7)

    def test_backward_works(self):
        """Gradients flow through the forecasting model."""
        backbone = self._make_mock_backbone(32)
        model = LLMForecastingModel(backbone, patch_dim=16, hidden_dim=32,
                                     horizon=96, n_features=7)
        loss = model(torch.randn(2, 6, 16)).sum()
        loss.backward()
        assert model.output_head.weight.grad is not None


# ═══════════════════════════════════════════════════════════════════════
# Classification Model (src/models/classification_model.py)
# ═══════════════════════════════════════════════════════════════════════

class TestClassificationModel:
    """Tests for LLMClassificationModel.

    Validates output shape (batch, n_classes), softmax-compatibility,
    and cross-entropy loss backward.
    """

    def _make_mock_backbone(self, hidden_dim=32):
        class MockBackbone(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.proj = nn.Linear(d, d)
            def forward(self, inputs_embeds=None, **kwargs):
                h = self.proj(inputs_embeds)
                class Out:
                    hidden_states = [h]
                return Out()
        return MockBackbone(hidden_dim)

    def test_output_shape(self):
        """Output: (batch, n_classes) logits."""
        backbone = self._make_mock_backbone(32)
        model = LLMClassificationModel(backbone, patch_dim=16, hidden_dim=32, n_classes=5)
        patches = torch.randn(4, 6, 16)
        assert model(patches).shape == (4, 5)

    def test_cross_entropy_loss(self):
        """Cross-entropy loss computes and backpropagates."""
        backbone = self._make_mock_backbone(32)
        model = LLMClassificationModel(backbone, patch_dim=16, hidden_dim=32, n_classes=3)
        patches = torch.randn(4, 6, 16)
        labels = torch.tensor([0, 1, 2, 0])
        logits = model(patches)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        assert model.classifier[0].weight.grad is not None

    def test_different_n_classes(self):
        """Works for binary and multi-class."""
        backbone = self._make_mock_backbone(32)
        for n_classes in [2, 5, 10, 50]:
            model = LLMClassificationModel(backbone, patch_dim=16, hidden_dim=32, n_classes=n_classes)
            assert model(torch.randn(2, 4, 16)).shape == (2, n_classes)


# ═══════════════════════════════════════════════════════════════════════
# Imputation Dataset (src/data/dataset.py)
# ═══════════════════════════════════════════════════════════════════════

class TestDataset:
    """Tests for TimeSeriesImputationDataset.

    Validates: windowing, masking rate, patch shape, deterministic masks,
    and ETT loader integration.
    """

    def _make_data(self, T=200, D=7):
        return np.random.randn(T, D).astype(np.float32)

    def test_window_count(self):
        """Number of windows = (T - window_size) // stride + 1."""
        data = self._make_data(200, 7)
        ds = TimeSeriesImputationDataset(data, window_size=96, stride=96)
        assert len(ds) == 2  # 200 // 96 = 2 full windows

    def test_sample_keys(self):
        """Each sample has patches, target, mask, masked_input."""
        ds = TimeSeriesImputationDataset(self._make_data(), window_size=96)
        sample = ds[0]
        assert set(sample.keys()) == {"patches", "target", "mask", "masked_input"}

    def test_patch_shape(self):
        """Patches shape: (n_patches, patch_size * n_features)."""
        ds = TimeSeriesImputationDataset(self._make_data(200, 7), window_size=96, patch_size=16)
        patches = ds[0]["patches"]
        assert patches.shape == (6, 16 * 7)  # 96/16=6 patches, each 16*7=112

    def test_masking_rate(self):
        """Actual masking rate is approximately the requested rate."""
        ds = TimeSeriesImputationDataset(self._make_data(1000, 7), window_size=96,
                                         missing_rate=0.3, seed=42)
        mask = ds[0]["mask"]
        observed_rate = 1 - mask.mean().item()
        assert abs(observed_rate - 0.3) < 0.1  # within 10% tolerance

    def test_deterministic_mask(self):
        """Same seed produces same mask."""
        data = self._make_data()
        ds1 = TimeSeriesImputationDataset(data, window_size=96, seed=42)
        ds2 = TimeSeriesImputationDataset(data, window_size=96, seed=42)
        torch.testing.assert_close(ds1[0]["mask"], ds2[0]["mask"])

    def test_load_ett(self):
        """ETT loader returns correct structure (requires ETT data on disk)."""
        ett_path = "data/time_series/ETDataset/ETT-small"
        if not os.path.exists(os.path.join(ett_path, "ETTh1.csv")):
            pytest.skip("ETT data not available")
        data = load_ett("h1", data_dir=ett_path)
        assert data["n_features"] == 7
        assert data["name"] == "ETT-h1"
        assert len(data["train"]) > 0


# ═══════════════════════════════════════════════════════════════════════
# Forecasting Dataset (src/data/forecasting.py)
# ═══════════════════════════════════════════════════════════════════════

class TestForecastingData:
    """Tests for ForecastingDataset.

    Validates: sample structure, lookback/horizon split, and patch shape.
    """

    def test_sample_structure(self):
        """Each sample has patches and target with correct shapes."""
        data = np.random.randn(500, 7).astype(np.float32)
        ds = ForecastingDataset(data, lookback=96, horizon=96, patch_size=16, stride=96)
        sample = ds[0]
        assert sample["patches"].shape == (6, 16 * 7)  # 96/16=6 patches
        assert sample["target"].shape == (96, 7)        # horizon

    def test_different_horizons(self):
        """Supports short (96), medium (192), long (336) horizons."""
        data = np.random.randn(1000, 7).astype(np.float32)
        for horizon in [96, 192, 336]:
            ds = ForecastingDataset(data, lookback=96, horizon=horizon, patch_size=16, stride=96)
            assert ds[0]["target"].shape == (horizon, 7)

    def test_load_ett_forecasting(self):
        """ETT forecasting loader returns correct structure."""
        ett_path = "data/time_series/ETDataset/ETT-small"
        if not os.path.exists(os.path.join(ett_path, "ETTh1.csv")):
            pytest.skip("ETT data not available")
        data = load_ett_forecasting("h1", horizon=96, data_dir=ett_path)
        assert data["horizon"] == 96
        assert data["lookback"] == 96
        assert data["n_features"] == 7


# ═══════════════════════════════════════════════════════════════════════
# Classification Dataset (src/data/classification.py)
# ═══════════════════════════════════════════════════════════════════════

class TestClassificationData:
    """Tests for TSClassificationDataset.

    Validates: sample structure, label type, and patch shape.
    Uses synthetic data (no UCR download required).
    """

    def test_sample_structure(self):
        """Each sample has patches (float) and label (long)."""
        data = np.random.randn(50, 96, 1).astype(np.float32)
        labels = np.random.randint(0, 3, size=50)
        ds = TSClassificationDataset(data, labels, patch_size=16)
        sample = ds[0]
        assert sample["patches"].dtype == torch.float32
        assert sample["label"].dtype == torch.long

    def test_patch_shape(self):
        """Patches shape: (n_patches, patch_size * n_features)."""
        data = np.random.randn(10, 64, 3).astype(np.float32)
        labels = np.zeros(10, dtype=int)
        ds = TSClassificationDataset(data, labels, patch_size=16)
        assert ds[0]["patches"].shape == (4, 16 * 3)  # 64/16=4 patches

    def test_label_range(self):
        """Labels are within [0, n_classes)."""
        data = np.random.randn(20, 32, 1).astype(np.float32)
        labels = np.array([0, 1, 2, 0, 1] * 4)
        ds = TSClassificationDataset(data, labels, patch_size=16)
        for i in range(len(ds)):
            assert 0 <= ds[i]["label"].item() < 3


# ═══════════════════════════════════════════════════════════════════════
# Metrics (scripts/experiments/train.py)
# ═══════════════════════════════════════════════════════════════════════

class TestMetrics:
    """Tests for evaluation metrics: masked MAE, MSE, MRE.

    Imported from train.py. Validates correctness on known inputs.
    """

    def setup_method(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "experiments"))
        from train import masked_mae, masked_mse, masked_mre, forecast_mae, forecast_mse, forecast_mre
        self.masked_mae = masked_mae
        self.masked_mse = masked_mse
        self.masked_mre = masked_mre
        self.forecast_mae = forecast_mae
        self.forecast_mse = forecast_mse
        self.forecast_mre = forecast_mre

    def test_masked_mae_known(self):
        """MAE on masked positions with known values."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.0, 4.0])
        mask = torch.tensor([0.0, 1.0, 0.0])  # positions 0,2 are missing
        mae = self.masked_mae(pred, target, mask)
        expected = (0.5 + 1.0) / 2  # |1-1.5| + |3-4| / 2
        assert abs(mae.item() - expected) < 1e-5

    def test_masked_mse_known(self):
        """MSE on masked positions with known values."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.0, 4.0])
        mask = torch.tensor([0.0, 1.0, 0.0])
        mse = self.masked_mse(pred, target, mask)
        expected = (0.25 + 1.0) / 2
        assert abs(mse.item() - expected) < 1e-5

    def test_masked_mre_known(self):
        """MRE on masked positions with known values."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 2.0, 4.0])
        mask = torch.tensor([0.0, 1.0, 0.0])
        mre = self.masked_mre(pred, target, mask)
        # pos 0: |1-2|/2 = 0.5, pos 2: |3-4|/4 = 0.25 -> avg = 0.375
        expected = (0.5 + 0.25) / 2
        assert abs(mre.item() - expected) < 1e-4

    def test_forecast_mae(self):
        """Forecast MAE on all positions."""
        pred = torch.tensor([1.0, 3.0])
        target = torch.tensor([2.0, 3.0])
        assert abs(self.forecast_mae(pred, target).item() - 0.5) < 1e-5

    def test_all_observed_returns_zero(self):
        """If mask is all 1 (no missing), masked metrics return 0."""
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([3.0, 4.0])
        mask = torch.ones(2)
        assert self.masked_mae(pred, target, mask).item() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
