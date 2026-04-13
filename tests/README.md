# NumLoRA Test Suite

Comprehensive tests for all NumLoRA components. Designed so that contributors
can validate changes without breaking existing functionality.

## Running Tests

```bash
# Run all tests
pytest tests/test_numlora.py -v

# Run a specific test class
pytest tests/test_numlora.py::TestNumLoRALinear -v

# Run a single test
pytest tests/test_numlora.py::TestMerge::test_merge_produces_same_output -v

# Run with coverage
pytest tests/test_numlora.py -v --cov=src --cov-report=term-missing
```

## Test Structure

Each test class maps to a specific source module:

| Test Class | Source Module | What It Tests |
|---|---|---|
| `TestNumLoRALinear` | `src/models/numlora.py` | Layer construction, init-identity, output shape, param counts, SSR/CTGS toggles |
| `TestMerge` | `src/models/numlora.py` | Weight merging for zero-overhead inference, unmerge restore, idempotency |
| `TestMAI` | `src/models/mai.py` | Magnitude-Aware Initialisation, inverse scaling, B reset |
| `TestApplyNumLoRA` | `src/models/apply.py` | Generic wrapper, target module replacement, param grouping, freezing |
| `TestGradientFlow` | `src/models/numlora.py` | CTGS backward hook, gradient flow to LoRA params, frozen weight isolation |
| `TestImputationModel` | `src/models/imputation_model.py` | Output shape (batch, window, features), backward pass |
| `TestForecastingModel` | `src/models/forecasting_model.py` | Output shape (batch, horizon, features), multiple horizons |
| `TestClassificationModel` | `src/models/classification_model.py` | Logit shape (batch, n_classes), cross-entropy backward |
| `TestDataset` | `src/data/dataset.py` | Windowing, masking rate, patch shape, deterministic seeds, ETT loader |
| `TestForecastingData` | `src/data/forecasting.py` | Lookback/horizon split, short/medium/long horizons |
| `TestClassificationData` | `src/data/classification.py` | Patch shape, label types, synthetic data (no download needed) |
| `TestMetrics` | `scripts/experiments/train.py` | Masked MAE/MSE/MRE correctness on known inputs, edge cases |

## Key Design Principles

1. **No GPU required.** All tests use CPU with small tensors and mock backbones.
   The mock backbone mimics HuggingFace output structure (`hidden_states`) without
   loading a real LLM.

2. **No data download required.** Dataset tests use synthetic numpy arrays.
   ETT-specific tests skip automatically if the CSV is not on disk
   (`pytest.skip("ETT data not available")`).

3. **No code modification.** Tests validate the public interface of each module
   without monkey-patching or modifying production code. If a test fails, the
   production code has a bug — not the test.

4. **Deterministic.** All random operations use fixed seeds. Running the same
   test twice produces the same result.

## File Dependencies

```
test_numlora.py
├── src/models/numlora.py       (NumLoRALinear, merge, CTGS hook)
├── src/models/mai.py           (calibrate_mai, collect_activation_variances)
├── src/models/apply.py         (apply_numlora, get_numlora_params, count_params)
├── src/models/imputation_model.py    (LLMImputationModel)
├── src/models/forecasting_model.py   (LLMForecastingModel)
├── src/models/classification_model.py (LLMClassificationModel)
├── src/data/dataset.py         (TimeSeriesImputationDataset, load_ett)
├── src/data/forecasting.py     (ForecastingDataset, load_ett_forecasting)
├── src/data/classification.py  (TSClassificationDataset)
└── scripts/experiments/train.py (masked_mae, masked_mse, masked_mre, forecast_*)
```

## Adding New Tests

When adding a new module, create a corresponding test class following the pattern:

```python
class TestNewModule:
    """Tests for NewModule (src/path/to/module.py).

    Validates: [list what this class tests].
    """

    def test_basic_functionality(self):
        """One-line description of what this test checks."""
        # Arrange
        module = NewModule(...)
        input = torch.randn(...)

        # Act
        output = module(input)

        # Assert
        assert output.shape == expected_shape
```

Keep tests small, focused, and independent. Each test should verify one behaviour.
