# NumLoRA: Calibrating Low-Rank Adaptation for Continuous-Valued Inputs

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**LoRA was not made for numbers.** Its gradient flow is calibrated for text token embeddings, where each token is a sparse one-hot lookup producing bounded gradients. When frozen LLMs are repurposed for numerical tasks (time-series imputation, forecasting, classification), the continuous input projections produce dense gradients whose magnitude scales with the patch embedding norm, causing training instability.

**NumLoRA** introduces **Continuous-Token Gradient Scaling (CTGS)**: a single learnable scalar per layer that normalises LoRA gradients by the input activation norm during the backward pass. CTGS adds just *L* parameters (e.g., 32 for a 32-layer model), incurs **zero inference overhead**, and requires **no domain-specific priors**.

## Key Results

On **SmolLM-360M** across 5 time-series imputation benchmarks at 3 missing rates (15 conditions):

| Dataset | MR=0.1 | MR=0.3 | MR=0.5 |
|---------|--------|--------|--------|
| ETT-h1 (energy) | **+21.2%** | -4.9% | **+15.8%** |
| Weather (meteorological) | -6.5% | **+2.8%** | **+1.5%** |
| Exchange (financial) | **+18.1%** | **+2.9%** | -5.0% |
| Traffic (transport) | **+19.4%** | **+16.4%** | **+12.7%** |
| ILI (epidemiological) | **+14.0%** | **+12.2%** | **+11.8%** |

**NumLoRA wins 12/15 conditions** (+8.3% average MAE improvement). Traffic and ILI are clean sweeps.

### Why CTGS Is the Key

Component ablation on SmolLM-360M (MR=0.3, averaged over ETT-h1, Exchange, Weather):

| Variant | vs LoRA | Extra params |
|---------|---------|-------------|
| **CTGS only** | **+8.6%** | **32** |
| MAI only | +7.0% | 0 |
| SSR only | +5.4% | 27,648 |
| Full NumLoRA (all 3) | +0.7% | 27,680 |

CTGS alone outperforms the full three-component method. Gradient instability, not scale miscalibration, is the dominant failure mode of LoRA on numerical data.

### DoRA Fails on Numbers

DoRA (2024 text-PEFT SOTA) performs *worse* than vanilla LoRA on numerical benchmarks (e.g., ETT-h1 MR=0.5: DoRA 0.736 vs LoRA 0.692), confirming that text-optimised PEFT innovations do not transfer.

## Method

Standard LoRA forward: `h = W₀x + BAx`

CTGS adds one backward-pass modification:

```
∇_A ← ∇_A · c / (||x|| + ε)
```

where `c` is a learnable scalar per layer (init 1.0) and `ε = 1e-8`. This normalises gradient magnitude by the input activation norm, preventing high-magnitude numerical patches from dominating updates. At inference, the hook is inactive and weights merge as standard LoRA — zero overhead.

## Supported Tasks

NumLoRA supports three downstream tasks out of the box:

| Task | Datasets | Metrics |
|------|----------|---------|
| **Imputation** | ETT (h1/h2/m1/m2), Weather, Exchange, Traffic, ILI | MAE, MSE, MRE |
| **Forecasting** | ETT short/medium/long (horizon 96/192/336) | MAE, MSE, MRE |
| **Classification** | UCR archive (ECG200, FordA, Wafer, etc.) | Accuracy |

## Backbones

| Backbone | Parameters | Architecture |
|----------|-----------|-------------|
| SmolLM-360M | 360M | Llama-family (2024) |
| Qwen2.5-0.5B | 494M | GQA attention |
| TinyLlama-1.1B | 1.1B | Llama-family |
| Phi-3-mini | 3.8B | Phi-family |

## Quick Start

```bash
pip install -r requirements.txt

# Apply NumLoRA (CTGS) to any HuggingFace model
from src.models.apply import apply_numlora
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")
replaced = apply_numlora(model, rank=8)  # Auto-detects architecture
```

```bash
# Imputation
python scripts/experiments/train.py --task imputation --dataset ett_h1 --missing-rate 0.3

# Forecasting (short/medium/long term)
python scripts/experiments/train.py --task forecasting --dataset ett_h1_96
python scripts/experiments/train.py --task forecasting --dataset ett_h1_336

# Classification
python scripts/experiments/train.py --task classification --dataset ecg200

# Full clean sweep (all backbones, all tasks)
bash scripts/experiments/launch_clean_sweep.sh
```

```bash
# Generate paper tables and figures from results
python scripts/analysis/generate_tables.py
```

## Project Structure

```
NumLoRA/
├── src/
│   ├── models/
│   │   ├── numlora.py              # Core NumLoRALinear (SSR + CTGS + merge)
│   │   ├── mai.py                  # Magnitude-Aware Initialisation
│   │   ├── apply.py                # apply_numlora() for any HF model
│   │   ├── imputation_model.py     # Frozen-LLM imputation wrapper
│   │   ├── forecasting_model.py    # Frozen-LLM forecasting wrapper
│   │   └── classification_model.py # Frozen-LLM classification wrapper
│   ├── data/
│   │   ├── dataset.py              # Imputation loaders + MCAR masking
│   │   ├── forecasting.py          # Forecasting loaders (ETT horizons)
│   │   └── classification.py       # Classification loaders (UCR)
│   ├── baselines/
│   └── utils/
├── scripts/
│   ├── experiments/
│   │   ├── train.py                # Unified entry (--task imputation|forecasting|classification)
│   │   ├── launch_clean_sweep.sh   # Full production sweep
│   │   ├── launch_smoke.sh         # Quick validation
│   │   └── launch_quick.sh         # Intermediate check
│   └── analysis/
│       └── generate_tables.py      # LaTeX tables + publication figures
├── configs/                        # Backbone, dataset, baseline configs
├── tests/                          # 16 unit tests
├── docs/
│   ├── method.md                   # Mathematical specification
│   ├── ablation-plan.md            # Ablation study design
│   └── architecture.md             # Figure drawing specifications
├── EXPERIMENT_STATUS.md            # Live experiment tracker
├── ROADMAP.md                      # Development roadmap
└── CHANGELOG.md                    # Version history
```

## Datasets

| Dataset | Domain | Features | Length | Task |
|---------|--------|----------|--------|------|
| ETT-h1/h2 | Electrical load | 7 | 17,420 | Imputation + Forecasting |
| ETT-m1/m2 | Electrical load | 7 | 69,680 | Imputation + Forecasting |
| Weather | Meteorology | 21 | 52,696 | Imputation |
| Exchange Rate | Finance | 8 | 7,588 | Imputation |
| Traffic | Freeway flow | 862 | 17,544 | Imputation |
| ILI | Epidemiology | 7 | 966 | Imputation |

## Related Work

This project is part of the LLM-for-Time-Series research programme at YuLab, Hosei University:

- **LLM4Imp** — Leveraging Frozen LLMs with Spectral Prompts for Time-Series Imputation (IEEE ICC 2026)
- **Spec2Llama** — Spectral-Aware Forecasting for Versatile Time Series Analysis (Expert Systems with Applications, 2026)
- **Spec2LLM** — Spectral-to-Language Reprogramming for Power Transformer Forecasting (Preprint)
- **Federated LoRA** — Federated Fine-Tuning of LLMs for Intelligent Automotive Systems (IEEE VTC-Spring 2025)

See also: [smartgrid-coopt](https://github.com/mesabo/smartgrid-coopt) | [Google Scholar](https://scholar.google.com/scholar?q=Franck+Junior+Aboya+Messou)

## Citation

```bibtex
@article{messou2026numlora,
  title={NumLoRA: Calibrating Low-Rank Adaptation for Continuous-Valued Inputs},
  author={Messou, Franck Junior Aboya and Liu, Tong and Wang, Weiyu
          and Chen, Jinhua and Zhang, Shilong and Yu, Tao and Yu, Keping},
  year={2026},
  note={Under review}
}
```

## Acknowledgments

Computational resources provided by [Hosei University](https://www.hosei.ac.jp/) through the Network Intelligence and Security Laboratory (YuLab), Graduate School of Science and Engineering.

## Contact

**Franck Junior Aboya Messou** — franckjunioraboya.messou@ieee.org

## License

[Apache License 2.0](LICENSE)
