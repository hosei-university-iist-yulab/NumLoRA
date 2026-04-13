# NumLoRA: Calibrating Low-Rank Adaptation for Continuous-Valued Inputs

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**NumLoRA** fixes a fundamental problem: LoRA was designed for text, but frozen LLMs are increasingly used for numerical tasks (time-series imputation, tabular regression, scientific computing). Text tokens produce sparse, bounded gradients through embedding lookups. Numerical patches produce dense gradients through continuous projections, whose magnitude scales with the input norm. This mismatch causes training instability and suboptimal adaptation.

NumLoRA introduces **Continuous-Token Gradient Scaling (CTGS)**, a single learnable scalar per layer that normalises LoRA gradients by the input activation norm during the backward pass. CTGS adds just *L* parameters (e.g., 32 for a 32-layer model), incurs **zero inference overhead** (backward-pass only), and requires **no domain-specific priors**.

## Key Results

On **SmolLM-360M** across 5 time-series benchmarks at 3 missing rates (15 conditions total):

| Dataset | MR=0.1 | MR=0.3 | MR=0.5 |
|---------|--------|--------|--------|
| ETT-h1 (energy) | **+21.2%** | -4.9% | **+15.8%** |
| Weather (meteorological) | -6.5% | **+2.8%** | **+1.5%** |
| Exchange (financial) | **+18.1%** | **+2.9%** | -5.0% |
| Traffic (transport) | **+19.4%** | **+16.4%** | **+12.7%** |
| ILI (epidemiological) | **+14.0%** | **+12.2%** | **+11.8%** |

**NumLoRA wins 12/15 conditions** with an average improvement of +8.3% MAE over vanilla LoRA. Losses are small (-5% to -7%) with overlapping error bars. Traffic and ILI are clean sweeps.

**DoRA** (2024 text-PEFT SOTA) performs *worse* than vanilla LoRA on numerical data, confirming that text-optimised PEFT innovations do not transfer.

### Ablation: CTGS Is the Key

Component ablation on SmolLM-360M (ETT-h1, Exchange, Weather at MR=0.3):

| Variant | vs LoRA | Extra params |
|---------|---------|-------------|
| CTGS only | **+8.6%** | 32 |
| MAI only | +7.0% | 0 |
| SSR only | +5.4% | 27,648 |
| Full NumLoRA (all 3) | +0.7% | 27,680 |

**CTGS alone outperforms the full three-component method.** Gradient instability, not scale miscalibration, is the dominant failure mode of LoRA on numerical data.

## Why This Matters

Every existing LoRA variant (LoRA, DoRA, PiSSA, VeRA, QLoRA, AdaLoRA) was designed and evaluated exclusively on text or vision-language tasks. None analysed whether LoRA's gradient flow is appropriate for continuous-valued embeddings. NumLoRA is the first to identify and fix this gap.

## Method

For each adapted layer, standard LoRA computes: `h = W₀x + BAx`

CTGS adds one backward-pass modification:

```
∇_A ← ∇_A · c / (||x|| + ε)
```

where `c` is a learnable scalar per layer (init 1.0) and `ε = 1e-8`. This normalises gradient magnitude by the input activation norm, preventing high-magnitude numerical patches from dominating updates. At inference, the hook is inactive and weights merge as standard LoRA.

## Backbones Tested

| Backbone | Parameters | Status |
|----------|-----------|--------|
| SmolLM-360M | 360M | Full results (12/15 wins) |
| Qwen2.5-0.5B | 494M | Full results (6/15 wins) |
| TinyLlama-1.1B | 1.1B | Running |
| Phi-3-mini | 3.8B | Running |

## Quick Start

```bash
# Install
conda activate llms  # or your environment
pip install -r requirements.txt

# Apply NumLoRA to any HuggingFace model
from src.models.apply import apply_numlora
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")
replaced = apply_numlora(model, rank=8)  # Auto-detects architecture
# That's it. Train as usual with standard LoRA training loop.
```

```bash
# Run experiments
bash scripts/experiments/launch_smoke.sh      # 5 epochs, ~10 min
bash scripts/experiments/launch_quick.sh      # 50 epochs, ~2 hours
bash scripts/experiments/launch_full.sh all   # Full sweep, ~hours
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
│   │   ├── numlora.py          # Core NumLoRALinear (SSR + CTGS + merge)
│   │   ├── mai.py              # Magnitude-Aware Initialisation
│   │   ├── apply.py            # apply_numlora() for any HF model
│   │   └── imputation_model.py # Frozen-LLM imputation wrapper
│   ├── data/
│   │   └── dataset.py          # TS loaders + MCAR masking
│   ├── baselines/
│   └── utils/
├── scripts/
│   ├── experiments/            # Tier launchers (smoke/quick/full)
│   │   ├── train.py            # Unified training entry point
│   │   ├── launch_smoke.sh
│   │   ├── launch_quick.sh
│   │   └── launch_full.sh
│   ├── analysis/
│   │   └── generate_tables.py  # LaTeX tables + figures from JSON
│   └── data/
│       └── download_datasets.sh
├── configs/                    # Backbone, dataset, baseline configs
├── tests/                      # 16 unit tests
├── results/                    # Experiment outputs (JSON per run)
├── docs/                       # Method spec, ablation plan
├── EXPERIMENT_STATUS.md        # Live experiment tracker
└── CHANGELOG.md
```

## Datasets

| Dataset | Domain | Features | Length | Source |
|---------|--------|----------|--------|--------|
| ETT-h1 | Electrical load | 7 | 17,420 | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| Weather | Meteorology | 21 | 52,696 | Max Planck Jena |
| Exchange Rate | Finance | 8 | 7,588 | [Lai et al.](https://github.com/laiguokun/multivariate-time-series-data) |
| Traffic | Freeway flow | 862 | 17,544 | PeMS |
| ILI | Epidemiology | 7 | 966 | CDC |

## Related Work

This project is part of the LLM-for-Time-Series research programme at YuLab, Hosei University:

- **LLM4Imp** — Leveraging Frozen LLMs with Spectral Prompts for Time-Series Imputation (IEEE ICC 2026)
- **Spec2Llama** — Spectral-Aware Forecasting for Versatile Time Series Analysis (Expert Systems with Applications, 2026)
- **Spec2LLM** — Spectral-to-Language Reprogramming for Power Transformer Forecasting (Preprint)
- **Federated LoRA** — Federated Fine-Tuning of LLMs for Intelligent Automotive Systems (IEEE VTC-Spring 2025)

See also our smart grid co-optimisation framework: [smartgrid-coopt](https://github.com/mesabo/smartgrid-coopt)

For the full list of publications: [Google Scholar](https://scholar.google.com/scholar?q=Franck+Junior+Aboya+Messou)

## Citation

```bibtex
@article{messou2026numlora,
  title={NumLoRA: Calibrating Low-Rank Adaptation for Continuous-Valued Inputs},
  author={Messou, Franck Junior Aboya and Liu, Tong and Wang, Weiyu
          and Chen, Jinhua and Zhang, Shilong and Yu, Tao and Yu, Keping},
  year={2026},
  note={Manuscript in preparation}
}
```

## Acknowledgments

This work is supported by computational resources provided by [Hosei University](https://www.hosei.ac.jp/) through the Network Intelligence and Security Laboratory (YuLab), Graduate School of Science and Engineering.

## Contact

**Franck Junior Aboya Messou** — franckjunioraboya.messou@ieee.org

## License

[Apache License 2.0](LICENSE)
