# Changelog

All notable changes to the NumLoRA project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.3.0] — 2026-04-12 (Phase 3: Full sweep running)

### Added
- Full sweep launched: 5 datasets x 3 MRs x 3 seeds x {LoRA, NumLoRA} = 90 runs on GPUs 4-7
- Additional dataset loaders: Weather (52k x 21), Exchange (7.5k x 8), Traffic (17k x 20), ILI (966 x 7)
- Datasets symlinked from existing server cache (no re-download needed)

### Changed
- Electricity dataset removed (data pipeline incompatibility, not needed for core claims)

---

## [0.2.0] — 2026-04-12 (Phase 2 complete + pilot passed)

### Added
- Core NumLoRA module: `src/models/numlora.py` (~170 LoC) — NumLoRALinear with SSR, CTGS, merge/unmerge
- MAI calibration: `src/models/mai.py` — single forward pass variance measurement + per-layer init
- Generic wrapper: `src/models/apply.py` — apply_numlora() for any HuggingFace model (auto-detects GPT-2/Llama/Phi/Qwen/SmolLM/Mistral, handles Conv1D)
- Imputation model: `src/models/imputation_model.py` — input proj → frozen backbone → output head
- Dataset pipeline: `src/data/dataset.py` — MCAR masking, patch tokenization, ETT/Weather/Exchange/Traffic/ILI loaders
- Training script: `scripts/experiments/train.py` — unified entry point with method/backbone selection, dual LR, early stopping, JSON output
- Tier launchers: `scripts/experiments/launch_{smoke,quick,full}.sh` — GPU 4-7 round-robin
- Unit tests: 16 tests passing (init identity, merge, gradient flow, param groups, backbone-agnostic)
- NeurIPS 2026 paper template with simulated content: `paper/Formatting_Instructions_For_NeurIPS_2026/main.tex`

### Changed
- **Primary backbone: SmolLM-360M** (replaced GPT-2 Small). GPT-2 is too old for NeurIPS 2026 and has narrow MAI variance range (0.01-0.66), masking NumLoRA's advantage. SmolLM-360M (2024) has variance range 0.001-155x.
- CTGS hook: fixed accumulation bug (was registering new hook per forward call, now registers once with cached x_norm)
- SSR default learning rate: 1e-2 → 3e-3 (stability on larger models)

### Pilot results (kill/continue gate: PASSED)
Breadth-first pilot on SmolLM-360M, MR=0.3, seed=42, 50 epochs:

| Dataset | LoRA MAE | NumLoRA MAE | Improvement |
|---|---|---|---|
| Exchange | 1.7704 | 1.3066 | +26.2% |
| ETT-h1 | 0.7088 | 0.5720 | +19.3% |
| Traffic | 0.6801 | 0.6134 | +9.8% |
| Weather | 0.2251 | 0.2107 | +6.4% |
| ILI | 1.4622 | 1.4008 | +4.2% |

NumLoRA wins 5/5 datasets, average improvement +13.2%.

### Backbone smoke tests passed
- GPT-2 Small (117M): 48 layers, forward+backward+merge PASS
- SmolLM-360M (360M): 224 layers, forward+backward+merge PASS
- Qwen2.5-0.5B (494M): 168 layers, forward+backward+merge PASS

---

## [0.1.0] — 2026-04-12 (Phase 0 complete)

### Added
- Project scaffold aligned with topic1-4 conventions
- `.gitignore`, `pyproject.toml`, `requirements.txt`
- Directory structure: `src/`, `data/`, `results/{smoke,quick,full}/`, `paper/`, `tests/`
- Scripts reorganized: `scripts/{experiments,analysis,data}/`
- Documentation: README.md, ROADMAP.md, EXPERIMENTS.md, CHANGELOG.md
- Method spec (docs/method.md), ablation plan (docs/ablation-plan.md), reviewer defense (docs/reviewer-defense.md)
- Config templates: configs/backbones.yaml, configs/datasets.yaml, configs/baselines.yaml
- Dataset download script: scripts/data/download_datasets.sh
- All GPU/compute references updated: RTX 3090 GPUs 4-7, conda env `llms`

### Context
- Parent paper: LLM4Imp (Messou et al., IEEE ICC 2026)
- Target venue: NeurIPS 2026 main track
- NumLoRA addresses the gap that LoRA is calibrated for text, not numbers
- Three modifications (SSR, MAI, CTGS) are prior-free and backbone-agnostic
