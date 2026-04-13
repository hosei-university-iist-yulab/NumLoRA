# Changelog

All notable changes to the NumLoRA project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.4.0] — 2026-04-13

### Added
- Multi-task support: `--task imputation|forecasting|classification` in train.py
- Forecasting task: ETT short/medium/long horizons (96/192/336) with LLMForecastingModel
- Classification task: UCR dataset loader with LLMClassificationModel (mean-pool + MLP)
- MRE (Mean Relative Error) metric added to all evaluations alongside MAE and MSE
- Task-specific output directories: `results/full/{imputation,forecasting,classification,ablation}/`
- Architecture diagram specifications in `docs/architecture.md`
- Clean sweep launcher `scripts/experiments/launch_clean_sweep.sh` with OOM-aware GPU allocation

### Changed
- Paper narrative pivoted to CTGS-focused: SSR and MAI demoted to "investigated but not recommended"
- All previous results cleared for clean re-run with 0% bias

---

## [0.3.0] — 2026-04-08

### Added
- Full experimental sweep: 4 backbones x 5 datasets x 3 MRs x 3 seeds
- Component ablation: 7 subsets of {SSR, MAI, CTGS} on SmolLM and TinyLlama
- DoRA baseline comparison (killed after confirming worse than LoRA on numerical data)
- Publication-quality figure generation matching topic4 style (bold, serif, 600 dpi)
- Results aggregation script: `scripts/analysis/generate_tables.py` (LaTeX tables + PDF figures)
- Radar charts, MR curves, ablation bars, efficiency plots
- NeurIPS 2026 paper draft with real results (9 pages, Algorithm 1, 3 figures)

### Key findings
- SmolLM-360M: NumLoRA wins 12/15 conditions (+8.3% avg MAE over LoRA)
- CTGS alone (+8.6%) outperforms full NumLoRA (+0.7%) — SSR interferes with CTGS
- DoRA degrades on numerical data, confirming text-PEFT innovations do not transfer
- Qwen-0.5B: mixed results (6/15 wins), SSR lr needs per-architecture tuning

---

## [0.2.0] — 2026-04-02

### Added
- Core NumLoRA module: `src/models/numlora.py` (~170 LoC) with SSR, CTGS, merge/unmerge
- MAI calibration: `src/models/mai.py` — single forward pass variance measurement
- Generic wrapper: `src/models/apply.py` — apply_numlora() for any HuggingFace model
- Auto-detection for GPT-2 (Conv1D), Llama, Phi, Qwen, SmolLM, Mistral architectures
- Imputation model: `src/models/imputation_model.py` — input proj -> backbone -> output head
- Dataset pipeline: `src/data/dataset.py` — MCAR masking, patch tokenization, ETT/Weather/Exchange/Traffic/ILI loaders
- Training script: `scripts/experiments/train.py` — unified entry with method/backbone selection, dual LR, early stopping
- Tier-based launchers: smoke (5 ep), quick (50 ep), full (100 ep)
- 16 unit tests (init identity, merge correctness, gradient flow, param groups, backbone-agnostic)
- Backbone smoke tests: GPT-2 (48 layers), SmolLM (224 layers), Qwen (168 layers) — all pass

### Changed
- Primary backbone: SmolLM-360M (GPT-2 dropped — too old, narrow MAI variance range)
- CTGS hook: fixed accumulation bug (was registering new hook per forward, now registers once)
- SSR learning rate: 1e-2 -> 3e-3 (stability on larger models)

### Pilot results (kill/continue gate: PASSED)
- SmolLM-360M breadth pilot: 5/5 datasets won (+4.2% to +26.2% over LoRA)

---

## [0.1.0] — 2026-03-24

### Added
- Project scaffold aligned with topic1-4 conventions
- `.gitignore`, `pyproject.toml`, `requirements.txt`
- Directory structure: `src/`, `data/`, `results/{smoke,quick,full}/`, `tests/`
- Scripts: `scripts/{experiments,analysis,data}/`
- Documentation: README.md, ROADMAP.md, EXPERIMENTS.md
- Method specification: `docs/method.md`
- Ablation plan: `docs/ablation-plan.md`
- Reviewer defense: `docs/reviewer-defense.md`
- Config templates: `configs/{backbones,datasets,baselines}.yaml`
- Dataset download script skeleton
- NeurIPS 2026 paper template

### Context
- Parent paper: LLM4Imp (Messou et al., IEEE ICC 2026)
- Direction D5 from the LLM4Imp future-directions roadmap
- Three modifications proposed: SSR, MAI, CTGS — all prior-free and backbone-agnostic
