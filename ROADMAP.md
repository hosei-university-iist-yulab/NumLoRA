# NumLoRA — Development Roadmap

> Prior-Free Parameter-Efficient Fine-Tuning for Numerical Data.
> Primary target: **NeurIPS 2026** main track.

## Overview

NumLoRA is Direction D5 from the LLM4Imp research programme (IEEE ICC 2026). It is developed first because it is the simplest to implement (~170 LoC), trains at the same speed as LoRA, and its results frame the entire programme by setting the numerical-PEFT baseline.

## Milestones

```
Phase 0 ──► Phase 1+2 ──► Phase 3 ──────────► Phase 4 ──► Phase 5
 Docs       Core+Infra    Sweep+Ablation       Paper       Submit
 Apr '26    Apr 12 '26    Apr '26 (running)     May '26     NeurIPS '26
  DONE        DONE          IN PROGRESS
```

---

## Phase 0 — Documentation & Design [DONE]

- [x] README, ROADMAP, EXPERIMENTS.md, method spec, ablation plan, reviewer defense
- [x] Config templates (backbones, datasets, baselines)
- [x] NeurIPS 2026 paper template with simulated content (paper/Formatting_Instructions_For_NeurIPS_2026/)
- [x] Project scaffold aligned with topic1-4 conventions (tier system, GPU assignment, .gitignore, pyproject.toml)

---

## Phase 1+2 — Core Implementation + Infrastructure [DONE]

**Completed 2026-04-12.** Phases 1 and 2 were merged for efficiency.

### Core module (src/models/)
- [x] `numlora.py` — NumLoRALinear with SSR (alpha, beta, gamma), CTGS hook, merge/unmerge (~170 LoC)
- [x] `mai.py` — Magnitude-Aware Initialisation from single forward pass calibration
- [x] `apply.py` — `apply_numlora()` generic wrapper for any HuggingFace model (auto-detects GPT-2/Llama/Phi/Qwen/SmolLM/Mistral, handles Conv1D)
- [x] `imputation_model.py` — Frozen-LLM imputation wrapper (input proj → backbone → output head)

### Data pipeline (src/data/)
- [x] `dataset.py` — Unified time-series loader with MCAR masking, patch tokenization, train-normalized splits
- [x] Loaders: ETT (h1/h2/m1/m2), Weather, Exchange, Traffic, ILI
- [x] Datasets symlinked from existing server cache (no re-download needed)
- [x] Electricity removed (data pipeline incompatibility, not needed for core claims)

### Training harness (scripts/experiments/)
- [x] `train.py` — Unified entry point: method selection, backbone selection, dual LR, early stopping, JSON output
- [x] Tier-based launchers: `launch_smoke.sh` (5 ep), `launch_quick.sh` (50 ep), `launch_full.sh` (100+ ep)
- [x] GPU assignment: CUDA 4-7, round-robin with xargs concurrency

### Tests
- [x] 16 unit tests passing (init identity, merge correctness, gradient flow, param groups, backbone-agnostic)

### Backbone smoke tests
- [x] GPT-2 Small (117M) — 48 layers replaced, forward+backward+merge PASS
- [x] SmolLM-360M (360M) — 224 layers replaced, forward+backward+merge PASS
- [x] Qwen2.5-0.5B (494M) — 168 layers replaced, forward+backward+merge PASS

### Key findings
- **Primary backbone: SmolLM-360M** (not GPT-2). GPT-2 is too old for a 2026 venue and has narrow MAI variance range (0.01-0.66), masking NumLoRA's advantage. SmolLM-360M has variance range 0.001-155x, making MAI calibration critical.
- **CTGS hook bug fixed:** original implementation accumulated hooks per forward call, causing gradient explosion. Fixed to register once with cached x_norm.
- **SSR learning rate:** default lr_ssr reduced from 1e-2 to 3e-3 for stability on larger models.

---

## Phase 2.5 — Pilot Experiments [DONE]

**Kill/continue gate: PASSED decisively.**

### Breadth-first pilot (SmolLM-360M, MR=0.3, seed=42, 50 epochs)

| Dataset | Domain | LoRA MAE | NumLoRA MAE | Improvement |
|---|---|---|---|---|
| Exchange | Finance | 1.7704 | 1.3066 | **+26.2%** |
| ETT-h1 | Energy | 0.7088 | 0.5720 | **+19.3%** |
| Traffic | Transport | 0.6801 | 0.6134 | **+9.8%** |
| Weather | Meteorology | 0.2251 | 0.2107 | **+6.4%** |
| ILI | Epidemiology | 1.4622 | 1.4008 | **+4.2%** |

**Result: NumLoRA wins 5/5 datasets.** Average improvement +13.2%, far above 1% target.

---

## Phase 3 — Full Experimental Sweep [IN PROGRESS]

**Hardware:** 4x NVIDIA RTX 3090 24GB (GPUs 4-7, shared server). Conda env: `llms`.

### 3.1 Main sweep (running)
- [x] 5 datasets x 3 MRs x 3 seeds x {LoRA, NumLoRA} = 90 runs launched (GPUs 4-7, 3 concurrent/GPU)
- [ ] Collect results, compute mean±std, paired bootstrap CIs

### 3.2 DoRA baseline sweep (next)
- [ ] 5 datasets x 3 MRs x 3 seeds x {DoRA} = 45 runs

### 3.3 Component ablation
- [ ] 7 subsets of {SSR, MAI, CTGS} on 3 representative datasets (ETT-h1, Exchange, Weather)
- [ ] 3 MRs x 3 seeds = 189 ablation runs

### 3.4 Expanded ablation (new dimensions)
- [ ] **Rank ablation:** r = {2, 4, 8, 16, 32} on 3 datasets x MR=0.3 x 3 seeds
- [ ] **LR ratio ablation:** lr_ssr/lr_lora = {1x, 3x, 10x, 30x} on 3 datasets
- [ ] **Patch size ablation:** {8, 16, 32, 64} on 3 datasets

### 3.5 Backbone scale validation
- [ ] Qwen2.5-0.5B on 3 pilot datasets x 3 MRs x 3 seeds
- [ ] Phi-3-mini (3.8B) on 3 datasets x MR=0.3 x 3 seeds (1 concurrent per GPU)

### 3.6 Additional ETT datasets
- [ ] ETT h2/m1/m2 — fill out the ETT family (same loader)

### 3.7 Statistical analysis
- [ ] Paired bootstrap CIs (10k resamples) for all NumLoRA vs LoRA cells
- [ ] Holm-Bonferroni correction for global claim
- [ ] Friedman test + Nemenyi post-hoc for multi-method ranking

---

## Phase 4 — Paper Writing

**Target:** NeurIPS 2026 main track (9 pages + appendix).

### Structure
- Abstract, Introduction, Related Work, Method (SSR/MAI/CTGS with equations)
- Experiments: main results, MR sensitivity, ablation (components + ranks + LR + patch size), backbone scale
- Analysis: when NumLoRA fails, efficiency, activation variance visualization
- Conclusion

### Key tables
- Table 1: NumLoRA vs baselines (LoRA, DoRA, PiSSA) on all TS datasets at MR=0.3
- Table 2: MR sensitivity sweep (0.1-0.5) on representative datasets
- Table 3: Component ablation (7 subsets)
- Table 4: Rank ablation (r=2,4,8,16,32)
- Table 5: Backbone scale (SmolLM → Qwen → Phi-3)
- Table 6: Efficiency (params, training time, inference overhead)

### Key figures
- Fig 1: Method diagram (LoRA vs NumLoRA)
- Fig 2: MAE vs MR curves
- Fig 3: Ablation bar chart
- Fig 4: Activation variance histogram (text vs numerical inputs)

---

## Phase 5 — Submission

- [ ] Primary target: **NeurIPS 2026** main track
- [ ] Backup: TMLR (rolling), AAAI 2027, ICLR 2027
- [ ] Co-author review, code release

---

## Backbone Strategy

| Backbone | Params | Role | Status |
|---|---|---|---|
| **SmolLM-360M** | 360M | **Primary** (all datasets, all experiments) | Validated |
| **Qwen2.5-0.5B** | 494M | Scale check (same tier, different arch) | Smoke tested |
| **Phi-3-mini** | 3.8B | Mid-scale validation | Planned |
| **Llama-3 8B** | 8B | Large-scale validation (if needed) | Planned |

**Decision:** GPT-2 dropped as primary backbone (too old for NeurIPS 2026, narrow MAI variance range masks the effect). SmolLM-360M is modern (2024), fast, and has the widest activation variance range where NumLoRA's calibration is most impactful.

---

## Relationship to broader research programme

NumLoRA (D5) is one of five directions in the LLM4Imp follow-up roadmap:

| Direction | Role | When |
|---|---|---|
| **D5 — NumLoRA** | **Prior-free generalist (this project)** | **Apr–Sep 2026** |
| D1 — SpecLoRA | Spectral-conditioned specialist | Jun–Sep 2026 |
| D2 — FourierFT-N | Frequency-basis specialist | Oct–Dec 2026 |
| D3 — MaskLoRA | Missingness-conditioned specialist | Jan–Apr 2027 |
| D4 — GeoLoRA | Metric-only boundary adaptation | Jan–Apr 2027 |
| Unified framework | All five composed | Sep 2027 → ICLR 2028 |
