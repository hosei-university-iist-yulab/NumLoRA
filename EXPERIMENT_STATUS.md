# NumLoRA Experiment Status

**Last updated:** 2026-04-13
**Total planned:** ~945 runs (main sweeps + CTGS-only + ablation)

## Main Sweep: 4 Backbones x 5 Datasets x 3 MRs x 3 Seeds x 3 Methods (LoRA, NumLoRA-full, CTGS-only)

|                | SmolLM-360M (3/GPU) | Qwen-0.5B (2/GPU) | TinyLlama-1.1B (3/GPU) | Phi-3-mini-3.8B (1/GPU) |
|----------------|---------------------|--------------------|------------------------|-------------------------|
| **ETT-h1**     | ✔                   | ✔                  | RUNNING                | RUNNING                 |
| **ETT-h2**     | QUEUED              | QUEUED             | QUEUED                 | QUEUED                  |
| **ETT-m1**     | QUEUED              | QUEUED             | QUEUED                 | QUEUED                  |
| **ETT-m2**     | QUEUED              | QUEUED             | QUEUED                 | QUEUED                  |
| **Weather**    | ✔                   | ✔                  | RUNNING                | RUNNING                 |
| **Exchange**   | ✔                   | ✔                  | RUNNING                | RUNNING                 |
| **Traffic**    | ✔                   | ✔                  | RUNNING                | RUNNING                 |
| **ILI**        | ✔                   | ✔                  | RUNNING                | RUNNING                 |

**CTGS-only** sweep added for all 4 backbones (chained after Phi-3 for SmolLM/Qwen).

## Ablation Studies

| Ablation | Backbone | Datasets | Runs | Status |
|----------|----------|----------|------|--------|
| Component (7 subsets of {SSR, MAI, CTGS}) | SmolLM-360M | ETT-h1, Exchange, Weather | 189 | ✔ DONE |
| Component (7 subsets of {SSR, MAI, CTGS}) | TinyLlama-1.1B | ETT-h1, Exchange, Weather | 189 | RUNNING |
| LR ratio (lr_ssr = 1x, 3x, 10x) | SmolLM-360M | ETT-h1, Exchange, Weather | 27 | QUEUED |

**Key finding from SmolLM ablation (MR=0.3):**
CTGS alone (+8.6% vs LoRA) outperforms full NumLoRA (+0.7%). SSR degrades when combined with CTGS. TinyLlama ablation will validate whether this is SmolLM-specific or universal.

## Completed Results

### SmolLM-360M: NumLoRA-full vs LoRA (12/15 wins, +8.3% avg)

| Dataset    | MR=0.1       | MR=0.3       | MR=0.5       |
|------------|--------------|--------------|--------------|
| ETT-h1     | +21.2% WIN   | -4.9% loss   | +15.8% WIN   |
| Weather    | -6.5% loss   | +2.8% WIN    | +1.5% WIN    |
| Exchange   | +18.1% WIN   | +2.9% WIN    | -5.0% loss   |
| Traffic    | +19.4% WIN   | +16.4% WIN   | +12.7% WIN   |
| ILI        | +14.0% WIN   | +12.2% WIN   | +11.8% WIN   |

### Qwen-0.5B: NumLoRA-full vs LoRA (6/15 wins — mixed, lr_ssr needs tuning)

| Dataset    | MR=0.1       | MR=0.3       | MR=0.5       |
|------------|--------------|--------------|--------------|
| ETT-h1     | -19.1% loss  | +1.1% WIN    | -24.4% loss  |
| Weather    | -2.2% loss   | -23.7% loss  | -38.0% loss  |
| Exchange   | -30.6% loss  | -22.9% loss  | -41.1% loss  |
| Traffic    | +16.7% WIN   | +6.1% WIN    | -11.8% loss  |
| ILI        | +32.8% WIN   | +31.3% WIN   | +18.5% WIN   |

### SmolLM Component Ablation (MR=0.3, avg over 3 datasets)

| Variant          | Avg MAE | vs LoRA |
|------------------|---------|---------|
| LoRA (baseline)  | 0.7799  | --      |
| CTGS only        | 0.7126  | +8.6%   |
| MAI only         | 0.7257  | +7.0%   |
| SSR only         | 0.7377  | +5.4%   |
| MAI + CTGS       | 0.7126  | +8.6%   |
| MAI + SSR        | 0.7377  | +5.4%   |
| SSR + CTGS       | 0.7743  | +0.7%   |
| NumLoRA (all 3)  | 0.7743  | +0.7%   |

### DoRA (killed — confirmed worse than LoRA on numerical data)

DoRA worse than LoRA at MR=0.5 on ETT-h1 (0.7357 vs 0.6915). Sweep killed.

## Current GPU Assignment

- GPUs 4-5: Phi-3-mini (135 jobs, 1 concurrent/GPU)
- GPUs 6-7: TinyLlama (316 jobs, 3 concurrent/GPU)
- Chained: SmolLM/Qwen CTGS-only (90 jobs) auto-starts after Phi-3

## Hardware

- 4x NVIDIA RTX 3090 24GB (GPUs 4-7)
- Conda env: `llms`
- Server: shared, GPUs 0-3 not used by this project
