# NumLoRA — Ablation Study Plan

## Purpose

Determine which of NumLoRA's three modifications (SSR, MAI, CTGS) carries the most weight. If one modification alone explains most of the gain, that simplifies the story and maximises impact.

## Ablation matrix

| ID | Variant | SSR | MAI | CTGS | Params (L=12, d=768, r=8) |
|---|---|---|---|---|---|
| A0 | LoRA (baseline) | - | - | - | 147,456 |
| A1 | LoRA + MAI | - | yes | - | 147,456 |
| A2 | LoRA + SSR | yes | - | - | 175,104 |
| A3 | LoRA + CTGS | - | - | yes | 147,468 |
| A4 | LoRA + MAI + SSR | yes | yes | - | 175,104 |
| A5 | LoRA + MAI + CTGS | - | yes | yes | 147,468 |
| A6 | LoRA + SSR + CTGS | yes | - | yes | 175,116 |
| A7 | **NumLoRA (full)** | **yes** | **yes** | **yes** | **175,116** |

## Datasets

3 representative datasets chosen to cover different failure modes:

| Dataset | Why chosen |
|---|---|
| **PhysioNet 2012** | Clinical, heterogeneous, high-dim (D=35). Tests whether SSR handles diverse variable scales. |
| **Solar Alabama** | High dynamic range (0–1400 W/m^2). Tests whether CTGS prevents gradient domination. |
| **Exchange Rate** | Aperiodic, low-dim, near-random-walk. Adversarial for any spectral method — tests whether NumLoRA wins even here. |

## Missing rates

3 levels: MR in {0.1, 0.3, 0.5} — covering low, medium, high missingness.

## Seeds

3 seeds (42, 123, 456) per configuration.

## Total runs

8 variants x 3 datasets x 3 MRs x 3 seeds = **216 runs**
At ~15-20 min/RTX 3090 each: **~54-72 GPU-hours on RTX 3090**

## Analysis plan

### 1. Contribution table

For each dataset x MR, compute:
```
delta_MAI  = MAE(A0) - MAE(A1)     # improvement from MAI alone
delta_SSR  = MAE(A0) - MAE(A2)     # improvement from SSR alone
delta_CTGS = MAE(A0) - MAE(A3)     # improvement from CTGS alone
delta_full = MAE(A0) - MAE(A7)     # improvement from all three
```

Report: what fraction of delta_full does each single modification explain?

### 2. Interaction effects

If delta_MAI + delta_SSR + delta_CTGS > delta_full: modifications overlap (redundant).
If delta_MAI + delta_SSR + delta_CTGS < delta_full: modifications synergise (super-additive).

### 3. Key questions to answer

| Question | How to answer | Implication |
|---|---|---|
| Is MAI alone enough? | Compare A1 vs A7 | If gap < 0.5%, headline = "fix the init" |
| Does SSR help without MAI? | Compare A2 vs A0 | If yes, SSR is independently valuable |
| Is CTGS essential for high-range data? | Compare Solar: A3 vs A0, A7 vs A4 | If CTGS-off drops Solar but not Exchange, it's magnitude-specific |
| Do modifications compose? | Compare sum of singles vs A7 | Sub- or super-additive? |
| Is the param overhead the story? | Compare A7 vs LoRA r=9 (A0 at r=9) | If LoRA r=9 = NumLoRA r=8, it's just capacity |

### 4. Visualisation

- Bar chart: 8 variants on x-axis, MAE on y-axis, one panel per dataset
- Heatmap: 8 variants x 3 datasets, colour = delta from LoRA baseline
- Line plot: MAE vs MR for each variant, showing degradation curves

## Possible outcomes and their headlines

| Outcome | Paper headline |
|---|---|
| MAI alone explains 80%+ | "LoRA's initialisation is wrong for numbers — a one-line fix" |
| SSR + MAI together explain 90%+ | "Affine calibration is all you need for numerical LoRA" |
| All three are needed (each contributes 30%+) | "Three complementary fixes for the text-to-numbers gap in LoRA" |
| No single modification helps much, but combined they do | "NumLoRA's strength is synergy, not any one trick" |
| NumLoRA barely beats LoRA overall | Negative result: "LoRA transfers well to numbers despite the modality gap" (still publishable) |
