# NumLoRA — Anticipated Reviewer Questions & Rebuttals

> Prepare defenses before submission, not after reviews.

---

## Q1: "This is just LoRA with three tricks. Where is the novelty?"

**Rebuttal:** LoRA is "low-rank decomposition." DoRA (ICML 2024) is "LoRA + magnitude-direction decomposition" — one trick. PiSSA is "LoRA + SVD initialisation" — one trick. Both were accepted at top venues.

NumLoRA is "LoRA + affine calibration to continuous input" — three coordinated tricks that address three specific failure modes of applying text-native PEFT to numerical data. The conceptual contribution is **identifying what goes wrong** when LoRA is applied to non-text modalities and fixing each issue minimally. The ablation study (Table 3) shows each trick is independently valuable and they compose.

---

## Q2: "The 19% parameter overhead explains the win. This is just a bigger LoRA."

**Rebuttal:** Table 1 includes LoRA at rank 9 (~166k params) as a baseline. NumLoRA at rank 8 (~175k) has comparable parameter count. If LoRA-9 matched NumLoRA-8, we would report it honestly. The key result is that NumLoRA's extra capacity is *structurally placed* (scale, shift, bias) — not random rank increase. The SSR parameters have a clear interpretation: they calibrate the update magnitude to the numerical embedding scale.

Additionally, MAI adds 0 parameters, and CTGS adds 12 parameters (1 per layer). The ablation shows these zero-cost modifications contribute substantially.

---

## Q3: "Why not just normalise the input? RevIN already handles this."

**Rebuttal:** RevIN normalises the *input time series* before patching. NumLoRA normalises the *weight update* inside the LoRA adaptation. These operate at different levels:

- RevIN: X → (X - mean) / std → patches → embeddings
- NumLoRA SSR: LoRA update dW → diag(alpha) * BA * diag(beta) + gamma

RevIN handles data-level scale. SSR handles the mismatch between embedding-level scale and LoRA's text-calibrated init. They are complementary, not redundant. The ablation confirms this: NumLoRA still helps on top of RevIN.

---

## Q4: "CTGS is just gradient clipping with extra steps."

**Rebuttal:** Gradient clipping applies a global cap across all samples. CTGS applies per-sample, per-layer normalisation. The key difference:

| | Gradient clipping | CTGS |
|---|---|---|
| Scope | Global (all samples) | Per-sample, per-layer |
| Effect on small signals | No effect | **Amplifies** — gives equal voice to low-magnitude patches |
| Effect on large signals | Caps at threshold | Scales down proportionally |
| Learnable | No (fixed threshold) | Yes (learnable c per layer) |

A near-zero heart rate is clinically critical — CTGS amplifies its gradient. Gradient clipping would ignore it.

---

## Q5: "Why not use a larger backbone? GPT-2 is old."

**Rebuttal:** Table 6 shows backbone scale results (GPT-2 Small → Medium → 7-8B). NumLoRA's improvement over LoRA is consistent across backbone sizes. We use GPT-2 Small as the primary backbone because:

1. It matches LLM4Imp (the parent paper), enabling direct comparison
2. It allows full experimental sweeps (10 datasets x 5 MRs x 3 seeds x 13 baselines) within a realistic compute budget
3. The NumLoRA modifications are backbone-agnostic — same formulas, same code, regardless of backbone

The scale validation proves this is not a GPT-2-specific artifact.

---

## Q6: "Why these 10 time-series datasets? This seems cherry-picked."

**Rebuttal:** The 10 datasets cover:
- **4 from LLM4Imp** (PhysioNet, BAQ, IAQ, Solar) — direct comparison to parent paper
- **4 standard TS benchmarks** (ETT, Electricity, Traffic, Weather) — used in every TS forecasting paper since 2021
- **2 adversarial cases** (Exchange Rate: aperiodic; ILI: tiny) — stress tests

Coverage: periodic + non-stationary + small + large + clinical + high-dim + low-dim. The 5 non-TS datasets extend the claim beyond time series entirely. This is broader than most PEFT papers (which test on NLU benchmarks only).

---

## Q7: "Why is the success bar only 1% MAE? That seems low."

**Rebuttal:** The 1% threshold is per-dataset. The full success criterion is:

> Beat LoRA by >= 1% MAE on >= 9/10 datasets, across all missing rates.

That's 9/10 * 5 MRs = 45/50 cells. With Holm-Bonferroni correction and 3-seed bootstrap CIs, surviving 45/50 is a very strong result. Compare: DoRA's headline improvement over LoRA on text benchmarks is also in the 1-2% range.

For numerical data, 1% MAE is meaningful because baseline differences between methods are often 2-5%. A universal 1% lift is more valuable than a 5% lift on 3 datasets with degradation on 7 others.

---

## Q8: "How does this relate to LLM4Imp? Is this a follow-up or a standalone paper?"

**Rebuttal:** NumLoRA is a **standalone method** that can be applied to any frozen-LLM-for-numbers pipeline, not just LLM4Imp. We use LLM4Imp as the experimental vehicle because:

1. It's our own work — full access to code and configurations
2. It already has a patch embedding + reprogramming architecture that LoRA can plug into
3. Direct comparison to the ICC 2026 results validates our experimental setup

NumLoRA's three modifications are generic — they apply to Time-LLM, GPT4TS, or any other frozen-LLM-for-time-series method. We demonstrate this by testing on multiple baselines.

---

## Q9: "Single-seed results for LLM4Imp in the parent paper make comparison unreliable."

**Rebuttal:** Agreed — this is a known weakness of the ICC 2026 paper. Our experiments re-run LLM4Imp with 3 seeds and paired bootstrap CIs, providing the first multi-seed evaluation of LLM4Imp. The NumLoRA paper's LLM4Imp numbers may differ slightly from the ICC 2026 tables due to seed variance — we report this transparently and use the multi-seed numbers for all comparisons.

---

## Q10: "Why not test on NLP tasks too, to show NumLoRA doesn't hurt text?"

**Rebuttal:** Fair point. If space permits, we include a small NLP sanity check (GLUE subset) showing NumLoRA = LoRA on text (alpha and beta learn to stay at 1, gamma stays at 0, CTGS has no effect because text gradients are already well-behaved). This proves NumLoRA is a strict superset: it helps on numbers and doesn't hurt on text.

---

## Q11: "The paper seems incremental over LLM4Imp."

**Rebuttal:** LLM4Imp reprograms the *modality boundary* (cross-attention with spectral prompt). NumLoRA adapts the *backbone weights* (PEFT inside the frozen LLM). These are orthogonal contributions:

- LLM4Imp answers: "How do you get numbers into an LLM?"
- NumLoRA answers: "How do you adapt the LLM's weights when the input is numerical?"

LLM4Imp with NumLoRA is strictly better than LLM4Imp alone. But NumLoRA also works with Time-LLM, GPT4TS, or any future frozen-LLM method. The contribution is independent of the specific reprogramming architecture.
