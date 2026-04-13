# NumLoRA — Full Method Specification

> Mathematical specification of NumLoRA for implementers. Every equation, every shape, every init value.

## Background: LLM4Imp architecture (parent paper)

NumLoRA operates inside the LLM4Imp pipeline. For context, the full LLM4Imp forward pass is:

```
Raw X (B x T x D)
    ├── Spectral Prompt Builder → E_prompt (B x D x L_p x d_LLM)
    │     ├── Statistical features: delta_min, delta_max, delta_med, sigma, tau
    │     ├── Spectral features: FFT → dominant freq, period, top-L autocorrelation lags
    │     └── Prompt assembly → BPE → GPT-2 embedding lookup
    │
    ├── RevIN normalisation → Patch segmentation → Patch projection
    │     → H_patch (B x D x L_patch x d_model)
    │
    └── Cross-domain reprogramming (cross-attention):
          Q = H_patch, K = V = E_prompt
          O = softmax(QK^T / sqrt(d_k)) V
          H_reprog = LN(H_patch + gamma_g * O)
              │
              ▼
          Frozen LLM forward pass → H_ctx
              │
              ▼
          Output projection + inverse RevIN → X_hat (B x T x D)
```

**Loss:** Masked MSE on imputed positions only:
```
L = sum((1 - M) * (X_hat - X)^2) / sum(1 - M)
```

**What NumLoRA modifies:** The weight matrices inside the frozen LLM. Everything else (prompt, patches, cross-attention, output projection) stays identical to LLM4Imp.

---

## 1. Standard LoRA (baseline)

For a frozen weight matrix W in R^{d x d} at layer l:

```
W_eff = W + B_l * A_l
```

where A_l in R^{r x d}, B_l in R^{d x r}, r << d.

**Init:** A ~ Kaiming-uniform, B = 0. So W_eff = W at step 0.

**Problem for numerical data:**
1. Init variance is calibrated for text-scale activations
2. No bias term — cannot express affine shifts
3. Gradient magnitude proportional to input norm — unstable for high-dynamic-range data

---

## 2. NumLoRA: three modifications

### 2.1 Modification 1 — Scale-Shift Renormalisation (SSR)

```
dW_l = diag(alpha_l) * B_l * A_l * diag(beta_l) + gamma_l
```

**Shapes:**
- alpha_l in R^d     — output scale (row-wise)
- beta_l in R^d      — input scale (column-wise)
- gamma_l in R^{d x 1} — bias (broadcast across columns)

**Init:**
- alpha_l = ones(d)
- beta_l = ones(d)
- gamma_l = zeros(d, 1)

**At init:** dW = diag(1) * B * A * diag(1) + 0 = BA → identical to LoRA.

**Parameters per layer:** 3d
**Total (L=12, d=768):** 27,648

### 2.2 Modification 2 — Magnitude-Aware Initialisation (MAI)

Before training begins:
1. Forward 1 batch through frozen backbone (no gradient)
2. At each layer l, record activation variance: sigma^2_l = Var[act_l(batch_0)]
3. Initialise A_l with calibrated variance:

```
A_l ~ N(0, 1 / (r * sigma^2_l))
B_l = 0
```

**Effect:** ||B*A|| starts at roughly the same scale regardless of the numerical data distribution. On text, Kaiming-uniform achieves this naturally because text embedding variance is predictable. On numbers, it's not — MAI fixes this.

**Parameters:** 0 (init-time only)
**Cost:** 1 forward pass (~2 seconds)

### 2.3 Modification 3 — Continuous-Token Gradient Scaling (CTGS)

Register a backward hook on A_l:

```
grad(A_l) <- grad(A_l) * c_l / (||h_patch||_2 + epsilon)
```

**Shapes:**
- c_l: scalar (1 per layer)
- h_patch: the input activation to this layer
- epsilon = 1e-8

**Init:** c_l = 1.0

**Effect:** All patches contribute equally to the LoRA gradient, regardless of their magnitude. Prevents large-scale patches (e.g., solar irradiance at 1400 W/m^2) from dominating over small-scale patches (e.g., near-zero heart rate).

**Parameters per layer:** 1
**Total (L=12):** 12

---

## 3. Combined forward and backward

### Forward:
```python
# At each layer l, for input h:
lora_out = B_l @ A_l @ h           # standard LoRA
scaled = diag(alpha_l) @ lora_out   # output scale
shifted = scaled + gamma_l          # bias
# Also apply input scale via: A_l_eff = A_l @ diag(beta_l)
# Full: diag(alpha_l) @ B_l @ A_l @ diag(beta_l) @ h + gamma_l

output = frozen_layer(h) + shifted
```

### Backward (CTGS hook):
```python
# Registered on A_l's gradient
def ctgs_hook(grad_A):
    patch_norm = h_patch.norm(dim=-1, keepdim=True)  # per-sample
    return grad_A * c_l / (patch_norm + 1e-8)
```

### Inference (merged):
```python
# Pre-compute once:
W_merged = W + diag(alpha_l) @ B_l @ A_l @ diag(beta_l)
b_merged = gamma_l

# At inference: output = W_merged @ h + b_merged
# Zero overhead — same speed as vanilla LoRA merge
```

---

## 4. Parameter summary

| Component | Formula | Per layer (d=768, r=8) | Total (L=12) |
|---|---|---|---|
| A | r * d | 6,144 | 73,728 |
| B | d * r | 6,144 | 73,728 |
| alpha | d | 768 | 9,216 |
| beta | d | 768 | 9,216 |
| gamma | d | 768 | 9,216 |
| c | 1 | 1 | 12 |
| **Total** | **2dr + 3d + 1** | **14,593** | **175,116** |

vs LoRA: 147,456 → **+19% overhead**
vs LoRA r=9: 165,888 → **+6% overhead**

---

## 5. Learning rates

| Parameter group | lr | Rationale |
|---|---|---|
| A, B | 1e-3 | Standard LoRA lr |
| alpha, beta, gamma | 1e-2 | Low-dim params; faster convergence |
| c | 1e-2 | Scalar; same group as aux params |

Use AdamW with weight_decay=0.01 on A, B only. No weight decay on alpha, beta, gamma, c (they are scale/shift, not weight matrices).

---

## 6. Where NumLoRA is applied

NumLoRA replaces LoRA in the frozen LLM's:
- Query projection (W_Q)
- Key projection (W_K)
- Value projection (W_V)
- Output projection (W_O)
- MLP up-projection (W_up)
- MLP down-projection (W_down)

Total: 6 weight matrices per layer x L layers.

The rest of LLM4Imp's trainable components (patch embedding, cross-attention reprogramming, output projection) remain unchanged.

---

## 7. Key invariants

1. **At init, NumLoRA = LoRA.** alpha=1, beta=1, gamma=0, B=0 → dW = 0 → W_eff = W.
2. **At inference, NumLoRA merges like LoRA.** W_merged = W + diag(alpha) B A diag(beta). Single matrix multiply, no runtime overhead.
3. **CTGS is backward-only.** Forward pass is identical to LoRA+SSR. CTGS only modifies gradient computation.
4. **MAI is init-only.** After the first forward pass sets A's variance, MAI has no further effect.
5. **Backbone is bit-identically frozen.** NumLoRA modifies only the LoRA adaptation, not the backbone weights.
