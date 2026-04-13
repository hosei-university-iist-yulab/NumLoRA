"""
NumLoRA Interactive Demo
========================

Gradio app comparing LoRA vs NumLoRA (CTGS) on time-series imputation.
Users select a dataset, missing rate, and backbone, then see MAE comparison
and the imputed time series side by side.

Run locally:
    pip install gradio
    python demo/app.py

Deploy to HuggingFace Spaces:
    Upload this directory as a Gradio Space.

Note: requires a GPU for backbone inference. On CPU, inference is slow
but functional for small datasets (ETT-h1).
"""

import os
import sys
import json
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not installed. Run: pip install gradio")

import numpy as np


def load_cached_results(results_dir="results/full/imputation"):
    """Load pre-computed results for the demo (no GPU needed)."""
    results = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            r = json.load(open(f))
            key = (r["dataset"], r["method"], r["missing_rate"], r.get("backbone", "smollm_360m"))
            if key not in results:
                results[key] = []
            results[key].append(r)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def get_comparison(dataset, missing_rate, backbone, results):
    """Compare LoRA vs CTGS-only from cached results."""
    mr = float(missing_rate)

    lora_key = (dataset, "lora_r8", mr, backbone)
    ctgs_key = (dataset, "numlora_ctgs_only", mr, backbone)

    lora_runs = results.get(lora_key, [])
    ctgs_runs = results.get(ctgs_key, [])

    if not lora_runs or not ctgs_runs:
        return "No results available for this configuration yet.", None

    lora_mae = np.mean([r["test_mae"] for r in lora_runs])
    lora_mse = np.mean([r["test_mse"] for r in lora_runs])
    lora_mre = np.mean([r.get("test_mre", 0) for r in lora_runs])

    ctgs_mae = np.mean([r["test_mae"] for r in ctgs_runs])
    ctgs_mse = np.mean([r["test_mse"] for r in ctgs_runs])
    ctgs_mre = np.mean([r.get("test_mre", 0) for r in ctgs_runs])

    imp_mae = (lora_mae - ctgs_mae) / lora_mae * 100
    imp_mse = (lora_mse - ctgs_mse) / lora_mse * 100

    summary = f"""## Results: {dataset} | MR={mr} | {backbone}

| Metric | LoRA | NumLoRA (CTGS) | Improvement |
|--------|------|----------------|-------------|
| MAE ↓  | {lora_mae:.4f} | {ctgs_mae:.4f} | {imp_mae:+.1f}% |
| MSE ↓  | {lora_mse:.4f} | {ctgs_mse:.4f} | {imp_mse:+.1f}% |
| MRE ↓  | {lora_mre:.4f} | {ctgs_mre:.4f} | -- |

**Verdict:** {"NumLoRA wins" if imp_mae > 0 else "LoRA wins"} ({abs(imp_mae):.1f}% {"improvement" if imp_mae > 0 else "gap"})

*Averaged over {len(lora_runs)} seeds.*
"""
    return summary, None


def build_demo():
    """Build the Gradio interface."""
    results = load_cached_results()

    datasets = sorted(set(k[0] for k in results.keys()))
    backbones = sorted(set(k[3] for k in results.keys()))
    missing_rates = sorted(set(str(k[2]) for k in results.keys()))

    if not datasets:
        datasets = ["ett_h1", "weather", "exchange", "traffic", "ili"]
    if not backbones:
        backbones = ["smollm_360m", "qwen_0.5b"]
    if not missing_rates:
        missing_rates = ["0.1", "0.3", "0.5"]

    with gr.Blocks(title="NumLoRA Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # NumLoRA: Calibrating Low-Rank Adaptation for Continuous-Valued Inputs

        **LoRA was not made for numbers.** This demo compares standard LoRA against
        NumLoRA (CTGS) on time-series imputation across different datasets, missing
        rates, and LLM backbones.

        CTGS (Continuous-Token Gradient Scaling) adds just one learnable scalar per
        layer to normalise LoRA gradients by input activation norms. Zero inference overhead.
        """)

        with gr.Row():
            dataset_dd = gr.Dropdown(choices=datasets, value=datasets[0], label="Dataset")
            mr_dd = gr.Dropdown(choices=missing_rates, value="0.3", label="Missing Rate")
            backbone_dd = gr.Dropdown(choices=backbones, value=backbones[0], label="Backbone")

        compare_btn = gr.Button("Compare", variant="primary")
        output_md = gr.Markdown()

        compare_btn.click(
            fn=lambda d, m, b: get_comparison(d, m, b, results)[0],
            inputs=[dataset_dd, mr_dd, backbone_dd],
            outputs=output_md,
        )

        gr.Markdown("""
        ---
        **Links:** [Paper (under review)]() |
        [Code](https://github.com/hosei-university-iist-yulab/NumLoRA) |
        [Google Scholar](https://scholar.google.com/scholar?q=Franck+Junior+Aboya+Messou)

        *YuLab, Hosei University*
        """)

    return demo


if __name__ == "__main__":
    if not GRADIO_AVAILABLE:
        print("Install gradio first: pip install gradio")
        sys.exit(1)
    demo = build_demo()
    demo.launch(share=False)
