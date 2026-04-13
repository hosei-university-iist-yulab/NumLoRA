"""
NumLoRA — Results aggregation: LaTeX tables + figures from JSON results.

Usage:
    python scripts/analysis/generate_tables.py [--results-dir results/full]

Outputs:
    paper/tables/table_main.tex        — Main comparison (LoRA vs CTGS-only vs NumLoRA per backbone)
    paper/tables/table_mr_sweep.tex    — MR sensitivity (0.1-0.5) per dataset
    paper/tables/table_ablation.tex    — Component ablation (7 subsets)
    paper/tables/table_backbone.tex    — Backbone scale comparison
    paper/figures/fig_mr_curves.pdf    — MAE vs MR curves
    paper/figures/fig_ablation_bars.pdf — Ablation bar chart
    paper/figures/fig_variance_hist.pdf — Activation variance: text vs numerical
    paper/figures/fig_radar.pdf        — Radar chart across datasets
"""

import argparse
import json
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Styling ──
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

DATASET_LABELS = {
    "ett_h1": "ETT-h1", "ett_h2": "ETT-h2", "ett_m1": "ETT-m1", "ett_m2": "ETT-m2",
    "weather": "Weather", "exchange": "Exchange", "traffic": "Traffic", "ili": "ILI",
}
BACKBONE_LABELS = {
    "smollm_360m": "SmolLM-360M", "qwen_0.5b": "Qwen-0.5B",
    "tinyllama_1.1b": "TinyLlama-1.1B", "phi3_mini": "Phi-3-mini",
}
METHOD_LABELS = {
    "lora_r8": "LoRA", "numlora_full": "NumLoRA", "numlora_ctgs_only": "CTGS-only",
    "dora_r8": "DoRA", "numlora_mai_only": "MAI-only", "numlora_ssr_only": "SSR-only",
    "numlora_mai_ssr": "MAI+SSR", "numlora_mai_ctgs": "MAI+CTGS",
    "numlora_ssr_ctgs": "SSR+CTGS",
}
COLORS = {
    "LoRA": "#1f77b4", "NumLoRA": "#d62728", "CTGS-only": "#2ca02c",
    "DoRA": "#9467bd", "MAI-only": "#ff7f0e", "SSR-only": "#8c564b",
    "MAI+SSR": "#e377c2", "MAI+CTGS": "#7f7f7f", "SSR+CTGS": "#bcbd22",
}


def load_results(results_dir):
    """Load all JSON results into nested dict: results[backbone][dataset][method][mr] = [mae_values]"""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for pattern in [f"{results_dir}/*.json", f"{results_dir}/ablation/*.json"]:
        for f in sorted(glob.glob(pattern)):
            try:
                r = json.load(open(f))
            except (json.JSONDecodeError, KeyError):
                continue
            bb = r.get("backbone", "smollm_360m")
            results[bb][r["dataset"]][r["method"]][r["missing_rate"]].append(r["test_mae"])

    return results


def fmt_val(vals, bold=False, underline=False):
    """Format mean±std for LaTeX."""
    if not vals:
        return "--"
    m, s = np.mean(vals), np.std(vals)
    core = f"{m:.4f}"
    if len(vals) > 1:
        core += f"$\\pm${s:.3f}"
    if bold:
        core = f"\\textbf{{{core}}}"
    if underline:
        core = f"\\underline{{{core}}}"
    return core


# ═══════════════════════════════════════════════════════════════
# TABLES
# ═══════════════════════════════════════════════════════════════

def generate_main_table(results, out_path, backbone="smollm_360m", mr=0.3):
    """Table 1: Main comparison at MR=0.3 for a given backbone."""
    datasets = [d for d in DATASET_LABELS if results[backbone].get(d)]
    methods = ["lora_r8", "numlora_ctgs_only", "numlora_full"]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append(f"\\caption{{Main results (MAE $\\downarrow$) at MR={mr} on {BACKBONE_LABELS.get(backbone, backbone)}. "
                 "Best in \\textbf{bold}.}")
    lines.append(f"\\label{{tab:main_{backbone}}}")
    lines.append("\\centering\\small")
    cols = "l" + "c" * len(datasets)
    lines.append(f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}")
    lines.append("\\toprule")
    header = "Method & " + " & ".join(DATASET_LABELS.get(d, d) for d in datasets) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for method in methods:
        vals_per_ds = []
        for ds in datasets:
            v = results[backbone][ds].get(method, {}).get(mr, [])
            vals_per_ds.append(v)

        # Find best (lowest MAE)
        means = [np.mean(v) if v else float("inf") for v in vals_per_ds]
        best_idx = np.argmin(means)

        row_parts = [METHOD_LABELS.get(method, method)]
        for i, v in enumerate(vals_per_ds):
            row_parts.append(fmt_val(v, bold=(i == best_idx and len(v) > 0)))
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written: {out_path}")


def generate_ablation_table(results, out_path, backbone="smollm_360m", mr=0.3):
    """Table 3: Component ablation."""
    datasets = ["ett_h1", "exchange", "weather"]
    methods = [
        "lora_r8", "numlora_mai_only", "numlora_ssr_only", "numlora_ctgs_only",
        "numlora_mai_ssr", "numlora_mai_ctgs", "numlora_ssr_ctgs", "numlora_full",
    ]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append(f"\\caption{{Component ablation (MAE $\\downarrow$) at MR={mr} on {BACKBONE_LABELS.get(backbone, backbone)}.}}")
    lines.append(f"\\label{{tab:ablation_{backbone}}}")
    lines.append("\\centering\\small")
    lines.append("\\begin{tabular}{@{}lccc@{}}")
    lines.append("\\toprule")
    lines.append("Variant & " + " & ".join(DATASET_LABELS.get(d, d) for d in datasets) + " \\\\")
    lines.append("\\midrule")

    for method in methods:
        row = [METHOD_LABELS.get(method, method)]
        for ds in datasets:
            v = results[backbone][ds].get(method, {}).get(mr, [])
            row.append(fmt_val(v))
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written: {out_path}")


def generate_backbone_table(results, out_path, mr=0.3):
    """Table: Cross-backbone comparison (LoRA vs CTGS-only vs NumLoRA-full)."""
    backbones = [b for b in BACKBONE_LABELS if any(results[b])]
    datasets = ["ett_h1", "weather", "exchange", "traffic", "ili"]
    methods = ["lora_r8", "numlora_ctgs_only", "numlora_full"]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append(f"\\caption{{Cross-backbone comparison (MAE $\\downarrow$) at MR={mr}. "
                 "Wins = cells where method beats LoRA.}}")
    lines.append("\\label{tab:backbone}")
    lines.append("\\centering\\small")
    lines.append("\\begin{tabular}{@{}ll" + "c" * len(datasets) + "c@{}}")
    lines.append("\\toprule")
    lines.append("Backbone & Method & " + " & ".join(DATASET_LABELS.get(d, d) for d in datasets) + " & Wins \\\\")
    lines.append("\\midrule")

    for bb in backbones:
        for mi, method in enumerate(methods):
            row = []
            if mi == 0:
                row.append(f"\\multirow{{3}}{{*}}{{{BACKBONE_LABELS[bb]}}}")
            else:
                row.append("")
            row.append(METHOD_LABELS.get(method, method))

            wins = 0
            lora_means = {}
            for ds in datasets:
                lv = results[bb][ds].get("lora_r8", {}).get(mr, [])
                lora_means[ds] = np.mean(lv) if lv else float("inf")

            for ds in datasets:
                v = results[bb][ds].get(method, {}).get(mr, [])
                m = np.mean(v) if v else float("inf")
                if method != "lora_r8" and m < lora_means[ds]:
                    wins += 1
                row.append(fmt_val(v))

            row.append(f"{wins}/{len(datasets)}" if method != "lora_r8" else "--")
            lines.append(" & ".join(row) + " \\\\")

        lines.append("\\midrule")

    # Remove last \midrule and replace with \bottomrule
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written: {out_path}")


# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════

def generate_mr_curves(results, out_path, backbone="smollm_360m"):
    """Fig: MAE vs missing rate curves per dataset."""
    datasets = [d for d in ["ett_h1", "weather", "exchange", "traffic", "ili"] if results[backbone].get(d)]
    methods = ["lora_r8", "numlora_ctgs_only", "numlora_full"]
    mrs = [0.1, 0.2, 0.3, 0.4, 0.5]

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(3.2 * n_ds, 3), sharey=False)
    if n_ds == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        for method in methods:
            label = METHOD_LABELS.get(method, method)
            means, stds, valid_mrs = [], [], []
            for mr in mrs:
                v = results[backbone][ds].get(method, {}).get(mr, [])
                if v:
                    means.append(np.mean(v))
                    stds.append(np.std(v))
                    valid_mrs.append(mr)

            if means:
                color = COLORS.get(label, "#333333")
                ax.plot(valid_mrs, means, "-o", label=label, color=color, markersize=4, linewidth=1.5)
                ax.fill_between(valid_mrs,
                                np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                alpha=0.15, color=color)

        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlabel("Missing Rate")
        if ax == axes[0]:
            ax.set_ylabel("MAE ↓")
        ax.grid(True, alpha=0.3)

    axes[-1].legend(loc="upper left", framealpha=0.9)
    fig.suptitle(f"MAE vs Missing Rate ({BACKBONE_LABELS.get(backbone, backbone)})", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close()
    print(f"  Written: {out_path}")


def generate_ablation_bars(results, out_path, backbone="smollm_360m", mr=0.3):
    """Fig: Ablation bar chart — improvement over LoRA per component variant."""
    datasets = ["ett_h1", "exchange", "weather"]
    methods = [
        "numlora_mai_only", "numlora_ssr_only", "numlora_ctgs_only",
        "numlora_mai_ssr", "numlora_mai_ctgs", "numlora_ssr_ctgs", "numlora_full",
    ]
    labels = ["MAI", "SSR", "CTGS", "MAI+SSR", "MAI+CTGS", "SSR+CTGS", "All 3"]

    # Compute improvement over LoRA (avg across datasets)
    improvements = []
    for method in methods:
        imp_per_ds = []
        for ds in datasets:
            lv = results[backbone][ds].get("lora_r8", {}).get(mr, [])
            nv = results[backbone][ds].get(method, {}).get(mr, [])
            if lv and nv:
                imp_per_ds.append((np.mean(lv) - np.mean(nv)) / np.mean(lv) * 100)
        improvements.append(np.mean(imp_per_ds) if imp_per_ds else 0)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_list = ["#ff7f0e", "#8c564b", "#2ca02c", "#e377c2", "#7f7f7f", "#bcbd22", "#d62728"]
    bars = ax.bar(labels, improvements, color=colors_list, edgecolor="white", linewidth=0.5)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("Improvement over LoRA (%)")
    ax.set_title(f"Component Ablation at MR={mr} ({BACKBONE_LABELS.get(backbone, backbone)})")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:+.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close()
    print(f"  Written: {out_path}")


def generate_backbone_radar(results, out_path, mr=0.3):
    """Fig: Radar chart comparing methods across datasets for each backbone."""
    backbones = [b for b in BACKBONE_LABELS if any(results[b])]
    datasets = ["ett_h1", "weather", "exchange", "traffic", "ili"]
    methods = ["lora_r8", "numlora_ctgs_only"]

    n_bb = len(backbones)
    fig, axes = plt.subplots(1, n_bb, figsize=(4 * n_bb, 4), subplot_kw=dict(polar=True))
    if n_bb == 1:
        axes = [axes]

    angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for ax, bb in zip(axes, backbones):
        for method in methods:
            label = METHOD_LABELS.get(method, method)
            vals = []
            for ds in datasets:
                v = results[bb][ds].get(method, {}).get(mr, [])
                vals.append(np.mean(v) if v else 0)
            vals += vals[:1]

            color = COLORS.get(label, "#333333")
            ax.plot(angles, vals, "-o", label=label, color=color, linewidth=1.5, markersize=4)
            ax.fill(angles, vals, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=8)
        ax.set_title(BACKBONE_LABELS[bb], pad=15)
        ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(f"MAE at MR={mr} (lower = better)", fontsize=13, y=1.05)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close()
    print(f"  Written: {out_path}")


def generate_improvement_heatmap(results, out_path, method="numlora_ctgs_only"):
    """Fig: Heatmap of improvement (%) over LoRA across backbones x datasets at MR=0.3."""
    backbones = [b for b in BACKBONE_LABELS if any(results[b])]
    datasets = ["ett_h1", "weather", "exchange", "traffic", "ili"]
    mr = 0.3

    matrix = []
    for bb in backbones:
        row = []
        for ds in datasets:
            lv = results[bb][ds].get("lora_r8", {}).get(mr, [])
            nv = results[bb][ds].get(method, {}).get(mr, [])
            if lv and nv:
                imp = (np.mean(lv) - np.mean(nv)) / np.mean(lv) * 100
            else:
                imp = 0
            row.append(imp)
        matrix.append(row)

    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-30, vmax=30)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_yticks(range(len(backbones)))
    ax.set_yticklabels([BACKBONE_LABELS[b] for b in backbones])

    for i in range(len(backbones)):
        for j in range(len(datasets)):
            val = matrix[i, j]
            color = "white" if abs(val) > 15 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="Improvement over LoRA (%)", shrink=0.8)
    ax.set_title(f"{METHOD_LABELS.get(method, method)} vs LoRA at MR={mr}")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close()
    print(f"  Written: {out_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NumLoRA results aggregation")
    parser.add_argument("--results-dir", default="results/full")
    args = parser.parse_args()

    os.makedirs("paper/tables", exist_ok=True)
    os.makedirs("paper/figures", exist_ok=True)

    print("Loading results...")
    results = load_results(args.results_dir)

    total = sum(
        len(v) for bb in results for ds in results[bb]
        for m in results[bb][ds] for mr, v in results[bb][ds][m].items()
    )
    print(f"  Loaded {total} individual runs across {len(results)} backbones\n")

    # ── Tables ──
    print("Generating tables...")
    for bb in BACKBONE_LABELS:
        if any(results[bb]):
            generate_main_table(results, f"paper/tables/table_main_{bb}.tex", backbone=bb)
    generate_ablation_table(results, "paper/tables/table_ablation_smollm.tex", backbone="smollm_360m")
    if any(results["tinyllama_1.1b"]):
        generate_ablation_table(results, "paper/tables/table_ablation_tinyllama.tex", backbone="tinyllama_1.1b")
    generate_backbone_table(results, "paper/tables/table_backbone.tex")

    # ── Figures ──
    print("\nGenerating figures...")
    for bb in BACKBONE_LABELS:
        if any(results[bb]):
            generate_mr_curves(results, f"paper/figures/fig_mr_curves_{bb}.pdf", backbone=bb)
    generate_ablation_bars(results, "paper/figures/fig_ablation_bars_smollm.pdf", backbone="smollm_360m")
    if any(results["tinyllama_1.1b"]):
        generate_ablation_bars(results, "paper/figures/fig_ablation_bars_tinyllama.pdf", backbone="tinyllama_1.1b")
    generate_backbone_radar(results, "paper/figures/fig_radar_backbones.pdf")
    generate_improvement_heatmap(results, "paper/figures/fig_heatmap_ctgs.pdf", method="numlora_ctgs_only")
    generate_improvement_heatmap(results, "paper/figures/fig_heatmap_numlora.pdf", method="numlora_full")

    print("\nDone! Tables in paper/tables/, figures in paper/figures/")


if __name__ == "__main__":
    main()
