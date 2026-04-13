"""
NumLoRA — Results aggregation: LaTeX tables + publication-quality figures.

Usage:
    python scripts/analysis/generate_tables.py [--results-dir results/full]

Style: matches topic4-multi-area-scaling conventions (bold, serif, 600 dpi).
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

# ═══════════════════════════════════════════════════════════════
# PUBLICATION STYLE (matching topic4-multi-area-scaling)
# ═══════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "font.weight": "bold",
    "text.color": "black",
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelcolor": "black",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.color": "black",
    "ytick.color": "black",
    "legend.fontsize": 11,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 2.5,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_orig_savefig = plt.savefig


def _savefig_bold(*args, **kwargs):
    kwargs.setdefault("dpi", 600)
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("pad_inches", 0.05)
    fig = plt.gcf()
    for ax in fig.get_axes():
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
            label.set_color("black")
        ax.title.set_color("black")
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_color("black")
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_color("black")
        ax.yaxis.label.set_fontweight("bold")
        for txt in ax.texts:
            txt.set_fontweight("bold")
            txt.set_color("black")
        if hasattr(ax, 'legend_') and ax.legend_:
            for t in ax.legend_.get_texts():
                t.set_fontweight("bold")
    if fig._suptitle:
        fig._suptitle.set_color("black")
        fig._suptitle.set_fontweight("bold")
    return _orig_savefig(*args, **kwargs)


plt.savefig = _savefig_bold

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
    "dora_r8": "DoRA",
}
COLORS = {
    "LoRA": "#1f77b4", "NumLoRA": "#d62728", "CTGS-only": "#2ca02c", "DoRA": "#9467bd",
}
MARKERS = {
    "LoRA": "o", "NumLoRA": "s", "CTGS-only": "D", "DoRA": "^",
}


def load_results(results_dir):
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    meta = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for pattern in [f"{results_dir}/*.json", f"{results_dir}/ablation/*.json"]:
        for f in sorted(glob.glob(pattern)):
            try:
                r = json.load(open(f))
            except (json.JSONDecodeError, KeyError):
                continue
            bb = r.get("backbone", "smollm_360m")
            results[bb][r["dataset"]][r["method"]][r["missing_rate"]].append(r["test_mae"])
            meta[bb][r["dataset"]][r["method"]][r["missing_rate"]] = {
                "elapsed": r.get("elapsed_seconds", 0),
                "params": r.get("trainable_params", 0),
                "pct": r.get("trainable_pct", 0),
            }
    return results, meta


# ═══════════════════════════════════════════════════════════════
# TABLES
# ═══════════════════════════════════════════════════════════════

def fmt_val(vals, bold=False):
    if not vals:
        return "--"
    m, s = np.mean(vals), np.std(vals)
    core = f"{m:.4f}" if len(vals) == 1 else f"{m:.3f}$\\pm${s:.3f}"
    if bold:
        core = f"\\textbf{{{core}}}"
    return core


def generate_main_table(results, out_path, backbone="smollm_360m", mr=0.3):
    datasets = [d for d in DATASET_LABELS if results[backbone].get(d)]
    methods = ["lora_r8", "numlora_ctgs_only", "numlora_full"]

    lines = [
        "\\begin{table}[t]",
        f"\\caption{{Main results (MAE $\\downarrow$) at MR={mr} on {BACKBONE_LABELS.get(backbone, backbone)}.}}",
        f"\\label{{tab:main_{backbone}}}",
        "\\centering\\small",
        f"\\begin{{tabular}}{{@{{}}l{'c' * len(datasets)}@{{}}}}",
        "\\toprule",
        "Method & " + " & ".join(DATASET_LABELS.get(d, d) for d in datasets) + " \\\\",
        "\\midrule",
    ]

    for method in methods:
        vals_per_ds = [results[backbone][ds].get(method, {}).get(mr, []) for ds in datasets]
        means = [np.mean(v) if v else float("inf") for v in vals_per_ds]
        best_idx = np.argmin(means)
        row = [METHOD_LABELS.get(method, method)]
        for i, v in enumerate(vals_per_ds):
            row.append(fmt_val(v, bold=(i == best_idx and len(v) > 0)))
        lines.append(" & ".join(row) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written: {out_path}")


def generate_ablation_table(results, out_path, backbone="smollm_360m", mr=0.3):
    datasets = ["ett_h1", "exchange", "weather"]
    methods = [
        "lora_r8", "numlora_ctgs_only", "numlora_mai_only", "numlora_ssr_only",
        "numlora_mai_ctgs", "numlora_mai_ssr", "numlora_ssr_ctgs", "numlora_full",
    ]
    labels = ["LoRA", "CTGS only", "MAI only", "SSR only",
              "MAI+CTGS", "MAI+SSR", "SSR+CTGS", "All three"]

    lines = [
        "\\begin{table}[t]",
        f"\\caption{{Component ablation (MAE $\\downarrow$) at MR={mr} on {BACKBONE_LABELS.get(backbone, backbone)}.}}",
        f"\\label{{tab:ablation_{backbone}}}",
        "\\centering\\small",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "Variant & " + " & ".join(DATASET_LABELS.get(d, d) for d in datasets) + " & vs LoRA \\\\",
        "\\midrule",
    ]

    lora_means = {}
    for ds in datasets:
        v = results[backbone][ds].get("lora_r8", {}).get(mr, [])
        lora_means[ds] = np.mean(v) if v else float("inf")
    lora_avg = np.mean(list(lora_means.values()))

    for method, label in zip(methods, labels):
        row = [label]
        method_means = []
        for ds in datasets:
            v = results[backbone][ds].get(method, {}).get(mr, [])
            row.append(fmt_val(v))
            if v:
                method_means.append(np.mean(v))
        avg = np.mean(method_means) if method_means else float("inf")
        if method == "lora_r8":
            row.append("--")
        else:
            imp = (lora_avg - avg) / lora_avg * 100
            row.append(f"+{imp:.1f}\\%")
        lines.append(" & ".join(row) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written: {out_path}")


def generate_backbone_table(results, out_path, mr=0.3):
    backbones = [b for b in BACKBONE_LABELS if any(results[b])]
    datasets = ["ett_h1", "weather", "exchange", "traffic", "ili"]
    methods = ["lora_r8", "numlora_ctgs_only", "numlora_full"]

    lines = [
        "\\begin{table}[t]",
        f"\\caption{{Cross-backbone comparison (MAE $\\downarrow$) at MR={mr}.}}",
        "\\label{tab:backbone}",
        "\\centering\\small",
        f"\\begin{{tabular}}{{@{{}}ll{'c' * len(datasets)}c@{{}}}}",
        "\\toprule",
        "Backbone & Method & " + " & ".join(DATASET_LABELS.get(d, d) for d in datasets) + " & Wins \\\\",
        "\\midrule",
    ]

    for bb in backbones:
        lora_means = {}
        for ds in datasets:
            lv = results[bb][ds].get("lora_r8", {}).get(mr, [])
            lora_means[ds] = np.mean(lv) if lv else float("inf")

        for mi, method in enumerate(methods):
            row = [BACKBONE_LABELS[bb] if mi == 0 else "", METHOD_LABELS.get(method, method)]
            wins = 0
            for ds in datasets:
                v = results[bb][ds].get(method, {}).get(mr, [])
                m = np.mean(v) if v else float("inf")
                if method != "lora_r8" and m < lora_means[ds]:
                    wins += 1
                row.append(fmt_val(v))
            row.append(f"{wins}/{len(datasets)}" if method != "lora_r8" else "--")
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\midrule")

    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written: {out_path}")


def generate_efficiency_table(results, meta, out_path):
    """Appendix: training time, parameters, memory per method."""
    lines = [
        "\\begin{table}[t]",
        "\\caption{Efficiency comparison on SmolLM-360M (ETT-h1, MR=0.3).}",
        "\\label{tab:efficiency}",
        "\\centering\\small",
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
        "Method & Trainable Params & Param \\% & Train Time (s) \\\\",
        "\\midrule",
    ]

    for method in ["lora_r8", "numlora_ctgs_only", "numlora_full"]:
        m = meta["smollm_360m"]["ett_h1"].get(method, {}).get(0.3, {})
        params = m.get("params", 0)
        pct = m.get("pct", 0)
        elapsed = m.get("elapsed", 0)
        lines.append(f"{METHOD_LABELS.get(method, method)} & {params:,} & {pct:.2f}\\% & {elapsed:.0f} \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written: {out_path}")


# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════

def generate_mr_curves(results, out_path, backbone="smollm_360m"):
    """MAE vs missing rate curves per dataset — publication quality."""
    datasets = [d for d in ["ett_h1", "weather", "exchange", "traffic", "ili"] if results[backbone].get(d)]
    methods = ["lora_r8", "numlora_full"]
    mrs = [0.1, 0.2, 0.3, 0.4, 0.5]

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(3.5 * n_ds, 3.5), sharey=False)
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
                marker = MARKERS.get(label, "o")
                ax.plot(valid_mrs, means, f"-{marker}", label=label, color=color,
                        markersize=7, linewidth=2.5, markeredgecolor="white", markeredgewidth=1.0)
                ax.fill_between(valid_mrs,
                                np.array(means) - np.array(stds),
                                np.array(means) + np.array(stds),
                                alpha=0.15, color=color)

        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=16, fontweight="bold")
        ax.set_xlabel("Missing Rate", fontsize=14, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel(r"MAE $\downarrow$", fontsize=14, fontweight="bold")
        ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])

    axes[-1].legend(loc="upper left", framealpha=0.95, edgecolor="black",
                    fontsize=12, fancybox=False)
    bb_label = BACKBONE_LABELS.get(backbone, backbone)
    fig.suptitle(f"MAE vs Missing Rate ({bb_label})", fontsize=18, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close()
    print(f"  Written: {out_path}")


def generate_radar(results, out_path, mr=0.3):
    """Radar chart comparing LoRA vs NumLoRA across datasets for each backbone."""
    backbones = [b for b in BACKBONE_LABELS if any(results[b])]
    datasets = ["ett_h1", "weather", "exchange", "traffic", "ili"]
    methods = ["lora_r8", "numlora_full"]

    n_bb = len(backbones)
    fig, axes = plt.subplots(1, n_bb, figsize=(4.5 * n_bb, 4.5), subplot_kw=dict(polar=True))
    if n_bb == 1:
        axes = [axes]

    angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False).tolist()
    angles += angles[:1]

    for ax, bb in zip(axes, backbones):
        for method in methods:
            label = METHOD_LABELS.get(method, method)
            vals = []
            for ds in datasets:
                v = results[bb][ds].get(method, {}).get(mr, [])
                vals.append(np.mean(v) if v else 0)
            vals += vals[:1]

            color = COLORS.get(label, "#333333")
            ax.plot(angles, vals, "-o", label=label, color=color,
                    linewidth=2.5, markersize=6, markeredgecolor="white", markeredgewidth=1.0)
            ax.fill(angles, vals, alpha=0.12, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets],
                           fontsize=11, fontweight="bold")
        ax.set_title(BACKBONE_LABELS[bb], pad=20, fontsize=15, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10, framealpha=0.95, edgecolor="black", fancybox=False)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle(f"MAE at MR={mr} (lower = better)", fontsize=18, fontweight="bold", y=1.06)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close()
    print(f"  Written: {out_path}")


def generate_ablation_bars(results, out_path, backbone="smollm_360m", mr=0.3):
    """Ablation bar chart — improvement over LoRA per component variant."""
    datasets = ["ett_h1", "exchange", "weather"]
    methods = [
        "numlora_ctgs_only", "numlora_mai_only", "numlora_ssr_only",
        "numlora_mai_ctgs", "numlora_mai_ssr", "numlora_ssr_ctgs", "numlora_full",
    ]
    labels = ["CTGS", "MAI", "SSR", "MAI+\nCTGS", "MAI+\nSSR", "SSR+\nCTGS", "All 3"]

    lora_means = {}
    for ds in datasets:
        v = results[backbone][ds].get("lora_r8", {}).get(mr, [])
        lora_means[ds] = np.mean(v) if v else float("inf")
    lora_avg = np.mean(list(lora_means.values()))

    improvements = []
    for method in methods:
        imp_per_ds = []
        for ds in datasets:
            v = results[backbone][ds].get(method, {}).get(mr, [])
            if v:
                imp_per_ds.append((lora_means[ds] - np.mean(v)) / lora_means[ds] * 100)
        improvements.append(np.mean(imp_per_ds) if imp_per_ds else 0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors_list = ["#2ca02c", "#ff7f0e", "#8c564b", "#7f7f7f", "#e377c2", "#bcbd22", "#d62728"]
    bars = ax.bar(labels, improvements, color=colors_list, edgecolor="black", linewidth=0.8, width=0.65)

    ax.axhline(y=0, color="black", linewidth=1.0)
    ax.set_ylabel("Improvement over LoRA (%)", fontsize=15, fontweight="bold")
    ax.set_title(f"Component Ablation at MR={mr} ({BACKBONE_LABELS.get(backbone, backbone)})",
                 fontsize=16, fontweight="bold")

    for bar, val in zip(bars, improvements):
        y_pos = bar.get_height() + 0.3 if bar.get_height() >= 0 else bar.get_height() - 0.8
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:+.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close()
    print(f"  Written: {out_path}")


def generate_training_efficiency(results, meta, out_path, backbone="smollm_360m"):
    """Appendix: training time and parameter count comparison."""
    datasets = [d for d in ["ett_h1", "weather", "exchange", "traffic", "ili"] if results[backbone].get(d)]
    methods = ["lora_r8", "numlora_ctgs_only", "numlora_full"]
    mr = 0.3

    # Collect elapsed times
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    x = np.arange(len(datasets))
    width = 0.25

    for i, method in enumerate(methods):
        label = METHOD_LABELS.get(method, method)
        times = []
        params = []
        for ds in datasets:
            m = meta[backbone][ds].get(method, {}).get(mr, {})
            times.append(m.get("elapsed", 0))
            params.append(m.get("params", 0))

        color = COLORS.get(label, "#333333")
        ax1.bar(x + i * width, times, width, label=label, color=color,
                edgecolor="black", linewidth=0.5)
        ax2.bar(x + i * width, [p / 1000 for p in params], width, label=label,
                color=color, edgecolor="black", linewidth=0.5)

    ax1.set_ylabel("Training Time (s)", fontsize=14, fontweight="bold")
    ax1.set_title("(a) Training Time", fontsize=15, fontweight="bold")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=11, fontweight="bold")
    ax1.legend(fontsize=11, framealpha=0.95, edgecolor="black", fancybox=False)

    ax2.set_ylabel("Trainable Params (K)", fontsize=14, fontweight="bold")
    ax2.set_title("(b) Parameter Count", fontsize=15, fontweight="bold")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], fontsize=11, fontweight="bold")
    ax2.legend(fontsize=11, framealpha=0.95, edgecolor="black", fancybox=False)

    fig.suptitle(f"Efficiency ({BACKBONE_LABELS.get(backbone, backbone)}, MR={mr})",
                 fontsize=17, fontweight="bold", y=1.03)
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
    results, meta = load_results(args.results_dir)

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
    generate_efficiency_table(results, meta, "paper/tables/table_efficiency.tex")

    # ── Main paper figures ──
    print("\nGenerating main paper figures...")
    generate_radar(results, "paper/figures/fig_radar_backbones.pdf")
    for bb in BACKBONE_LABELS:
        if any(results[bb]):
            generate_mr_curves(results, f"paper/figures/fig_mr_curves_{bb}.pdf", backbone=bb)

    # ── Appendix figures ──
    print("\nGenerating appendix figures...")
    generate_ablation_bars(results, "paper/figures/fig_ablation_bars_smollm.pdf", backbone="smollm_360m")
    if any(results["tinyllama_1.1b"]):
        generate_ablation_bars(results, "paper/figures/fig_ablation_bars_tinyllama.pdf", backbone="tinyllama_1.1b")
    for bb in BACKBONE_LABELS:
        if any(results[bb]):
            generate_training_efficiency(results, meta, f"paper/figures/fig_efficiency_{bb}.pdf", backbone=bb)

    print("\nDone! Tables in paper/tables/, figures in paper/figures/")


if __name__ == "__main__":
    main()
