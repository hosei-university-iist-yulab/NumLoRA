"""
NumLoRA training script — unified entry point for all methods and datasets.

Usage:
    python scripts/experiments/train.py \
        --method numlora_full \
        --dataset ett_h1 \
        --missing-rate 0.3 \
        --seed 42 \
        --epochs 200 \
        --output results/full/numlora/numlora_full_ett_h1_mr0.3_seed42.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.dataset import load_ett, load_weather, load_exchange, load_traffic, load_ili, create_datasets
from src.models.imputation_model import LLMImputationModel
from src.models.apply import apply_numlora, get_numlora_params, count_params
from src.models.mai import calibrate_numlora


# ── Method configurations ──

METHODS = {
    "frozen":       {"peft": None,     "rank": 0,  "ssr": False, "ctgs": False},
    "lora_r8":      {"peft": "lora",   "rank": 8,  "ssr": False, "ctgs": False},
    "lora_r9":      {"peft": "lora",   "rank": 9,  "ssr": False, "ctgs": False},
    "dora_r8":      {"peft": "dora",   "rank": 8,  "ssr": False, "ctgs": False},
    "numlora_full": {"peft": "numlora","rank": 8,  "ssr": True,  "ctgs": True},
    # Ablations
    "numlora_mai_only":  {"peft": "numlora", "rank": 8, "ssr": False, "ctgs": False},
    "numlora_ssr_only":  {"peft": "numlora", "rank": 8, "ssr": True,  "ctgs": False},
    "numlora_ctgs_only": {"peft": "numlora", "rank": 8, "ssr": False, "ctgs": True},
    "numlora_mai_ssr":   {"peft": "numlora", "rank": 8, "ssr": True,  "ctgs": False},
    "numlora_mai_ctgs":  {"peft": "numlora", "rank": 8, "ssr": False, "ctgs": True},
    "numlora_ssr_ctgs":  {"peft": "numlora", "rank": 8, "ssr": True,  "ctgs": True},
}

DATASETS = {
    "ett_h1": lambda: load_ett("h1"),
    "ett_h2": lambda: load_ett("h2"),
    "ett_m1": lambda: load_ett("m1"),
    "ett_m2": lambda: load_ett("m2"),
    "weather": lambda: load_weather(),
    "exchange": lambda: load_exchange(),
    "traffic": lambda: load_traffic(),
    "ili": lambda: load_ili(),
}

BACKBONES = {
    "gpt2_small":    "gpt2",
    "smollm_360m":   "HuggingFaceTB/SmolLM-360M",
    "qwen_0.5b":     "Qwen/Qwen2.5-0.5B-Instruct",
    "tinyllama_1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi3_mini":     "microsoft/Phi-3-mini-4k-instruct",
}


def setup_model(backbone_name, method_name, device):
    """Load backbone, apply PEFT method, return imputation model."""
    hf_id = BACKBONES[backbone_name]
    config = METHODS[method_name]

    backbone = AutoModelForCausalLM.from_pretrained(hf_id).to(device)
    hidden_dim = backbone.config.hidden_size

    if config["peft"] == "numlora":
        apply_numlora(
            backbone, rank=config["rank"],
            enable_ssr=config["ssr"], enable_ctgs=config["ctgs"],
        )
    elif config["peft"] == "lora":
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=config["rank"],
            lora_alpha=config["rank"],
            target_modules=_get_target_modules(backbone),
            bias="none",
        )
        backbone = get_peft_model(backbone, lora_config)
    elif config["peft"] == "dora":
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=config["rank"],
            lora_alpha=config["rank"],
            target_modules=_get_target_modules(backbone),
            use_dora=True,
            bias="none",
        )
        backbone = get_peft_model(backbone, lora_config)
    elif config["peft"] is None:
        # Frozen — freeze everything
        for p in backbone.parameters():
            p.requires_grad_(False)

    return backbone, hidden_dim


def _get_target_modules(model):
    """Auto-detect target modules for peft LoRA."""
    model_type = getattr(model.config, "model_type", "").lower()
    if "gpt2" in model_type:
        return ["c_attn", "c_proj", "c_fc"]
    else:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]


def masked_mae(pred, target, mask):
    """MAE computed only on masked (imputed) positions."""
    missing = (1 - mask)  # 1 where values were missing
    n_missing = missing.sum()
    if n_missing == 0:
        return torch.tensor(0.0)
    return (torch.abs(pred - target) * missing).sum() / n_missing


def masked_mse(pred, target, mask):
    """MSE computed only on masked positions."""
    missing = (1 - mask)
    n_missing = missing.sum()
    if n_missing == 0:
        return torch.tensor(0.0)
    return ((pred - target) ** 2 * missing).sum() / n_missing


def train_epoch(model, loader, optimizer, device):
    """One training epoch. Returns average loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        patches = batch["patches"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)

        pred = model(patches)
        loss = masked_mse(pred, target, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on a dataset. Returns dict of metrics."""
    model.eval()
    all_mae, all_mse = [], []

    for batch in loader:
        patches = batch["patches"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)

        pred = model(patches)
        all_mae.append(masked_mae(pred, target, mask).item())
        all_mse.append(masked_mse(pred, target, mask).item())

    mae = np.mean(all_mae)
    mse = np.mean(all_mse)
    return {"mae": mae, "mse": mse, "rmse": np.sqrt(mse)}


def main():
    parser = argparse.ArgumentParser(description="NumLoRA training")
    parser.add_argument("--method", type=str, default="numlora_full", choices=list(METHODS.keys()))
    parser.add_argument("--dataset", type=str, default="ett_h1", choices=list(DATASETS.keys()))
    parser.add_argument("--backbone", type=str, default="gpt2_small", choices=list(BACKBONES.keys()))
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-ssr", type=float, default=3e-3)
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tier", type=str, default="full")
    args = parser.parse_args()

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Method: {args.method} | Dataset: {args.dataset} | MR: {args.missing_rate} | Seed: {args.seed}")

    # ── Data ──
    data_dict = DATASETS[args.dataset]()
    train_ds, val_ds, test_ds = create_datasets(
        data_dict, window_size=args.window_size, patch_size=args.patch_size,
        missing_rate=args.missing_rate, seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    patch_dim = args.patch_size * data_dict["n_features"]
    print(f"Data: {data_dict['name']} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} | Patch dim: {patch_dim}")

    # ── Model ──
    backbone, hidden_dim = setup_model(args.backbone, args.method, device)
    model = LLMImputationModel(
        backbone=backbone, patch_dim=patch_dim, hidden_dim=hidden_dim,
        window_size=args.window_size, n_features=data_dict["n_features"],
        patch_size=args.patch_size,
    ).to(device)

    # MAI calibration (for NumLoRA methods)
    method_config = METHODS[args.method]
    if method_config["peft"] == "numlora":
        sample_batch = next(iter(train_loader))
        cal_embeds = model.input_proj(sample_batch["patches"].to(device))
        calibrate_numlora(backbone, {"inputs_embeds": cal_embeds}, device)
        print("MAI calibration complete")

    # ── Optimizer ──
    param_counts = count_params(model)
    print(f"Params: {param_counts['trainable']:,} trainable / {param_counts['frozen']:,} frozen ({param_counts['trainable_pct']:.2f}%)")

    if method_config["peft"] == "numlora":
        groups = get_numlora_params(backbone)
        optimizer = torch.optim.AdamW([
            {"params": groups["lora"] + list(model.input_proj.parameters()) + list(model.output_head.parameters()), "lr": args.lr},
            {"params": groups["ssr"], "lr": args.lr_ssr},
            {"params": groups["ctgs"], "lr": args.lr_ssr},
        ], weight_decay=1e-5)
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training ──
    best_val_mae = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {train_loss:.6f} | Val MAE: {val_metrics['mae']:.6f} | Best: {best_val_mae:.6f} | Pat: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start_time

    # ── Test ──
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest: MAE={test_metrics['mae']:.6f} | MSE={test_metrics['mse']:.6f} | RMSE={test_metrics['rmse']:.6f}")
    print(f"Time: {elapsed:.1f}s | Best epoch: {epoch - patience_counter}")

    # ── Save results ──
    result = {
        "method": args.method,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "missing_rate": args.missing_rate,
        "seed": args.seed,
        "epochs_trained": epoch - patience_counter,
        "elapsed_seconds": round(elapsed, 1),
        "test_mae": round(test_metrics["mae"], 6),
        "test_mse": round(test_metrics["mse"], 6),
        "test_rmse": round(test_metrics["rmse"], 6),
        "best_val_mae": round(best_val_mae, 6),
        "trainable_params": param_counts["trainable"],
        "trainable_pct": round(param_counts["trainable_pct"], 2),
        "tier": args.tier,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
