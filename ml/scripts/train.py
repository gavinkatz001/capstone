"""Main training entry point."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falldet.data.dataset import FallDetectionDataset, compute_pos_weight, create_dataloader
from falldet.data.preprocessing import build_dataset
from falldet.data.unified import load_and_harmonize
from falldet.models.factory import create_model
from falldet.training.losses import get_loss_fn
from falldet.training.trainer import Trainer
from falldet.utils.config import load_config
from falldet.utils.device import get_device
from falldet.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train a fall detection model")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config YAML"
    )
    args, overrides = parser.parse_known_args()

    # Load config
    config = load_config(args.config, overrides)
    seed = config.get("seed", 42)
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    dc = config.get("data", {})
    tc = config.get("training", {})
    mc = config.get("model", {})
    oc = config.get("output", {})

    # Data pipeline
    raw_dir = Path("data/raw")
    splits_dir = Path("data/splits")

    print("Loading and harmonizing datasets...")
    samples = load_and_harmonize(raw_dir, dc.get("datasets", ["nhoyh"]))

    print("Building windows and splits...")
    train_win, val_win, test_win, norm_stats = build_dataset(
        samples,
        window_sec=dc.get("window_sec", 60.0),
        stride_sec=dc.get("stride_sec", 30.0),
        rate_hz=dc.get("target_rate_hz", 50.0),
        val_fraction=dc.get("val_fraction", 0.15),
        test_fraction=dc.get("test_fraction", 0.15),
        seed=seed,
        splits_dir=splits_dir,
    )

    # Augmentation config
    aug_cfg = dc.get("augmentation", {})
    aug_enabled = aug_cfg.get("enabled", False)

    train_ds = FallDetectionDataset(
        train_win, augment=aug_enabled, aug_config=aug_cfg if aug_enabled else None
    )
    val_ds = FallDetectionDataset(val_win)

    batch_size = tc.get("batch_size", 32)
    train_loader = create_dataloader(train_ds, batch_size=batch_size, weighted_sampling=True)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model_name = mc.get("name", "cnn1d")
    model_kwargs = mc.get(model_name, {})
    model = create_model(model_name, **model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} ({n_params:,} parameters)")

    # Loss
    pos_weight = compute_pos_weight(train_ds) if tc.get("pos_weight_auto", True) else 1.0
    criterion = get_loss_fn(
        tc.get("loss", "bce"),
        pos_weight=pos_weight,
        focal_gamma=tc.get("focal_gamma", 2.0),
    )
    print(f"Loss: {tc.get('loss', 'bce')} (pos_weight={pos_weight:.2f})")

    # Train
    output_dir = Path(oc.get("dir", "outputs"))
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    print(f"\nTraining for up to {tc.get('epochs', 100)} epochs...")
    print("-" * 80)
    best_metrics = trainer.train()
    print("-" * 80)
    print(f"\nBest validation AUC: {best_metrics.get('auc', 0):.4f}")
    print(f"  TPR: {best_metrics.get('tpr', 0):.3f}, FPR: {best_metrics.get('fpr', 0):.3f}")
    print(f"Checkpoint saved to {output_dir}/best.pt")


if __name__ == "__main__":
    main()
