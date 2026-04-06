"""Training loop with validation, early stopping, and checkpointing."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from falldet.evaluation.metrics import compute_metrics
from falldet.tracking.logger import TrainingLogger


class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.counter = 0

    def step(self, score: float) -> bool:
        """Returns True if training should stop."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def is_best(self, score: float) -> bool:
        return score >= self.best_score


class Trainer:
    """Handles the full training lifecycle."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: dict,
        device: torch.device,
        output_dir: str | Path = "outputs",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)

        tc = config.get("training", {})
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tc.get("lr", 1e-3),
            weight_decay=tc.get("weight_decay", 1e-4),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=tc.get("epochs", 100)
        )
        self.early_stopping = EarlyStopping(patience=tc.get("patience", 15))
        self.logger = TrainingLogger(self.output_dir)
        self.logger.save_config(config)

        self.epochs = tc.get("epochs", 100)

    def train(self) -> dict:
        """Run full training. Returns best validation metrics."""
        best_metrics = {}

        for epoch in range(self.epochs):
            train_metrics = self._train_epoch()
            val_metrics = self._validate()
            lr = self.optimizer.param_groups[0]["lr"]

            self.logger.log_epoch(epoch, train_metrics, val_metrics, lr)
            self.scheduler.step()

            val_auc = val_metrics.get("auc", 0.0)
            tpr = val_metrics.get("tpr", 0.0)
            fpr = val_metrics.get("fpr", 1.0)

            print(
                f"  Epoch {epoch:3d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_auc={val_auc:.4f} | "
                f"val_tpr={tpr:.3f} val_fpr={fpr:.3f} | "
                f"lr={lr:.6f}"
            )

            if self.early_stopping.is_best(val_auc):
                best_metrics = val_metrics
                self._save_checkpoint("best.pt", epoch, val_metrics)

            if self.early_stopping.step(val_auc):
                print(f"  Early stopping at epoch {epoch}")
                break

        self._save_checkpoint("last.pt", epoch, val_metrics)
        self.logger.save_summary(self.config, best_metrics)
        self.logger.close()

        return best_metrics

    def _train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y, _meta in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_labels = []
        n_batches = 0

        for x, y, _meta in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy().ravel())
            n_batches += 1

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)

        metrics = compute_metrics(y_true, y_prob, threshold=0.5)
        metrics["loss"] = total_loss / max(n_batches, 1)
        metrics["auc"] = metrics.pop("roc_auc", 0.0)

        return metrics

    def _save_checkpoint(self, filename: str, epoch: int, metrics: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
            path,
        )
