"""Simple CSV/JSON experiment logger with zero external dependencies."""

import csv
import json
import time
from pathlib import Path


class TrainingLogger:
    """Log training metrics to CSV and save run summaries as JSON."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / "metrics.csv"
        self.summary_path = self.output_dir / "summary.json"
        self._csv_file = None
        self._csv_writer = None
        self._start_time = time.time()
        self._best_val_auc = 0.0
        self._best_epoch = 0

    def _ensure_csv(self, fieldnames: list[str]) -> None:
        if self._csv_file is None:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._csv_writer.writeheader()

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict, lr: float) -> None:
        """Log one epoch's metrics."""
        row = {"epoch": epoch, "lr": f"{lr:.6f}"}
        for k, v in train_metrics.items():
            row[f"train_{k}"] = f"{v:.6f}" if isinstance(v, float) else v
        for k, v in val_metrics.items():
            row[f"val_{k}"] = f"{v:.6f}" if isinstance(v, float) else v

        self._ensure_csv(list(row.keys()))
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # Track best
        val_auc = val_metrics.get("auc", 0.0)
        if val_auc > self._best_val_auc:
            self._best_val_auc = val_auc
            self._best_epoch = epoch

    def save_summary(self, config: dict, final_metrics: dict | None = None) -> None:
        """Save run summary JSON."""
        summary = {
            "training_time_sec": time.time() - self._start_time,
            "best_val_auc": self._best_val_auc,
            "best_epoch": self._best_epoch,
            "config": config,
        }
        if final_metrics:
            summary["test_metrics"] = final_metrics

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def save_config(self, config: dict) -> None:
        """Save the config used for this run."""
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.close()
