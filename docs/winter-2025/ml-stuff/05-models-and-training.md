# Models and Training

## Model Architecture: 1D-CNN (Implemented)

**Location**: `falldet/models/cnn1d.py`

This is the first and currently only neural network model. It processes raw 6-channel time-series windows.

### Architecture Diagram

```
Input: (batch, 6, 3000)                           # 6 IMU channels, 3000 timesteps

Conv1d(6, 32, kernel=7, stride=2, pad=3)           # large kernel catches impact signature
BatchNorm1d(32) + ReLU                             -> (batch, 32, 1500)

Conv1d(32, 64, kernel=5, stride=2, pad=2)
BatchNorm1d(64) + ReLU                             -> (batch, 64, 750)

Conv1d(64, 128, kernel=5, stride=2, pad=2)
BatchNorm1d(128) + ReLU                            -> (batch, 128, 375)

Conv1d(128, 128, kernel=3, stride=2, pad=1)
BatchNorm1d(128) + ReLU                            -> (batch, 128, 188)

AdaptiveAvgPool1d(1)                               -> (batch, 128, 1)
Squeeze                                            -> (batch, 128)
Dropout(0.3)
Linear(128, 1)                                     -> (batch, 1) logits
```

### Why This Architecture

- **Large first kernel (7)**: At 50 Hz, a kernel of 7 spans 0.14 seconds. Fall impacts produce sharp acceleration spikes over ~0.1-0.3 seconds. The first layer is sized to capture this.
- **Progressive stride-2 downsampling**: Reduces the 3000-length sequence to 188 in 4 layers. This is more parameter-efficient than pooling layers and learns the downsampling.
- **BatchNorm**: Stabilizes training and allows higher learning rates.
- **AdaptiveAvgPool1d(1)**: Reduces any remaining sequence length to 1, making the network input-length-agnostic (though we always feed 3000).
- **Dropout(0.3)**: Regularization to prevent overfitting on our small dataset (267 training windows).
- **Single output logit**: Binary classification. Sigmoid is applied externally (in the loss function via `BCEWithLogitsLoss`, or in evaluation via `torch.sigmoid`).

### Parameter Count

102,881 total parameters. This is intentionally small -- the dataset is small, so a larger model would overfit.

### Configuration

Configurable via `configs/default.yaml` under `model.cnn1d`:

```yaml
model:
  name: "cnn1d"
  cnn1d:
    channels: [32, 64, 128, 128]     # output channels per conv layer
    kernel_sizes: [7, 5, 5, 3]       # kernel size per conv layer
    dropout: 0.3
```

## Planned Models (Not Yet Implemented)

### Baseline: Logistic Regression

**File**: `falldet/models/baseline.py` (placeholder)

A sklearn `LogisticRegression` trained on engineered features (magnitude, jerk, FFT, etc.) extracted per-window. This will set the performance floor -- if a linear model on hand-crafted features works well, the neural nets need to beat it to justify their complexity.

### BiLSTM

**File**: `falldet/models/lstm.py` (placeholder)

```
Input: (batch, 6, 3000)
Downsample 5x -> (batch, 6, 600) -- reduces LSTM computational cost
Transpose -> (batch, 600, 6) -- LSTM expects (batch, seq, features)
LSTM(input=6, hidden=128, layers=2, bidirectional=True, dropout=0.2)
Last hidden -> (batch, 256) -- 128*2 for bidirectional
Dropout(0.3)
Linear(256, 1)
```

Why planned: LSTMs are the go-to for sequential pattern recognition. Bidirectional because we have the full window (not streaming).

### Transformer Encoder

**File**: `falldet/models/transformer.py` (placeholder)

```
Input: (batch, 6, 3000)
Patch embedding: Conv1d(6, 64, kernel=50, stride=25) -> (batch, 64, 119 patches)
  Each patch = 0.5 seconds of data
Transpose -> (batch, 119, 64)
Positional encoding (learnable)
TransformerEncoder(d_model=64, nhead=4, layers=3, ff=256, dropout=0.1)
Mean pool -> (batch, 64)
Linear(64, 1)
```

Why planned: Self-attention can learn to focus on the fall impact region within the 60-second window, regardless of where it occurs. No compute constraints since we run on cloud.

### Model Factory

**Location**: `falldet/models/factory.py`

```python
MODEL_REGISTRY = {"cnn1d": CNN1D}  # add more as implemented

model = create_model("cnn1d", channels=[32, 64, 128, 128], dropout=0.3)
```

When adding a new model, import it in `factory.py` and add it to `MODEL_REGISTRY`.

---

## Training Infrastructure

### Trainer Class

**Location**: `falldet/training/trainer.py`

The `Trainer` class handles the entire training lifecycle:

```python
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    config=config,
    device=device,
    output_dir="outputs",
)
best_metrics = trainer.train()
```

What it does each epoch:
1. **Train**: Forward pass, loss, backward, gradient clipping (max_norm=1.0), optimizer step
2. **Validate**: Forward pass on val set, compute loss + full metrics (TPR, FPR, AUC, etc.)
3. **Log**: Write epoch row to `metrics.csv`
4. **Checkpoint**: If val AUC improves, save `best.pt`
5. **Early stop**: If val AUC hasn't improved for `patience` epochs, stop

### Early Stopping

Monitors validation AUC. Default patience = 15 epochs. The counter resets whenever a new best AUC is achieved. When the counter reaches patience, training stops.

### Gradient Clipping

`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` is applied every step. This prevents exploding gradients, especially important for time-series data which can have sharp spikes.

### Loss Functions

**Location**: `falldet/training/losses.py`

Two options, selectable via config:

1. **BCEWithLogitsLoss** (default): Standard binary cross-entropy. Uses `pos_weight` to handle class imbalance -- the loss for fall samples is multiplied by `n_negative / n_positive` (currently 2.26).

2. **FocalLoss**: Down-weights easy examples with `(1 - p_t)^gamma` factor. Focuses training on hard-to-classify windows. Gamma default = 2.0.

### Optimizer and Scheduler

- **Optimizer**: AdamW with lr=1e-3, weight_decay=1e-4
- **Scheduler**: CosineAnnealingLR over the total epochs. Smoothly decays the learning rate from initial to near-zero.

### Checkpointing

Each checkpoint `.pt` file contains:
```python
{
    "epoch": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": OrderedDict,
    "metrics": dict,          # val metrics at this epoch
    "config": dict,           # full config used for this run
}
```

Two checkpoints are saved:
- `best.pt`: the epoch with the highest validation AUC
- `last.pt`: the final epoch (useful for debugging or resuming)

---

## Running Training

### Basic Run

```bash
cd ml
python scripts/train.py --config configs/default.yaml
```

### With Overrides

CLI overrides use dot notation -- they merge into the YAML config:

```bash
# Change model
python scripts/train.py --config configs/default.yaml --model.name=lstm

# Change hyperparameters
python scripts/train.py --config configs/default.yaml --training.lr=0.0005 --training.epochs=50

# Use focal loss
python scripts/train.py --config configs/default.yaml --training.loss=focal

# Quick test run
python scripts/train.py --config configs/default.yaml --training.epochs=3
```

### What Training Produces

After a run, `ml/outputs/` contains:

| File | Contents |
|------|----------|
| `best.pt` | Best model checkpoint (by val AUC) |
| `last.pt` | Last epoch checkpoint |
| `metrics.csv` | Per-epoch: train_loss, val_loss, val_tpr, val_fpr, val_auc, lr |
| `config.json` | Exact config used for this run |
| `summary.json` | Training time, best epoch, best AUC |

### Observed Performance (5-epoch quick run, CPU)

```
Epoch 0: val_auc=0.9295, val_tpr=1.000, val_fpr=0.958
Epoch 1: val_auc=0.9327, val_tpr=1.000, val_fpr=0.667
Epoch 2: val_auc=0.9359, val_tpr=1.000, val_fpr=0.542
Epoch 3: val_auc=0.9423, val_tpr=1.000, val_fpr=0.417
Epoch 4: val_auc=0.9455, val_tpr=1.000, val_fpr=0.375
```

AUC is steadily increasing. TPR is perfect at 1.0, but FPR at the default 0.5 threshold is still high (0.375). This is because:
1. Only 5 epochs of training (needs ~50-100 for convergence)
2. The default threshold of 0.5 may not be optimal (threshold sweep will find a better one)
3. The dataset is small and most recordings are padded to 60s (lots of zero-padding)

A full training run (100 epochs with early stopping) and proper threshold tuning should significantly improve FPR while maintaining high TPR.
