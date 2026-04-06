# Evaluation and Metrics

## What We Measure and Why

The model outputs a raw logit per window. After applying sigmoid, we get a probability in [0, 1]. A threshold converts this to a binary prediction. The choice of threshold determines the trade-off between TPR and FPR.

### Primary Metrics

| Metric | Definition | Target | Why It Matters |
|--------|-----------|--------|----------------|
| **TPR** (True Positive Rate / Sensitivity / Recall) | TP / (TP + FN) | >= 95% ideal, >= 90% marginal | Missing a real fall can be fatal. This is the most important metric. |
| **FPR** (False Positive Rate) | FP / (FP + TN) | <= 10% ideal, <= 15% marginal | False alarms cause alarm fatigue. Staff stop trusting the system and ignore alerts. |
| **ROC AUC** | Area under the ROC curve | As high as possible | Threshold-independent measure of overall discrimination ability. 1.0 = perfect, 0.5 = random. |

### Secondary Metrics

| Metric | Definition | Notes |
|--------|-----------|-------|
| Precision | TP / (TP + FP) | What fraction of positive predictions are real falls |
| Specificity | TN / (TN + FP) = 1 - FPR | How well ADL is correctly identified |
| F1 | Harmonic mean of precision and recall | Balanced metric, but we care more about recall |
| F2 | Weighted F-score favoring recall 2:1 over precision | Better than F1 for our use case because missed falls are worse than false alarms |

## Metrics Implementation

**Location**: `falldet/evaluation/metrics.py`

### `compute_metrics(y_true, y_prob, threshold=0.5)`

Takes ground truth labels, predicted probabilities, and a threshold. Returns a dict with all metrics above plus the raw confusion matrix values (TP, FP, TN, FN).

```python
from falldet.evaluation.metrics import compute_metrics

metrics = compute_metrics(y_true, y_prob, threshold=0.5)
# {'tpr': 1.0, 'fpr': 0.375, 'precision': 0.59, 'roc_auc': 0.945, ...}
```

### `find_best_threshold(y_true, y_prob, target_tpr=0.95, max_fpr=0.10)`

Sweeps thresholds from 0.0 to 1.0 in 101 steps. Strategy:

1. **Primary goal**: Find thresholds where TPR >= target (0.95)
2. **Among those**: Pick the one with the lowest FPR
3. **Fallback**: If no threshold achieves target TPR within max FPR, maximize F2 score

Returns `(best_threshold, metrics_dict)`.

This threshold should be stored alongside the model checkpoint so the cloud inference service uses the same operating point.

### `compute_roc_curve(y_true, y_prob)`

Returns the full ROC curve data (arrays of FPR, TPR, thresholds) for plotting. Uses `sklearn.metrics.roc_curve` internally.

## How Evaluation Fits Into the Workflow

During training, the Trainer runs validation metrics at every epoch using threshold=0.5. This gives a rough sense of progress but is NOT the final evaluation.

For final evaluation on the test set:

1. Load the best checkpoint
2. Run inference on all test windows to get probabilities
3. Use `find_best_threshold` on the **validation** set to pick the operating threshold
4. Apply that threshold to the **test** set to get final metrics
5. Generate the full evaluation report

This two-step threshold selection prevents overfitting to the test set -- the threshold is chosen on val, then evaluated on test.

## Mapping to QFD Engineering Specs

The QFD chart from BME 362 defined 8 engineering requirements. Here's how the ML metrics map:

| QFD Requirement | ML Metric | Status |
|-----------------|-----------|--------|
| ER5: Maximize fall detection performance | TPR (sensitivity) | Measured in every training run |
| ER8: Minimize false alarm rate | FPR (1 - specificity) | Measured in every training run |
| ER2: Minimize system response latency | Inference time per packet | Not yet measured (need to benchmark) |
| ER7: Maximize fall type coverage | Per-fall-type TPR | Not yet implemented (need per-type eval) |

### What's Not Yet Implemented

- **Per-fall-type metrics**: Break down TPR by fall type (forward, backward, lateral, seated, syncope). The metadata is tracked per window but the evaluation code doesn't yet group by fall type.
- **Evaluation report generation**: `evaluation/report.py` is a placeholder. Should generate a markdown file with confusion matrix heatmap, ROC curve plot, per-type bar chart, and a table comparing results to QFD targets.
- **Inference latency benchmark**: Time how long a single 60s packet takes to process (forward pass only). This contributes to the ER2 system response latency target.
- **Dedicated `evaluate.py` script**: The `scripts/evaluate.py` entry point is not yet implemented. Should load a checkpoint, run on the test set, and generate the full report.
