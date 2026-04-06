# Known Issues and Next Steps

## Known Issues

### I1: All Recordings Are Shorter Than 60 Seconds

Every recording in the current datasets is 1.1s to 51.6s. All windows are zero-padded to 3000 samples (60s). This means:

- The model is learning on data that is mostly zeros at the tail end
- Real-world 60-second packets from the actual wearable will have continuous motion data for the full window
- The model may develop a bias toward detecting falls in the first ~20 seconds of a window (where data exists) and ignoring the rest

**Mitigation**: When the team collects their own data with the actual wearable, recordings will be naturally 60 seconds. Fine-tuning on this data will correct the bias. In the meantime, data augmentation (time jitter -- randomly placing the recording within the 60s window) would help.

### I2: Small Dataset Size

369 total windows (114 falls, 255 ADL) across 21 subjects is small for deep learning. For context, ImageNet has 1.2 million images. Our dataset is closer to "toy dataset" scale.

**Mitigation priorities**:
1. Add more dataset loaders (EdgeFall has ~1500 trials)
2. Implement data augmentation (noise, scaling, time warping, rotation)
3. Use 30s stride for overlap (already done -- but only helps if recordings are > 60s)
4. Collect custom data with the actual wearable hardware
5. Consider self-supervised pre-training or transfer learning from activity recognition

### I3: Sensor Placement Mismatch

nhoyh data is wrist-mounted. The actual wearable's placement is TBD (could be chest, wrist, hip, or pendant). Fall acceleration signatures differ significantly by body placement.

**Mitigation**: Track placement as metadata (already done). Eventually train placement-specific models or validate cross-placement generalization. Collecting data from the actual device at the actual body position is the most important mitigation.

### I4: No Gyro in Microchip Dataset

The Microchip dataset has accelerometer only. Gyro channels are zero-filled. The model might learn that zeros in gyro channels = certain class, which is spurious.

**Mitigation**: The `has_gyro` flag is tracked per window. Future models could mask gyro channels for accel-only samples or train separate heads. For now, the Microchip dataset is small (20 samples) so its impact is limited.

### I5: Validation and Test Sets Are Small

Val = 37 windows (13 falls), Test = 65 windows (19 falls). Per-epoch validation metrics are noisy because a single misclassification changes TPR by ~7%.

**Mitigation**: Use AUC (threshold-independent) as the primary early-stopping metric rather than TPR at a fixed threshold. Increase dataset size (I2).

### I6: No Real Elderly Fall Data

All public datasets are simulated falls performed by young volunteers. The biomechanics of a 20-year-old simulating a fall are different from an 80-year-old actually falling (slower onset, less arm flailing, more crumpling).

**Mitigation**: This is a fundamental limitation of the field, not just our project. The best mitigation is collecting real fall data in LTC environments (requires ethics approval). Until then, test on as diverse a set of simulated falls as possible and focus on reducing false alarms (which use real ADL data).

---

## Prioritized Next Steps

### P0: Critical Path (Needed for a Working System)

**1. Run a full training cycle (100 epochs)**
- The 5-epoch smoke test showed AUC=0.945 and climbing. A full run should converge to a much better model.
- Command: `python scripts/train.py --config configs/default.yaml`
- Expected: AUC > 0.97, TPR > 0.95 at a properly tuned threshold

**2. Implement `scripts/evaluate.py`**
- Load best checkpoint, run on test set, run threshold sweep, generate metrics
- Needs: `find_best_threshold` from `evaluation/metrics.py` (already implemented)

**3. Implement `scripts/predict.py`**
- Load a checkpoint, take a single 60-second CSV, output a fall probability
- This is the interface the cloud app will use (conceptually)

### P1: High Priority (Significantly Improves Results)

**4. Add EdgeFall dataset loader**
- ~1500 trials would roughly 4x our data volume
- Requires IEEE DataPort access (may need academic login)
- Most impactful single thing for improving model performance

**5. Implement data augmentation transforms**
- Time jitter: randomly offset recording start within the 60s window
- Noise injection: Gaussian noise on accel/gyro channels (partially implemented)
- Random scaling: per-axis scaling (partially implemented)
- Time warping: stretch/compress via interpolation
- 3D rotation: simulate different sensor orientations

**6. Implement feature engineering pipeline**
- `falldet/features/time_domain.py`: magnitude, jerk, SMA, zero-crossings, per-channel stats
- `falldet/features/freq_domain.py`: FFT, spectral energy, dominant frequency, band energy ratios
- `falldet/features/extract.py`: orchestrate feature extraction per window
- Needed for the baseline logistic regression model

**7. Implement baseline logistic regression model**
- `falldet/models/baseline.py`: sklearn pipeline on engineered features
- Sets the performance floor. If logistic regression gets 90% AUC, the neural net needs to beat it.

### P2: Medium Priority (Useful for Experimentation)

**8. Implement LSTM model**
- `falldet/models/lstm.py`: BiLSTM with temporal downsampling
- May capture long-range temporal patterns the CNN misses

**9. Implement Transformer model**
- `falldet/models/transformer.py`: Patch-based Transformer encoder
- Best at learning where in the 60s window to attend to
- Since cloud-hosted, no compute constraints

**10. Implement evaluation report generation**
- `falldet/evaluation/report.py`: Generate markdown report with plots
- Confusion matrix heatmap, ROC curve, per-fall-type breakdown
- Comparison against QFD targets (TPR >= 95%, FPR <= 10%)

**11. Implement per-fall-type evaluation**
- Break down TPR by fall type (forward, backward, lateral, seated, syncope)
- The metadata is already tracked per window -- just needs grouping in the evaluation code

**12. Add UR Fall dataset loader**
- Accel-only, 30 falls. Supplementary data.
- Small but adds diversity in fall types

### P3: Nice to Have

**13. Create exploration notebook**
- `notebooks/01_data_exploration.ipynb`
- Plot sample fall vs ADL waveforms
- Histogram of recording durations
- Per-channel statistics
- Class distribution visualizations

**14. Write unit tests**
- `tests/test_preprocessing.py`: Verify windowing produces correct shapes, normalization math
- `tests/test_features.py`: Verify feature extraction on synthetic signals
- `tests/test_dataset.py`: Verify DataLoader batch shapes and label distribution

**15. Experiment tracking with MLflow**
- Optional: `pip install -e ".[tracking]"` enables MLflow
- `falldet/tracking/mlflow_logger.py`: Wrap existing logging calls
- Provides a web UI for comparing experiments

**16. Hyperparameter sweep infrastructure**
- Grid search or random search over lr, dropout, channel sizes, loss function
- Could be a simple bash script or a Python script with config generation

### P4: Future (After Prototype Hardware Exists)

**17. Collect real data from the actual wearable**
- Most important thing for real-world performance
- Will need ethics approval for data collection with LTC residents
- Fine-tune the model on this data

**18. Cloud inference service**
- Wrap the predict script in a REST API (Flask/FastAPI)
- Endpoint: POST /predict with a 60s packet, returns {is_fall: bool, confidence: float}
- Latency target: < 1 second per packet

**19. Model monitoring and retraining pipeline**
- Track inference-time metrics (prediction distribution, anomaly detection)
- Periodic retraining as more data is collected

---

## Files That Need to Be Created

| File | Status | Priority |
|------|--------|----------|
| `falldet/data/loaders/edgefall.py` | Empty placeholder | P1 |
| `falldet/data/loaders/ur_fall.py` | Empty placeholder | P2 |
| `falldet/features/time_domain.py` | Empty placeholder | P1 |
| `falldet/features/freq_domain.py` | Empty placeholder | P1 |
| `falldet/features/extract.py` | Empty placeholder | P1 |
| `falldet/models/baseline.py` | Empty placeholder | P1 |
| `falldet/models/lstm.py` | Empty placeholder | P2 |
| `falldet/models/transformer.py` | Empty placeholder | P2 |
| `falldet/evaluation/threshold.py` | Empty placeholder | P0 (logic exists in metrics.py) |
| `falldet/evaluation/report.py` | Empty placeholder | P2 |
| `scripts/evaluate.py` | Not created | P0 |
| `scripts/predict.py` | Not created | P0 |
| `scripts/explore_data.py` | Not created | P3 |
| `notebooks/01_data_exploration.ipynb` | Not created | P3 |
| `tests/test_preprocessing.py` | Not created | P3 |
| `tests/test_features.py` | Not created | P3 |
