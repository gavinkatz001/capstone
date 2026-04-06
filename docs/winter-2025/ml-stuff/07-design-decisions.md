# Design Decisions

Every significant choice made in the ML infrastructure, the alternatives considered, and why we chose what we did.

---

## D1: PyTorch over TensorFlow

**Decision**: Use PyTorch as the ML framework.

**Alternatives considered**:
- TensorFlow/Keras: More mature TFLite export for edge devices, but heavier API
- JAX: Cutting edge but steep learning curve
- PyTorch + TFLite export: Train in PyTorch, convert for deployment

**Why PyTorch**:
- Most flexible for custom model architectures and experimentation
- Dominant in research -- most published fall detection models have PyTorch code
- Clean, Pythonic API that's easier to learn for a student team
- Since inference is cloud-hosted (not edge), we don't need TFLite export
- If edge deployment becomes needed later, ONNX export from PyTorch is straightforward

---

## D2: 50 Hz Unified Sampling Rate

**Decision**: Resample all datasets to 50 Hz.

**Alternatives considered**:
- 100 Hz (keep Microchip native rate)
- 25 Hz (more aggressive downsampling for smaller data)
- Variable rate (keep each dataset at its native rate, handle in the model)

**Why 50 Hz**:
- nhoyh (our largest dataset at 349 records) is already at 50 Hz -- no resampling needed for 94% of data
- Falls are low-frequency events. The impact spike has frequency content mainly below 20 Hz. Nyquist at 50 Hz (25 Hz) captures everything.
- A 60-second window at 50 Hz = 3000 samples. At 100 Hz it would be 6000 -- double the memory, double the compute, no accuracy benefit.
- 25 Hz would work for fall detection but might miss subtle pre-fall patterns (tremor, shuffling) that occur at 10-15 Hz.

---

## D3: 60-Second Window Size (Not a Choice -- An Architectural Constraint)

**Decision**: All data is windowed to exactly 60 seconds.

**This is not a design choice -- it's dictated by the system architecture.** The wearable device transmits 60-second packets. The cloud model receives 60-second packets. The ML pipeline must match.

**Implications**:
- Most existing recordings are 15-50 seconds, so they are zero-padded to 60 seconds
- The model must learn to handle windows where a fall occupies only a small fraction of the 60 seconds
- Short recordings with zero-padding mean the model sees a lot of zeros. This could be a problem but hasn't been in initial tests (AUC > 0.94 after 5 epochs)
- If we later collect our own data with the actual wearable, recordings will naturally be 60 seconds

---

## D4: Subject-Based Train/Val/Test Splits

**Decision**: Never put the same subject's data in both train and test sets.

**Alternative**: Random split at the window level (simpler, more balanced).

**Why subject-based**:
- Each person has unique movement patterns, sensor attachment angles, and biomechanics
- A random split would let the model memorize "subject_07 walks this way" and classify their fall correctly just by recognizing the subject, not the fall pattern
- Subject-based splits give a pessimistic but realistic estimate of performance on new, unseen users
- This is standard practice in the fall detection literature and required for publishable results
- With 21 subjects, we get roughly 15/3/3 for train/val/test -- enough for meaningful evaluation

**Trade-off**: The val and test sets are small (~37 and ~65 windows). This makes per-epoch metrics noisy. But honest evaluation is worth the noise.

---

## D5: Channels-First Tensor Format (6, 3000)

**Decision**: Store tensors as `(batch, channels, time)` = `(batch, 6, 3000)`.

**Alternative**: Channels-last `(batch, time, channels)` = `(batch, 3000, 6)` -- more natural for LSTMs and reading as a table.

**Why channels-first**:
- PyTorch's `Conv1d` expects `(batch, channels, length)`. This is the primary model architecture.
- Avoids a transpose in the forward pass of the most-used model
- LSTMs expect `(batch, seq, features)` so they need a transpose -- but that's a single `.permute(0, 2, 1)` in the LSTM model's forward method
- Consistency with PyTorch conventions (Conv2d also uses channels-first)

---

## D6: Zero-Dependency CSV Logger (Not MLflow/W&B)

**Decision**: Use a simple CSV + JSON logger for experiment tracking.

**Alternatives considered**:
- MLflow: Full experiment tracking server with UI
- Weights & Biases: Cloud-hosted tracking with collaboration features
- TensorBoard: PyTorch-native logging with web UI

**Why CSV/JSON**:
- Zero setup, zero accounts, zero servers to run
- Works offline on any machine
- Files are human-readable and version-controllable
- Sufficient for a 4-person capstone team doing at most a few dozen experiments
- MLflow is available as an optional extra (`pip install -e ".[tracking]"`) if the team wants it later
- CSV files can be loaded into pandas for analysis: `pd.read_csv("outputs/metrics.csv")`

---

## D7: No PyTorch Lightning

**Decision**: Write a raw PyTorch training loop in `trainer.py` (~150 lines).

**Alternative**: Use PyTorch Lightning which abstracts the training loop.

**Why raw PyTorch**:
- For a team learning ML, understanding the raw training loop is educational and valuable
- The Trainer class is ~150 lines of straightforward code -- simple enough to read, modify, and debug
- Lightning adds a significant abstraction layer and its own set of conventions to learn
- Our training loop has no exotic requirements (no multi-GPU, no mixed precision, no distributed training)
- Debugging is easier when you can see every line of the training loop
- If the team later wants Lightning, wrapping the existing code into a LightningModule is a ~30 minute task

---

## D8: BCEWithLogitsLoss with pos_weight (Not Focal Loss)

**Decision**: Default to weighted BCE loss. Focal loss available as an option.

**Why weighted BCE as default**:
- Standard, well-understood, and effective for moderate class imbalance
- `pos_weight = n_negative / n_positive` (currently 2.26) makes the loss treat each fall window as worth ~2.26 ADL windows
- Combined with weighted sampling in the DataLoader, this provides two layers of imbalance handling
- Focal loss is more complex and harder to tune (extra gamma hyperparameter). It's better when the imbalance is extreme (e.g., 1:100). Our imbalance is ~1:2.3, which is moderate.

---

## D9: SampleRecord Dataclass (Not pandas DataFrame)

**Decision**: Use Python dataclasses for per-sample data throughout the pipeline.

**Alternative**: Use a single large DataFrame with columns for all metadata.

**Why dataclasses**:
- Each sample has a variable-length numpy array for accel/gyro data. DataFrames handle this awkwardly.
- Type hints make the expected fields explicit and IDE-navigable
- Easier to pass through function pipelines than DataFrame rows
- No risk of column name typos or type confusion
- The SampleRecord -> UnifiedSample -> Window progression makes the data transformation stages explicit

---

## D10: Editable Package Install (Not sys.path Hacks)

**Decision**: Install `falldet` as an editable package via `pip install -e .`

**Alternative**: Add `sys.path.insert(0, ...)` at the top of every script.

**Why editable install**:
- Clean imports from anywhere: `from falldet.data.unified import load_and_harmonize`
- No `sys.path` manipulation needed (though scripts include it as a fallback)
- Works identically in scripts, notebooks, tests, and the Python REPL
- The `pyproject.toml` defines the package, making it installable and distributable

---

## D11: Git-Clone for Dataset Download (Not ZIP/Tar Archives)

**Decision**: Use `git clone --depth 1` to download GitHub-hosted datasets.

**Alternative**: Download ZIP archives via HTTP.

**Why git clone**:
- Simpler code -- one subprocess call vs. HTTP download + unzip
- `--depth 1` avoids downloading full git history (saves bandwidth)
- The `.git` directory is deleted after cloning to save space
- Works reliably through corporate firewalls that might block direct file downloads

**Windows-specific fix**: Git objects on Windows are sometimes read-only. The download script uses a custom `shutil.rmtree` handler (`_force_remove_readonly`) that `os.chmod(path, stat.S_IWRITE)` before deleting.

---

## D12: Config via YAML with CLI Dot-Notation Overrides

**Decision**: Single `default.yaml` config file with CLI overrides like `--model.name=lstm`.

**Alternatives considered**:
- Hydra (Facebook's config framework): Powerful but complex, steep learning curve
- argparse only: Flat, hard to organize hierarchically
- Environment variables: Not structured enough for nested configs

**Why YAML + dot overrides**:
- YAML is human-readable and supports nesting naturally
- One file to see all settings at once
- CLI overrides allow quick experiments without editing the YAML file
- The config loader is ~50 lines of code (`config.py`), no external dependencies beyond `pyyaml`
- The full config is saved alongside each run's outputs (`config.json`) for reproducibility
