# Notebooks

Colab-ready Jupyter notebooks for the fall detection training pipeline.

## Files

| File | What it does |
|------|-------------|
| `fall_detection_train.ipynb` | End-to-end training: downloads datasets, preprocesses, trains a 1D-CNN, evaluates against QFD targets, saves a checkpoint. **This is the main notebook.** |
| `fall_detection_inference_demo.ipynb` | Loads a saved checkpoint and demonstrates the `predict()` function on real test windows bundled in the checkpoint. Simulates how the cloud service calls the model. |
| `build_notebooks.py` | Source-of-truth Python script that generates the two `.ipynb` files. Edit this, not the notebooks directly. |

## How to run in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File -> Upload notebook -> choose `fall_detection_train.ipynb`
3. **Runtime -> Change runtime type -> Hardware accelerator -> GPU** (gives you a free T4)
4. Runtime -> Run all

The training notebook takes ~5 minutes on a T4 GPU. It will:
- Clone two public IMU datasets from GitHub (nhoyh + microchip)
- Harmonize them into 60-second windows at 50 Hz
- Split train/val/test by subject
- Train a 1D-CNN with early stopping
- Pick the operating threshold on the validation set
- Report final test metrics against the QFD targets
- Save everything (weights, config, norm stats, threshold, sample test windows) to `fall_detection_model.pt`

Download `fall_detection_model.pt` from Colab's file browser (left sidebar) if you want to use it locally.

## How to run the inference demo

1. First run `fall_detection_train.ipynb` to produce `fall_detection_model.pt`
2. Upload both files to a new Colab notebook (or the same session)
3. Open `fall_detection_inference_demo.ipynb` and run all cells

The demo uses real windows from the held-out test set (bundled in the checkpoint) to show that the `predict()` function correctly classifies them. It also benchmarks inference latency, which should be under 10 ms on a T4 GPU.

## Editing

**Do not edit the `.ipynb` files directly.** They are generated from `build_notebooks.py`. Edit that script and regenerate:

```bash
cd ml
source .venv/Scripts/activate    # or the equivalent for your OS
python notebooks/build_notebooks.py
```

This ensures the notebooks stay consistent and version-control-friendly (single source of truth).

## Relationship to the `falldet` package

The notebooks are **standalone** -- they re-define every function they need as a cell, with no imports from the `falldet` package. This keeps them easy to upload to Colab without any project setup.

The canonical, tested version of this code lives in `ml/falldet/` as an installable Python package. If you're iterating on the codebase locally, use that. If you want free GPU compute for a training run, use these notebooks.

If you change the source files in `ml/falldet/`, you should also update `build_notebooks.py` to keep the notebooks in sync. The logic is intentionally straightforward to translate: each cell in `build_notebooks.py` corresponds to one source file or one clear pipeline stage.
