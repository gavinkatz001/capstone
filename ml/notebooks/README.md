# Notebooks

Colab-ready Jupyter notebooks for the fall detection training pipeline.

For the current preliminary six-axis baseline, use `preliminary_fall_training.ipynb` and read [`../DATA_SOURCE_PLAN.md`](../DATA_SOURCE_PLAN.md) for the source assessment and archive staging instructions.

## Files

| File | What it does |
|------|-------------|
| `preliminary_fall_training.ipynb` | Standalone Colab notebook for a 4-second, six-axis preliminary model. It fetches HR_IMU and JasonKau and accepts staged FallAllD, UP-Fall, SisFall, SmartFallMM, and UNIVPM data. |
| `fall_detection_train.ipynb` | Earlier 60-second training experiment. Its current Microchip path pads accelerometer-only data into six channels; use the preliminary notebook for a verified six-axis baseline. |
| `fall_detection_inference_demo.ipynb` | Loads a saved checkpoint and demonstrates the `predict()` function on real test windows bundled in the checkpoint. Simulates how the cloud service calls the model. |
| `build_notebooks.py` | Source-of-truth Python script for the earlier training and inference notebooks. |
| `build_preliminary_colab.py` | Source-of-truth Python script for the preliminary notebook. |
| `multisource_loaders.py` | Source adapters embedded into the preliminary notebook during generation. |

## How to run in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File -> Upload notebook -> choose `preliminary_fall_training.ipynb`
3. **Runtime -> Change runtime type -> Hardware accelerator -> GPU** (gives you a free T4)
4. Runtime -> Run all

The preliminary notebook will:
- Clone HR_IMU and JasonKau, then load any additional archives staged under its `DATA_ROOT`
- Harmonize accepted recordings into 4-second windows at 50 Hz
- Split train/val/test by subject within each source (JasonKau is training only)
- Train a 1D-CNN with early stopping
- Pick the operating threshold on the validation set
- Report overall and per-source held-out test metrics
- Save weights, normalization, threshold, and split metadata to `preliminary_fall_model.pt`

The notebook offers `preliminary_fall_model.pt` for download after training.

## How to run the inference demo

1. First run `fall_detection_train.ipynb` to produce `fall_detection_model.pt`
2. Upload both files to a new Colab notebook (or the same session)
3. Open `fall_detection_inference_demo.ipynb` and run all cells

The demo uses real windows from the held-out test set (bundled in the checkpoint) to show that the `predict()` function correctly classifies them. It also benchmarks inference latency, which should be under 10 ms on a T4 GPU.

## Editing

**Do not edit the `.ipynb` files directly.** For the preliminary notebook, edit `build_preliminary_colab.py` or `multisource_loaders.py` and regenerate:

```bash
cd ml
source .venv/Scripts/activate    # or the equivalent for your OS
python notebooks/build_preliminary_colab.py
```

This ensures the notebooks stay consistent and version-control-friendly (single source of truth).

## Relationship to the `falldet` package

The notebooks are **standalone** -- they re-define every function they need as a cell, with no imports from the `falldet` package. This keeps them easy to upload to Colab without any project setup.

The canonical, tested version of this code lives in `ml/falldet/` as an installable Python package. If you're iterating on the codebase locally, use that. If you want free GPU compute for a training run, use these notebooks.

If you change the source files in `ml/falldet/`, you should also update `build_notebooks.py` to keep the notebooks in sync. The logic is intentionally straightforward to translate: each cell in `build_notebooks.py` corresponds to one source file or one clear pipeline stage.
