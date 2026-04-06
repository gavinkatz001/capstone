# ML Infrastructure Documentation Index

> Written: 2026-04-06 | Author: Gavin Katz + Claude Code | Status: Initial build complete

This documentation suite covers the ML training infrastructure built for the fall detection capstone project. It is written so that any developer (or AI agent) picking up this codebase can understand what exists, why it was built this way, and what remains to be done.

## Documents

| # | File | Covers |
|---|------|--------|
| 01 | [system-architecture.md](01-system-architecture.md) | End-to-end system architecture, how the ML model fits into the wearable-gateway-cloud pipeline, and the constraints that shaped every design decision |
| 02 | [repo-structure.md](02-repo-structure.md) | Complete directory tree of `ml/`, what every file does, and how the Python package is organized |
| 03 | [environment-setup.md](03-environment-setup.md) | How to set up the dev environment from scratch, dependencies, and why `uv` was chosen |
| 04 | [data-pipeline.md](04-data-pipeline.md) | Dataset sources, loaders, harmonization, windowing, normalization, splitting -- the full journey from raw .mat/.csv to PyTorch tensors |
| 05 | [models-and-training.md](05-models-and-training.md) | Model architectures (CNN1D and planned LSTM/Transformer), training loop, loss functions, checkpointing, and how to run training |
| 06 | [evaluation-and-metrics.md](06-evaluation-and-metrics.md) | How models are evaluated, what metrics matter, threshold selection strategy, and how results map to the QFD engineering specs |
| 07 | [design-decisions.md](07-design-decisions.md) | Every significant design choice, the alternatives considered, and the reasoning. Read this to understand *why*, not just *what* |
| 08 | [known-issues-and-next-steps.md](08-known-issues-and-next-steps.md) | What's incomplete, known limitations, and the prioritized roadmap for what to build next |

## How to read these docs

- **New to the project?** Start with `01-system-architecture.md` for the big picture, then `02-repo-structure.md` to understand the codebase layout.
- **Want to run training?** Jump to `03-environment-setup.md` then `05-models-and-training.md`.
- **Want to add a new dataset?** Read `04-data-pipeline.md` -- it explains the loader interface and harmonization process.
- **Want to understand why something was done a certain way?** `07-design-decisions.md` is the authoritative reference.
- **Want to know what to build next?** `08-known-issues-and-next-steps.md` has the prioritized backlog.
