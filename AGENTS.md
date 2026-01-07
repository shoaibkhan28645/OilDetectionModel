# Repository Guidelines

## Project Structure & Module Organization

- Code: `src/` (`data_preprocessing.py`, `model.py`, `train.py`, `predict.py`)
- Data: `data/train/`, `data/validation/`, `data/test/` with class folders `coriander_oil/` and `mustard_oil/`
- Models: `models/` for saved `.h5` files
- Dependencies: `requirements.txt` (TensorFlow/Keras, OpenCV, NumPy, etc.)
- Note: Scripts use relative paths from `src/` (e.g., `../data`, `../models`). Run commands from inside `src/` unless you adjust paths.

## Build, Test, and Development Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train (recommended transfer learning)
cd src && python train.py --transfer

# Alternative: simple CNN
python train.py --simple

# Inspect model summary
python model.py

# Predict (single image or directory)
python predict.py ../models/oil_detection_transfer_learning_20250901_033505.h5 ../data/test/image1.jpg
```

## Coding Style & Naming Conventions

- Follow PEP 8 (4-space indentation, 88–100 char lines where reasonable).
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- Dataset folder names are fixed: `coriander_oil`, `mustard_oil`.
- Include concise docstrings and helpful print logs for CLI UX.

## Testing Guidelines

- No formal unit tests yet. Validate via:
  - Small sample runs: `train.py` with few images per class.
  - Architecture checks: `python model.py`.
  - Prediction sanity: run `predict.py` on known-labeled images.
- If adding tests, prefer `pytest`, place files under `tests/` named `test_*.py`.

## Commit & Pull Request Guidelines

- Commit messages: imperative mood, concise scope (e.g., "train: add early stopping config").
- PRs must include: summary, rationale, commands to reproduce, and before/after metrics or screenshots (plots/confusion matrix when relevant).
- Link related issues and note any data assumptions or path changes.
- Exclude large datasets and generated models from commits; keep artifacts in `models/` locally (use Git LFS if versioning large files).

## Security & Configuration Tips

- Do not commit personal datasets; ensure images are anonymized as needed.
- GPU/CPU selection is managed by TensorFlow; set `CUDA_VISIBLE_DEVICES` if required.
- For less noisy logs, you can export `TF_CPP_MIN_LOG_LEVEL=2` during experiments.
