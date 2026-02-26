# Sentry

A Python repository providing scripts-only utilities for YOLO dataset ingestion, validation, auditing, merging, training, and inference.

## Setup

1. Create a virtual environment:
   ```nu
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install the package in editable mode with dependencies:
   ```nu
   pip install -e .
   ```

## Workflow

This repository uses [Ultralytics YOLO](https://docs.ultralytics.com/) underneath but wraps it in robust data pipelines.

1. **Ingest Datasets**:
   - From GitHub: `python scripts/ingest_github.py --repo <URL> --path <SUBPATH> --branch main`
   - From Local: `python scripts/ingest_local.py --source_dir <PATH>`

2. **Validate & Audit**:
   - Validate structure & labels: `python scripts/validate.py --dataset_dir data/raw/custom_dataset`
   - Print dataset statistics: `python scripts/audit.py --dataset_dir data/raw/custom_dataset`

3. **Merge**:
   - Merge GitHub and custom collections: `python scripts/merge.py --config configs/default.yaml`

4. **Train**:
   - Train YOLO model on merged dataset: `python scripts/train.py --config configs/default.yaml`

5. **Inference**:
   - Run inference on images or videos: `python scripts/infer.py --config configs/default.yaml --source data/samples/video.mp4`

## Project Structure
- `configs/`: YAML configuration files.
- `src/sentry_ai/`: Core Python library modules.
- `scripts/`: Entrypoint scripts.
- `data/`: Local storage for raw and processed datasets (ignored in git).
- `runs/`: Output directory for Ultralytics YOLO training/inference (ignored in git).
