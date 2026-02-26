# Sentry AI - Comprehensive Code Documentation

This document provides a detailed overview of the modules, functions, and scripts implemented in the `sentry` repository.

## 1. Core Library (`src/sentry_ai`)

### 1.1 `config.py`
**Purpose**: Loads configuration settings from YAML files.

- `load_config(config_path: str | Path) -> dict`
  - Reads a YAML file from the specified path.
  - Returns a dictionary representing the configuration. Used centrally across all entrypoint scripts to parse runtime parameters.

### 1.2 `dataset/ingest.py`
**Purpose**: Handles the ingestion of raw datasets from remote and local sources.

- `ingest_github(repo_url: str, subfolder_path: str, target_dir: str | Path, branch: str = "main")`
  - Clones the specified GitHub repository to a temporary directory.
  - Extracts the requested `subfolder_path` and copies it to `target_dir` using `shutil`.
  - Cleans up the temporary clone automatically.
- `ingest_local(source_dir: str | Path, target_dir: str | Path)`
  - Copies an entire local dataset directory to the specified `target_dir`.

### 1.3 `dataset/validate.py`
**Purpose**: Enforces strict format adherence for YOLO datasets.

- `validate_yolo_dataset(dataset_dir: str | Path, max_class_id: int = 1000) -> list[str]`
  - Validates the presence of `images/` and `labels/` directories.
  - Checks for orphaned images (missing labels) and orphaned labels (missing images).
  - Parses each `.txt` label file to ensure exactly 5 fields exist per line: `<class_id> <x_center> <y_center> <width> <height>`.
  - Enforces boundary rules: `0 <= x <= 1`, `0 <= y <= 1`, `0 < width <= 1`, `0 < height <= 1`.
  - Ensures `class_id` is non-negative and $\le$ `max_class_id`.

### 1.4 `dataset/audit.py`
**Purpose**: Gathers and computes dataset statistics.

- `audit_yolo_dataset(dataset_dir: str | Path) -> dict`
  - Iterates through the splits (`train`, `val`, `test`).
  - Counts the total number of images, labeled images, and empty images (background images with no labels).
  - Calculates the total number of bounding boxes and computes a frequency distribution of classes.
  - Calculates the percentile distributions for bounding box sizes based on width $\times$ height areas.

### 1.5 `dataset/merge.py`
**Purpose**: Safely combines multiple datasets (e.g., GitHub ingested + Custom local).

- `collect_items(dataset_dir: Path, prefix: str) -> list`
  - Discovers all images and their associated label files across all splits in a given dataset, assigning a unique prefix to prevent filename collisions during the merge.
- `remap_and_copy_label(src_label: Path, dst_label: Path, remap: dict)`
  - Reads a source label file and optionally translates the `class_id` using the user-provided `remap` dictionary, then writes it to the destination.
- `merge_datasets(config: dict)`
  - Uses the configuration to orchestrate merging. Supports two modes:
    1. **Preseve**: Keeps the original `train/val/test` splits of the incoming datasets.
    2. **Rebuild**: Shuffles all combined data and redistributes it according to `rebuild_ratios` specified in `configs/default.yaml`.

### 1.6 `yolo/train.py`
**Purpose**: Wraps the Ultralytics YOLO training procedure.

- `generate_data_yaml(dataset_dir: Path, output_yaml: Path, num_classes: int = 80)`
  - Automatically generates the required `data.yaml` file pointing to the merged dataset splits.
- `train_yolo(config: dict)`
  - Parses `training` config block (model architecture, epochs, batch size, target image size, device).
  - Initializes the Ultralytics model and starts the training phase, storing outputs in `runs/`.

### 1.7 `yolo/infer.py`
**Purpose**: Wraps Ultralytics for inference tasks.

- `infer_yolo(config: dict, source: str)`
  - Loads trained YOLO weights (default path from `configs/default.yaml`).
  - Predicts bounding boxes on the specified `source` (image, video, or directory).
  - Uses configured confidence thresholds and IoU limits, saving results under the `runs/` project directory.

---

## 2. Pipeline Scripts (`scripts/`)

The repository exposes thin CLI wrappers around the core library.

- **`ingest_github.py`**: Invokes `ingest_github` for remote data gathering.
- **`ingest_local.py`**: Invokes `ingest_local` for local staging.
- **`validate.py`**: Invokes `validate_yolo_dataset`, formatting and printing errors, returning exit code `1` if the dataset is invalid.
- **`audit.py`**: Invokes `audit_yolo_dataset`, displaying the comprehensive statistics as a formatted JSON payload.
- **`merge.py`**: Reads central config and executes `merge_datasets`.
- **`train.py`**: Reads central config and initiates `train_yolo`.
- **`infer.py`**: Reads central config and initiates `infer_yolo`.

---

## 3. Configuration Management

**`configs/default.yaml`** controls the entire pipeline state:
- **`dataset`**: Pointers to raw and merged data directories, merge behavior toggles, split ratios, and optional class mapping tables.
- **`training`**: YOLO architecture size (`yolov8n.yaml` default/pre-trained weights), hyper-parameters, and hardware targeting (`device`).
- **`inference`**: Checkpoint pathing, `conf_threshold`, and `iou_threshold`.
