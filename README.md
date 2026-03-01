<p align="center">
  <img src="assets/logo.svg" alt="Sentry Logo" width="200"/>
</p>

<h1 align="center">Sentry — Model Pipeline</h1>

<p align="center">
  <b>YOLO-based weapon detection for real-world surveillance conditions</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12"/>
  <img src="https://img.shields.io/badge/YOLO-Ultralytics-orange?logo=data:image/png;base64,&logoColor=white" alt="Ultralytics"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status"/>
</p>

---

## The Story Behind Sentry

Sentry started as a question: *what does it actually take to build a weapon detector that works in the real world?*

Not a demo that works on clean stock photos. Not a model that performs on a curated benchmark and falls apart the moment someone points a CCTV camera at a poorly-lit parking lot. A real detector — one that generalises across camera angles, lighting conditions, and object sizes that are awkward for most architectures to handle.

The answer, it turned out, involved a lot of dataset engineering before a single training epoch ever started.

This repository is the AI core of Sentry. It contains the tooling we built to go from a pile of heterogeneous, incompatible datasets with different class schemas, annotation formats, and split structures — to a single, clean, validated YOLO dataset ready for training.

---

## What This Branch Contains

This is the `main` branch — it holds the **pipeline library and scripts**. No frontend. No UI. Pure AI infrastructure.

| Stage        | What it does                                                                           |
| ------------ | -------------------------------------------------------------------------------------- |
| **Ingest**   | Pull datasets from GitHub repos or local paths into a unified staging area             |
| **Validate** | Enforce YOLO format rules — catch orphaned files, malformed labels, out-of-bound boxes |
| **Audit**    | Collect statistics: class distribution, bounding box size percentiles, split counts    |
| **Merge**    | Combine multiple datasets; remap class IDs; rebuild or preserve train/val/test splits  |
| **Train**    | Kick off a YOLO training run with a single config file                                 |
| **Infer**    | Run inference on images, videos, or directories                                        |

---

## Design Philosophy

The objective is not just to train a detector. The objective is to build a **robust, production-grade CCTV threat detection model** with the following properties:

- Detects small weapons in large, high-resolution frames
- Handles occlusion, motion blur, and inconsistent lighting
- Maintains reproducible, auditable dataset provenance
- Minimises silent annotation errors before they corrupt a training run
- Generalises beyond the training distribution

---

## The Training Notebook

> [`notebooks/merged_dataset_training.ipynb`](notebooks/merged_dataset_training.ipynb)

This notebook is where everything comes together.

After weeks of collecting and cleaning data across four different sources, converting annotation formats, repairing corrupt XML files, resolving class ID conflicts, and running validation and audit passes — the training notebook is the moment of truth. It runs end-to-end on a dedicated remote GPU server and produces the final Sentry model weights.

### What it does

1. **Downloads the final dataset** — a 2.78 GB packaged archive of the fully merged and validated dataset, pulled directly from cloud storage onto the remote machine.

2. **Pulls the Sentry AI library** — downloads the latest version of this repository and adds it to the Python path so all pipeline modules are available without installation.

3. **Loads the pipeline config** — reads `config/model_pipeline.yaml` which defines dataset paths, training hyperparameters, and device targeting in one place.

4. **Runs dataset analysis** — before training starts, the notebook calls `analyze_dataset()` to generate three diagnostic visualisations and saves them to `analyses/`:
   - **Class distribution** — confirms only one class (`Weapon`) is present across all splits
   - **Bounding box centre heatmap** — reveals whether weapon detections are spatially biased (e.g., always centred, always on the left)
   - **Box size category breakdown** — shows the proportion of small, medium, and large bounding boxes, critical for tuning anchor settings

5. **Trains the model** — calls `train_yolo(config)`, which initialises an Ultralytics YOLO model and kicks off the full training run. Training outputs, checkpoints, and metrics land in `runs/`.

### Why this notebook matters

Most object detection projects treat training as the starting line. For Sentry, training is the finish line of an extensive data pipeline. The notebook is intentionally minimal in code — one config load, one analysis call, one training call — because all the complexity has already been absorbed upstream by the library. This makes the training run reproducible: anyone with access to the config and the dataset archive can reproduce the exact same run from scratch.

---

## Dataset Engineering

Training a good weapon detector is 80% dataset work. This pipeline exists to make that work systematic.

We actively work on:

- Merging compatible YOLO-format datasets from different sources
- Cleaning inconsistent annotations and fixing label schema mismatches
- Validating bounding box normalisation and boundary rules
- Auditing class distribution to catch severe imbalance early
- Removing corrupted or unreadable samples
- Ensuring that every image has a corresponding label (and vice versa)

### Focus Areas

- Small-object detection at surveillance-typical resolutions
- CCTV-like oblique camera angles
- Realistic weapon positioning (holstered, partially occluded, handheld)
- Negative examples — frames with no weapons, to prevent over-triggering

---

## Datasets Used

The final merged training dataset was assembled from four sources. All class labels were unified under a single class: **`Weapon (class 0)`**.

---

### 1. Kaggle — Weapons in Images (Segmented Videos)

**Source:** https://www.kaggle.com/datasets/jubaerad/weapons-in-images-segmented-videos  
**Author:** @jubaerad on Kaggle  
**Format:** YOLO `.txt` labels with corresponding images  
**Content:** Weapon images extracted from segmented video footage. Includes a held-out test set used as our evaluation split.

This dataset provided a large volume of diverse weapon imagery and formed the backbone of our training set. We are grateful for the effort put into curating this resource.

---

### 2. Ari-DASCI — OD-WeaponDetection (Knife Detection)

**Source:** https://github.com/ari-dasci/OD-WeaponDetection  
**Authors:** AI & Data Science Research Group, University of Cádiz  
**Format:** Pascal VOC XML → converted to YOLO during ingestion  
**Content:** Knife detection images with bounding box annotations.

This sub-dataset required a format conversion from Pascal VOC (XML) to YOLO (normalised `.txt`). The conversion was handled via `pylabel` as part of the data gathering notebook.

> **Citation:**  
> Castillo-Cara, M., et al. *OD-WeaponDetection: A weapon detection dataset for computer vision and deep learning.*  
> https://github.com/ari-dasci/OD-WeaponDetection

---

### 3. Ari-DASCI — OD-WeaponDetection (Pistol Detection)

**Source:** https://github.com/ari-dasci/OD-WeaponDetection  
**Authors:** AI & Data Science Research Group, University of Cádiz  
**Format:** Pascal VOC XML → converted to YOLO during ingestion  
**Content:** Pistol detection images with bounding box annotations.

Note: the XML annotation files in this sub-dataset had a structural defect — the `<filename>` field was missing its file extension. Our data gathering notebook includes an automatic repair step that scans the image directory and patches the XML before conversion.

Same citation as above.

---

### 4. Sentry Internal Dataset

**Format:** YOLO (already structured with `images/` and `labels/` splits)  
**Content:** Manually annotated images collected and labelled by the Sentry team specifically for this project. Covers scenarios not well-represented in the public datasets — including realistic surveillance angles and partial occlusions.

This dataset is not publicly released.

---

### Merge Configuration

All four datasets were merged using Sentry's own `merge_datasets` pipeline in **rebuild** mode:

| Split   | Ratio |
| ------- | ----- |
| `train` | 70%   |
| `val`   | 15%   |
| `test`  | 15%   |

Class IDs were remapped to `0` (Weapon) across all sources, and split boundaries were rebuilt from scratch by random shuffle to ensure balanced distribution.

---

## Repository Structure

```
Sentry/
├── assets/                  # Logos and visual assets
├── configs/
│   └── default.yaml         # Central pipeline configuration
├── notebooks/
│   ├── full_dataset_gathering.ipynb   # Data collection and preparation (Colab)
│   └── merged_dataset_training.ipynb  # Final training run on remote server
├── scripts/                 # Thin CLI wrappers around the library
│   ├── ingest_github.py
│   ├── ingest_local.py
│   ├── validate.py
│   ├── audit.py
│   ├── merge.py
│   ├── train.py
│   └── infer.py
├── src/
│   └── sentry_ai/
│       ├── config.py
│       ├── dataset/
│       │   ├── ingest.py
│       │   ├── validate.py
│       │   ├── audit.py
│       │   └── merge.py
│       └── yolo/
│           ├── train.py
│           └── infer.py
└── DOCUMENTATION.md         # Full API documentation
```

---

## Getting Started

```bash
# Install the library
pip install -e .

# Validate a dataset
python scripts/validate.py --config configs/default.yaml

# Merge datasets
python scripts/merge.py --config configs/default.yaml

# Start training
python scripts/train.py --config configs/default.yaml
```

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for the full API reference.

---

## License

Apache License 2.0. See [`LICENSE`](LICENSE) for details.

Third-party datasets retain their original licenses. Please refer to each dataset's source for terms of use.
