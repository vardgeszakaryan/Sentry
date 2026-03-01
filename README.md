<p align="center">
  <img src="assets/logo.svg" alt="Sentry Logo" width="180"/>
</p>

<h1 align="center">Sentry — Model Pipeline</h1>

<p align="center">
  <b>Dataset Engineering · Training · Validation · Inference</b>
</p>

---

## 🎯 Purpose of This Branch

This branch contains the **AI core** of Sentry.

It provides a structured, modular, and reproducible pipeline for:

- Dataset ingestion
- Dataset validation & auditing
- Dataset merging
- YOLO training
- Real-time inference

This branch is strictly focused on the model layer.  
No UI. No frontend. Only the AI engine.

---

## 🧠 Design Philosophy

The objective is not just to “train a detector.”

The objective is to build a **robust, real-world CCTV threat detection model** that:

- Detects small weapons in large frames
- Handles occlusion and poor lighting
- Maintains clean dataset integrity
- Produces reproducible training results
- Minimizes silent annotation errors

---

## 📊 Dataset Engineering Strategy

We actively work on:

- Merging compatible YOLO-format datasets
- Cleaning inconsistent annotations
- Validating bounding box normalization
- Auditing class distribution imbalance
- Removing corrupted samples
- Ensuring image-to-label integrity

### Focus Areas

- Small-object detection
- CCTV-like camera angles
- Surveillance resolution conditions
- Realistic weapon positioning

---

## 📚 External Dataset Reference

One of the structured datasets currently utilized:

**Ari-DASCI – OD-WeaponDetection**
https://github.com/ari-dasci/OD-WeaponDetection/tree/master

We acknowledge and credit this research contribution.

Our pipeline extends it by:

- Integrating additional compatible datasets
- Running validation & audit passes
- Normalizing splits
- Preparing data for scalable YOLO training

---

## 🏗 Repository Structure
