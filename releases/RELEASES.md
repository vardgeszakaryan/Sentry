# Sentry — Release History

All model releases for the Sentry weapon detection system are documented here.
Weights are stored in their respective version folder under `releases/`.

---

## [v1.0.0] — 2026-03-04 🎉 Initial Release

### Model
| Property        | Value                     |
| --------------- | ------------------------- |
| File            | `releases/v1.0.0/best.pt` |
| Architecture    | YOLOv8                    |
| Input size      | 640 × 640                 |
| Classes         | 1 — `Weapon`              |
| Training epochs | 100                       |

### Dataset
Trained on the merged Sentry dataset assembled from four sources:
- Kaggle — Weapons in Images (Segmented Videos)
- Ari-DASCI OD-WeaponDetection (Knife Detection)
- Ari-DASCI OD-WeaponDetection (Pistol Detection)
- Sentry Internal Dataset (proprietary)

**Split:** 70 % train · 15 % val · 15 % test  
**Total class:** Weapon (class 0) — all sources unified

### Notes
- First production-ready checkpoint
- Intended for real-time inference at 640 × 640 with ByteTrack tracking

---

## How to Add a New Release

1. Place the new weights file(s) in `releases/vX.Y.Z/`
2. Add a new section at the **top** of this file following the template above
3. Bump the version badge in `README.md`

