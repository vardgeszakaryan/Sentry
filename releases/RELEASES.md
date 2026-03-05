# Sentry — Release History

All model releases for the Sentry weapon detection system are documented here.
Weights are stored in their respective version folder under `releases/`.

---

## [v1.0.0] — 2026-03-04 🎉 Initial Release

### Model
| Property     | Value                             |
| ------------ | --------------------------------- |
| Weapon Model | `releases/v1.0.0/best.pt`         |
| Pose Model   | `releases/v1.0.0/yolo26n-pose.pt` |
| Resolution   | 1280 × 1280 (Cinematic)           |
| Tracking     | ByteTrack (Threaded)              |
| Classes      | Weapon, Human Pose                |

### Dataset
Trained on the merged Sentry dataset (Weapon class) and utilizing pre-trained COCO pose weights.
- Weapon Sources: Kaggle, Ari-DASCI (Knife/Pistol), Sentry Internal.
- Pose Specs: 17-point COCO landmarks.

### Notes
- **Initial Production Release (v1.0.0)**
- Features multi-threaded inference for ~20 FPS.
- Includes Cinematic FBI visual FX: Teal-orange LUT, Vignette, Glowing Skeleton.
- Weapon stabilization via ENTER/EXIT hysteresis and EMA smoothing.

---

## How to Add a New Release

1. Place the new weights file(s) in `releases/vX.Y.Z/`
2. Add a new section at the **top** of this file following the template above
3. Bump the version badge in `README.md`

