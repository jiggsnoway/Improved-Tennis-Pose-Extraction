# Improved tennis pose extraction

Shot direction prediction, posture quality scoring, and stroke-type classification from tennis match video using MediaPipe and a temporal transformer — with cross-player generalisation experiments across self-recorded and professional player footage.

## Overview

This project extends Shimizu et al. (MMSports 2019), which predicted tennis shot direction from video using OpenPose and LSTM (66.8% accuracy). Four improvements are made:

- Replaces OpenPose with **MediaPipe Pose** using world landmarks — metric 3D coordinates independent of camera angle, requiring no GPU
- Replaces LSTM with a **causal temporal transformer** with multi-head self-attention, allowing the model to learn which frames in a swing matter most
- Adds a **posture quality scoring head** trained on good/bad stroke labels alongside direction and stroke type
- Tests **cross-player generalisation** across self-recorded footage and professional match video (Federer, Alcaraz)

## Results

| Split | Direction | Stroke type | Posture quality |
|---|---|---|---|
| Random split | 43% | 80% | 53% |
| Cross-player | 21% | 47% | 39% |
| Chance baseline | 33% | — | 50% |

The cross-player generalisation gap (−22pp direction, −33pp stroke type) is a finding in itself. It quantifies how much player-specific movement style the model learns when trained on a single player, and establishes a concrete baseline for future cross-player work.

Stroke type classification at 80% on random split confirms the pose pipeline and transformer backbone are working well. Direction prediction is harder — it depends on subtle late-swing cues in the final 5–7 frames before impact — and remains the open problem.

## Pipeline

```
Raw video
  └── MediaPipe PoseLandmarker (world landmarks, hip-origin normalised, torso-scaled)
        └── keypoints.csv
              └── Feature extraction (stroke segmentation, missing joint interpolation)
                    └── Temporal transformer (causal mask, 3 layers, d_model=128)
                          ├── Direction head     → left / straight / right
                          ├── Stroke type head   → forehand / backhand / serve
                          └── Posture head       → quality score 0–1
```

## Repository structure

```
improved-tennis-pose-extraction/
├── notebooks/
│   ├── 01_keypoint_extraction.ipynb
│   ├── 02_annotation_and_labels.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation.ipynb
├── data/
│   ├── keypoints.csv
│   └── labels_to_do.csv
├── models/
│   └── temporal_transformer.py
└── README.md
```

## Key design decisions

**World landmarks over image landmarks**

`pose_world_landmarks` returns coordinates in metres relative to the body centre, unaffected by camera distance or angle. All landmarks are further normalised to a hip-midpoint origin and divided by torso length (mid-hip to mid-shoulder distance), making players of different heights directly comparable in the same embedding space.

**Causal self-attention mask**

The transformer applies a causal mask so each frame only attends to past frames. This matches real prediction conditions — at inference time the shot outcome has not yet occurred and future frames are unavailable.

**Shared backbone, three output heads**

Direction, stroke type, and posture quality share the transformer backbone and are trained jointly with a combined loss. This regularises the direction head, which benefits from the stronger gradient signal coming from stroke type classification during early training epochs.

**Stroke-level train/test split**

All frames belonging to a given stroke are kept together in either train or test — never split across both. Splitting on raw frame rows would leak mid-swing context into the test set and produce artificially high random-split accuracy.

## Data

`keypoints.csv` columns per frame:

| Column | Description |
|---|---|
| `frame` | Raw frame index in source video |
| `time_sec` | Timestamp in seconds |
| `video` | Source video label |
| `player_id` | Player identifier (used for cross-player split) |
| `stroke_id` | Stroke group ID (−1 = unannotated) |
| `impact_frame` | Boolean, manually annotated |
| `pose_detected` | Boolean, MediaPipe detection success |
| `{joint}_wx/wy/wz` | Hip-origin normalised world coordinates |
| `{joint}_vis` | MediaPipe visibility confidence |

Joints tracked: nose, left/right shoulder, elbow, wrist, hip, knee, ankle (13 total).

`labels_to_do.csv` columns:

| Column | Values |
|---|---|
| `stroke_id` | Matches keypoints.csv |
| `direction` | `left` / `straight` / `right` |
| `stroke_type` | `forehand` / `backhand` / `serve` |
| `posture_quality` | `good` / `bad` |

## Requirements

```
mediapipe>=0.10
torch>=2.0
opencv-python
pandas
numpy
scikit-learn
```

## Known limitations

- Direction accuracy (43%) is only marginally above the 33% chance baseline on random split. The signal lives in subtle wrist and elbow dynamics in the final frames before impact — a larger dataset with higher frame rate video (60fps+) is the most direct path to improvement.
- The cross-player generalisation gap is large. The model learns player-specific movement patterns rather than generalised swing mechanics. Torso-length normalisation and domain-adversarial training are identified as next steps but not yet implemented.
- Impact frame annotation is manual, which limits dataset scale. Automatic impact detection from wrist velocity peaks is left for future work.

## Reference

Shimizu, T., Hachiuma, R., Saito, H., Yoshikawa, T., & Lee, C. (2019). Prediction of future shot direction using pose and position of tennis player. *Proceedings of the 2nd International Workshop on Multimedia Content Analysis in Sports (MMSports '19)*. https://doi.org/10.1145/3347318.3355523
```
