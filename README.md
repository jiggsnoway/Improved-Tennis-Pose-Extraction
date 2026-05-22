# Tennis Stroke Biomechanics Analysis

Multi-task ML pipeline for analyzing tennis stroke mechanics from monocular RGB video. Given a video, it automatically detects strokes, classifies them (forehand/backhand), predicts shot direction, and flags posture quality — all from pose alone, no ball tracking, no court markers.

Built as independent research extending [Shimizu et al. (2019)](https://doi.org/10.1145/3347318.3355527), with a controlled ablation study showing world-coordinate pose features transfer across players where image-normalized landmarks don't.

---

## What it does

The pipeline has four stages:

1. **Pose extraction** — MediaPipe Pose Landmarker (full model, float16) extracts 33 landmarks per frame, including metric world coordinates in a hip-origin frame
2. **Stroke segmentation** — velocity-based detector scores each frame as `0.5·wrist + 0.3·elbow + 0.2·shoulder` displacement; peaks above a player-adaptive threshold mark stroke events, no manual annotation needed
3. **Sequence construction** — each stroke becomes a fixed `30×39` window (±15 frames around the peak)
4. **Multi-task inference** — `TennisTransformerGPU` runs three heads in parallel: stroke type, shot direction, posture quality

A rule-based layer on top of direction + posture outputs one of four coaching messages. No extra labels required for that part.

---

## Results

Trained on 1,281 strokes from 7 professional players (Alcaraz, Federer, Djokovic, Nadal, Zverev, Sinner, Wawrinka) and 1 amateur across 11 videos.

| Task | Random 80/20 | Cross-player | Majority baseline |
|---|---|---|---|
| Stroke type | 83.7% | 82.9% | 77.0% |
| Direction | 61.9% | — (collapses)† | 61.3% |
| Posture | 62.6% | 63.4% | 63.4% |

†Direction does not genuinely transfer cross-player; the model defaults to the majority class (center). This is the main open problem.

The stroke-type cross-player result (82.9%, 0.8% drop from random split) is the finding I'm most confident in. A model trained entirely on professional footage deploys to an amateur player with negligible accuracy loss.

**Ablation — world coordinates vs image landmarks (cross-player):**

| Task | Image landmarks | World coordinates | Δ |
|---|---|---|---|
| Stroke type | 47% | 83% | +36% |
| Direction | 21% | 68% | +47% |
| Posture | 39% | 63% | +24% |

The gap closes substantially on random split (same players in train/test), which confirms the benefit is specifically about generalization, not just fitting.

---

## Model

```python
class TennisTransformerGPU(nn.Module):
    # input_dim=39, d_model=128, nhead=4, num_layers=4, dropout=0.3
    # three heads: direction (3 classes), stroke type (2), posture (2)
    # 564,103 parameters total
```

Trained jointly on all three heads:

```
L = L_type + L_dir + L_posture
```

Posture head uses 2.5× class weight on "bad" to counter imbalance (63/37 split). Adam, lr=5e-4, weight decay 1e-3, StepLR (step=25, γ=0.5), 80 epochs, batch 16, gradient clip 1.0.

---

## Setup

Runs on Kaggle free-tier (T4 GPU). No local GPU needed.

```bash
pip install mediapipe torch torchvision opencv-python numpy pandas scikit-learn matplotlib seaborn
```

MediaPipe Pose Landmarker model file: download `pose_landmarker_full.task` from the [MediaPipe Models page](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) and place it in the project root.

---

## Usage

**Extract pose from a video:**
```python
python extract_poses.py --video my_video.mp4 --output keypoints.csv
```

**Detect strokes:**
```python
python detect_strokes.py --keypoints keypoints.csv --output stroke_windows.csv
```

**Run inference:**
```python
python inference.py --video my_video.mp4 --model model_cross_player.pt
```

The inference script overlays pose, per-task predictions with confidence scores, and the coaching feedback message on each frame.

---

## Files

```
├── extract_poses.py          # Stage 1: MediaPipe pose extraction
├── detect_strokes.py         # Stage 2: velocity-based stroke segmentation
├── build_sequences.py        # Stage 3: fixed-length sequence construction
├── model.py                  # TennisTransformerGPU definition
├── train.py                  # Training script (random + cross-player splits)
├── inference.py              # Per-frame inference with visualization
├── ablation.py               # World vs image coordinate ablation
├── data/
│   ├── keypoints.csv         # Extracted pose landmarks
│   ├── stroke_windows.csv    # Detected stroke boundaries
│   ├── labels_clean.csv      # Manual annotations
│   └── X_sequences.npy       # 1281×30×39 sequence array
├── models/
│   ├── model_random_split.pt
│   └── model_cross_player.pt
└── figures/                  # Paper figures
```

---

## Limitations worth knowing

- **Direction task does not work cross-player.** 61.9% on random split barely clears the 61.3% majority-class ceiling. The fix is contact-window modeling (2-3 frames around peak wrist velocity) rather than the full 30-frame sequence.
- **Posture labels are single-annotator.** Binary good/bad assessed by me on hip-shoulder separation, stance balance, and racket preparation timing. No inter-rater reliability.
- **Camera angle matters.** The pipeline was trained predominantly on behind-the-baseline footage. Side/front views degrade landmark visibility and drop accuracy.
- **Amateur test set is one player** (me). Cross-player results should be treated as a directional finding, not a general benchmark.

---

## Paper

Submitted to *AI* (MDPI), Special Issue: AI and Computer Vision in Real-World and Industrial Applications.

> Hazarika, J. (2026). Multi-Task Tennis Stroke Biomechanics Analysis Using MediaPipe Pose.

---

## License

MIT
