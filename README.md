# Robust BEV Tracking with Uncertainty-Aware Pseudo-Points and Cross-Modal Transformer

Official implementation for ISCTIS 2025 paper.

## Architecture

The proposed method performs multi-object tracking in Bird's-Eye View (BEV) by fusing monocular camera and 4D millimeter-wave radar. Key components:

- **Uncertainty-Aware Pseudo-Point Module**: Generates pseudo-LiDAR points from monocular depth estimation with confidence weighting
- **Cross-Modal Transformer**: Aligns radar and visual features in BEV space
- **Confidence-Driven Kalman Tracker**: Adapts observation noise based on fused detection confidence

## Project Structure

```
├── models/               # Network architecture
│   ├── base_model.py     # Main fusion model
│   ├── radar_branch.py   # Radar PointPillars-style encoder
│   ├── image_branch.py   # Camera ResNet34 + depth estimation
│   ├── pseudo_lidar.py   # Pseudo-LiDAR generation with YOLOv5
│   └── fusion.py         # Multi-modal fusion strategies
├── data/
│   └── dataset.py        # Dataset loading and preprocessing
├── utils/
│   ├── tracker.py        # Kalman filter tracking and data association
│   └── metrics.py        # MOTA/MOTP evaluation metrics
├── config/
│   └── base.py           # Configuration
├── scripts/
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── fusion_model_v42_with_yolo.py  # Production model
└── evaluate_epoch73.py            # Paper results evaluation
```

## Requirements

```bash
pip install -r requirements.txt
```

YOLOv5 is loaded automatically via `torch.hub`.

## Pre-trained Weights

Download the pre-trained model checkpoint (epoch 73):

- [Google Drive](TODO) | [Baidu Pan](TODO)

Place the downloaded `model_epoch_73.pth` in the `checkpoints/` directory.

## Evaluation

```bash
python evaluate_epoch73.py
```

## Training

```bash
python scripts/train.py
```

## Dataset

We use a self-collected multi-sensor dataset with a monocular camera (960x510), OCULii 4D millimeter-wave radar, and 128-beam LiDAR for ground-truth annotation. The dataset contains 9,726 frames (7,482 training / 2,244 validation).

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{bevtrack2025,
  title={Robust BEV Tracking with Uncertainty-Aware Pseudo-Points and Cross-Modal Transformer},
  booktitle={ISCTIS 2025},
  year={2025}
}
```
