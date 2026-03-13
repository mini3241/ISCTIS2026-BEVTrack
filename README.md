# Robust BEV Tracking with Uncertainty-Aware Pseudo-Points and Cross-Modal Transformer

Official implementation for ISCTIS 2026 paper.

## Architecture

The proposed method performs multi-object tracking in Bird's-Eye View (BEV) by fusing monocular camera and 4D millimeter-wave radar. Key components:

- **Uncertainty-Aware Pseudo-Point Module**: Generates pseudo-LiDAR points from monocular depth estimation with confidence weighting
- **Cross-Modal Transformer**: Pre-fuses pseudo-point and radar BEV as Keys/Values, image BEV as Queries for cross-attention alignment
- **Confidence-Driven Kalman Tracker**: Adapts observation noise based on fused detection confidence

## Project Structure

```
├── config/
│   └── base.py               # Configuration
├── models/
│   ├── base_model.py         # Main fusion model
│   ├── radar_branch.py       # Radar voxelization and BEV encoder
│   ├── image_branch.py       # Camera ResNet34 + depth estimation
│   ├── pseudo_lidar.py       # Pseudo-LiDAR generation with YOLOv5
│   └── fusion.py             # Cross-Modal Transformer fusion
├── data/
│   └── dataset.py            # Dataset loading and preprocessing
├── utils/
│   ├── tracker.py            # Kalman filter tracking and data association
│   ├── metrics.py            # MOTA/MOTP evaluation metrics
│   └── focal_loss.py         # Gaussian Focal Loss for heatmap detection
├── scripts/
│   ├── train.py              # Training script
│   ├── train_v2.py           # Training script with Focal Loss
│   ├── evaluate.py           # Evaluation script
│   ├── debug_visualization.py    # Model output visualization
│   └── test_visualization.py     # Raw data visualization
```

## Requirements

- PyTorch >= 1.10
- torchvision
- numpy, scipy, opencv-python
- matplotlib, tqdm, pandas

YOLOv5 is loaded automatically via `torch.hub`.

## Training

```bash
python scripts/train.py
```

## Evaluation

```bash
python scripts/evaluate.py
```

## Dataset

We use a self-collected multi-sensor dataset with a monocular camera (960x510), OCULii 4D millimeter-wave radar, and 128-beam LiDAR for ground-truth annotation. The dataset contains 9,726 frames (7,482 training / 2,244 validation).

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{bevtrack2026,
  title={Robust BEV Tracking with Uncertainty-Aware Pseudo-Points and Cross-Modal Transformer},
  booktitle={ISCTIS 2026},
  year={2026}
}
```
