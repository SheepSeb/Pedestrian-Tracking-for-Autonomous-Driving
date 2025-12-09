# Deep SORT

## Overview

This repo contains the base Deep SORT tracker plus two self-contained stacks:

- `tf_implementation`: scripts that use the TensorFlow appearance model (`models_tf/mars-small128.pb`).
- `pytorch_implementation`: scripts that use the PyTorch appearance model (ResNet18 default, optional checkpoint).
- Base/core code remains under `deep_sort/`, `application_util/`, and shared tools in `tools/`.

## Setup

1) Install deps with uv:
```
uv sync
```

2) Download MOT16 (train/test) to `./MOT16` (images under `./MOT16/train/*/img1`).
   Official link: https://motchallenge.net/data/MOT16/

3) Ensure TF frozen graph is present for TF workflows: `models_tf/mars-small128.pb`
   (included in `models_tf/` per your attached files).

## Layout
- Base tracker & utilities: `deep_sort/`, `application_util/`, `deep_sort_app.py`, `generate_videos.py`, `tools/*`.
- TensorFlow stack: `tf_implementation/`
  - `generate_detections.py` (TF .pb)
  - `run_tracker.py` (tracker with TF detections)
  - `make_video.py` (render videos from TF tracks)
- PyTorch stack: `pytorch_implementation/`
  - `generate_detections.py` (Torch encoder)
  - `run_tracker.py` (tracker with Torch detections)
  - `make_video.py` (render videos from Torch tracks)
  - Training remains at `tools/train_mot16.py` (torch)

## Quick commands

### TensorFlow pipeline
Generate detections (TF):
```
PYTHONPATH=. uv run python tf_implementation/generate_detections.py \
  --mot_dir ./MOT16/train \
  --output_dir ./resources/detections_tf \
  --model ./models_tf/mars-small128.pb
```

Track a sequence with TF detections:
```
PYTHONPATH=. uv run python tf_implementation/run_tracker.py \
  --sequence MOT16-02 \
  --detections_dir ./resources/detections_tf \
  --tracks_dir ./tracks_tf \
  --mot_dir ./MOT16/train
```

Render video from TF tracks:
```
PYTHONPATH=. uv run python tf_implementation/make_video.py \
  --result_dir ./tracks_tf \
  --mot_dir ./MOT16/train \
  --output_dir ./videos_tf \
  --convert_h264
```

### PyTorch pipeline
Generate detections (Torch, default ResNet18):
```
PYTHONPATH=. uv run python pytorch_implementation/generate_detections.py \
  --mot_dir ./MOT16/train \
  --output_dir ./resources/detections_torch
```
Optional: add `--model /path/to/state_dict.pt` to use a custom checkpoint.

Track a sequence with Torch detections:
```
PYTHONPATH=. uv run python pytorch_implementation/run_tracker.py \
  --sequence MOT16-02 \
  --detections_dir ./resources/detections_torch \
  --tracks_dir ./tracks_torch \
  --mot_dir ./MOT16/train
```

Render video from Torch tracks:
```
PYTHONPATH=. uv run python pytorch_implementation/make_video.py \
  --result_dir ./tracks_torch \
  --mot_dir ./MOT16/train \
  --output_dir ./videos_torch \
  --convert_h264
```

### Train PyTorch appearance model (1+ epochs)
```
PYTHONPATH=. uv run python tools/train_mot16.py \
  --data-root ./MOT16 \
  --output-dir ./checkpoints/mot16 \
  --log-dir ./runs/mot16 \
  --epochs 1 \
  --batch-size 64 \
  --device auto
```
View logs: `uv run tensorboard --logdir ./runs/mot16`
W&B: add `--use-wandb --wandb-mode offline` (or `online`).

### Torch/TF embedding comparison (optional)
```
PYTHONPATH=. uv run python tools/train_and_compare.py \
  --data-root ./MOT16 \
  --output-dir ./checkpoints/mot16 \
  --log-dir ./runs/mot16
```
Picks `models_tf/mars-small128.pb` automatically if present; prints basic Torch vs TF embedding stats on sample frames.

## Notes
- Outputs are organized separately: `resources/detections_tf`, `resources/detections_torch`, `tracks_tf`, `tracks_torch`, `videos_tf`, `videos_torch`.
- Base tracker code is shared; the two stacks differ only in the appearance model and helper wrappers.
- Legacy TF tools (e.g., `tools/freeze_model.py`) remain available; PyTorch is the primary path for new work.

## Citing DeepSORT

If you find this repo useful in your research, please consider citing the following papers:

    @inproceedings{Wojke2017simple,
      title={Simple Online and Realtime Tracking with a Deep Association Metric},
      author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
      booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
      year={2017},
      pages={3645--3649},
      organization={IEEE},
      doi={10.1109/ICIP.2017.8296962}
    }

    @inproceedings{Wojke2018deep,
      title={Deep Cosine Metric Learning for Person Re-identification},
      author={Wojke, Nicolai and Bewley, Alex},
      booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
      year={2018},
      pages={748--756},
      organization={IEEE},
      doi={10.1109/WACV.2018.00087}
    }
