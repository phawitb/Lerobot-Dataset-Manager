# LeRobot Dataset Manager

Web-based tool for managing, visualizing, augmenting, and preparing robot learning datasets for the [LeRobot](https://github.com/huggingface/lerobot) framework.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Dataset Management** — Import, rename, delete, combine, and browse datasets
- **Episode Visualizer** — Synchronized multi-camera video playback with joint trajectory charts
- **Segment Editor** — Mark idle vs movement phases, auto-detect or manual annotation
- **Data Augmentation** — Camera shifts, lighting changes, robot noise, language paraphrasing with live preview
- **Remove Idle Frames** — Auto-trim idle frames from episode start/end
- **Random Sampling** — Create subsets by randomly sampling episodes
- **HuggingFace Integration** — Download from and push datasets to HuggingFace Hub
- **Training Guide** — Auto-generated training commands for ACT, SmolVLA, and pi0.5
- **Inference Guide** — Rollout commands with optional real-time camera augmentation (`fakecam_inject.py`)

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for video processing)

### Setup

```bash
# Clone the repository
git clone https://github.com/phawitb/Lerobot-Dataset-Manager.git
cd Lerobot-Dataset-Manager

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
fastapi>=0.100.0
uvicorn>=0.23.0
pyarrow>=12.0.0
numpy>=1.24.0
opencv-python>=4.8.0
```

### Install LeRobot (for Training / Inference on GPU server)

```bash
# Create conda environment
conda create -n lerobot python=3.12 -y
conda activate lerobot

# Install FFmpeg
conda install ffmpeg=7.1.1 -c conda-forge -y

# Clone and install LeRobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
pip install -e ".[all]"

# Install additional dependencies
pip install "transformers>=4.48.0" "huggingface-hub>=1.5.0,<2.0"
pip install python-dateutil wandb
```

## Usage

### Start the application

```bash
python main.py
```

Open your browser at **http://localhost:8080**

### Options

```bash
python main.py --port 8080    # Change port (default: 8080)
python main.py --host 0.0.0.0 # Change host (default: 0.0.0.0)
```

### Data directory

Datasets are stored in the `./data/` directory (created automatically on first run). Each dataset follows the LeRobot format:

```
data/
  my_dataset/
    meta/
      info.json
      episodes.jsonl
      tasks.json
      stats.json
    data/
      chunk-000/
        episode_000000.parquet
        ...
    videos/
      chunk-000/
        top/
          episode_000000.mp4
        wrist/
          episode_000000.mp4
```

## Workflow

1. **Collect** — Record episodes on the robot using LeRobot recording tools
2. **Import** — Import datasets into the manager via the web UI
3. **Visualize & Clean** — Review episodes, remove bad data, trim idle frames
4. **Augment** — Multiply data with camera, lighting, robot noise, and language variations
5. **Train** — Push to HuggingFace, then use the Training tab commands on a GPU server
6. **Inference** — Download trained model and run on the robot

## Supported Models

| Model | Type | GPU | Best For |
|-------|------|-----|----------|
| **ACT** | Action Chunking Transformer | 8 GB+ | Simple tasks, fast training |
| **SmolVLA** | Vision-Language-Action | 16 GB+ | Language-conditioned tasks |
| **pi0.5** | Large VLA | 24 GB+ | Best generalization |

## fakecam_inject.py

A wrapper that monkey-patches `cv2.VideoCapture` to apply real-time augmentation during inference — useful for testing policy robustness to visual perturbations.

```bash
python fakecam_inject.py --params-file fakecam_params.json -- \
  lerobot-rollout \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --policy.path=./models/act_my_dataset
```

**Parameters:** `rotation`, `translate_x`, `translate_y`, `scale`, `shear`, `brightness`, `contrast`, `saturation`, `noise`, `blur`

**Hot-reload:** Edit `fakecam_params.json` while running — changes apply automatically every 2 seconds.
