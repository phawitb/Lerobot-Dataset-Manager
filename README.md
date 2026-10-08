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
- **Remove Idle Frames** — Trim idle frames from episode start, end, or both (middle idle segments are never removed)
- **Random Sampling** — Create subsets by randomly sampling episodes
- **HuggingFace Integration** — Auto-list and one-click download/push datasets with real-time terminal logs
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
python-multipart
huggingface_hub[cli]
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

Datasets are stored in the `./data/` directory (created automatically on first run). The current data directory path is displayed in the sidebar. Each dataset follows the LeRobot format:

```
data/
  my_dataset/
    meta/
      info.json          # Dataset metadata (fps, robot_type, etc.)
      episodes/          # Episode metadata (timestamps, lengths)
      tasks.parquet      # Task descriptions for language conditioning
      stats.json         # Normalization statistics
    data/
      chunk-000/         # Frame data (actions, states, timestamps)
        episode_000000.parquet
        ...
    videos/
      observation.images.top/
        chunk-000/
          episode_000000.mp4
      observation.images.wrist/
        chunk-000/
          episode_000000.mp4
```

## Workflow

1. **Collect** — Record episodes on the robot using LeRobot recording tools
2. **Import & Download** — Import datasets from local directories, drag & drop folders, or download directly from HuggingFace Hub
3. **Visualize & Clean** — Play synchronized multi-camera videos, view joint trajectories, remove idle frames, delete bad episodes
4. **Augment** — Multiply data with camera shifts, lighting changes, robot noise, and language variations
5. **Train** — Push dataset to HuggingFace, then train on a GPU server using auto-generated commands
6. **Inference** — Download trained model and run on the robot (with optional real-time augmentation)

## Sidebar

The sidebar is shared across all tabs and provides dataset management and processing tools.

### Dataset

- **Dataset Selector** — Dropdown to select a dataset. Shows total count in parentheses. The data directory path is displayed below the heading.
- **Rename** — Rename the selected dataset folder.
- **Delete** — Permanently delete the selected dataset and all its files.

### Import & Download

- **Import Dataset** — Import from a local directory via file browser or drag & drop. The Import button only activates when a valid dataset folder (containing `meta/info.json`) is selected.
- **Download from HF** — Enter your HuggingFace username to auto-list all your datasets. Click Download to fetch any dataset directly. Already-downloaded datasets show a green checkmark. Terminal log shows real-time download progress. Public datasets require no login; for private/gated datasets, run `huggingface-cli login` first.
- **Combine Datasets** — Merge 2+ datasets into one, consolidating episodes, tasks, and videos.

### Process

- **Augment** — Multiply dataset with configurable augmentation: camera perspective/affine transforms, brightness/contrast/saturation/noise/blur, joint noise, and language paraphrasing. Preview before applying.
- **Remove Idle** — Create a new dataset with idle frames trimmed. Choose to remove from: **Start only** (default), **End only**, or **Both**. Idle segments in the middle are never removed.
- **Random Sample** — Create a subset by randomly sampling N episodes from the dataset.

### Export

- **Push to HF** — Upload dataset to HuggingFace Hub. Automatically checks login status — if authenticated, push directly with one click. Shows the CLI command with a copy button for manual use. Terminal log shows real-time upload progress.

## Tab: Dataset

Main workspace for visualizing and editing episodes.

- **Episode Visualizer** — Play synchronized multi-camera videos (top, wrist, etc.) with joint trajectory chart. Scrub through frames with real-time sync across all video streams. For augmented datasets, original and augmented videos play side-by-side in sync.
- **Segment Editor** — Mark idle vs movement phases in episodes. Auto-detect segments using velocity-based analysis, or manually split and adjust segment boundaries by dragging handles.
- **Edit Task** — Assign or update task descriptions for selected episodes (used as language conditioning for VLA models like SmolVLA and pi0.5).
- **Delete Episodes** — Select and permanently remove bad or unwanted episodes from the dataset.

## Tab: Training

Step-by-step commands for training a policy model on a GPU server.

- **Setup** — Create conda environment, install LeRobot, FFmpeg, and all dependencies (one-time setup).
- **Download** — Download dataset from HuggingFace Hub to the GPU server.
- **Training** — Auto-generated `lerobot-train` command for the selected model (ACT / SmolVLA / pi0.5) with W&B logging toggle.
- **Tips** — GPU memory recommendations, batch size tuning, multi-GPU training, checkpoint management.

## Tab: Inference

Commands for running a trained model on the robot.

- **Setup** — Same environment setup as training (shared conda env).
- **Download Model** — Download trained model checkpoint from HuggingFace Hub.
- **Inference** — Run `lerobot-rollout` with robot port, camera config, and task instruction.
- **Inference + Augmentation** — Run inference with `fakecam_inject.py` wrapper that applies real-time camera augmentation to test policy robustness. Supports hot-reload of parameters.
- **Tips** — Camera setup, serial port config, headless mode, hot-reload usage.

## Data Augmentation

The augmentation system multiplies your dataset by creating new episodes from existing ones with controlled variations:

| Category | Parameters | Description |
|----------|-----------|-------------|
| **Camera Shift** | Perspective | Random perspective warp simulating camera viewpoint changes |
| | Affine (translate, scale, shear) | Shift, zoom, and skew camera frames |
| | Rotation | Rotate camera frame by random degrees |
| **Light & Quality** | Brightness / Contrast / Saturation | Simulate different lighting conditions |
| | Color Jitter / Shadow | Random color shifts and directional shadow overlay |
| | Noise / Blur | Gaussian noise and motion blur for image quality variation |
| **Robot Noise** | Start Trim | Random trim of initial idle frames |
| | Joint Offset / Jitter | Small random perturbations to action and state trajectories |
| **Language** | Paraphrase / Typos | Generate task text variations for VLA language conditioning robustness |

## Supported Models

| Model | Type | GPU | Best For |
|-------|------|-----|----------|
| **ACT** | Action Chunking Transformer | 8 GB+ | Simple tasks, fast training |
| **SmolVLA** | Vision-Language-Action | 16 GB+ | Language-conditioned tasks |
| **pi0.5** | Large VLA | 24 GB+ | Best generalization |

## HuggingFace Integration

| Feature | Description |
|---------|-------------|
| **Download** | Auto-list datasets by username, one-click download with real-time terminal log. No auth needed for public datasets. |
| **Push** | Upload datasets with auth status check, one-click push or manual command with copy button. Terminal log shows progress. |
| **Authentication** | Public datasets require no login. For private/gated datasets or pushing, run `huggingface-cli login` in terminal. |

## API Endpoints

The backend exposes a REST API at `http://localhost:8080`:

| Endpoint | Description |
|----------|-------------|
| `GET /api/datasets` | List all local datasets |
| `GET /api/datasets/{name}/info` | Get dataset metadata and tasks |
| `GET /api/datasets/{name}/frames/{ep}` | Get frame data for an episode |
| `POST /api/datasets/import` | Import dataset from local path |
| `POST /api/datasets/combine` | Combine multiple datasets |
| `POST /api/datasets/{name}/augment` | Augment dataset with variations |
| `POST /api/datasets/{name}/remove-idle` | Remove idle frames (mode: start/end/both) |
| `POST /api/datasets/{name}/random-sample` | Random sample episodes |
| `GET /api/hf/datasets` | List HF datasets by username |
| `POST /api/hf/download` | Download dataset from HF Hub |
| `GET /api/hf/download/status` | Check download progress and logs |
| `POST /api/hf/push` | Push dataset to HF Hub |
| `GET /api/hf/push/status` | Check push progress and logs |
| `GET /api/hf/auth-status` | Check HF login status |
| `GET /api/browse` | Browse local filesystem for import |

## fakecam_inject.py

A wrapper that monkey-patches `cv2.VideoCapture` to apply real-time augmentation during inference — useful for testing policy robustness to visual perturbations without modifying inference code.

```bash
python fakecam_inject.py --params-file fakecam_params.json -- \
  lerobot-rollout \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --policy.path=./models/act_my_dataset
```

**Parameters:** `rotation`, `translate_x`, `translate_y`, `scale`, `shear`, `brightness`, `contrast`, `saturation`, `noise`, `blur`

**Key features:**
- **Hot-reload** — Edit `fakecam_params.json` while running; changes apply automatically every 2 seconds
- **Multiple param sources** — JSON file (`--params-file`), remote server (`--from-server`), or direct JSON (`--params`)
- **Black camera mode** — Replace specific camera indices with black frames (`--black-cameras`) to test missing sensor robustness

## Tech Stack

- **Backend:** FastAPI, Python, PyArrow/Parquet, OpenCV, FFmpeg
- **Frontend:** Vanilla JavaScript, Chart.js
- **Integration:** HuggingFace Hub
