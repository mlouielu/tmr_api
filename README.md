# TMR API & Mesh Generator

Derived from TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis demo: https://huggingface.co/spaces/Mathux/TMR. This project provides tools for text-to-motion retrieval and SMPL mesh generation. It includes a web API for interactive search and a CLI tool (`tmrgen`) for batch processing and mesh generation.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)
- CUDA-capable GPU (recommended)
- **Data:**
    - `AMASS` dataset (raw .npz files) required for mesh generation.
    - `SMPL` models required for mesh generation.

## Installation

1.  **Initialize Environment with uv:**
    ```bash
    uv sync --all-extras
    ```

2.  **Data Setup:**
    The system requires model weights and embeddings. By default, it looks in `tmr_data/`. On the first run, `launch.py` will attempt to download these automatically. You can override the location by setting `TMR_DATA_DIR`.

## Usage

### 1. API Server (`launch.py`)

Starts the Gradio web server and exposes the retrieval API.

```bash
python launch.py
```
**Endpoint:** `http://127.0.0.1:7860/`

**API Specification:**
- **Endpoint:** `/predict`
- **Parameters:**
    - `query` (str): Text description of the motion.
    - `gallery` (str): "All motions" or "Unseen motions".
    - `videos` (int): Number of results to retrieve.
- **Output:** JSON list of objects containing:
    - `score`: Similarity score.
    - `corresponding text`: Text annotation from the dataset.
    - `AMASS path`: Relative path to the motion file in AMASS.
    - `start_time`, `end_time`: Temporal segment of the motion.
    - `video link`: URL to the rendered preview (if available).
    - `fps`: Frame rate of the motion.

### 2. CLI Tool (`tmrgen`)

`tmrgen` allows you to search for motions and generate SMPL meshes locally. It supports both standalone local inference and remote inference via the API.

**Location:** `src.tmr_api.tools.tmrgen`

**Command:**
```bash
python -m src.tmr_api.tools.tmrgen [OPTIONS]
```

**Arguments:**
- `--query`: Text prompt (e.g., "A person doing a backflip").
- `--amass_root`: Path to your local AMASS dataset root.
- `--smpl_path`: Path to your SMPL models directory.
- `--output_dir`: Directory to save results (default: `output_meshes`).
- `--limit`: Number of top results to process.
- `--remote [URL]`: Enable remote mode. Optional URL (default: `http://127.0.0.1:7860/`).

**Examples:**

**Local Mode (Default):**
Runs the retrieval model locally. Requires no running server.
```bash
python -m src.tmr_api.tools.tmrgen \
  --query "Walk forward and turn" \
  --amass_root path/to/AMASS \
  --smpl_path path/to/SMPL \
  --limit 3
```

**Remote Mode:**
Connects to a running `launch.py` instance to perform the search, then generates meshes locally.
```bash
python -m src.tmr_api.tools.tmrgen \
  --remote \
  --query "Jump up" \
  --amass_root path/to/AMASS \
  --smpl_path path/to/SMPL
```

## Output Structure

For each result, `tmrgen` creates a directory named after the motion and time segment:
```
output_meshes/
  └── Dataset_Path_To_Motion_Start_End/
      ├── metadata.json       # Query, score, ranking, FPS, and file info
      ├── frame_0000.obj      # Generated mesh sequence
      ├── frame_0001.obj
      └── ...
```
