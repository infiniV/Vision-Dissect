# Vision Inference Server

FastAPI-based multi-model inference server for computer vision tasks.

## Features

- Multiple vision models (DepthPro, Depth Anything V2, YOLO, SAM)
- ONNX runtime optimization for YOLO models
- Lazy model loading with automatic warmup
- Flexible response formats (full/downsampled/stats)
- Request queue monitoring
- CPU-optimized with asyncio locks

## Quick Start

```bash
cd server
cp .env.example .env
python serve.py
```

Server runs at `http://localhost:8000`

## API Endpoints

### Depth Estimation

- `POST /api/v1/depth/depthpro` - Apple DepthPro
- `POST /api/v1/depth/depth-anything-v2?variant=vits` - Depth Anything V2

### Detection

- `POST /api/v1/detect` - YOLO object detection

### Segmentation

- `POST /api/v1/segment/yolo` - YOLO instance segmentation
- `POST /api/v1/segment/sam` - Mobile SAM

### Pose Estimation

- `POST /api/v1/pose` - YOLO pose detection

### Monitoring

- `GET /health` - Health check and loaded models
- `GET /queue` - Request queue depths

## Configuration

Edit `.env` file:

```
DEVICE=cpu
MAX_IMAGE_SIZE=10485760
DEPTH_ANYTHING_VARIANT=vits
YOLO_CONFIDENCE=0.5
USE_ONNX=true
DEFAULT_DEPTH_FORMAT=downsampled
WARMUP_MODELS=depthpro,yolo_detect
```

## Adding Models

See `ADDING_MODELS.md` for detailed instructions.

## Dependencies

Install via workspace `pyproject.toml`:

- fastapi
- uvicorn
- pydantic-settings
- torch
- transformers
- ultralytics
- onnxruntime
- opencv-python
- pillow
