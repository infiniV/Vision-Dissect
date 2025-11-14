# CLABSIGuard

Real-time healthcare monitoring system using computer vision for depth estimation, instance segmentation, and pose estimation.

## Overview

CLABSIGuard is a privacy-preserving healthcare compliance monitoring system that uses three pretrained state-of-the-art computer vision models running in parallel:

- **Depth Estimation**: DepthAnything V2 Small (24.8M params)
- **Instance Segmentation**: YOLO11n-Seg (2.87M params)
- **Pose Estimation**: YOLO11n-Pose (2.87M params)

The system operates at 5-6 FPS on real-time webcam feeds, providing immediate visual feedback through a 2K fullscreen display.

## Project Structure

```
health-monitor/
├── src/                     # Core source code
│   ├── clabsi_guard.py      # V1 model (ResNet50 backbone)
│   ├── clabsi_guard_v2.py   # V2 model (YOLO backbone)
│   ├── monitor.py           # Compliance monitoring
│   └── utils.py             # Utility functions
├── heads/                   # Model prediction heads
│   ├── pretrained_heads.py  # PyTorch pretrained models
│   └── onnx_heads.py        # ONNX runtime models
├── backbones/               # Backbone networks
│   ├── yolo_backbone.py     # YOLO11 backbone
│   └── feature_adapter.py   # Feature adaptation layers
├── benchmarks/              # Performance benchmarking
│   ├── benchmark.py         # Single model benchmark
│   └── benchmark_v1_vs_v2.py # V1 vs V2 comparison
├── tests/                   # Test suite
│   ├── test_camera.py
│   ├── test_fps_lock.py
│   ├── test_pretrained_integration.py
│   ├── test_visualization.py
│   └── test_visual_outputs.py
├── scripts/                 # Utility scripts
│   ├── debug_predictions.py # Debug model outputs
│   └── validate_all.py      # Run all validations
├── demos/                   # Interactive demos
│   └── webcam_demo.py       # Real-time webcam demo
├── models/                  # Model weights
│   ├── yolo11n-seg.pt       # Segmentation model
│   └── yolo11n-pose.pt      # Pose estimation model
└── docs/                    # Documentation
    └── ARCHITECTURE.md      # Detailed architecture
```

## Installation

### Requirements

- Python 3.11+
- CUDA-capable GPU (recommended)
- Webcam

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd tasks/health-monitor
```

2. Install dependencies:
```bash
pip install torch torchvision
pip install ultralytics
pip install transformers
pip install opencv-python
pip install numpy
```

3. Models will auto-download on first run:
   - DepthAnything V2 Small (~100MB) via Hugging Face
   - YOLO11n-Seg and YOLO11n-Pose (~6MB each) via Ultralytics

## Quick Start

### Run the real-time webcam demo:

```bash
# From the health-monitor directory
python -m demos.webcam_demo
```

Options:
```bash
# Use V1 model (ResNet50 backbone)
python -m demos.webcam_demo --v1

# Specify camera ID
python -m demos.webcam_demo --camera 1

# Disable fullscreen
python -m demos.webcam_demo --no-fullscreen
```

### Run benchmarks:

```bash
# Benchmark V1 model
python -m benchmarks.benchmark

# Compare V1 vs V2
python -m benchmarks.benchmark_v1_vs_v2
```

### Run tests:

```bash
# Test camera access
python -m tests.test_camera

# Test FPS locking
python -m tests.test_fps_lock

# Test pretrained models
python -m tests.test_pretrained_integration

# Test visualization
python -m tests.test_visualization
```

## Usage

### Basic Integration

```python
from src.clabsi_guard_v2 import CLABSIGuardV2
from heads.pretrained_heads import (
    PretrainedDepthHead,
    PretrainedSegmentationHead,
    PretrainedKeypointsHead
)
from src.monitor import ComplianceMonitor

# Initialize models
depth_head = PretrainedDepthHead()
seg_head = PretrainedSegmentationHead()
kp_head = PretrainedKeypointsHead()
monitor = ComplianceMonitor()

# Process image (numpy array, RGB format)
depth = depth_head.predict(image_rgb)  # Returns (H, W) depth map
masks = seg_head.predict(image_rgb)    # Returns (8, H, W) instance masks
keypoints = kp_head.predict(image_rgb) # Returns (21, H, W) heatmaps

# Update compliance monitor
import torch
monitor.update(
    torch.from_numpy(depth).unsqueeze(0),
    torch.from_numpy(masks).unsqueeze(0),
    torch.from_numpy(keypoints).unsqueeze(0)
)
```

### Model Variants

**V1 (ResNet50 Backbone)**:
- Custom trained backbone
- 85-90% parameters in shared backbone
- Tiny prediction heads (2-3 layers)
- ~25M total parameters

**V2 (Pretrained YOLO Backbone)**:
- Transfer learning from YOLO11
- Pretrained feature extraction
- State-of-the-art performance
- ~10M parameters

## Architecture

CLABSIGuard V2 uses three independent pretrained models running in parallel:

1. **DepthAnything V2 Small**: Transformer-based monocular depth estimation
   - Input: RGB image (640x480)
   - Output: Depth map (480x640) normalized to [0, 1]

2. **YOLO11n-Seg**: Instance segmentation
   - Input: RGB image (640x480)
   - Output: 8 instance masks (8x480x640) with confidence values

3. **YOLO11n-Pose**: Pose estimation
   - Input: RGB image (640x480)
   - Output: 21 keypoint heatmaps (21x480x640)

See `docs/ARCHITECTURE.md` for detailed diagrams and component descriptions.

## Performance

**System Specifications**:
- GPU: CUDA-capable (tested on NVIDIA RTX)
- Resolution: 640x480 input, 2560x1440 display
- Target FPS: 5 FPS (achieves 5.5 avg)

**Model Performance** (V2):
- Total parameters: ~30M (combined)
- Inference time: ~180ms per frame
- Memory usage: ~2GB VRAM

## Development

### Code Organization

- **src/**: Core model implementations and utilities
- **heads/**: Modular prediction heads (pretrained PyTorch and ONNX)
- **backbones/**: Shared feature extractors
- **benchmarks/**: Performance measurement scripts
- **tests/**: Unit and integration tests
- **demos/**: Interactive applications
- **scripts/**: Development utilities

### Adding Custom Heads

To add a new prediction head:

1. Create a class in `heads/` that inherits from `nn.Module`
2. Implement `forward()` for PyTorch compatibility
3. Implement `predict()` for numpy input/output
4. Update `heads/__init__.py` to export the new class

Example:
```python
class CustomHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Load your model

    def forward(self, x):
        # PyTorch tensor processing
        pass

    def predict(self, image_np):
        # Numpy array processing
        # Returns numpy array output
        pass
```

### Running Validation

```bash
# Run all validation tests
python -m scripts.validate_all
```

## Privacy

CLABSIGuard is designed with privacy in mind:
- No video recording or storage
- No network transmission of frames
- Real-time processing only
- Immediate disposal of processed frames

## Troubleshooting

### Camera Issues
```bash
# Test camera access
python -m tests.test_camera
```

### Model Loading Issues
- Ensure CUDA is available: `torch.cuda.is_available()`
- Check model files in `models/` directory
- Verify Hugging Face transformers installation

### Performance Issues
- Reduce input resolution (640x480 → 480x360)
- Process every 2nd frame
- Use ONNX runtime for faster inference
- Enable mixed precision (FP16)

## Citation

Based on:
- DepthAnything V2: https://github.com/DepthAnything/Depth-Anything-V2
- YOLO11: https://github.com/ultralytics/ultralytics
- TEO-1 Architecture: Task-Efficient One-backbone design

## License

See LICENSE file for details.
