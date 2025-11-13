# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vision-Dissect is a comprehensive benchmarking and layer-by-layer analysis framework for computer vision models. It systematically evaluates depth estimation (DepthPro, Depth Anything V2), object detection (YOLO11), segmentation (YOLO11n-Segment, MobileSAM), and pose estimation (YOLO11n-Pose) models.

## Common Commands

### Python Environment

```bash
# Install dependencies
pip install -e .

# Activate virtual environment (if not already active)
# Windows:
.venv\Scripts\activate
# Unix/MacOS:
source .venv/bin/activate
```

### Running Benchmarks

```bash
# Run comprehensive benchmark on all models
python vision-bench/unified_benchmark.py

# Dissect specific models
python dissect/dissect_depthpro.py
python dissect/dissect_depthanything.py
```

### Interactive Applications

```bash
# Launch Streamlit apps
streamlit run apps/depthpro_app.py
streamlit run apps/depth_viz_app.py
```

### Web Explorer (Next.js)

```bash
cd explorer
npm install
npm run dev        # Development server at http://localhost:3000
npm run build      # Production build
npm run start      # Production server
npm run lint       # Run ESLint
```

### Comparison Tools

```bash
# Compare model outputs
python compare/compare_models.py
python compare/compare_channels.py
python compare/visualize_depth.py
```

## Architecture

### Core Components

**vision-bench/unified_benchmark.py** - Main benchmarking framework
- Abstract `ModelBenchmark` base class with hooks for load, infer, dissect, cleanup
- Model-specific implementations: DepthProBenchmark, DepthAnythingV2*Benchmark, YOLO11n*Benchmark, MobileSAMBenchmark
- Sequential execution with memory-efficient cleanup between models
- Generates JSON, CSV, and Markdown reports with comprehensive metrics
- Layer dissection using PyTorch forward hooks to capture intermediate activations
- Saves visualizations as PNG (4x4 grid of 16 channels) and NPY (8 channels for numerical analysis)

**dissect/** - Model dissection tools
- `dissect_depthpro.py`: Extract and visualize DepthPro encoder layers, fusion stages, attention patterns
- `dissect_depthanything.py`: Layer-by-layer analysis of Depth Anything V2 variants

**apps/** - Interactive Streamlit applications
- `depthpro_app.py`: Real-time depth estimation with DepthPro
- `depth_viz_app.py`: Advanced depth visualization and comparison

**explorer/** - Professional Next.js web interface
- React 18.3 with Next.js 14.2
- Radix UI components for tabs, dialogs, scroll areas
- Recharts for performance metrics visualization
- XYFlow for computational graph visualization
- Fetches benchmark results from GitHub raw URLs in production
- Displays layer visualizations, performance metrics, and live monitoring

**compare/** - Model comparison utilities
- Channel-level output comparison
- ONNX model analysis
- Depth map visualization with multiple colormaps

### Key Model Implementations

**DepthPro** (Apple, ~1.3B params, 57 layers)
- Uses HuggingFace transformers: `apple/DepthPro-hf`
- Dual Dinov2 encoders (patch + image) with feature fusion
- Metric depth estimation with field-of-view prediction
- Large model requires selective layer filtering: only Conv and main Attention layers

**Depth Anything V2** (Small/Base/Large variants)
- Requires external module: `depth_anything_v2`
- Setup: Clone https://github.com/DepthAnything/Depth-Anything-V2
- ViT-based encoder (ViT-S: 24.8M, ViT-B: 97.5M, ViT-L: 335M params)
- Relative depth estimation with 33 Conv2d layers per variant

**YOLO11n** (Ultralytics)
- Detection: 89 Conv2d layers, 2.62M params
- Segment: 102 Conv2d/ConvTranspose2d layers, 2.87M params
- Pose: 98 Conv2d layers, 2.87M params
- Auto-downloads weights if missing from `models/` directory

**MobileSAM** (138 layers, 10.1M params)
- Uses Ultralytics SAM wrapper
- Segment Anything Model optimized for mobile

### Benchmark Workflow

1. **Initialization**: Detect CUDA device, find test images in `test_data/`, set random seed (42)
2. **Sequential Processing**: Load model → Run inference (5 runs per image) → Layer dissection → Cleanup → Sleep 2s
3. **Memory Tracking**: Monitor usage before load, after load, peak during inference, after cleanup
4. **Layer Dissection**: Register forward hooks on filtered layers (skip containers, capture Conv/Attention only)
5. **Visualization**: Save first 16 channels as PNG grid, first 8 channels as NPY for numerical analysis
6. **Reporting**: Generate JSON (raw data), CSV (summary table), Markdown (readable report with analysis)

### Output Structure

```
vision-bench/
  results/
    benchmark_results_{timestamp}.json      # Complete raw data
    benchmark_summary_{timestamp}.csv       # Summary table
    benchmark_report_{timestamp}.md         # Readable report
  viz/{timestamp}/
    DepthPro/
      layer_000.png, layer_000.npy          # Feature maps
      layers_metadata.json                  # Layer statistics
    DepthAnythingV2-Small/
    YOLO11n-Detect/
    ...
```

### Layer Filtering Strategy

**DepthPro & Depth Anything** (Large models):
- Only capture Conv layers and main Attention modules
- Skip container modules, normalization, activation layers
- Reduces dissection from 1000+ to ~50-60 meaningful layers

**YOLO11 & MobileSAM**:
- Capture Conv, Linear, Norm, activation layers
- Filter out container Sequential/ModuleList modules

### Explorer Web App Data Flow

**Development**: Reads from local `vision-bench/results/` and `vision-bench/viz/`
**Production**: Fetches from GitHub raw URLs:
- `https://raw.githubusercontent.com/{owner}/{repo}/{branch}/vision-bench/results/benchmark_results_{timestamp}.json`
- PNG/NPY files fetched similarly from viz directory

## Key Dependencies

**Python (>=3.11)**:
- torch >= 2.9.0 (CUDA 12.6 support)
- transformers >= 4.47.0 (DepthPro)
- ultralytics >= 8.3.225 (YOLO11, MobileSAM)
- timm >= 1.0.22 (Vision models)
- streamlit >= 1.40.0 (Interactive apps)
- onnx >= 1.19.1, onnxruntime >= 1.23.2 (Model export)

**Node.js**:
- next ^14.2.0
- react ^18.3.1
- @xyflow/react ^12.9.2 (Graph visualization)
- recharts ^2.12.0 (Charts)
- Radix UI components (Tabs, Dialog, Accordion, etc.)

## Important Notes

- Models auto-download on first run (DepthPro: ~1.2GB, YOLO11: ~5-6MB each)
- CUDA highly recommended; CPU fallback available but 100x+ slower
- Depth Anything V2 requires external git clone (not pip installable)
- Benchmark runs sequentially to avoid GPU memory conflicts
- Each model is fully unloaded before next one loads (prevents OOM)
- Random seed (42) ensures reproducible inference times
- Layer dissection uses first test image only (multiple inference runs use all images)
