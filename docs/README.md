# Documentation Index

## Overview
This folder contains comprehensive documentation for all vision models analyzed in this project.

## Generated Documentation

### 1. Main Analysis Document
📄 **[MODEL_ANALYSIS.md](MODEL_ANALYSIS.md)**
- Comprehensive overview of all models
- Quick comparison matrix
- Recommendations for model selection
- All visualization references
- **Read this first for a complete overview**

### 2. Model-Specific Deep Dives

#### Depth Estimation Models

📄 **[DEPTHPRO_DETAILED.md](DEPTHPRO_DETAILED.md)**
- Apple DepthPro architecture breakdown
- Dual DINOv2 encoder analysis
- Feature extraction experiments
- Performance benchmarks
- Optimization strategies
- **Best for: Understanding metric depth estimation**

📄 **[DEPTH_ANYTHING_V2_DETAILED.md](DEPTH_ANYTHING_V2_DETAILED.md)**
- Depth Anything V2 (vits/vitb/vitl) analysis
- Speed vs quality tradeoffs
- Colormap comparison study
- Statistical depth analysis
- Real-time deployment guide
- **Best for: Fast relative depth estimation**

#### Object Detection/Segmentation/Pose

📄 **[YOLO11_FAMILY_DETAILED.md](YOLO11_FAMILY_DETAILED.md)**
- YOLO11n variants (Detection/Segmentation/Pose)
- Layer-by-layer architecture breakdown
- Shared backbone analysis
- Task-specific head comparison
- ONNX export deep dive
- Performance optimization guide
- **Best for: Understanding multi-task vision models**

## Quick Reference

### Model Selection Guide

| Task | Model | Speed | Quality | Use Case |
|------|-------|-------|---------|----------|
| **Metric Depth** | DepthPro | ⭐⭐ | ⭐⭐⭐⭐⭐ | Offline/High-quality |
| **Relative Depth** | Depth Anything V2 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Real-time |
| **Object Detection** | YOLO11n | ⭐⭐⭐⭐ | ⭐⭐⭐ | Real-time |
| **Segmentation** | YOLO11n-seg | ⭐⭐⭐⭐ | ⭐⭐⭐ | Fast masks |
| **Segmentation (Quality)** | Mobile SAM | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Promptable |
| **Pose Estimation** | YOLO11n-pose | ⭐⭐⭐⭐ | ⭐⭐⭐ | Multi-person |

## Scripts Analyzed

### Exploration Scripts (`explore/`)
1. ✅ **explore_depthpro.py** - DepthPro model exploration
2. ✅ **explore_sam.py** - Mobile SAM exploration
3. ✅ **onnx_runtime_tutorial.py** - ONNX conversion basics

### Dissection Scripts (`dissect/`)
1. ✅ **dissect_depthanything.py** - Depth Anything V2 internals
2. ✅ **dissect_depthpro.py** - DepthPro internals

### Comparison Scripts (`compare/`)
1. ✅ **compare_channels.py** - RGB/HSV/Gradient analysis
2. ✅ **compare_models.py** - YOLO11 variant comparison
3. ✅ **compare_onnx.py** - PyTorch vs ONNX analysis
4. ✅ **visualize_depth.py** - Depth visualization techniques
5. ✅ **visualize_outputs.py** - Model output visualization

## Visualizations Generated

All visualizations are stored in `../viz/`:

### Depth Models
- `depthpro_features.png` - DepthPro intermediate features
- `depthpro_layer_comparison.png` - Layer progression
- `depth_estimation_comparison.png` - Depth Anything V2 output
- `depth_colormaps_comparison.png` - Colormap comparison
- `depth_detailed_analysis.png` - Statistical analysis with 3D plot
- `depth_side_by_side.png` - Side-by-side comparison

### YOLO Models
- `model_comparison.png` - Layer 22 features across variants
- `model_outputs.png` - Detection/Segmentation/Pose outputs

### Feature Analysis
- `channels_rgb.png` - RGB channel separation
- `channels_hsv.png` - HSV channel separation
- `gradients.png` - Sobel and Laplacian edge detection
- `canny.png` - Canny edge detection
- `sam_masks.png` - Mobile SAM segmentation masks

### ONNX Analysis
- `pytorch_layer_22.png` - PyTorch feature visualization
- `pytorch_layer_23.png` - Detection head output

## Key Findings Summary

### DepthPro
- **Architecture**: Dual DINOv2 encoders (24 layers each)
- **Output**: 1536×1536 metric depth maps
- **Speed**: ~2-3 seconds per image (GPU)
- **Best for**: High-quality metric depth, offline processing

### Depth Anything V2
- **Architecture**: Single DINOv2 encoder with DPT head
- **Output**: Maintains input aspect ratio, relative depth
- **Speed**: <1 second per image (vits variant)
- **Best for**: Real-time applications, general depth sensing

### YOLO11 Family
- **Architecture**: 24-layer unified backbone + task-specific heads
- **Parameters**: 2.6M (nano variant)
- **Speed**: ~50ms per image (GPU)
- **Best for**: Real-time detection/segmentation/pose

### Mobile SAM
- **Architecture**: Promptable segmentation
- **Output**: High-quality instance masks
- **Speed**: Slower than YOLO, faster than original SAM
- **Best for**: Interactive segmentation, zero-shot

### Channel Analysis
- **RGB/HSV**: Standard color space analysis
- **Gradients**: Sobel, Laplacian for edge detection
- **Canny**: Best edge detection quality
- **Use**: Understanding model preprocessing

### ONNX Export
- **Expansion**: 24 PyTorch layers → 320 ONNX nodes
- **Benefits**: Cross-platform, optimization, quantization
- **Trade-off**: Larger file size, more complex graph

## Recommendations by Use Case

### Real-time Video Processing
1. **Depth**: Depth Anything V2 (vits)
2. **Detection**: YOLO11n
3. **Segmentation**: YOLO11n-seg
4. **Pose**: YOLO11n-pose

### Offline High-Quality Processing
1. **Depth**: DepthPro
2. **Detection**: YOLO11l (large variant)
3. **Segmentation**: Mobile SAM
4. **Pose**: YOLO11l-pose

### Edge/Mobile Deployment
1. **Depth**: Depth Anything V2 (vits) + TensorRT
2. **Detection**: YOLO11n + ONNX/TensorRT
3. **Avoid**: DepthPro (too heavy)

### Research/Benchmarking
1. **Depth**: Both DepthPro and Depth Anything V2
2. **Detection**: Full YOLO11 family
3. **Segmentation**: Mobile SAM for quality baseline

## Testing Environment

### Hardware
- **GPU**: CUDA-enabled (tested on modern GPU)
- **CPU**: AMD Ryzen 7 6800HS
- **RAM**: Sufficient for all models

### Software
- **Python**: 3.11.9
- **PyTorch**: 2.9.0+cu126
- **Ultralytics**: 8.3.225
- **Transformers**: Latest compatible with DepthPro
- **ONNX**: 1.19.1

### Test Data
- **Image**: 640×480 RGB test image
- **Content**: Mixed objects, varying depths
- **Location**: `../test_data/test_image.jpg`

## Future Work

### Potential Improvements
1. **Model Fusion**: Combine depth + detection for 3D object detection
2. **Multi-task Learning**: Single model for multiple tasks
3. **Quantization**: INT8 optimization for deployment
4. **Mobile Optimization**: CoreML, TFLite exports
5. **Temporal Consistency**: Video-specific optimizations

### Additional Analysis
1. **Attention Visualization**: Deep dive into transformer attention
2. **Adversarial Testing**: Robustness analysis
3. **Cross-dataset Evaluation**: Test on multiple datasets
4. **Speed Profiling**: Layer-by-layer timing analysis

## How to Use This Documentation

### For Beginners
1. Start with [MODEL_ANALYSIS.md](MODEL_ANALYSIS.md)
2. Review the Quick Reference table above
3. Read model-specific docs for your use case

### For Researchers
1. Review all detailed docs
2. Check experimental results sections
3. Examine visualizations in `../viz/`
4. Reference architecture deep dives

### For Engineers
1. Focus on optimization strategies sections
2. Review ONNX export analysis
3. Check performance metrics
4. Use code examples for implementation

### For Students
1. Read architecture overviews
2. Understand component functions
3. Study visualization results
4. Compare different approaches

## Contact & Contributions

This documentation was generated from extensive testing and analysis of multiple vision models. All scripts used for analysis are available in the repository.

**Analysis Date**: November 10, 2025  
**Repository**: Vision-Dissect  
**Branch**: main

---

*For questions or suggestions, please open an issue in the repository.*
