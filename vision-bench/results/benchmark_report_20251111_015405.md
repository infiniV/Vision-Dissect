# Vision Model Benchmark Report

**Timestamp:** 20251111_015405

## System Information

| Property | Value |
|----------|-------|
| OS | Windows 10 |
| Python | 3.11.9 |
| PyTorch | 2.9.0+cu126 |
| CUDA Available | True |
| CUDA Version | 12.6 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| Device | cuda |
| Random Seed | 42 |

## Benchmark Configuration

- Test Images: 1
- Inference Runs per Image: 5
- Total Inference Runs per Model: 5
- Total Execution Time: 1283.00s

## Results

| Model | Load Time (s) | Avg Inference (s) | Std (s) | FPS | Peak Memory (MB) | Parameters | Layers |
|-------|---------------|-------------------|---------|-----|------------------|------------|--------|
| DepthPro | 15.16 | 14.811 | 1.201 | 0.07 | 3643 | 951,991,330 | 57 |
| YOLO11n-Detect | 0.36 | 0.178 | 0.283 | 5.63 | 19 | 2,616,248 | 89 |
| YOLO11n-Segment | 0.11 | 0.071 | 0.062 | 14.16 | 21 | 2,868,664 | 102 |
| YOLO11n-Pose | 0.08 | 0.089 | 0.102 | 11.21 | 20 | 2,866,468 | 98 |
| MobileSAM | 0.25 | 7.238 | 2.302 | 0.14 | 67 | 10,130,092 | 138 |
| DepthAnythingV2-Small | 0.28 | 0.058 | 0.047 | 17.13 | 105 | 24,785,089 | 33 |
| DepthAnythingV2-Base | 0.94 | 0.111 | 0.055 | 9.02 | 381 | 97,470,785 | 33 |
| DepthAnythingV2-Large | 3.07 | 0.262 | 0.009 | 3.82 | 1290 | 335,315,649 | 33 |

## Analysis

- **Fastest Model:** DepthAnythingV2-Small (0.058s, 17.13 FPS)
- **Slowest Model:** DepthPro (14.811s, 0.07 FPS)
- **Most Memory Intensive:** DepthPro (3643 MB)
- **Most Parameters:** DepthPro (951,991,330)

## Visualizations

Layer dissection visualizations are saved in:

- `vision-bench/viz/20251111_015405/DepthPro/`
- `vision-bench/viz/20251111_015405/YOLO11n-Detect/`
- `vision-bench/viz/20251111_015405/YOLO11n-Segment/`
- `vision-bench/viz/20251111_015405/YOLO11n-Pose/`
- `vision-bench/viz/20251111_015405/MobileSAM/`
- `vision-bench/viz/20251111_015405/DepthAnythingV2-Small/`
- `vision-bench/viz/20251111_015405/DepthAnythingV2-Base/`
- `vision-bench/viz/20251111_015405/DepthAnythingV2-Large/`
