# Vision Model Benchmark Report

**Timestamp:** 20251113_153353

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
- Total Execution Time: 1667.42s

## Results

| Model | Load Time (s) | Avg Inference (s) | Std (s) | FPS | Peak Memory (MB) | Parameters | Layers | Zero-Shot |
|-------|---------------|-------------------|---------|-----|------------------|------------|--------|----------|
| GroundingDino | 18.64 | 1.028 | 1.423 | 0.97 | 669 | 172,249,090 | 17 | ✓ |
| YOLO11n-Detect | 1.12 | 0.292 | 0.504 | 3.43 | 19 | 2,616,248 | 89 |  |
| YOLO11n-Segment | 0.28 | 0.222 | 0.334 | 4.49 | 22 | 2,868,664 | 102 |  |
| YOLO11n-Pose | 0.25 | 0.141 | 0.179 | 7.09 | 20 | 2,866,468 | 98 |  |
| MobileSAM | 1.21 | 5.272 | 1.052 | 0.19 | 189 | 10,130,092 | 138 |  |
| DepthAnythingV2-Small | 0.90 | 0.301 | 0.436 | 3.33 | 104 | 24,785,089 | 33 |  |
| DepthAnythingV2-Base | 1.42 | 0.196 | 0.171 | 5.10 | 382 | 97,470,785 | 33 |  |
| DepthAnythingV2-Large | 3.81 | 0.258 | 0.007 | 3.87 | 1291 | 335,315,649 | 33 |  |
| DepthPro | 8.65 | 17.164 | 3.580 | 0.06 | 3643 | 951,991,330 | 57 |  |

## Analysis

- **Fastest Model:** YOLO11n-Pose (0.141s, 7.09 FPS)
- **Slowest Model:** DepthPro (17.164s, 0.06 FPS)
- **Most Memory Intensive:** DepthPro (3643 MB)
- **Most Parameters:** DepthPro (951,991,330)

## Prompt Variation Analysis

### GroundingDino

| Prompt | Inference Time (s) | Detections |
|--------|-------------------|------------|
| a car, a person, a building, a tree, a window, a door | 0.373 | 1 |
| a vehicle, a human, architecture, vegetation | 0.341 | 5 |
| a vintage car, a person walking, a wooden door, a yellow wall | 0.329 | 6 |

## Visualizations

Layer dissection visualizations are saved in:

- `vision-bench/viz/20251113_153353/GroundingDino/`
- `vision-bench/viz/20251113_153353/YOLO11n-Detect/`
- `vision-bench/viz/20251113_153353/YOLO11n-Segment/`
- `vision-bench/viz/20251113_153353/YOLO11n-Pose/`
- `vision-bench/viz/20251113_153353/MobileSAM/`
- `vision-bench/viz/20251113_153353/DepthAnythingV2-Small/`
- `vision-bench/viz/20251113_153353/DepthAnythingV2-Base/`
- `vision-bench/viz/20251113_153353/DepthAnythingV2-Large/`
- `vision-bench/viz/20251113_153353/DepthPro/`
