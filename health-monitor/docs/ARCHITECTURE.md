# CLABSIGuard V2 Architecture

## System Overview

CLABSIGuard V2 is a real-time healthcare monitoring system that uses three pretrained state-of-the-art computer vision models running in parallel to provide depth estimation, instance segmentation, and pose estimation.

## High-Level Architecture

```mermaid
graph TB
    subgraph Input["Input Layer"]
        Webcam[Webcam Feed<br/>640x480 RGB]
    end

    subgraph Preprocessing["Preprocessing"]
        Webcam --> Resize[Resize to 640x480]
        Resize --> BGR2RGB[Convert BGR to RGB]
    end

    subgraph Models["Pretrained Models - Parallel Inference"]
        BGR2RGB --> Depth[DepthAnything V2 Small<br/>Hugging Face<br/>24.8M params]
        BGR2RGB --> Seg[YOLO11n-Seg<br/>2.87M params]
        BGR2RGB --> Pose[YOLO11n-Pose<br/>2.87M params]
    end

    subgraph Outputs["Model Outputs"]
        Depth --> DepthMap["Depth Map<br/>(H, W)<br/>float32 [0,1]"]
        Seg --> SegMasks["Instance Masks<br/>(8, H, W)<br/>float32 [0,1]"]
        Pose --> Heatmaps["Keypoint Heatmaps<br/>(21, H, W)<br/>float32"]
    end

    subgraph Visualization["Visualization Layer"]
        DepthMap --> DepthVis[Depth Colormap<br/>MAGMA]
        SegMasks --> SegVis[Instance Overlay<br/>Multi-color]
        Heatmaps --> PoseVis[Heatmap Overlay<br/>HOT colormap]
    end

    subgraph Display["Display - 2K Fullscreen"]
        DepthVis --> Grid[2x2 Grid Layout<br/>2560x1440]
        SegVis --> Grid
        PoseVis --> Grid
        BGR2RGB --> Grid
        Grid --> Monitor[Compliance Monitor<br/>Status & FPS]
        Monitor --> Screen[Fullscreen Display]
    end

    style Webcam fill:#e1f5ff
    style Depth fill:#fff3e0
    style Seg fill:#fff3e0
    style Pose fill:#fff3e0
    style Screen fill:#c8e6c9
```

## Detailed Component Architecture

```mermaid
graph LR
    subgraph WebcamDemo["WebcamDemo Class"]
        Init[Initialize<br/>- Camera<br/>- Models<br/>- Monitor]

        Init --> Loop{Main Loop}

        Loop --> Capture[Capture Frame<br/>640x480 BGR]
        Capture --> Convert[Convert to RGB]

        Convert --> DepthHead[PretrainedDepthHead.predict]
        Convert --> SegHead[PretrainedSegmentationHead.predict]
        Convert --> KeypointHead[PretrainedKeypointsHead.predict]

        DepthHead --> DepthOut[depth_np<br/>480x640]
        SegHead --> SegOut[seg_np<br/>8x480x640]
        KeypointHead --> KeyOut[kp_np<br/>21x480x640]

        DepthOut --> ToTensor[Convert to PyTorch Tensors]
        SegOut --> ToTensor
        KeyOut --> ToTensor

        ToTensor --> CompMon[ComplianceMonitor.update]
        CompMon --> VisGrid[create_visualization_grid]

        VisGrid --> StatusPanel[draw_status_panel]
        StatusPanel --> Scale[Scale to 2560x1440]
        Scale --> Display[cv2.imshow]

        Display --> FPS[Track FPS<br/>5.5 avg]
        FPS --> Loop
    end

    style Init fill:#e1f5ff
    style DepthHead fill:#fff3e0
    style SegHead fill:#fff3e0
    style KeypointHead fill:#fff3e0
    style Display fill:#c8e6c9
```

## Model Details

```mermaid
graph TB
    subgraph DepthModel["DepthAnything V2 Small"]
        DepthInput["Input: PIL Image<br/>RGB uint8<br/>Any size"]
        DepthPipeline["HF Pipeline<br/>depth-estimation"]
        DepthArch["Architecture:<br/>DINOv2 ViT-S Encoder<br/>DPT Decoder"]
        DepthParams["24.8M parameters<br/>~1.2GB VRAM"]
        DepthTime["Inference: ~80-100ms"]
        DepthOutput["Output: Depth Map<br/>(H, W) float32<br/>Normalized [0, 1]"]

        DepthInput --> DepthPipeline
        DepthPipeline --> DepthArch
        DepthArch --> DepthParams
        DepthParams --> DepthTime
        DepthTime --> DepthOutput
    end

    subgraph SegModel["YOLO11n-Seg"]
        SegInput["Input: RGB uint8<br/>(H, W, 3)"]
        SegResize["Auto-resize to 640x640"]
        SegArch["Architecture:<br/>CSPDarknet + C2PSA<br/>102 Conv layers"]
        SegParams["2.87M parameters<br/>~0.5GB VRAM"]
        SegTime["Inference: ~30-40ms"]
        SegDetect["Detect Objects<br/>80 COCO classes"]
        SegMask["Generate Masks<br/>Per instance"]
        SegOutput["Output: (8, H, W)<br/>8 instance masks<br/>float32 [0, 1]"]

        SegInput --> SegResize
        SegResize --> SegArch
        SegArch --> SegParams
        SegParams --> SegTime
        SegTime --> SegDetect
        SegDetect --> SegMask
        SegMask --> SegOutput
    end

    subgraph PoseModel["YOLO11n-Pose"]
        PoseInput["Input: RGB uint8<br/>(H, W, 3)"]
        PoseResize["Auto-resize to 640x640"]
        PoseArch["Architecture:<br/>CSPDarknet + C2PSA<br/>98 Conv layers"]
        PoseParams["2.87M parameters<br/>~0.5GB VRAM"]
        PoseTime["Inference: ~30-40ms"]
        PoseDetect["Detect People"]
        PoseKP["Extract 17 COCO<br/>Keypoints per person"]
        PoseGauss["Generate Gaussian<br/>Heatmaps (sigma=15)"]
        PoseOutput["Output: (21, H, W)<br/>21 keypoint heatmaps<br/>float32"]

        PoseInput --> PoseResize
        PoseResize --> PoseArch
        PoseArch --> PoseParams
        PoseParams --> PoseTime
        PoseTime --> PoseDetect
        PoseDetect --> PoseKP
        PoseKP --> PoseGauss
        PoseGauss --> PoseOutput
    end

    style DepthModel fill:#fff3e0
    style SegModel fill:#fff3e0
    style PoseModel fill:#fff3e0
```

## Visualization Pipeline

```mermaid
graph TB
    subgraph GridLayout["2x2 Grid Visualization"]

        subgraph TL["Top-Left: Input + Keypoints"]
            InputFrame[Original Frame<br/>640x480 RGB]
            DrawKP[Draw Keypoint Circles<br/>Green dots at peaks]
            DrawLines[Draw Skeleton<br/>Cyan lines]
            TLOut[Input with Overlay]

            InputFrame --> DrawKP
            DrawKP --> DrawLines
            DrawLines --> TLOut
        end

        subgraph TR["Top-Right: Depth"]
            DepthMap[Depth Map<br/>480x640 float32]
            NormDepth[Normalize to [0, 255]]
            ApplyMAGMA[Apply MAGMA Colormap<br/>Purple to Yellow]
            TROut[Colored Depth]

            DepthMap --> NormDepth
            NormDepth --> ApplyMAGMA
            ApplyMAGMA --> TROut
        end

        subgraph BL["Bottom-Left: Segmentation"]
            SegMasks[8 Instance Masks<br/>8x480x640]
            ThreshMasks[Threshold at 0.5<br/>Binary masks]
            ColorMasks["Apply Colors:<br/>Red, Green, Blue, Yellow,<br/>Magenta, Cyan, Orange, Purple"]
            BLOut[Colored Instances]

            SegMasks --> ThreshMasks
            ThreshMasks --> ColorMasks
            ColorMasks --> BLOut
        end

        subgraph BR["Bottom-Right: Keypoint Heatmaps"]
            KPHeatmaps[21 Keypoint Heatmaps<br/>21x480x640]
            SumHeatmaps[Sum all channels]
            NormHM[Normalize to [0, 255]]
            ApplyHOT[Apply HOT Colormap<br/>Black to Yellow to White]
            BROut[Keypoint Visualization]

            KPHeatmaps --> SumHeatmaps
            SumHeatmaps --> NormHM
            NormHM --> ApplyHOT
            ApplyHOT --> BROut
        end

        TLOut --> Combine[Combine to 2x2 Grid<br/>1280x960]
        TROut --> Combine
        BLOut --> Combine
        BROut --> Combine

        Combine --> Labels[Add Labels<br/>Scaled text]
        Labels --> Status[Draw Status Panel<br/>FPS, Device, Violations]
        Status --> Scale[Scale to 2560x1440]
        Scale --> Final[Final Display]
    end

    style TL fill:#e3f2fd
    style TR fill:#fff3e0
    style BL fill:#f3e5f5
    style BR fill:#fce4ec
    style Final fill:#c8e6c9
```

## Data Flow Sequence

```mermaid
sequenceDiagram
    participant User
    participant Webcam
    participant Demo as WebcamDemo
    participant Depth as DepthHead
    participant Seg as SegHead
    participant Pose as PoseHead
    participant Monitor as ComplianceMonitor
    participant Display

    User->>Demo: Start demo (python webcam_demo.py)
    Demo->>Depth: Load DepthAnything V2 (1-2 min first time)
    Demo->>Seg: Load YOLO11n-Seg
    Demo->>Pose: Load YOLO11n-Pose
    Demo->>Monitor: Initialize compliance monitor

    loop Every Frame (~180ms)
        Webcam->>Demo: Capture frame (640x480 BGR)
        Demo->>Demo: Convert BGR to RGB

        par Parallel Inference
            Demo->>Depth: predict(rgb)
            Depth-->>Demo: depth_map (480, 640)
        and
            Demo->>Seg: predict(rgb)
            Seg-->>Demo: seg_masks (8, 480, 640)
        and
            Demo->>Pose: predict(rgb)
            Pose-->>Demo: keypoints (21, 480, 640)
        end

        Demo->>Demo: Convert to PyTorch tensors
        Demo->>Monitor: update(outputs)
        Monitor-->>Demo: violations, status

        Demo->>Demo: create_visualization_grid()
        Demo->>Demo: draw_status_panel()
        Demo->>Demo: scale to 2560x1440
        Demo->>Display: cv2.imshow()

        Display-->>User: Display fullscreen
        User->>Demo: Key press (q/s/o/f/r)
    end

    User->>Demo: Press 'q'
    Demo->>Display: Close window
    Demo->>User: Show statistics
```

## Performance Characteristics

```mermaid
graph LR
    subgraph Timing["Frame Processing Time (~180ms)"]
        T1[Capture: ~2ms]
        T2[Depth Inference: ~100ms]
        T3[Seg Inference: ~35ms]
        T4[Pose Inference: ~35ms]
        T5[Visualization: ~5ms]
        T6[Display: ~3ms]

        T1 --> T2
        T2 --> T3
        T3 --> T4
        T4 --> T5
        T5 --> T6

        T6 --> FPS[Result: 5.5 FPS avg]
    end

    subgraph Memory["GPU Memory Usage (~2.2GB)"]
        M1[Depth Model: ~1.2GB]
        M2[YOLO Seg: ~0.5GB]
        M3[YOLO Pose: ~0.5GB]
        M4[Overhead: ~0.1GB]

        M1 --> Total
        M2 --> Total
        M3 --> Total
        M4 --> Total[Total: ~2.3GB]
    end

    style FPS fill:#c8e6c9
    style Total fill:#ffecb3
```

## File Structure

```mermaid
graph TB
    subgraph Project["health-monitor/"]

        subgraph Core["Core Files"]
            Demo[webcam_demo.py<br/>Main application<br/>Webcam capture & display]
            Heads[pretrained_heads.py<br/>Model wrappers<br/>DepthHead, SegHead, PoseHead]
            Monitor[monitor.py<br/>ComplianceMonitor<br/>Violation tracking]
        end

        subgraph Models["Model Files"]
            YOLO1[../models/yolo11n-seg.pt<br/>2.87M params]
            YOLO2[../models/yolo11n-pose.pt<br/>2.87M params]
            HF[Hugging Face Cache<br/>depth-anything/...<br/>24.8M params]
        end

        subgraph Tests["Test Scripts"]
            TestCam[test_camera.py<br/>Camera & CUDA check]
            TestVis[test_visual_outputs.py<br/>Visual verification]
            TestDebug[debug_predictions.py<br/>Debug raw outputs]
            TestInteg[test_pretrained_integration.py<br/>Integration test]
        end

        subgraph Docs["Documentation"]
            README[README.md<br/>Quick start guide]
            ARCH[ARCHITECTURE.md<br/>This file]
            FIXES[FIXES_APPLIED.md<br/>Issue fixes]
            PRETRAINED[PRETRAINED_MODELS.md<br/>Model details]
        end

        Demo --> Heads
        Demo --> Monitor
        Heads --> YOLO1
        Heads --> YOLO2
        Heads --> HF
    end

    style Core fill:#e1f5ff
    style Models fill:#fff3e0
    style Tests fill:#f3e5f5
    style Docs fill:#c8e6c9
```

## System Requirements

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.11+
- **GPU**: NVIDIA GPU with 3GB+ VRAM (CUDA 12.6)
- **CPU**: Fallback available (50x slower)
- **RAM**: 8GB minimum, 16GB recommended
- **Display**: 2K (2560x1440) recommended, supports 1080p to 4K
- **Webcam**: Any USB/built-in camera (640x480 minimum)

## Dependencies

```mermaid
graph TB
    subgraph Python["Python 3.11+"]
        PyTorch[torch 2.9.0+cu126<br/>Deep learning framework]
        Transformers[transformers<br/>Hugging Face models]
        Ultralytics[ultralytics 8.3.225+<br/>YOLO models]
        OpenCV[opencv-python<br/>Image processing]
        NumPy[numpy<br/>Array operations]
    end

    subgraph External["External Models"]
        HFModel[depth-anything/Depth-Anything-V2-Small-hf<br/>Auto-downloads on first run]
        YOLOSeg[yolo11n-seg.pt<br/>Included in models/]
        YOLOPose[yolo11n-pose.pt<br/>Included in models/]
    end

    PyTorch --> WebcamDemo
    Transformers --> DepthHead
    Ultralytics --> SegHead
    Ultralytics --> PoseHead
    OpenCV --> WebcamDemo
    NumPy --> All[All Components]

    HFModel --> DepthHead
    YOLOSeg --> SegHead
    YOLOPose --> PoseHead

    style PyTorch fill:#e1f5ff
    style HFModel fill:#fff3e0
```

## Compliance Monitoring

```mermaid
graph TB
    subgraph MonitorSystem["ComplianceMonitor"]
        Buffer[Temporal Buffer<br/>30 frames deque]

        Buffer --> Rules{Violation Rules}

        Rules --> Critical[CRITICAL<br/>Bare hand in sterile zone]
        Rules --> Warning[WARNING<br/>No sanitizer use<br/>Timeout: 60s]
        Rules --> Caution[CAUTION<br/>Person proximity<br/>without PPE]

        Critical --> Log[Log Violation<br/>Timestamp + Message]
        Warning --> Log
        Caution --> Log

        Log --> Status[Update Status<br/>Display on screen]
        Status --> Summary[Session Summary<br/>Count per type]
    end

    style Critical fill:#ffcdd2
    style Warning fill:#fff9c4
    style Caution fill:#fff3e0
    style Summary fill:#c8e6c9
```

## Key Design Decisions

1. **Pretrained Models Instead of TEO-1**
   - Trade-off: Higher memory usage and slower FPS
   - Benefit: Immediate high-quality predictions without training data
   - Rationale: Proof-of-concept for healthcare monitoring

2. **Sequential Model Execution**
   - Trade-off: 5.5 FPS instead of potential 30 FPS
   - Benefit: Simpler architecture, no threading complexity
   - Rationale: Sufficient for demonstration purposes

3. **Fullscreen 2K Display**
   - Benefit: Clear visualization for presentations
   - Feature: Runtime toggle (f key) for flexibility
   - Auto-scaling: All text/labels scale with resolution

4. **Grid Layout Default**
   - Shows all four outputs simultaneously
   - Easy comparison of model predictions
   - Professional presentation format

## Future Enhancements

```mermaid
graph TB
    Current[Current: 5.5 FPS<br/>Sequential Inference]

    Current --> Option1[Option 1: Lighter Models<br/>MiDaS Small, YOLO8n<br/>Target: 15-20 FPS]
    Current --> Option2[Option 2: Multi-Threading<br/>Parallel model execution<br/>Target: 10-15 FPS]
    Current --> Option3[Option 3: Frame Skipping<br/>Different rates per model<br/>Target: 15-20 FPS]
    Current --> Option4[Option 4: Return to TEO-1<br/>Shared backbone + tiny heads<br/>Target: 30+ FPS<br/>Requires: Training data]

    style Current fill:#e1f5ff
    style Option1 fill:#fff3e0
    style Option2 fill:#f3e5f5
    style Option3 fill:#fce4ec
    style Option4 fill:#c8e6c9
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-14
**Performance**: 5.5 FPS average on RTX 3060 Laptop GPU
**Status**: Production-ready for demonstration
