"""
ONNX Runtime Tutorial - Learn the Basics
=========================================

ONNX Runtime (ORT) is a high-performance inference engine for running
machine learning models. It's optimized for edge devices and production
deployments.

Why ONNX Runtime?
-----------------
1. Cross-platform: Works on CPU, GPU, mobile, and edge devices
2. Fast: Optimized kernels for various hardware accelerators
3. Lightweight: Smaller memory footprint than training frameworks
4. Universal: Supports models from PyTorch, TensorFlow, etc.
5. Production-ready: Used by Microsoft, Facebook, and others at scale

Key Concepts:
-------------
- InferenceSession: Main object for running models
- Execution Providers: Hardware backends (CPU, CUDA, TensorRT, etc.)
- Session Options: Configuration for threading, optimization, memory
- IO Binding: Zero-copy inference for GPU acceleration
"""

import onnxruntime as ort
import numpy as np
from pathlib import Path


def example_1_basic_inference():
    """
    Example 1: Basic Model Inference

    InferenceSession is the core object. It loads your ONNX model
    and provides methods to run inference.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Inference")
    print("=" * 60)

    model_path = "yolo11n.onnx"

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return

    # Create inference session with default settings
    session = ort.InferenceSession(model_path)

    # Inspect model metadata
    print("\n--- Model Inputs ---")
    for input_meta in session.get_inputs():
        print(f"  Name: {input_meta.name}")
        print(f"  Shape: {input_meta.shape}")
        print(f"  Type: {input_meta.type}")

    print("\n--- Model Outputs ---")
    for output_meta in session.get_outputs():
        print(f"  Name: {output_meta.name}")
        print(f"  Shape: {output_meta.shape}")
        print(f"  Type: {output_meta.type}")

    # Prepare dummy input (batch=1, channels=3, height=640, width=640)
    input_name = session.get_inputs()[0].name
    input_shape = (1, 3, 640, 640)
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Run inference - returns list of outputs
    outputs = session.run(None, {input_name: input_data})

    print(f"\n--- Inference Result ---")
    print(f"  Output shape: {outputs[0].shape}")
    print(f"  Output dtype: {outputs[0].dtype}")
    print(f"  Value range: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")


def example_2_execution_providers():
    """
    Example 2: Execution Providers (Hardware Backends)

    Execution providers determine where your model runs:
    - CPUExecutionProvider: Standard CPU
    - CUDAExecutionProvider: NVIDIA GPUs
    - TensorRTExecutionProvider: Optimized NVIDIA inference
    - CoreMLExecutionProvider: Apple devices
    - DirectMLExecutionProvider: Windows GPU (any vendor)

    Edge Device Benefits:
    - Lower power consumption
    - Reduced latency (no cloud roundtrip)
    - Privacy (data stays on device)
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Execution Providers")
    print("=" * 60)

    # Check what's available on your system
    available = ort.get_available_providers()
    print(f"\nAvailable providers: {available}")

    model_path = "yolo11n.onnx"
    if not Path(model_path).exists():
        return

    # Priority order: ORT tries first provider, falls back if unavailable
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    print(f"Active providers: {session.get_providers()}")

    # Configure provider-specific options
    cuda_options = {
        "device_id": 0,  # GPU device ID
        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB memory limit
        "arena_extend_strategy": "kSameAsRequested",  # Memory allocation
        "cudnn_conv_algo_search": "DEFAULT",  # Conv algorithm
    }

    # Create session with provider options
    session_configured = ort.InferenceSession(
        model_path,
        providers=[("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"],
    )


def example_3_session_options():
    """
    Example 3: Session Options for Performance Tuning

    SessionOptions control how the model executes:
    - Threading: Parallelize operations
    - Optimization: Graph transformations for speed
    - Memory: Reduce memory footprint
    - Profiling: Measure performance bottlenecks

    Critical for edge devices with limited resources.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Session Options")
    print("=" * 60)

    model_path = "yolo11n.onnx"
    if not Path(model_path).exists():
        return

    options = ort.SessionOptions()

    # Threading Configuration
    # ------------------------
    # intra_op_num_threads: Threads for parallel operations within a layer
    # inter_op_num_threads: Threads for parallel execution of layers
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 2

    # Execution mode: Sequential vs Parallel
    # ORT_SEQUENTIAL: Layers run one after another (lower memory)
    # ORT_PARALLEL: Layers run in parallel when possible (faster)
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Graph Optimization Levels
    # -------------------------
    # ORT_DISABLE_ALL: No optimizations (debugging)
    # ORT_ENABLE_BASIC: Constant folding, redundant node removal
    # ORT_ENABLE_EXTENDED: Node fusion (Conv+BN+ReLU -> single op)
    # ORT_ENABLE_ALL: All optimizations including layout transforms
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Save optimized graph for inspection
    options.optimized_model_filepath = "optimized_model.onnx"

    # Memory Optimization
    # -------------------
    options.enable_mem_pattern = True  # Reuse memory patterns
    options.enable_cpu_mem_arena = True  # Pool allocator for CPU

    # Logging (0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal)
    options.log_severity_level = 3

    # Create session with options
    session = ort.InferenceSession(
        model_path, sess_options=options, providers=["CPUExecutionProvider"]
    )

    print("\n--- Session Configuration ---")
    print(f"  Intra-op threads: {options.intra_op_num_threads}")
    print(f"  Inter-op threads: {options.inter_op_num_threads}")
    print(f"  Optimization level: {options.graph_optimization_level}")
    print(f"  Memory arena: {options.enable_cpu_mem_arena}")


def example_4_performance_comparison():
    """
    Example 4: Measure Inference Performance

    ONNX Runtime is typically 2-10x faster than PyTorch/TensorFlow
    for inference, especially on CPUs and edge devices.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Performance Measurement")
    print("=" * 60)

    model_path = "yolo11n.onnx"
    if not Path(model_path).exists():
        return

    import time

    # Setup
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)

    # Warmup (first run is slower due to initialization)
    for _ in range(5):
        session.run(None, {input_name: input_data})

    # Measure latency
    num_runs = 100
    latencies = []

    for _ in range(num_runs):
        start = time.perf_counter()
        outputs = session.run(None, {input_name: input_data})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    throughput = 1000 / avg_latency  # FPS

    print(f"\n--- Performance Metrics (n={num_runs}) ---")
    print(f"  Average latency: {avg_latency:.2f} ms")
    print(f"  Std deviation: {std_latency:.2f} ms")
    print(f"  Throughput: {throughput:.2f} FPS")
    print(f"  Min latency: {np.min(latencies):.2f} ms")
    print(f"  Max latency: {np.max(latencies):.2f} ms")


def example_5_edge_device_tips():
    """
    Example 5: Edge Device Optimization Tips

    Best practices for deploying on resource-constrained devices:
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Edge Device Optimization")
    print("=" * 60)

    tips = """
    1. Model Quantization
       - Convert FP32 → INT8 (4x smaller, 2-4x faster)
       - Use dynamic quantization for post-training
       - Example: 100MB model → 25MB model
    
    2. Use Appropriate Execution Providers
       - Mobile: CoreML (iOS), NNAPI (Android), QNN (Qualcomm)
       - Edge devices: CPU or vendor-specific accelerators
       - Raspberry Pi: CPUExecutionProvider with low thread count
    
    3. Optimize Session Options
       - Use ORT_ENABLE_ALL optimization level
       - Set intra_op_num_threads based on available cores
       - Enable memory arena for efficient allocation
    
    4. Input Resolution
       - Lower resolution = faster inference
       - YOLO: 640x640 → 320x320 (4x faster)
       - Trade-off: speed vs accuracy
    
    5. Batch Size
       - Edge devices: Use batch_size=1
       - Reduces memory and latency
       - Higher batches for throughput, not latency
    
    6. Model Selection
       - Use mobile-optimized models (e.g., MobileNet, EfficientNet)
       - YOLO11n (nano) vs YOLO11x (extra large)
       - Fewer parameters = faster inference
    
    7. Memory Management
       - Monitor RAM usage with enable_cpu_mem_arena
       - Set gpu_mem_limit for GPU providers
       - Release sessions when not in use
    """

    print(tips)

    # Example: Lightweight session for edge device
    model_path = "yolo11n.onnx"
    if Path(model_path).exists():
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2  # Low thread count for edge
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = True
        options.log_severity_level = 3  # Minimal logging

        session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )

        print("\n--- Edge-Optimized Session Created ---")
        print(f"  Provider: {session.get_providers()}")
        print(f"  Threads: {options.intra_op_num_threads}")


def example_6_model_inspection():
    """
    Example 6: Inspect Model Metadata

    Understanding your model's structure helps optimize deployment.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Model Inspection")
    print("=" * 60)

    model_path = "yolo11n.onnx"
    if not Path(model_path).exists():
        return

    session = ort.InferenceSession(model_path)

    # Model metadata
    metadata = session.get_modelmeta()
    print(f"\n--- Model Metadata ---")
    print(f"  Producer: {metadata.producer_name}")
    print(f"  Version: {metadata.version}")
    print(f"  Graph name: {metadata.graph_name}")

    # Detailed input/output information
    print(f"\n--- Detailed Input Information ---")
    for idx, inp in enumerate(session.get_inputs()):
        print(f"  [{idx}] {inp.name}")
        print(f"      Shape: {inp.shape}")
        print(f"      Type: {inp.type}")
        # Dynamic dimensions show as strings like 'batch'

    print(f"\n--- Detailed Output Information ---")
    for idx, out in enumerate(session.get_outputs()):
        print(f"  [{idx}] {out.name}")
        print(f"      Shape: {out.shape}")
        print(f"      Type: {out.type}")


def main():
    """
    Run all examples to learn ONNX Runtime basics.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "ONNX RUNTIME TUTORIAL")
    print("=" * 70)
    print("\nThis tutorial covers the fundamentals of ONNX Runtime,")
    print("a high-performance inference engine optimized for edge devices.")

    try:
        example_1_basic_inference()
        example_2_execution_providers()
        example_3_session_options()
        example_4_performance_comparison()
        example_5_edge_device_tips()
        example_6_model_inspection()

        print("\n" + "=" * 70)
        print("Tutorial completed! Key takeaways:")
        print("=" * 70)
        print(
            """
1. InferenceSession: Core object for model execution
2. Execution Providers: Choose hardware backend (CPU/GPU/Mobile)
3. SessionOptions: Tune threading, optimization, memory
4. Edge Optimization: Quantization, low resolution, efficient models
5. Performance: Measure latency and throughput for your use case

Next steps:
- Export your PyTorch/TensorFlow model to ONNX
- Profile with enable_profiling for bottlenecks
- Test on target hardware (Raspberry Pi, Jetson, mobile)
- Consider quantization for 4x size reduction
        """
        )

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have an ONNX model in the current directory.")


if __name__ == "__main__":
    main()
