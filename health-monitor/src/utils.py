"""
Utility functions for CLABSIGuard healthcare monitoring system
"""
import torch
import numpy as np


def count_model_parameters(model):
    """
    Count total and trainable parameters in a model
    Returns dict with counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def print_model_summary(model):
    """Print model architecture summary"""
    print("\nModel Architecture Summary")
    print("="*50)

    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name:25s}: {params:>15,} params")

    total = count_model_parameters(model)
    print("="*50)
    print(f"{'Total':25s}: {total['total']:>15,} params")
    print(f"{'Trainable':25s}: {total['trainable']:>15,} params")
    print("="*50)


def measure_inference_time(model, input_shape=(1, 3, 480, 640), num_runs=100, warmup=10):
    """
    Measure average inference time
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        num_runs: Number of inference runs
        warmup: Number of warmup runs
    Returns:
        dict with timing statistics
    """
    device = next(model.parameters()).device
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Measure
    if device.type == 'cuda':
        torch.cuda.synchronize()

    import time
    times = []

    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'fps': 1.0 / np.mean(times)
    }


def check_gpu_memory():
    """Check GPU memory usage"""
    if not torch.cuda.is_available():
        return None

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9

    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated
    }


def benchmark_model(model, input_shape=(1, 3, 480, 640), num_runs=100):
    """
    Complete benchmark of model
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        num_runs: Number of runs for timing
    """
    print("\n" + "="*50)
    print("Model Benchmark")
    print("="*50)

    # Parameter count
    params = count_model_parameters(model)
    print(f"\nTotal parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")

    # Device info
    device = next(model.parameters()).device
    print(f"\nDevice: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Inference timing
    print(f"\nMeasuring inference time ({num_runs} runs)...")
    timing = measure_inference_time(model, input_shape, num_runs)

    print(f"Mean: {timing['mean_ms']:.2f} ms")
    print(f"Std: {timing['std_ms']:.2f} ms")
    print(f"Min: {timing['min_ms']:.2f} ms")
    print(f"Max: {timing['max_ms']:.2f} ms")
    print(f"FPS: {timing['fps']:.1f}")

    # Memory usage
    if device.type == 'cuda':
        mem = check_gpu_memory()
        print(f"\nGPU Memory:")
        print(f"Allocated: {mem['allocated_gb']:.2f} GB")
        print(f"Reserved: {mem['reserved_gb']:.2f} GB")
        print(f"Peak: {mem['max_allocated_gb']:.2f} GB")

    print("="*50)

    return {
        'parameters': params,
        'timing': timing,
        'device': str(device)
    }


def parse_detection_output(detection_tensor, conf_threshold=0.5, nms_threshold=0.4):
    """
    Parse detection head output to bounding boxes
    Args:
        detection_tensor: [batch, anchors*(5+classes), h, w]
        conf_threshold: Confidence threshold
        nms_threshold: NMS IoU threshold
    Returns:
        List of detections per image
    """
    # Placeholder implementation
    # Real version would parse YOLO-style outputs
    # For now, return empty list
    return []


def parse_keypoints_heatmaps(heatmaps, threshold=0.5):
    """
    Extract keypoint coordinates from heatmaps
    Args:
        heatmaps: [batch, num_keypoints, h, w]
        threshold: Confidence threshold
    Returns:
        List of keypoint coordinates
    """
    batch_size, num_kp, h, w = heatmaps.shape
    keypoints = []

    for b in range(batch_size):
        kp_batch = []
        for k in range(num_kp):
            heatmap = heatmaps[b, k].cpu().numpy()

            # Find peak
            max_val = heatmap.max()
            if max_val > threshold:
                max_idx = heatmap.argmax()
                y, x = np.unravel_index(max_idx, heatmap.shape)
                kp_batch.append((x, y, max_val))
            else:
                kp_batch.append(None)

        keypoints.append(kp_batch)

    return keypoints


if __name__ == "__main__":
    print("Utility functions loaded successfully")
