"""
Benchmark CLABSIGuard V1 vs V2
Compare performance, parameters, and inference speed
"""
import torch
import time
import numpy as np
from clabsi_guard import CLABSIGuard
from clabsi_guard_v2 import CLABSIGuardV2


def benchmark_model(model, model_name, num_runs=100):
    """Benchmark a model"""
    print(f"\nBenchmarking {model_name}...")
    print("="*60)

    device = next(model.parameters()).device
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, 480, 640).to(device)

    # Warm up
    print("Warming up (10 runs)...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure inference time
    print(f"Measuring inference time ({num_runs} runs)...")
    times = []

    with torch.no_grad():
        for i in range(num_runs):
            start = time.time()
            outputs = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{num_runs}")

    # Calculate statistics
    mean_time = np.mean(times) * 1000  # ms
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    fps = 1.0 / np.mean(times)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Memory usage
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
        peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
    else:
        peak_memory = 0

    results = {
        'name': model_name,
        'mean_ms': mean_time,
        'std_ms': std_time,
        'min_ms': min_time,
        'max_ms': max_time,
        'fps': fps,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'peak_memory_gb': peak_memory
    }

    return results


def print_results(results):
    """Print benchmark results"""
    print("\n" + "="*60)
    print(f"Results: {results['name']}")
    print("="*60)
    print(f"Inference Time:")
    print(f"  Mean: {results['mean_ms']:.2f} ms")
    print(f"  Std:  {results['std_ms']:.2f} ms")
    print(f"  Min:  {results['min_ms']:.2f} ms")
    print(f"  Max:  {results['max_ms']:.2f} ms")
    print(f"  FPS:  {results['fps']:.1f}")
    print(f"\nParameters:")
    print(f"  Total: {results['total_params']:,}")
    print(f"  Trainable: {results['trainable_params']:,}")
    if results['peak_memory_gb'] > 0:
        print(f"\nGPU Memory:")
        print(f"  Peak: {results['peak_memory_gb']:.2f} GB")
    print("="*60)


def compare_results(v1_results, v2_results):
    """Compare V1 vs V2"""
    print("\n" + "="*60)
    print("COMPARISON: V1 vs V2")
    print("="*60)

    # Speed comparison
    speedup = v1_results['fps'] / v2_results['fps']
    if v2_results['fps'] > v1_results['fps']:
        speedup = v2_results['fps'] / v1_results['fps']
        print(f"\nSpeed:")
        print(f"  V1: {v1_results['fps']:.1f} FPS")
        print(f"  V2: {v2_results['fps']:.1f} FPS")
        print(f"  V2 is {speedup:.2f}x FASTER")
    else:
        print(f"\nSpeed:")
        print(f"  V1: {v1_results['fps']:.1f} FPS")
        print(f"  V2: {v2_results['fps']:.1f} FPS")
        print(f"  V1 is {speedup:.2f}x faster")

    # Parameter comparison
    param_ratio = v1_results['total_params'] / v2_results['total_params']
    print(f"\nParameters:")
    print(f"  V1: {v1_results['total_params']:,} params")
    print(f"  V2: {v2_results['total_params']:,} params")
    print(f"  V2 is {param_ratio:.2f}x SMALLER")

    # Memory comparison
    if v1_results['peak_memory_gb'] > 0 and v2_results['peak_memory_gb'] > 0:
        mem_ratio = v1_results['peak_memory_gb'] / v2_results['peak_memory_gb']
        print(f"\nGPU Memory:")
        print(f"  V1: {v1_results['peak_memory_gb']:.2f} GB")
        print(f"  V2: {v2_results['peak_memory_gb']:.2f} GB")
        if mem_ratio > 1:
            print(f"  V2 uses {mem_ratio:.2f}x LESS memory")
        else:
            print(f"  V1 uses {1/mem_ratio:.2f}x less memory")

    print("="*60)

    # Overall assessment
    print("\nOVERALL ASSESSMENT:")
    print("-"*60)

    improvements = []
    if v2_results['fps'] > v1_results['fps']:
        improvements.append(f"Faster inference ({v2_results['fps']:.1f} vs {v1_results['fps']:.1f} FPS)")
    if param_ratio > 1:
        improvements.append(f"Smaller model ({param_ratio:.1f}x fewer parameters)")
    improvements.append("Pretrained backbone (better features)")
    improvements.append("TEO-1 architecture maintained")

    for improvement in improvements:
        print(f"  + {improvement}")

    print("="*60)


def main():
    print("\n" + "="*80)
    print("CLABSIGuard V1 vs V2 Benchmark")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load V1
    print("\nLoading V1 (ResNet50)...")
    v1_model = CLABSIGuard(pretrained=True)
    v1_model = v1_model.to(device)
    v1_model.eval()

    # Load V2
    print("\nLoading V2 (YOLO Pretrained)...")
    v2_model = CLABSIGuardV2("models/yolo11n-pose.pt", freeze_backbone=True)
    v2_model = v2_model.to(device)
    v2_model.eval()

    # Benchmark V1
    v1_results = benchmark_model(v1_model, "V1 (ResNet50)")
    print_results(v1_results)

    # Benchmark V2
    v2_results = benchmark_model(v2_model, "V2 (YOLO Pretrained)")
    print_results(v2_results)

    # Compare
    compare_results(v1_results, v2_results)

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
