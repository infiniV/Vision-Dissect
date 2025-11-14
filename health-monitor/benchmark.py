"""
Benchmark CLABSIGuard model performance
"""
import torch
from clabsi_guard import CLABSIGuard
from utils import benchmark_model, print_model_summary


def main():
    print("Loading CLABSIGuard model...")

    # Create model
    model = CLABSIGuard(pretrained=True)

    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Enable optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Print architecture
    print_model_summary(model)

    # Run benchmark
    results = benchmark_model(model, input_shape=(1, 3, 480, 640), num_runs=100)

    # Performance check
    print("\n" + "="*50)
    print("Performance Check")
    print("="*50)

    target_fps = 5.0
    actual_fps = results['timing']['fps']

    if actual_fps >= target_fps:
        print(f"SUCCESS: {actual_fps:.1f} FPS >= {target_fps} FPS target")
    else:
        print(f"WARNING: {actual_fps:.1f} FPS < {target_fps} FPS target")
        print("\nOptimization suggestions:")
        print("1. Reduce input size (640x480 → 480x360 → 320x240)")
        print("2. Enable FP16 inference with model.half()")
        print("3. Process every 2nd frame")
        print("4. Use ResNet34 instead of ResNet50")

    print("="*50)


if __name__ == "__main__":
    main()
