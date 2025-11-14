"""
Test webcam availability and system capabilities
"""
import cv2
import torch
import sys

def test_camera():
    """Test if webcam is accessible"""
    print("Testing webcam access...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot access webcam")
        return False

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame from webcam")
        cap.release()
        return False

    height, width, channels = frame.shape
    print(f"SUCCESS: Webcam accessible")
    print(f"  Resolution: {width}x{height}")
    print(f"  Channels: {channels}")

    cap.release()
    return True

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA availability...")
    cuda_available = torch.cuda.is_available()

    if cuda_available:
        print(f"SUCCESS: CUDA available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA not available, will use CPU")
        print("  Expected performance: <1 FPS on CPU")

    return cuda_available

def test_pytorch():
    """Test PyTorch installation"""
    print("\nTesting PyTorch...")
    print(f"  PyTorch version: {torch.__version__}")

    # Quick tensor operation
    x = torch.randn(1, 3, 224, 224)
    if torch.cuda.is_available():
        x = x.cuda()
        print("  GPU tensor creation: OK")
    else:
        print("  CPU tensor creation: OK")

    return True

def main():
    print("="*50)
    print("System Capabilities Test")
    print("="*50)

    camera_ok = test_camera()
    cuda_ok = test_cuda()
    pytorch_ok = test_pytorch()

    print("\n" + "="*50)
    print("Summary:")
    print(f"  Webcam: {'PASS' if camera_ok else 'FAIL'}")
    print(f"  CUDA: {'PASS' if cuda_ok else 'CPU-only'}")
    print(f"  PyTorch: {'PASS' if pytorch_ok else 'FAIL'}")
    print("="*50)

    if not camera_ok:
        print("\nERROR: Cannot proceed without webcam access")
        sys.exit(1)

    if not cuda_ok:
        print("\nWARNING: Without CUDA, expect <1 FPS. Consider using a GPU-enabled machine.")

    print("\nSystem ready for implementation!")

if __name__ == "__main__":
    main()
