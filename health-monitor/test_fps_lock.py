"""
Test FPS locking functionality
"""
import time
import torch
import cv2
import numpy as np
from clabsi_guard import CLABSIGuard


def test_fps_lock():
    """Test that FPS limiting works correctly"""
    print("Testing FPS lock at 30 FPS...")

    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLABSIGuard(pretrained=True)
    model = model.to(device)
    model.eval()

    target_fps = 30
    frame_time = 1.0 / target_fps

    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    # Test 60 frames with FPS limiting
    fps_measurements = []
    last_time = time.time()

    print(f"Running 60 frames with {target_fps} FPS target...")

    for i in range(60):
        frame_start = time.time()

        # Inference
        with torch.no_grad():
            outputs = model(tensor)

        # Measure actual FPS
        current_time = time.time()
        actual_frame_time = current_time - last_time
        last_time = current_time
        actual_fps = 1.0 / actual_frame_time if actual_frame_time > 0 else 0
        fps_measurements.append(actual_fps)

        # FPS limiting
        processing_time = time.time() - frame_start
        sleep_time = frame_time - processing_time

        if sleep_time > 0:
            time.sleep(sleep_time)

        if (i + 1) % 10 == 0:
            recent_avg = np.mean(fps_measurements[-10:])
            print(f"  Frame {i+1}/60: Recent avg FPS = {recent_avg:.1f}")

    # Analyze results
    avg_fps = np.mean(fps_measurements)
    std_fps = np.std(fps_measurements)
    min_fps = np.min(fps_measurements)
    max_fps = np.max(fps_measurements)

    print("\nResults:")
    print(f"  Target FPS: {target_fps}")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Std Dev: {std_fps:.1f}")
    print(f"  Min FPS: {min_fps:.1f}")
    print(f"  Max FPS: {max_fps:.1f}")

    # Check if we're within tolerance
    tolerance = 3  # Allow ±3 FPS
    if abs(avg_fps - target_fps) <= tolerance:
        print(f"\nSUCCESS: Average FPS ({avg_fps:.1f}) is within ±{tolerance} of target ({target_fps})")
        print("FPS locking is working correctly!")
    else:
        diff = abs(avg_fps - target_fps)
        print(f"\nWARNING: Average FPS ({avg_fps:.1f}) is {diff:.1f} FPS away from target")

    # Check stability
    if std_fps <= 5:
        print(f"SUCCESS: FPS is stable (std dev = {std_fps:.1f})")
    else:
        print(f"WARNING: FPS is unstable (std dev = {std_fps:.1f})")

    print("\nFPS lock test completed!")


if __name__ == "__main__":
    test_fps_lock()
