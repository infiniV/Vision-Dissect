"""
Test visualization functions without launching full webcam demo
"""
import cv2
import torch
import numpy as np
from src.clabsi_guard import CLABSIGuard
from src.monitor import ComplianceMonitor


def test_visualization():
    """Test all visualization functions"""
    print("Testing visualization functions...")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLABSIGuard(pretrained=True)
    model = model.to(device)
    model.eval()

    # Create dummy frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Preprocess
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    # Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(tensor)

    print("Model outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # Test monitor (should not trigger violations now)
    print("\nTesting monitor (should show NONE status)...")
    monitor = ComplianceMonitor()
    violations = monitor.update(outputs)

    if len(violations) == 0:
        print("  SUCCESS: No false violations detected")
    else:
        print(f"  WARNING: {len(violations)} violations detected")
        for v in violations:
            print(f"    {v}")

    status = monitor.get_current_status()
    print(f"  Status: {status.value}")

    if status.value == "NONE":
        print("  SUCCESS: Status is NONE (no false alarms)")
    else:
        print(f"  WARNING: Status is {status.value} (expected NONE)")

    # Test keypoint extraction
    print("\nTesting keypoint visualization...")
    kp = outputs['keypoints'][0].cpu().numpy()
    peaks_found = 0
    for i in range(kp.shape[0]):
        heatmap = kp[i]
        if heatmap.max() > 0.3:
            peaks_found += 1

    print(f"  Keypoint peaks found: {peaks_found}/{kp.shape[0]}")

    # Test depth visualization
    print("\nTesting depth visualization...")
    depth = outputs['depth'][0, 0].cpu().numpy()
    print(f"  Depth range: {depth.min():.3f} to {depth.max():.3f}")

    # Test segmentation
    print("\nTesting segmentation...")
    seg = outputs['segmentation'][0].cpu().numpy()
    seg_mask = np.argmax(seg, axis=0)
    unique_classes = np.unique(seg_mask)
    print(f"  Segmentation classes present: {unique_classes.tolist()}")

    print("\nAll visualization tests completed successfully!")


if __name__ == "__main__":
    test_visualization()
