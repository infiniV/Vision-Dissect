"""Visual test of all model outputs"""
import cv2
import numpy as np
from heads.pretrained_heads import PretrainedDepthHead, PretrainedSegmentationHead, PretrainedKeypointsHead

print("Loading models...")
depth_head = PretrainedDepthHead()
seg_head = PretrainedSegmentationHead()
kp_head = PretrainedKeypointsHead()

print("\nCapturing frame from webcam...")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to capture frame")
    exit(1)

# Resize to 640x480
frame = cv2.resize(frame, (640, 480))
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

print("\nRunning predictions...")
depth = depth_head.predict(rgb)
seg = seg_head.predict(rgb)
kp = kp_head.predict(rgb)

print(f"Depth: {depth.shape}, range [{depth.min():.3f}, {depth.max():.3f}]")
print(f"Segmentation: {seg.shape}, active channels: {[i for i in range(8) if seg[i].max() > 0]}")
print(f"Keypoints: {kp.shape}, active channels: {[i for i in range(21) if kp[i].max() > 0]}")

# Visualize depth
depth_vis = (depth * 255).astype(np.uint8)
depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

# Visualize segmentation (instance masks)
seg_colored = np.zeros((480, 640, 3), dtype=np.uint8)
colors = [
    [255, 0, 0],     # Red
    [0, 255, 0],     # Green
    [0, 0, 255],     # Blue
    [255, 255, 0],   # Yellow
    [255, 0, 255],   # Magenta
    [0, 255, 255],   # Cyan
    [255, 128, 0],   # Orange
    [128, 0, 255]    # Purple
]
for i in range(min(seg.shape[0], len(colors))):
    mask = seg[i] > 0.5
    if mask.any():
        seg_colored[mask] = colors[i]
        print(f"  Instance {i}: {mask.sum()} pixels in {colors[i]}")

# Visualize keypoints (sum all heatmaps)
kp_sum = np.sum(kp, axis=0)
if kp_sum.max() > 0:
    kp_sum = kp_sum / kp_sum.max()
kp_vis = (kp_sum * 255).astype(np.uint8)
kp_colored = cv2.applyColorMap(kp_vis, cv2.COLORMAP_HOT)

# Create grid
grid = np.zeros((480*2, 640*2, 3), dtype=np.uint8)
grid[0:480, 0:640] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Input
grid[0:480, 640:1280] = depth_colored  # Depth
grid[480:960, 0:640] = seg_colored  # Segmentation
grid[480:960, 640:1280] = kp_colored  # Keypoints

# Add labels
cv2.putText(grid, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(grid, "Depth", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(grid, "Segmentation", (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(grid, "Keypoints", (650, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Convert RGB to BGR for OpenCV display
grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)

# Display
cv2.imshow('Visual Test', grid_bgr)
print("\nDisplaying visual test. Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Visual test complete!")
