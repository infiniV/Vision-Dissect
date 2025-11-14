"""Debug pretrained model predictions"""
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

print(f"Input shape: {rgb.shape}, dtype: {rgb.dtype}, range: [{rgb.min()}, {rgb.max()}]")

# Test depth
print("\n=== DEPTH ===")
depth = depth_head.predict(rgb)
print(f"Output shape: {depth.shape}")
print(f"Output dtype: {depth.dtype}")
print(f"Output range: [{depth.min():.6f}, {depth.max():.6f}]")
print(f"Output mean: {depth.mean():.6f}")
print(f"Output std: {depth.std():.6f}")
print(f"Unique values count: {len(np.unique(depth))}")

# Test segmentation
print("\n=== SEGMENTATION ===")
seg = seg_head.predict(rgb)
print(f"Output shape: {seg.shape}")
print(f"Output dtype: {seg.dtype}")
print(f"Output range: [{seg.min():.6f}, {seg.max():.6f}]")
for i in range(8):
    channel_max = seg[i].max()
    channel_sum = seg[i].sum()
    print(f"  Channel {i}: max={channel_max:.6f}, sum={channel_sum:.1f}")

# Test keypoints
print("\n=== KEYPOINTS ===")
kp = kp_head.predict(rgb)
print(f"Output shape: {kp.shape}")
print(f"Output dtype: {kp.dtype}")
print(f"Output range: [{kp.min():.6f}, {kp.max():.6f}]")
non_zero_channels = [i for i in range(21) if kp[i].max() > 0]
print(f"Non-zero channels: {non_zero_channels}")
