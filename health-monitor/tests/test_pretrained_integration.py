"""Quick test to verify pretrained heads integration"""
import numpy as np
from heads.pretrained_heads import PretrainedDepthHead, PretrainedSegmentationHead, PretrainedKeypointsHead

print("Testing pretrained heads integration...")

# Create dummy image
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

print("\n1. Loading Depth Head...")
try:
    depth_head = PretrainedDepthHead()
    print("   [+] Depth head loaded")
except Exception as e:
    print(f"   [-] Error: {e}")

print("\n2. Loading Segmentation Head...")
try:
    seg_head = PretrainedSegmentationHead()
    print("   [+] Segmentation head loaded")
except Exception as e:
    print(f"   [-] Error: {e}")

print("\n3. Loading Keypoints Head...")
try:
    kp_head = PretrainedKeypointsHead()
    print("   [+] Keypoints head loaded")
except Exception as e:
    print(f"   [-] Error: {e}")

print("\n4. Running predictions...")
try:
    depth = depth_head.predict(image)
    print(f"   [+] Depth: {depth.shape}")

    seg = seg_head.predict(image)
    print(f"   [+] Segmentation: {seg.shape}")

    kp = kp_head.predict(image)
    print(f"   [+] Keypoints: {kp.shape}")
except Exception as e:
    print(f"   [-] Error: {e}")
    import traceback
    traceback.print_exc()

print("\nIntegration test complete!")
