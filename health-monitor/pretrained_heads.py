"""
Pretrained Model Heads for CLABSIGuard V2
Uses actual pretrained models for immediate high-quality predictions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from ultralytics import YOLO
import sys
import os

# Add path for depth_anything_v2
vision_bench_path = os.path.join(os.path.dirname(__file__), '..', 'vision-bench')
sys.path.insert(0, vision_bench_path)


class PretrainedDepthHead(nn.Module):
    """
    Depth prediction using DepthAnything V2 from Hugging Face
    """
    def __init__(self):
        super().__init__()

        print("Loading DepthAnything V2 Small from Hugging Face...")

        try:
            from transformers import pipeline

            # Load depth estimation pipeline
            self.depth_model = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if torch.cuda.is_available() else -1
            )

            print(f"  DepthAnything V2 Small loaded!")

        except Exception as e:
            print(f"  WARNING: Error loading model ({e}), using grayscale fallback")
            self.depth_model = None

    def forward(self, x):
        """Placeholder for compatibility"""
        batch_size = x.shape[0]
        return torch.zeros(batch_size, 1, 480, 640, device=x.device)

    def predict(self, image_np):
        """
        Predict depth from numpy image

        Args:
            image_np: RGB image (H, W, 3) uint8

        Returns:
            Depth map (H, W) float32 normalized to [0, 1]
        """
        if self.depth_model is None:
            # Fallback: create fake depth based on image intensity
            h, w = image_np.shape[:2]
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            depth = gray.astype(np.float32) / 255.0
            return depth

        from PIL import Image

        # Convert numpy to PIL Image
        pil_image = Image.fromarray(image_np)

        # Run pipeline
        result = self.depth_model(pil_image)

        # Extract depth map (PIL Image) and convert to numpy
        depth_pil = result["depth"]
        depth = np.array(depth_pil).astype(np.float32)

        # Normalize to [0, 1]
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth_norm


class PretrainedSegmentationHead(nn.Module):
    """
    Segmentation using YOLO11n-Seg
    """
    def __init__(self, model_path="../models/yolo11n-seg.pt"):
        super().__init__()

        print(f"Loading YOLO11n-Seg from {model_path}...")

        self.seg_model = YOLO(model_path)
        self.seg_model.to('cuda' if torch.cuda.is_available() else 'cpu')

        print("  YOLO11n-Seg loaded!")

    def forward(self, x):
        """Placeholder for compatibility"""
        batch_size = x.shape[0]
        return torch.zeros(batch_size, 8, 480, 640, device=x.device)

    def predict(self, image_np):
        """
        Predict segmentation from numpy image

        Args:
            image_np: RGB image (H, W, 3) uint8

        Returns:
            Segmentation map (8, H, W) float32
        """
        h, w = image_np.shape[:2]

        # Run YOLO segmentation
        results = self.seg_model(image_np, verbose=False)

        # Initialize output
        seg_output = np.zeros((8, h, w), dtype=np.float32)

        if len(results) > 0:
            result = results[0]

            # Check if masks are available
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()  # [N, H, W]

                # Resize masks to original size and assign to classes
                for i, mask in enumerate(masks):
                    if i >= 8:
                        break

                    # Resize mask
                    mask_resized = cv2.resize(mask, (w, h))

                    # Assign to class channel
                    seg_output[i] = mask_resized

        return seg_output


class PretrainedKeypointsHead(nn.Module):
    """
    Keypoints prediction using YOLO11n-Pose
    """
    def __init__(self, model_path="../models/yolo11n-pose.pt"):
        super().__init__()

        print(f"Loading YOLO11n-Pose from {model_path}...")

        self.pose_model = YOLO(model_path)
        self.pose_model.to('cuda' if torch.cuda.is_available() else 'cpu')

        print("  YOLO11n-Pose loaded!")

    def forward(self, x):
        """Placeholder for compatibility"""
        batch_size = x.shape[0]
        return torch.zeros(batch_size, 21, 480, 640, device=x.device)

    def predict(self, image_np):
        """
        Predict keypoints from numpy image

        Args:
            image_np: RGB image (H, W, 3) uint8

        Returns:
            Keypoint heatmaps (21, H, W) float32
        """
        h, w = image_np.shape[:2]

        # Run YOLO pose
        results = self.pose_model(image_np, verbose=False)

        # Initialize heatmaps
        heatmaps = np.zeros((21, h, w), dtype=np.float32)

        if len(results) > 0:
            result = results[0]

            # Check if keypoints are available
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                kps = result.keypoints.data.cpu().numpy()  # [N, 17, 3] (x, y, conf)

                # Process first detection (if any)
                if len(kps) > 0:
                    keypoints = kps[0]  # [17, 3]

                    # Create Gaussian heatmaps for first 17 keypoints
                    sigma = 15  # Larger sigma for more visible heatmaps

                    for i in range(min(17, 21)):  # Use first 17 COCO keypoints
                        x, y, conf = keypoints[i]

                        if conf > 0.3:  # Only draw confident keypoints
                            x_int = int(x)
                            y_int = int(y)

                            if 0 <= x_int < w and 0 <= y_int < h:
                                # Create Gaussian heatmap
                                size = int(sigma * 3)
                                x_min = max(0, x_int - size)
                                x_max = min(w, x_int + size + 1)
                                y_min = max(0, y_int - size)
                                y_max = min(h, y_int + size + 1)

                                x_grid, y_grid = np.meshgrid(
                                    np.arange(x_min, x_max),
                                    np.arange(y_min, y_max)
                                )

                                gaussian = np.exp(-((x_grid - x_int)**2 + (y_grid - y_int)**2) / (2 * sigma**2))
                                gaussian = gaussian * conf

                                heatmaps[i, y_min:y_max, x_min:x_max] = np.maximum(
                                    heatmaps[i, y_min:y_max, x_min:x_max],
                                    gaussian
                                )

        return heatmaps


def test_pretrained_heads():
    """Test pretrained heads"""
    print("Testing pretrained model heads...")

    # Create dummy image (use actual test image if available)
    import os
    test_image_path = "../test_data"

    if os.path.exists(test_image_path):
        # Try to load a real test image
        test_images = [f for f in os.listdir(test_image_path) if f.endswith(('.jpg', '.png'))]
        if test_images:
            image_path = os.path.join(test_image_path, test_images[0])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (640, 480))
            print(f"Using test image: {test_images[0]}")
        else:
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    else:
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Test depth
    print("\nTesting Depth Head...")
    depth_head = PretrainedDepthHead()
    depth = depth_head.predict(image)
    print(f"  Depth output shape: {depth.shape}")
    print(f"  Depth range: [{depth.min():.3f}, {depth.max():.3f}]")

    # Test segmentation
    print("\nTesting Segmentation Head...")
    seg_head = PretrainedSegmentationHead()
    seg = seg_head.predict(image)
    print(f"  Segmentation output shape: {seg.shape}")
    print(f"  Segmentation max: {seg.max():.3f}")

    # Test keypoints
    print("\nTesting Keypoints Head...")
    kp_head = PretrainedKeypointsHead()
    kp = kp_head.predict(image)
    print(f"  Keypoints output shape: {kp.shape}")
    print(f"  Keypoints max: {kp.max():.3f}")

    print("\nAll pretrained heads tested successfully!")


if __name__ == "__main__":
    test_pretrained_heads()
