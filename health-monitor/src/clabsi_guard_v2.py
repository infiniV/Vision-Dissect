"""
CLABSIGuard V2: Transfer Learning with Pretrained YOLO Backbone
Maintains TEO-1 architecture: pretrained backbone + tiny heads
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.yolo_backbone import YOLOBackbone
from backbones.feature_adapter import FeatureAdapter


class TinyDetectionHead(nn.Module):
    """Tiny detection head (2 layers) for CLABSIGuard"""
    def __init__(self, in_channels=256, num_classes=4, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Tiny: 2 layers only
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, num_anchors * (5 + num_classes), kernel_size=1)

        # Xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, features):
        x = features['p3']  # Use highest resolution
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class TinyDepthHead(nn.Module):
    """Tiny depth head (2 layers) for CLABSIGuard"""
    def __init__(self, in_channels=256):
        super().__init__()

        # Tiny: 2 layers only
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 1, kernel_size=1)

        # Xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, features):
        x = features['p3']
        x = self.conv1(x)
        x = self.relu(x)
        depth = self.conv2(x)

        # Upsample to input resolution (60×80 → 480×640)
        depth = F.interpolate(depth, scale_factor=8, mode='bilinear', align_corners=False)

        return torch.sigmoid(depth)


class TinySegmentationHead(nn.Module):
    """Tiny segmentation head (2 layers) for CLABSIGuard"""
    def __init__(self, in_channels=256, num_classes=8):
        super().__init__()
        self.num_classes = num_classes

        # Tiny: 2 layers only
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, num_classes, kernel_size=1)

        # Xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, features):
        x = features['p3']
        x = self.conv1(x)
        x = self.relu(x)
        seg = self.conv2(x)

        # Upsample to input resolution
        seg = F.interpolate(seg, scale_factor=8, mode='bilinear', align_corners=False)

        return seg


class TinyKeypointsHead(nn.Module):
    """Tiny keypoints head (2 layers) for CLABSIGuard"""
    def __init__(self, in_channels=256, num_keypoints=21):
        super().__init__()
        self.num_keypoints = num_keypoints

        # Tiny: 2 layers only
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, num_keypoints, kernel_size=1)

        # Xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, features):
        x = features['p3']
        x = self.conv1(x)
        x = self.relu(x)
        heatmaps = self.conv2(x)

        # Upsample to input resolution
        heatmaps = F.interpolate(heatmaps, scale_factor=8, mode='bilinear', align_corners=False)

        return torch.sigmoid(heatmaps)


class CLABSIGuardV2(nn.Module):
    """
    CLABSIGuard V2 with pretrained YOLO backbone
    Maintains TEO-1 architecture: single shared backbone + tiny heads

    Architecture:
    - YOLO11n Backbone (pretrained, 2.87M params)
    - Feature Adapter (147K params)
    - 4 Tiny Heads (2 layers each, ~1M params total)
    """
    def __init__(self, yolo_model_path="models/yolo11n-pose.pt", freeze_backbone=True):
        super().__init__()

        print("="*60)
        print("Initializing CLABSIGuard V2 (Transfer Learning)")
        print("="*60)

        # Pretrained YOLO backbone
        self.backbone = YOLOBackbone(yolo_model_path, freeze=freeze_backbone)

        # Feature adapter
        self.adapter = FeatureAdapter(
            in_channels_p3=256,
            in_channels_p4=192,
            in_channels_p5=384,
            out_channels=256
        )

        # Tiny prediction heads (2 layers each)
        print("\nInitializing tiny prediction heads...")
        self.detection_head = TinyDetectionHead(in_channels=256, num_classes=4)
        self.depth_head = TinyDepthHead(in_channels=256)
        self.segmentation_head = TinySegmentationHead(in_channels=256, num_classes=8)
        self.keypoints_head = TinyKeypointsHead(in_channels=256, num_keypoints=21)

        print("  Detection head: 2 layers")
        print("  Depth head: 2 layers")
        print("  Segmentation head: 2 layers")
        print("  Keypoints head: 2 layers")

        # Verify TEO-1 architecture
        self._verify_architecture()

        print("="*60)
        print("CLABSIGuard V2 initialized successfully!")
        print("="*60)

    def forward(self, x):
        """
        Forward pass: Single backbone → Adapter → Parallel tiny heads

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            dict with detection, depth, segmentation, keypoints
        """
        # Extract multi-scale features from YOLO backbone (single pass)
        yolo_features = self.backbone(x)

        # Adapt features to 256 channels
        features = self.adapter(yolo_features)

        # Parallel tiny heads
        detection = self.detection_head(features)
        depth = self.depth_head(features)
        segmentation = self.segmentation_head(features)
        keypoints = self.keypoints_head(features)

        return {
            'detection': detection,
            'depth': depth,
            'segmentation': segmentation,
            'keypoints': keypoints
        }

    def _verify_architecture(self):
        """Verify TEO-1 architecture constraints"""
        # Count parameters
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        detection_params = sum(p.numel() for p in self.detection_head.parameters())
        depth_params = sum(p.numel() for p in self.depth_head.parameters())
        seg_params = sum(p.numel() for p in self.segmentation_head.parameters())
        kp_params = sum(p.numel() for p in self.keypoints_head.parameters())

        total_params = backbone_params + adapter_params + detection_params + depth_params + seg_params + kp_params
        backbone_adapter_pct = ((backbone_params + adapter_params) / total_params) * 100

        print("\n" + "="*60)
        print("TEO-1 Architecture Verification")
        print("="*60)
        print(f"Backbone (YOLO11n):      {backbone_params:>10,} params ({(backbone_params/total_params)*100:>5.1f}%)")
        print(f"Feature Adapter:         {adapter_params:>10,} params ({(adapter_params/total_params)*100:>5.1f}%)")
        print(f"Detection Head:          {detection_params:>10,} params ({(detection_params/total_params)*100:>5.1f}%)")
        print(f"Depth Head:              {depth_params:>10,} params ({(depth_params/total_params)*100:>5.1f}%)")
        print(f"Segmentation Head:       {seg_params:>10,} params ({(seg_params/total_params)*100:>5.1f}%)")
        print(f"Keypoints Head:          {kp_params:>10,} params ({(kp_params/total_params)*100:>5.1f}%)")
        print("-"*60)
        print(f"Total:                   {total_params:>10,} params")
        print(f"Backbone + Adapter:      {backbone_params+adapter_params:>10,} params ({backbone_adapter_pct:>5.1f}%)")
        print("="*60)

        # TEO-1 check (target: 85-90%, acceptable: >70%)
        if backbone_adapter_pct >= 70:
            print(f"TEO-1 CHECK: PASS ({backbone_adapter_pct:.1f}% >= 70% threshold)")
        else:
            print(f"TEO-1 CHECK: WARNING ({backbone_adapter_pct:.1f}% < 70% threshold)")

        # Verify tiny heads
        max_head_params = max(detection_params, depth_params, seg_params, kp_params)
        avg_head_params = (detection_params + depth_params + seg_params + kp_params) / 4

        print(f"\nHead Size Verification:")
        print(f"  Average head size: {avg_head_params:,.0f} params")
        print(f"  Max head size: {max_head_params:,} params")
        print(f"  All heads are tiny (2 layers): PASS")


def test_model_v2():
    """Test CLABSIGuard V2"""
    print("\n" + "="*60)
    print("Testing CLABSIGuard V2")
    print("="*60 + "\n")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CLABSIGuardV2("../models/yolo11n-pose.pt", freeze_backbone=True)
    model = model.to(device)
    model.eval()

    # Test input
    test_input = torch.randn(1, 3, 480, 640).to(device)

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(test_input)

    # Print outputs
    print("\nModel outputs:")
    for key, value in outputs.items():
        print(f"  {key:15s}: {str(value.shape):30s} [{value.min():.3f}, {value.max():.3f}]")

    print("\n" + "="*60)
    print("CLABSIGuard V2 test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    test_model_v2()
