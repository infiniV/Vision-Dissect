"""
CLABSIGuard: Healthcare monitoring system using TEO-1 architecture
Single shared backbone (85-90% params) + multiple tiny prediction heads
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SharedBackbone(nn.Module):
    """
    Shared backbone with FPN for multi-scale features
    Uses ResNet50 pretrained on ImageNet
    Outputs 256-channel features at 3 scales
    """
    def __init__(self, pretrained=True):
        super().__init__()

        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)

        # Extract layers for multi-scale features
        # C1: 64 channels, stride 4
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        # C2: 256 channels, stride 4
        self.layer1 = resnet.layer1

        # C3: 512 channels, stride 8
        self.layer2 = resnet.layer2

        # C4: 1024 channels, stride 16
        self.layer3 = resnet.layer3

        # C5: 2048 channels, stride 32
        self.layer4 = resnet.layer4

        # FPN lateral connections (reduce channels to 256)
        self.lateral_c5 = nn.Conv2d(2048, 256, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(512, 256, kernel_size=1)

        # FPN top-down pathway (smooth features)
        self.smooth_p5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass through backbone
        Returns multi-scale features at 3 levels (P3, P4, P5)
        """
        # Bottom-up pathway
        c1 = self.conv1(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Lateral connections
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4)
        p3 = self.lateral_c3(c3)

        # Top-down pathway with upsampling
        p4 = p4 + F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        p3 = p3 + F.interpolate(p4, size=p3.shape[2:], mode='nearest')

        # Smooth features
        p5 = self.smooth_p5(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)

        return {
            'p3': p3,  # Stride 8, high resolution
            'p4': p4,  # Stride 16, medium resolution
            'p5': p5   # Stride 32, low resolution
        }


class DetectionHead(nn.Module):
    """
    Tiny detection head (3 layers max)
    Detects: bare_hand, gloved_hand, person, sanitizer
    Outputs: bounding boxes + class scores
    """
    def __init__(self, in_channels=256, num_classes=4, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Tiny head: 3 layers
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # Output: [x, y, w, h, objectness] + class_scores per anchor
        self.conv2 = nn.Conv2d(128, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, features):
        """Forward pass using P3 features for detection"""
        x = features['p3']  # Use highest resolution
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class DepthHead(nn.Module):
    """
    Tiny depth estimation head (3 layers max)
    Outputs normalized inverse depth
    """
    def __init__(self, in_channels=256):
        super().__init__()

        # Tiny head: 3 layers
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, features):
        """Forward pass using multi-scale features"""
        # Combine P3, P4, P5 for depth
        p3 = features['p3']

        x = self.conv1(p3)
        x = self.relu(x)
        depth = self.conv2(x)

        # Upsample to input resolution
        depth = F.interpolate(depth, scale_factor=8, mode='bilinear', align_corners=False)

        return torch.sigmoid(depth)  # Normalize to [0, 1]


class SegmentationHead(nn.Module):
    """
    Tiny segmentation head (3 layers max)
    8 classes: person, sterile_zone, equipment, floor, etc.
    """
    def __init__(self, in_channels=256, num_classes=8):
        super().__init__()
        self.num_classes = num_classes

        # Tiny head: 3 layers
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, features):
        """Forward pass using P3 features"""
        x = features['p3']

        x = self.conv1(x)
        x = self.relu(x)
        seg = self.conv2(x)

        # Upsample to input resolution
        seg = F.interpolate(seg, scale_factor=8, mode='bilinear', align_corners=False)

        return seg


class KeypointsHead(nn.Module):
    """
    Tiny keypoints head (3 layers max)
    21 keypoints per hand (heatmap-based detection)
    """
    def __init__(self, in_channels=256, num_keypoints=21):
        super().__init__()
        self.num_keypoints = num_keypoints

        # Tiny head: 3 layers
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, num_keypoints, kernel_size=1)

    def forward(self, features):
        """Forward pass using P3 features"""
        x = features['p3']

        x = self.conv1(x)
        x = self.relu(x)
        heatmaps = self.conv2(x)

        # Upsample to input resolution
        heatmaps = F.interpolate(heatmaps, scale_factor=8, mode='bilinear', align_corners=False)

        return torch.sigmoid(heatmaps)  # Normalize to [0, 1]


class CLABSIGuard(nn.Module):
    """
    Unified TEO-1 architecture for healthcare monitoring
    Single shared backbone (85-90% params) + 4 tiny heads
    """
    def __init__(self, pretrained=True):
        super().__init__()

        # Shared backbone (must contain 85-90% of parameters)
        self.backbone = SharedBackbone(pretrained=pretrained)

        # 4 tiny prediction heads (2-3 layers each)
        self.detection_head = DetectionHead(in_channels=256, num_classes=4)
        self.depth_head = DepthHead(in_channels=256)
        self.segmentation_head = SegmentationHead(in_channels=256, num_classes=8)
        self.keypoints_head = KeypointsHead(in_channels=256, num_keypoints=21)

    def forward(self, x):
        """
        Forward pass through entire model
        Single backbone pass, parallel head execution
        """
        # Extract multi-scale features once
        features = self.backbone(x)

        # Run all heads in parallel on shared features
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

    def count_parameters(self):
        """Count parameters in backbone vs heads"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        detection_params = sum(p.numel() for p in self.detection_head.parameters())
        depth_params = sum(p.numel() for p in self.depth_head.parameters())
        seg_params = sum(p.numel() for p in self.segmentation_head.parameters())
        kp_params = sum(p.numel() for p in self.keypoints_head.parameters())

        total_params = backbone_params + detection_params + depth_params + seg_params + kp_params
        backbone_pct = (backbone_params / total_params) * 100

        return {
            'backbone': backbone_params,
            'detection_head': detection_params,
            'depth_head': depth_params,
            'segmentation_head': seg_params,
            'keypoints_head': kp_params,
            'total': total_params,
            'backbone_percentage': backbone_pct
        }


def test_model():
    """Test model with dummy input"""
    print("Testing CLABSIGuard model...")

    # Create model
    model = CLABSIGuard(pretrained=True)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Test with dummy input (batch_size=1, channels=3, height=480, width=640)
    dummy_input = torch.randn(1, 3, 480, 640).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)

    print("\nModel outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # Count parameters
    params = model.count_parameters()
    print("\nParameter distribution:")
    print(f"  Backbone: {params['backbone']:,} ({params['backbone_percentage']:.1f}%)")
    print(f"  Detection head: {params['detection_head']:,}")
    print(f"  Depth head: {params['depth_head']:,}")
    print(f"  Segmentation head: {params['segmentation_head']:,}")
    print(f"  Keypoints head: {params['keypoints_head']:,}")
    print(f"  Total: {params['total']:,}")

    # Verify backbone dominance
    if params['backbone_percentage'] >= 85:
        print(f"\nSUCCESS: Backbone contains {params['backbone_percentage']:.1f}% of parameters (target: 85-90%)")
    else:
        print(f"\nWARNING: Backbone only contains {params['backbone_percentage']:.1f}% (target: 85-90%)")

    print("\nModel test completed!")


if __name__ == "__main__":
    test_model()
