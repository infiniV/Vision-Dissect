"""
Feature Adapter for YOLO Backbone
Converts YOLO multi-scale features to uniform 256-channel format
"""
import torch
import torch.nn as nn


class FeatureAdapter(nn.Module):
    """
    Adapts YOLO backbone features to uniform 256-channel format
    Uses 1×1 convolutions for channel adaptation (lightweight)

    Input: YOLO features
      - p3: 256ch @ 60×80
      - p4: 192ch @ 30×40
      - p5: 384ch @ 15×20

    Output: Adapted features
      - p3: 256ch @ 60×80 (no change)
      - p4: 256ch @ 30×40 (expand)
      - p5: 256ch @ 15×20 (compress)
    """
    def __init__(self, in_channels_p3=256, in_channels_p4=192, in_channels_p5=384, out_channels=256):
        super().__init__()

        self.out_channels = out_channels

        # P3: Already 256 channels, no adaptation needed (identity)
        if in_channels_p3 == out_channels:
            self.adapt_p3 = nn.Identity()
        else:
            self.adapt_p3 = nn.Conv2d(in_channels_p3, out_channels, kernel_size=1, bias=False)

        # P4: 192 → 256 channels (expand)
        self.adapt_p4 = nn.Conv2d(in_channels_p4, out_channels, kernel_size=1, bias=False)

        # P5: 384 → 256 channels (compress)
        self.adapt_p5 = nn.Conv2d(in_channels_p5, out_channels, kernel_size=1, bias=False)

        print(f"Feature adapter created:")
        print(f"  P3: {in_channels_p3} -> {out_channels} channels")
        print(f"  P4: {in_channels_p4} -> {out_channels} channels")
        print(f"  P5: {in_channels_p5} -> {out_channels} channels")

    def forward(self, features):
        """
        Adapt features to uniform channel count

        Args:
            features: dict with keys 'p3', 'p4', 'p5'

        Returns:
            dict with adapted features, all with out_channels channels
        """
        return {
            'p3': self.adapt_p3(features['p3']),
            'p4': self.adapt_p4(features['p4']),
            'p5': self.adapt_p5(features['p5'])
        }

    def count_parameters(self):
        """Count parameters in adapter"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total,
            'trainable': trainable
        }


def test_feature_adapter():
    """Test feature adapter"""
    print("Testing Feature Adapter...")

    # Create adapter
    adapter = FeatureAdapter(
        in_channels_p3=256,
        in_channels_p4=192,
        in_channels_p5=384,
        out_channels=256
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adapter = adapter.to(device)

    # Create dummy YOLO features (matching real output from YOLO backbone)
    dummy_features = {
        'p3': torch.randn(1, 256, 60, 80).to(device),
        'p4': torch.randn(1, 192, 30, 40).to(device),
        'p5': torch.randn(1, 384, 15, 20).to(device)
    }

    print("\nInput feature shapes:")
    for key, value in dummy_features.items():
        print(f"  {key}: {value.shape}")

    # Forward pass
    with torch.no_grad():
        adapted = adapter(dummy_features)

    print("\nOutput feature shapes:")
    for key, value in adapted.items():
        print(f"  {key}: {value.shape}")

    # Verify all outputs are 256 channels
    assert adapted['p3'].shape[1] == 256, "P3 should have 256 channels"
    assert adapted['p4'].shape[1] == 256, "P4 should have 256 channels"
    assert adapted['p5'].shape[1] == 256, "P5 should have 256 channels"

    # Count parameters
    params = adapter.count_parameters()
    print(f"\nParameter count:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")

    print("\nFeature adapter test passed!")

    return adapter


if __name__ == "__main__":
    test_feature_adapter()
