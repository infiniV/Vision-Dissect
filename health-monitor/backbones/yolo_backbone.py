"""
YOLO11n Backbone Extractor
Extracts pretrained backbone (layers 0-22) from YOLO11n models
Provides multi-scale features for TEO-1 architecture
"""
import torch
import torch.nn as nn
from ultralytics import YOLO


class YOLOBackbone(nn.Module):
    """
    Pretrained YOLO11n backbone extractor
    Uses forward hooks to extract multi-scale features
    Outputs features at 3 scales for TEO-1 architecture
    """
    def __init__(self, model_path="models/yolo11n-pose.pt", freeze=True):
        super().__init__()

        print(f"Loading YOLO11n backbone from {model_path}...")

        # Load full YOLO model
        yolo_model = YOLO(model_path)

        # Extract PyTorch model
        self.model = yolo_model.model

        # Freeze weights if specified
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            print("Backbone weights frozen")

        # Storage for intermediate features
        self.features = {}

        # Register hooks to capture multi-scale features
        self._register_hooks()

        print(f"YOLO11n backbone extracted successfully")
        print(f"  Trainable: {not freeze}")

    def _register_hooks(self):
        """Register forward hooks to extract multi-scale features"""
        # YOLO11n architecture (from vision-bench analysis):
        # Layer 15: P3 output (high resolution, stride 8)
        # Layer 18: P4 output (medium resolution, stride 16)
        # Layer 21: P5 output (low resolution, stride 32)

        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        # Register hooks at specific layers
        self.model.model[15].register_forward_hook(get_hook('p3'))
        self.model.model[18].register_forward_hook(get_hook('p4'))
        self.model.model[21].register_forward_hook(get_hook('p5'))

        print(f"  Multi-scale feature extraction points:")
        print(f"    P3 (stride 8): layer 15")
        print(f"    P4 (stride 16): layer 18")
        print(f"    P5 (stride 32): layer 21")

    def forward(self, x):
        """
        Forward pass through YOLO model
        Extracts multi-scale features via hooks

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            dict with keys 'p3', 'p4', 'p5' containing features at different scales
        """
        # Clear previous features
        self.features = {}

        # Forward through full model (hooks will capture features)
        _ = self.model(x)

        # Return captured features
        return {
            'p3': self.features['p3'],
            'p4': self.features['p4'],
            'p5': self.features['p5']
        }

    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }


def test_yolo_backbone():
    """Test YOLO backbone extraction"""
    print("Testing YOLO11n backbone extraction...")

    # Create backbone
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = YOLOBackbone("../models/yolo11n-pose.pt", freeze=True)
    backbone = backbone.to(device)
    backbone.eval()

    # Test input
    test_input = torch.randn(1, 3, 480, 640).to(device)

    # Forward pass
    print("\nForward pass test...")
    with torch.no_grad():
        features = backbone(test_input)

    # Print output shapes
    print("\nOutput feature shapes:")
    for key, value in features.items():
        print(f"  {key}: {value.shape}")

    # Count parameters
    params = backbone.count_parameters()
    print(f"\nParameter count:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")

    print("\nBackbone extraction test passed!")

    return backbone, features


if __name__ == "__main__":
    test_yolo_backbone()
