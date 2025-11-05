import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO, SAM


def extract_layer(model, image_path, layer_idx):
    activation = {}

    def hook(module, input, output):
        if torch.is_tensor(output):
            activation["data"] = output.detach().cpu()
        elif isinstance(output, (tuple, list)):
            activation["data"] = (
                output[0].detach().cpu() if torch.is_tensor(output[0]) else None
            )

    layer = model.model.model[layer_idx]
    handle = layer.register_forward_hook(hook)

    model(image_path, verbose=False)
    handle.remove()

    return activation.get("data")


def visualize_models(image_path="test_image.jpg", layer_idx=7, max_channels=8):
    models = {
        "v11n": "yolo11n.pt",
        "v11n-seg": "yolo11n-seg.pt",
        "v11n-pose": "yolo11n-pose.pt",
    }

    fig, axes = plt.subplots(len(models), max_channels, figsize=(16, len(models) * 2))

    for row, (name, model_path) in enumerate(models.items()):
        print(f"\nProcessing {name}...")
        model = YOLO(model_path)

        num_layers = len(model.model.model)
        print(f"  Total layers: {num_layers}")
        print(f"  Layer types: {[type(layer).__name__ for layer in model.model.model]}")
        print(f"  Extracting layer {layer_idx}")

        feature_map = extract_layer(model, image_path, layer_idx)

        if feature_map is None:
            print(f"  Skipping: No feature map extracted")
            continue

        if feature_map.dim() != 4:
            print(f"  Skipping: Output is {feature_map.dim()}D (expected 4D)")
            for col in range(max_channels):
                ax = axes[row, col] if len(models) > 1 else axes[col]
                ax.axis("off")
            continue

        feature_map = feature_map[0]
        num_channels = min(feature_map.shape[0], max_channels)
        print(f"  Feature map shape: {feature_map.shape}")

        for col in range(max_channels):
            ax = axes[row, col] if len(models) > 1 else axes[col]

            if col < num_channels:
                channel = feature_map[col].numpy()
                ax.imshow(channel, cmap="viridis")
                ax.set_title(f"Ch {col}", fontsize=10)
                if col == 0:
                    ax.set_ylabel(
                        name,
                        rotation=0,
                        ha="right",
                        va="center",
                        fontsize=12,
                        weight="bold",
                    )

            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(f"YOLOv11 Models - Layer {layer_idx} Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig("viz/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: model_comparison.png")


def visualize_outputs(image_path="test_image.jpg"):
    models = {
        "Detection": "yolo11n.pt",
        "Segmentation": "yolo11n-seg.pt",
        "Pose": "yolo11n-pose.pt",
    }

    fig, axes = plt.subplots(1, len(models), figsize=(18, 6))

    for idx, (name, model_path) in enumerate(models.items()):
        print(f"Processing {name}...")
        model = YOLO(model_path)

        results = model(image_path, verbose=False)[0]
        plotted = results.plot()

        ax = axes[idx]
        ax.imshow(plotted[..., ::-1])
        ax.set_title(name, fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("viz/model_outputs.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: model_outputs.png")


# Layers 0-9: Backbone (feature extraction, downsampling)
# Layers 10-16: Neck (feature fusion, upsampling)
# Layers 17-22: Additional feature processing
# Layer 23: Detection/Segmentation/Pose head
if __name__ == "__main__":
    visualize_models(image_path="test_image.jpg", layer_idx=22, max_channels=3)
    visualize_outputs(image_path="test_image.jpg")
