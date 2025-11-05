import onnx
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import torch


def load_models(pt_model_path, export_onnx=True):
    pt_model = YOLO(pt_model_path)
    onnx_path = None

    if export_onnx:
        onnx_path = pt_model.export(format="onnx", imgsz=640)

    return pt_model, onnx_path


def print_architecture(pt_model, onnx_path=None):
    print("\nPyTorch Layers:")
    print("=" * 60)
    for idx, layer in enumerate(pt_model.model.model):
        print(f"  [{idx:2d}] {type(layer).__name__}")
    print(f"Total: {len(pt_model.model.model)}")

    if onnx_path:
        onnx_model = onnx.load(onnx_path)
        print("\nONNX Nodes:")
        print("=" * 60)
        for idx, node in enumerate(onnx_model.graph.node):
            print(
                f"  [{idx:3d}] {node.op_type:15s} in={len(node.input)} out={len(node.output)}"
            )
        print(f"Total: {len(onnx_model.graph.node)}")


def extract_pytorch_layer(pt_model, image_path, layer_idx):
    activation = {}

    def hook(module, input, output):
        if torch.is_tensor(output):
            activation["data"] = output.detach().cpu()
        elif isinstance(output, (tuple, list)):
            activation["data"] = [
                o.detach().cpu() if torch.is_tensor(o) else o for o in output
            ]

    layer = pt_model.model.model[layer_idx]
    handle = layer.register_forward_hook(hook)
    _ = pt_model(image_path, verbose=False)
    handle.remove()

    return activation.get("data"), type(layer).__name__


def preprocess_image(image_path, size=640):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((size, size), Image.BILINEAR)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def run_onnx_inference(onnx_path, image_array):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: image_array})[0]
    return output


def inspect_layer(data, layer_name, layer_idx):
    print(f"\nLayer {layer_idx} ({layer_name}):")
    print("-" * 60)

    if isinstance(data, list):
        print(f"Outputs: {len(data)} tensors")
        for i, tensor in enumerate(data):
            if torch.is_tensor(tensor):
                print(
                    f"  [{i}] shape={tensor.shape} range=[{tensor.min():.3f}, {tensor.max():.3f}]"
                )
    elif torch.is_tensor(data):
        print(f"Shape: {data.shape}")
        print(f"Range: [{data.min():.3f}, {data.max():.3f}]")
    elif isinstance(data, np.ndarray):
        print(f"Shape: {data.shape}")
        print(f"Range: [{data.min():.3f}, {data.max():.3f}]")


def visualize_feature_map(feature_map, title, save_path=None, max_channels=8):
    if isinstance(feature_map, list):
        feature_map = feature_map[0]

    if torch.is_tensor(feature_map) and feature_map.dim() == 4:
        feature_map = feature_map[0]

    if not torch.is_tensor(feature_map):
        feature_map = torch.from_numpy(feature_map)

    if feature_map.dim() != 3:
        print(f"Cannot visualize: expected 3D tensor, got {feature_map.dim()}D")
        return

    num_channels = min(max_channels, feature_map.shape[0])
    rows = 2
    cols = max_channels // rows

    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(max_channels):
        if i < num_channels:
            axes[i].imshow(feature_map[i].numpy(), cmap="viridis")
            axes[i].set_title(f"Ch {i}", fontsize=10)
        else:
            axes[i].axis("off")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


if __name__ == "__main__":
    pt_model, onnx_path = load_models("yolo11n.pt")

    print_architecture(pt_model, onnx_path)

    image_array = preprocess_image("test_image.jpg")

    pt_layer_data, pt_layer_name = extract_pytorch_layer(pt_model, "test_image.jpg", 22)
    inspect_layer(pt_layer_data, pt_layer_name, 22)
    visualize_feature_map(pt_layer_data, "PyTorch Layer 22", "viz/pytorch_layer_22.png")

    pt_layer_data, pt_layer_name = extract_pytorch_layer(pt_model, "test_image.jpg", 23)
    inspect_layer(pt_layer_data, pt_layer_name, 23)
    visualize_feature_map(pt_layer_data, "PyTorch Layer 23", "viz/pytorch_layer_23.png")

    onnx_output = run_onnx_inference(onnx_path, image_array)
    inspect_layer(onnx_output, "Final Output", 319)
